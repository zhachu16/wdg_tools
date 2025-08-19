import math
from pathlib import Path
from datetime import datetime
from typing import Union, Iterable, Tuple, Dict

import numpy as np
import pandas as pd
import yfinance as yf
from pandas import DataFrame

PRINTING_FEE_RATE = 1.5
SHIPPING_FEE_RATE = 1.0
BASE_TRANSACTION_FEE = 0.30
TRANSACTION_FEE_RATE = 0.029
UNIT_PRINTING_COST = 50.0

# --- Paths: cross-platform ---
CONFIG_DIR = Path("config")
SHIPPING_FILE = CONFIG_DIR / "shipping_price_brackets.csv"
MATERIALS_FILE = CONFIG_DIR / "material_catalogue.csv"

# Load shipping table
SHIPPING_DATA = np.loadtxt(SHIPPING_FILE.as_posix(), delimiter=",")
SHIPPING_UPPER_BOUND = SHIPPING_DATA[0, :]*1e-3
SHIPPING_COSTS = SHIPPING_DATA[1, :]

DEFAULT_DENSITY = 1.5e-6

BASE_CURRENCY = "GBP"
COST_CURRENCIES = {"SHIPPING_COST": "USD",
                   "PRINTING_COST": "CNY",
                   "TRANSACTION_COST": "USD"}
_FX_CACHE: Dict[str, float] = {}


def _normalize_weights(weight: Union[float, int, Iterable[float]]) -> np.ndarray:
    """Accept scalar int/float or iterable; return 1D float array."""
    if isinstance(weight, (int, float)) and not isinstance(weight, bool):
        return np.array([float(weight)], dtype=float)
    arr = np.asarray(list(weight), dtype=float)
    if arr.ndim != 1:
        raise ValueError("weight must be a scalar or 1D iterable of numbers.")
    return arr


def _get_customized_costs(n_items: int, indices: Union[int, Iterable[int]],
                          values: Union[float, Iterable[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (custom_values_array, default_mask)
    - custom_values_array: shape (n_items,), np.nan where default should be computed
    - default_mask: True where default should be computed; False where a custom value is provided
    """
    custom = np.full(n_items, np.nan, dtype=float)
    if isinstance(indices, int):
        vals = [values] if isinstance(values, (int, float)) else list(values)
        if len(vals) != 1:
            raise ValueError("Single index provided but multiple values given.")
        custom[indices] = float(vals[0])
    else:
        idxs = list(indices)
        vals = list(values) if not isinstance(values, (int, float)) else [values]
        if len(idxs) != len(vals):
            raise ValueError("indices and custom_values must have the same length.")
        for i, v in zip(idxs, vals):
            custom[i] = float(v)
    default_mask = np.isnan(custom)  # True means: compute default for these rows
    return custom, default_mask


def get_fx_rates() -> Dict[str, float]:
    """
    Get fresh FX rates with caching (updates once per hour).
    Returns a dict mapping currency code -> rate in BASE_CURRENCY.
    Example: {'USD': 0.78, 'CNY': 0.11}
    """
    now = datetime.now()
    last = _FX_CACHE.get("last_updated")
    if (not last) or (now - last).total_seconds() > 3600:
        try:
            for currency in set(COST_CURRENCIES.values()):
                if currency == BASE_CURRENCY:
                    _FX_CACHE[currency] = 1.0
                else:
                    ticker = f"{currency}{BASE_CURRENCY}=X"
                    info = yf.Ticker(ticker).fast_info
                    rate = float(info["last_price"])
                    if not (rate > 0 and math.isfinite(rate)):
                        raise ValueError(f"Invalid FX rate for {ticker}: {rate}")
                    _FX_CACHE[currency] = rate
            _FX_CACHE["last_updated"] = now
        except Exception as e:
            raise RuntimeError(f"FX update failed. Error: {e}") from e

    # Return only currency keys
    return {k: v for k, v in _FX_CACHE.items() if k not in {"last_updated"}}


def get_shipping_cost_estimate(weight: Union[float, int, Iterable[float]]) -> np.ndarray:
    """
    Calculates shipping cost based on weight using predefined price brackets.
    Always returns a numpy array (scalar inputs return shape (1,) arrays).
    """
    if np.isscalar(weight):
        idx = np.searchsorted(SHIPPING_UPPER_BOUND, weight, side="right")
        return np.array([SHIPPING_COSTS[idx]]) if idx < len(SHIPPING_COSTS) else np.array([1000.0])
    weight_arr = np.asarray(weight, dtype=float)
    idx = np.searchsorted(SHIPPING_UPPER_BOUND, weight_arr, side="right")
    idx_clipped = np.clip(idx, 0, len(SHIPPING_COSTS) - 1)
    results = SHIPPING_COSTS[idx_clipped]
    results[idx >= len(SHIPPING_COSTS)] = 1000.0
    return results


def get_price(weight: Union[float, int, Iterable[float]],
              override_printing_cost: Union[np.ndarray, list, None] = None,
              override_shipping_cost: Union[np.ndarray, list, None] = None,
              override_other_cost: Union[np.ndarray, list, None] = None) -> DataFrame:
    """
    Calculate itemized costs (printing, shipping, other) and total costs,
    converted into the BASE_CURRENCY using live FX rates.

    :param weight: Single value or iterable of floats/ints representing
        the weight(s) of the item(s). If a scalar is provided, it will be
        broadcast into a single-element array.
        Units must be consistent with `UNIT_PRINTING_COST` and
        `get_shipping_cost_estimate`.

    :param override_printing_cost: Optional. Custom printing costs for
        selected items. Must be a pair of lists/arrays:
        ``[indices, values]``. Example:
        ``[[0, 2], [42.0, 69.0]]`` sets item 0 to 42.0 and item 2 to 69.0,
        while other items fall back to the computed default formula:
        ``UNIT_PRINTING_COST * weight * FX * (1 + PRINTING_FEE_RATE) * (1 + TRANSACTION_FEE_RATE)``.

    :param override_shipping_cost: Optional. Custom shipping costs for
        selected items, same format as `override_printing_cost`.
        Items not overridden use the default formula:
        ``get_shipping_cost_estimate(weight) * FX * (1 + SHIPPING_FEE_RATE) * (1 + TRANSACTION_FEE_RATE)``.

        To remove shipping entirely, set all overrides to zero, e.g.:
        ``[list(range(n_items)), [0.0] * n_items]``.

    :param override_other_cost: Optional. Custom miscellaneous/transaction
        costs for selected items. Same structure as above. If not provided,
        only the last item gets a default transaction fee:
        ``BASE_TRANSACTION_FEE * FX``.

    :return: pandas.DataFrame with the following columns:
        - ``printing_cost``: per-item printing costs
        - ``shipping_cost``: per-item shipping costs
        - ``other_cost``: per-item other costs
        - ``total_cost``: row-wise sum of the three categories

    :notes:
        - All values are converted into BASE_CURRENCY using live FX rates.
        - Overrides let you selectively replace computed costs for specific items.
        - Useful for discounts, waived shipping, or known fixed costs.
    """
    fx_rates = get_fx_rates()

    weight_arr = _normalize_weights(weight)
    n_items = weight_arr.size

    printing_cost = np.zeros(n_items, dtype=float)
    shipping_cost = np.zeros(n_items, dtype=float)
    other_cost = np.zeros(n_items, dtype=float)

    # --- Printing ---
    if override_printing_cost is not None:
        custom_vals, default_mask = _get_customized_costs(n_items=n_items,
                                                          indices=override_printing_cost[0],
                                                          values=override_printing_cost[1])
        # Compute defaults where no custom value was given
        printing_cost[default_mask] = (UNIT_PRINTING_COST * weight_arr[default_mask]
                                       * fx_rates[COST_CURRENCIES["PRINTING_COST"]]
                                       * (1 + PRINTING_FEE_RATE)
                                       * (1 + TRANSACTION_FEE_RATE))
        # Use provided custom values
        printing_cost[~default_mask] = custom_vals[~default_mask]
    else:
        printing_cost = (UNIT_PRINTING_COST * weight_arr
                         * fx_rates[COST_CURRENCIES["PRINTING_COST"]]
                         * (1 + PRINTING_FEE_RATE)
                         * (1 + TRANSACTION_FEE_RATE))

    # --- Shipping ---
    if override_shipping_cost is not None:
        custom_vals, default_mask = _get_customized_costs(n_items=n_items,
                                                          indices=override_shipping_cost[0],
                                                          values=override_shipping_cost[1])
        base_ship = get_shipping_cost_estimate(weight_arr)
        shipping_cost[default_mask] = (base_ship[default_mask]
                                       * fx_rates[COST_CURRENCIES["SHIPPING_COST"]]
                                       * (1 + SHIPPING_FEE_RATE)
                                       * (1 + TRANSACTION_FEE_RATE))
        shipping_cost[~default_mask] = custom_vals[~default_mask]
    else:
        shipping_cost = (get_shipping_cost_estimate(weight_arr)
                         * fx_rates[COST_CURRENCIES["SHIPPING_COST"]]
                         * (1 + SHIPPING_FEE_RATE)
                         * (1 + TRANSACTION_FEE_RATE))

    # --- Other / transaction fee ---
    if override_other_cost is not None:
        custom_vals, default_mask = _get_customized_costs( n_items=n_items,
                                                           indices=override_other_cost[0],
                                                           values=override_other_cost[0])
        other_cost[~default_mask] = custom_vals[~default_mask]
    else:
        other_cost[-1] = BASE_TRANSACTION_FEE * fx_rates[COST_CURRENCIES["TRANSACTION_COST"]]

    df = pd.DataFrame({"printing_cost": printing_cost,
                       "shipping_cost": shipping_cost,
                       "other_cost": other_cost,
                       'total_cost': printing_cost + shipping_cost + other_cost})
    return df

# example usage
if __name__ == "__main__":
    price = get_price(weight=[1.3,5.6,2.5])
    override_price = get_price(weight=[1.3, 5.6, 2.5], override_printing_cost=[[0,2],[42,69]],
                               override_shipping_cost=[np.arange(3).tolist(),
                                                       np.zeros(3).tolist()])
    print(override_price)
    single_item_price = get_price(weight=3.14)
