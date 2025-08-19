import os
import copy

from prices import get_price
from meshinspector import get_mesh_info
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Union
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PyPDF2 import PdfReader, PdfWriter

# Page size for initial layout guesses
pw, ph = A4

# ---- HYPERPARAMETERS (tune these) ----
LAYOUT = {
    # Positions (top blocks)
    "CLIENT_INFO_POSITION": (pw - 5 * mm, ph - 34 * mm),  # right-edge X, top Y (right-aligned text)
    "INVOICE_INFO_POSITION": (5 * mm, ph - 34 * mm),  # left-aligned (we do NOT draw the "Invoice" heading)

    # Column X positions
    "ITEM_X_POSITION": 8 * mm,
    "DESCRIPTION_X_POSITION": 23 * mm,
    "QUANTITY_X_POSITION": 105* mm,
    "PRICE_X_POSITION": 142 * mm,  # also VAT/TOTAL text
    "AMOUNT_X_POSITION": 193 * mm,  # also VAT/TOTAL amounts (right-aligned)

    # First item Y and vertical separations
    "FIRST_ITEM_Y_POSITION": ph - 112 * mm,
    "ITEM_Y_SEPARATION": 25 * mm,  # distance between top lines of successive items
    "LAST_ITEM_VAT_Y_SEPARATION": 15 * mm,  # gap from last item to VAT
    "VAT_TOTAL_Y_SEPARATION": 6 * mm,  # gap from VAT to TOTAL
    "TOTAL_BANK_INFO_YSEPARATION": 18 * mm,  # gap from TOTAL to banking info

    # Banking info X
    "BANK_INFO_X_POSITION": 20 * mm,

    # Pagination cutoff (start new page before drawing an item that would drop below this Y)
    "PAGE_BOTTOM_Y_CUTOFF": 80 * mm,

    # Optional: detail line leading and wrapping for single-line descriptions
    "DETAIL_LINE_LEADING": 11,  # px; if omitted uses FONT_SIZE_MAIN+2
    "WRAP_CHARS": 70,

    # Fonts (paths or built-in names). If paths are invalid, falls back to Helvetica.
    "MAIN_FONT_REGULAR": None,  # e.g., "/path/to/Main-Regular.otf"
    "MAIN_FONT_BOLD": None,  # e.g., "/path/to/Main-Bold.otf"
    "ACCENT_FONT_REGULAR": None,  # e.g., "/path/to/Accent-Regular.otf"
    "ACCENT_FONT_BOLD": None,  # e.g., "/path/to/Accent-Bold.otf"

    # Sizes
    "FONT_SIZE_MAIN": 9,
    "FONT_SIZE_SMALL": 8,
}

# -------- Convenience aliases for layout, DO NOT touch, modify from the dict LAYOUT --------
INV_X, INV_Y = LAYOUT["INVOICE_INFO_POSITION"]
CLI_X_RIGHT, CLI_Y_TOP = LAYOUT["CLIENT_INFO_POSITION"]  # right-aligned

X_ITEM = LAYOUT["ITEM_X_POSITION"]
X_DESC = LAYOUT["DESCRIPTION_X_POSITION"]
X_QTY = LAYOUT["QUANTITY_X_POSITION"]
X_PRICE = LAYOUT["PRICE_X_POSITION"]  # also VAT/TOTAL text
X_AMOUNT = LAYOUT["AMOUNT_X_POSITION"]  # also VAT/TOTAL amounts (right aligned)

Y_FIRST_ITEM = LAYOUT["FIRST_ITEM_Y_POSITION"]
SEP_ITEM = LAYOUT["ITEM_Y_SEPARATION"]
SEP_LAST_VAT = LAYOUT["LAST_ITEM_VAT_Y_SEPARATION"]
SEP_VAT_TOTAL = LAYOUT["VAT_TOTAL_Y_SEPARATION"]
SEP_TOTAL_BANK = LAYOUT["TOTAL_BANK_INFO_YSEPARATION"]

BANK_X = LAYOUT["BANK_INFO_X_POSITION"]
BANK_LINES = LAYOUT.get("BANK_INFO_LINES", [
    "ACCOUNT #: 90590803",
    "SORT CODE: 04-00-03",
    "IBAN: GB10 MONZ 0400 0390 5058 03",
    "BIC: MONZGB21",
])

FS_MAIN = int(LAYOUT.get("FONT_SIZE_MAIN", 9))
FS_SMALL = int(LAYOUT.get("FONT_SIZE_SMALL", FS_MAIN))
DETAIL_LEADING = float(LAYOUT.get("DETAIL_LINE_LEADING", FS_MAIN + 2))
ITEM_LEADING = float(LAYOUT.get("ITEM_Y_SEPARATION", FS_MAIN + 6))
WRAP_CHARS = int(LAYOUT.get("WRAP_CHARS", 70))

# Pagination cutoff: if the next row would go below this Y, start a new page
PAGE_BOTTOM_Y_CUTOFF = float(LAYOUT.get("PAGE_BOTTOM_Y_CUTOFF", 80 * mm))

# Paths
BASE_DIR = os.path.dirname(__file__)
BLANK_PDF = os.path.join(BASE_DIR, "config/templates/blank_invoice.pdf")
OUTPUT_PDF = os.path.join(BASE_DIR, "example_generated_invoice.pdf")


# -----------------------------
# Helpers
# -----------------------------
def _resolve_font(alias_name: str, layout_value: Optional[str], bold: bool = False) -> str:
    """
    Returns a font name that ReportLab can use with setFont().
    - If layout_value is a path to .otf/.ttf and exists: register under alias_name and return alias_name.
    - If layout_value is a built-in font name (e.g. 'Helvetica', 'Times-Roman'): return it directly.
    - Otherwise, return a safe built-in fallback (Helvetica or Helvetica-Bold).
    """
    # 1) If a valid font file path was provided -> register to alias and use alias
    if layout_value and isinstance(layout_value, str) and os.path.isfile(layout_value):
        pdfmetrics.registerFont(TTFont(alias_name, layout_value))
        return alias_name

    # 2) If user provided a built-in font name, just use it
    if layout_value and isinstance(layout_value, str) and layout_value in pdfmetrics.standardFonts:
        return layout_value

    # 3) Fallback to safe built-ins
    return "Helvetica-Bold" if bold else "Helvetica"


def _draw_right_aligned(c: canvas.Canvas, x_right: float, y: float, text: str, font: str, size: int):
    c.setFont(font, size)
    tw = c.stringWidth(text, font, size)
    c.drawString(x_right - tw, y, text)


def _draw_lines(c: canvas.Canvas, x: float, y_start: float, lines: List[str], leading: float, font: str, size: int):
    """
    Draw lines downward (decreasing y). Returns the y below the last line.
    """
    c.setFont(font, size)
    y = y_start
    for line in lines:
        if line is not None and str(line).strip() != "":
            c.drawString(x, y, str(line))
            y -= leading
    return y


def _draw_lines_right(c: canvas.Canvas,
                      x_right: float,
                      y_start: float,
                      lines: List[str],
                      leading: float,
                      font: str,
                      size: int):
    c.setFont(font, size)
    y = y_start
    for line in lines:
        if line is not None and str(line).strip() != "":
            _draw_right_aligned(c, x_right, y, str(line), font, size)
            y -= leading
    return y


def _currency(n: float) -> str:
    return f"{n:,.2f}"


def _wrap_text(text: str, max_chars: int) -> List[str]:
    """Simple greedy wrapper by character count (works fine for compact invoices)."""
    if text is None:
        return []
    s = str(text).strip()
    if not s or max_chars <= 0:
        return [s]
    words = s.split()
    lines: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur_len + add <= max_chars:
            cur.append(w)
            cur_len += add
        else:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
    if cur:
        lines.append(" ".join(cur))
    return lines


# -----------------------------
# Core generator
# -----------------------------
def generate_invoice(data: Dict[str, Any],
                     blank_pdf_path: str,
                     output_path: str,
                     due_in: int = 7):
    """
    data: {
      "customer_info": {...},
      "invoice_ref": "...",
      "date": <datetime or str>,
      "items": { "1": { "description": dict|str, "quantity": float, "price": float }, ... },
      "vat": 0.2
    }
    """

    # -------- Fonts (from LAYOUT) --------
    main_reg = _resolve_font("MainFont", LAYOUT.get("MAIN_FONT_REGULAR"), bold=False)
    main_bold = _resolve_font("MainFont-Bold", LAYOUT.get("MAIN_FONT_BOLD"), bold=True)
    acc_reg = _resolve_font("AccentFont", LAYOUT.get("ACCENT_FONT_REGULAR"), bold=False)
    acc_bold = _resolve_font("AccentFont-Bold", LAYOUT.get("ACCENT_FONT_BOLD"), bold=True)

    # -------- Load page size from blank --------
    reader_blank = PdfReader(blank_pdf_path)
    first_page = reader_blank.pages[0]
    page_w = float(first_page.mediabox.width)
    page_h = float(first_page.mediabox.height)

    # -------- Parse dates --------
    dt = data["date"]
    if isinstance(dt, str):
        parsed = None
        for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
            try:
                parsed = datetime.strptime(dt, fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            raise ValueError("Unrecognized date format for 'date'. Try 'DD-MM-YYYY'.")
        dt = parsed
    due_dt = dt + timedelta(days=due_in)

    # -------- Prepare items (sorted by numeric key) --------
    def _key_int(k: str) -> int:
        try:
            return int(k)
        except Exception:
            return 10**9

    items_sorted = [(k, data["items"][k]) for k in sorted(data["items"].keys(), key=_key_int)]

    # -------- Overlay canvas setup --------
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    overlay_path = os.path.join(os.path.dirname(output_path) or ".", "_overlay_tmp.pdf")
    c = canvas.Canvas(overlay_path, pagesize=(page_w, page_h))

    # -------- Pagination: split items into pages --------
    def estimate_item_height(it: Dict[str, Any]) -> float:
        """Estimate vertical space consumed by one item block."""
        h = 0.0
        # Title line
        h += ITEM_LEADING
        # If dict description, add 4 detail lines
        desc = it.get("description")
        if isinstance(desc, dict):
            h += 4 * DETAIL_LEADING
        return h

    pages: List[List[Tuple[str, Dict[str, Any]]]] = []
    cur_page: List[Tuple[str, Dict[str, Any]]] = []
    cur_y = Y_FIRST_ITEM

    for k, it in items_sorted:
        need_h = estimate_item_height(it) + LAYOUT.get("ROW_GAP", 0.0)
        # If this item would cross the bottom cutoff, start a new page
        if cur_page and (cur_y - need_h < PAGE_BOTTOM_Y_CUTOFF):
            pages.append(cur_page)
            cur_page = []
            cur_y = Y_FIRST_ITEM
        cur_page.append((k, it))
        cur_y -= need_h
    if cur_page:
        pages.append(cur_page)

    # -------- Draw pages --------
    subtotal = 0.0

    def draw_header_info():
        """Draws invoice & client blocks (NOT the headings)."""
        # Invoice info (left-aligned)
        c.setFont(main_reg, FS_MAIN)
        c.drawString(INV_X, INV_Y, f"Invoice #: {data['invoice_ref']}")
        c.drawString(INV_X, INV_Y - (FS_MAIN + 2), f"Date: {dt.strftime('%d/%m/%Y')}")
        c.drawString(INV_X, INV_Y - 2*(FS_MAIN + 2), f"Due: {due_dt.strftime('%d/%m/%Y')}")

        # Client info (RIGHT-aligned to CLI_X_RIGHT)
        cust = data.get("customer_info", {})
        lines = [
            cust.get("name", ""),
            cust.get("VAT_number") if cust.get("VAT_number") else "",
            cust.get("address_line_1", ""),
            cust.get("address_line_2", "") if cust.get("address_line_2", "") else "",
            " ".join(v for v in [cust.get("city", ""), cust.get("post_code", "")] if v) + ", " + cust.get("country", "")
        ]
        _draw_lines_right(c, CLI_X_RIGHT, CLI_Y_TOP, [ln for ln in lines if ln], FS_MAIN + 2, main_reg, FS_MAIN)

    for page_idx, page_items in enumerate(pages):
        # Header info on each page
        draw_header_info()

        # Draw items on this page
        y = Y_FIRST_ITEM
        for (k, it) in page_items:
            desc = it.get("description")
            qty = float(it.get("quantity", 0.0))
            price = float(it.get("price", 0.0))
            amount = qty * price
            subtotal += amount

            # Left cols
            c.setFont(main_reg, FS_MAIN)
            c.drawString(X_ITEM, y, str(k))

            if isinstance(desc, dict):
                # Title (bold)
                title = desc.get("project_name", "")
                c.setFont(main_bold, FS_MAIN)
                c.drawString(X_DESC, y, title)
                # Details (regular)
                meta = [
                    f"Size: {desc.get('size','')}",
                    f"Bounding vol.: {desc.get('bounding_vol','')} L",
                    f"Surface: {desc.get('surface','')} m²",
                    f"Weight: {desc.get('weight','')} kg",
                ]
                _draw_lines(c, X_DESC, y - (FS_MAIN + 2), meta, DETAIL_LEADING, main_reg, FS_MAIN)
            else:
                # Single-line (wrap if needed)
                c.setFont(main_bold, FS_MAIN)
                for i, line in enumerate(_wrap_text(str(desc), WRAP_CHARS)):
                    c.drawString(X_DESC, y - i * DETAIL_LEADING, line)

            # Qty / Price / Amount (amount right-aligned)
            c.setFont(main_reg, FS_MAIN)
            c.drawString(X_QTY, y, f"{qty:g}")
            c.drawString(X_PRICE, y, _currency(price))
            _draw_right_aligned(c, X_AMOUNT, y, _currency(amount), main_reg, FS_MAIN)

            # Advance Y to next row
            y -= SEP_ITEM

        # If last page, add VAT/TOTAL and banking info
        is_last = (page_idx == len(pages) - 1)
        if is_last:
            vat_amount = subtotal * float(data.get("vat", 0.0))
            total_amount = subtotal + vat_amount

            y_vat = y - SEP_LAST_VAT
            y_total = y_vat - SEP_VAT_TOTAL
            y_bank = y_total - SEP_TOTAL_BANK

            c.setFont(acc_bold, FS_MAIN)
            c.drawString(X_PRICE, y_vat, f"VAT ({int(round(float(data['vat'])*100))}%)")
            _draw_right_aligned(c, X_AMOUNT, y_vat, _currency(vat_amount), acc_bold, FS_MAIN)

            c.drawString(X_PRICE, y_total, "TOTAL")
            _draw_right_aligned(c, X_AMOUNT, y_total, _currency(total_amount), acc_bold, FS_MAIN)

            # Banking info (accent regular)
            c.setFont(acc_reg, FS_MAIN)
            _draw_lines(c, BANK_X, y_bank, BANK_LINES, FS_MAIN + 2, acc_bold, FS_MAIN)

        c.showPage()

    c.save()

    # -------- Merge overlay with blank (PyPDF2) --------
    reader_overlay = PdfReader(overlay_path)
    writer = PdfWriter()

    blank_count = len(reader_blank.pages)
    overlay_count = len(reader_overlay.pages)

    for i in range(overlay_count):
        base = reader_blank.pages[min(i, blank_count - 1)]
        # Deep-copy so we don't mutate the original template page across iterations
        page = copy.deepcopy(base)
        page.merge_page(reader_overlay.pages[i])
        writer.add_page(page)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f_out:
        writer.write(f_out)

    try:
        os.remove(overlay_path)
    except FileNotFoundError:
        pass


def get_invoice_info(customer_info: dict,
                     invoice_ref: str,
                     item_info: Union[List[str], Dict[str, dict]],
                     date: Union[datetime, None]=None,
                     shipping_cost: float = 0,
                     vat: float = 0.20,
                     quantity_list: Optional[List[int]] = None,
                     override_price_list: Optional[List[float]] = None) -> dict:
    """
    Generate structured invoice information for 3D-printed items.

    :param customer_info: Dictionary containing customer details such as
        name, address, and contact information.
    :param invoice_ref: Unique invoice reference string.
    :param item_info: Either:
        - **List[str]**: list of mesh URLs. Each entry will be processed
          into an item description using `get_mesh_info`.
        - **Dict[str, dict]**: pre-built mapping of item IDs to item details
          (with keys: description, quantity, price).
    :param date: Optional invoice date. If ``None``, defaults to
        ``datetime.now()``.
    :param shipping_cost: Shipping cost to be added as a separate line item.
        Defaults to 0.
    :param vat: VAT rate applied to the invoice.
        Expressed as a decimal fraction (e.g., 0.20 for 20%).
    :param quantity_list: Optional list of integers specifying quantities
        corresponding to `item_info` entries (only when `item_info` is a list).
        Must match the number of items. Defaults to 1 for each item.
    :param override_price_list: Optional list of floats specifying per-item
        unit prices corresponding to `item_info` entries (only when `item_info`
        is a list). Must match the number of items. If not provided, price is
        computed from mesh weight.
    :return: Dictionary containing the complete invoice structure with keys:
        - ``customer_info``: customer details (dict)
        - ``invoice_ref``: invoice reference (str)
        - ``date``: invoice date (datetime)
        - ``items``: dict of line items, where each entry includes
          description, quantity, and price
        - ``vat``: VAT rate applied (float)
    """
    n_items = len(item_info)
    if isinstance(item_info, dict):
        items = item_info
    else:
        if quantity_list is not None:
            assert len(quantity_list) == n_items, "quantity_list length must match item_list"
        if override_price_list is not None:
            assert len(override_price_list) == n_items, "override_price_list length must match item_list"
        quantity_list = quantity_list if quantity_list is not None else [1] * n_items
        items = {}
        for i, url in enumerate(item_info):
            description = get_mesh_info(mesh_url=url)
            price = (override_price_list[i] if override_price_list is not None
                     else get_price(weight=description["weight"]))
            items[str(i + 1)] = {"description": description,
                                 "quantity": quantity_list[i],
                                 "price": price}
    items[str(n_items + 1)] = {"description": "Shipping",
                               "quantity": 1,
                               "price": shipping_cost}
    if date is None:
        date = datetime.now()
    return {"customer_info": customer_info,
            "invoice_ref": invoice_ref,
            "date": date,
            "items": items,
            "vat": vat}


# Example usage
if __name__ == "__main__":
    # Example data (replace with your inputs)
    customer_info = {"name": "Ben Dover",
                     "VAT_nr": "VT123456789",
                     "address_line_1": "Example Ave.",
                     "address_line_2": None,
                     "city": "London",
                     "country": "UK",
                     "post_code": "AB1C 2DE"}
    # Example item list (mesh URLs – replace with your real ones)
    mesh_urls = ["https://example.com/meshes/church_hollow.stl"]
    # Generate invoice info from meshes
    invoice_info = get_invoice_info(customer_info=customer_info,
                                    invoice_ref="THC20250716",
                                    item_info=mesh_urls,
                                    date=datetime(2025, 7, 16),
                                    shipping_cost=94.5,
                                    vat=0.35,
                                    quantity_list=[1, 1],  # quantities must match items
                                    override_price_list=[894.5, 94.5])  # override prices if known
    # Write invoice PDF
    generate_invoice(invoice_info, BLANK_PDF, OUTPUT_PDF)
    print(f"Invoice written to: {OUTPUT_PDF}")

    sample = {
        "customer_info": customer_info,
        "invoice_ref": "THC20250716",
        "date": "16-07-2025",
        "items": {
            "1": {
                "description": {
                    "project_name": "250709-Church 1_50_Hollow",
                    "size": "468x468x440 mm",
                    "bounding_vol": 97.25,
                    "surface": 2.14,
                    "weight": 27.1,
                },
                "quantity": 1.0,
                "price": 894.5,
            },
            "2": {
                "description": "packaging",
                "quantity": 1.0,
                "price": 94.5
            }
        },
        "vat": 0.35,
    }
    generate_invoice(sample, BLANK_PDF, OUTPUT_PDF)
    print(f"Invoice written to: {OUTPUT_PDF}")