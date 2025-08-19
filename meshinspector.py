import numpy as np
import trimesh

from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from trimesh.proximity import ProximityQuery

from tools import get_mesh_from_url

VOXEL_SIZE = 10.0
SIZE_LARGE = 1000.0
SIZE_MEDIUM = 500.0
MIN_WALL_THICKNESS_SMALL = 10.0
MIN_WALL_THICKNESS_MEDIUM = 50.0
MIN_WALL_THICKNESS_LARGE = 150.0
MAX_PROTRUSION_ASPECT_RATIO = 1.0
MAX_PROTRUSION_FRAGILE_RADIUS = 5.0
CURVATURE_THRESHOLD = 50.0
DESNSITY = 1.5e-6 # kg/mm3
UNIT_PRICE = 15.0 # in GBP/kg

DEFAULT_DENSITY = 1.5  # g/cm3

def inspect_mesh(mesh_url: str,
                 voxel_size=VOXEL_SIZE,
                 min_wall_thickness_small=MIN_WALL_THICKNESS_SMALL,
                 min_wall_thickness_medium=MIN_WALL_THICKNESS_MEDIUM,
                 min_wall_thickness_large=MIN_WALL_THICKNESS_LARGE,
                 size_medium=SIZE_MEDIUM,
                 size_large=SIZE_LARGE,
                 max_protrusion_fragile_radius=MAX_PROTRUSION_FRAGILE_RADIUS,
                 max_protrusion_aspect_ratio=MAX_PROTRUSION_ASPECT_RATIO,
                 curvature_threshold=CURVATURE_THRESHOLD):
    """
    Check if a mesh is 3D printable based on:
    - Wall thickness (hybrid EDT + raycasting verification)
    - Fragile protrusions (skeletonization with aspect ratio checks)
    - Structural fragility (high curvature regions)

    :param mesh: trimesh.mesh Input mesh.
    :param voxel_size: float Resolution for voxelization (mm).
    :param min_wall_thickness_small: float Minimum allowed wall thickness (mm) for a small-sized mesh.
    :param min_wall_thickness_medium: float Minimum allowed wall thickness (mm) for a medium-sized mesh.
    :param min_wall_thickness_large: float Minimum allowed wall thickness (mm) for a large-sized mesh.
    :param size_medium: float Lower bound in size (mm) for a mesh to be classified as medium-sized.
    :param size_large: float Lower bound in size (mm) for a mesh to be classified as large-sized.
    :param max_protrusion_fragile_radius: float Maximum radius (mm) to consider for fragile protrusions.
    :param max_protrusion_aspect_ratio: float Max L/R ratio for protrusions.
    :param curvature_threshold: float Curvature value to flag fragile tips.
    :return: tuple (is_printable: bool, info: dict) Printability status and info regarding printability.
    """
    mesh = get_mesh_from_url(mesh_url)
    is_printable = True
    info = {}

    # --- Size Classification ---
    max_dimension = max(mesh.bounding_box.extents)
    if max_dimension > size_large:
        min_wall_thickness = min_wall_thickness_large
    elif max_dimension >= size_medium:
        min_wall_thickness = min_wall_thickness_medium
    else:
        min_wall_thickness = min_wall_thickness_small

    # --- Voxelization ---
    try:
        voxels = mesh.voxelized(pitch=voxel_size).fill()
        voxel_grid = voxels.matrix.astype(np.uint8)

    except Exception as e:
        return {"is_printable": False,
                "info": {"Error": f"Voxelization failed: {str(e)}, please check project file"}}

    # --- Wall Thickness Check ---
    try:
        # Fast scan with EDT first
        edt = distance_transform_edt(voxel_grid)
        local_thickness = edt * voxel_size * 2  # Convert to mm diameter
        min_edt_thickness = np.min(local_thickness[voxel_grid == 1])

        # If potential thin point is found, use raycasting to get a precise thickness measure
        if min_edt_thickness < min_wall_thickness:
            # Pinpoint thinnest point
            min_loc = np.unravel_index(np.argmin(edt), edt.shape)
            min_point_voxel = np.array(min_loc) + 0.5  # Voxel center
            min_point_mesh = (min_point_voxel * voxel_size) @ voxels.transform[:3, :3].T + voxels.transform[:3, 3]
            # Raycast
            true_thickness = trimesh.proximity.thickness(mesh, [min_point_mesh])[0]

            if true_thickness < min_wall_thickness:
                is_printable = False
                info['Thin walls'] = (f"Wall thickness below minimum: "
                                        f"{true_thickness:.2f}mm < {min_wall_thickness}mm")

    except Exception as e:
        return {"is_printable": False,
                "info": {"Error": f"Wall thickness analysis failed: {str(e)}, please check project file"}}

    # --- Fragile Protrusion Check ---
    try:
        skeleton = skeletonize(voxel_grid)
        skeleton_points = np.argwhere(skeleton)
        if len(skeleton_points) > 0:
            transform = voxels.transform

            # Convert voxel indices to mesh coordinates:
            # 1. Add 0.5 to get voxel center coordinates
            # 2. Scale by voxel size
            # 3. Apply transformation matrix
            skeleton_coords = trimesh.transform_points(
                (skeleton_points + 0.5) * voxel_size,
                transform
            )

            prox = ProximityQuery(mesh)
            for point in skeleton_coords:
                radius = prox.signed_distance([point])[0]
                if 0 < radius < max_protrusion_fragile_radius:
                    distances = np.linalg.norm(skeleton_coords - point, axis=1)
                    max_length = np.max(distances)
                    aspect_ratio = max_length / radius
                    if aspect_ratio > max_protrusion_aspect_ratio:
                        is_printable = False
                        info['Protrusions'] = (f"Fragile protrusion detected: "
                                                 f"length={max_length:.2f}mm, "
                                                 f"aspect_ratio={aspect_ratio:.2f}")

    except Exception as e:
        return {"is_printable": False,
                "info": {"Error": f"Protrusion analysis failed: {str(e)}, please check project file"}}

    # --- Curvature Check ---
    try:
        if mesh.vertices.shape[0] > 0:
            curvature = trimesh.curvature.discrete_gaussian_curvature_measure(
                mesh, mesh.vertices, radius=min_wall_thickness_small
            )
            if np.any(curvature > curvature_threshold) and max_dimension<100.0:
                is_printable = False
                info['Structural fragility'] = (f"Potential fragile strucutre detected,"
                                                f"please check for sharp tips, thin stripes etc.")


    except Exception as e:
        return {"is_printable": "Error",
                "info": {"Error": f"Curvature analysis failed: {str(e)}, please check project file"}}

    if is_printable:
        info['Printable'] = 'No obvious fault detected, mesh is likely printable'

    return {"is_printable":is_printable, "info": info}


def get_mesh_info(mesh_url: str) -> dict:
    """
    Get mesh information including size, volume, surface area, weight, and density.
    """
    try:
        mesh = get_mesh_from_url(mesh_url)
        bounding_box = mesh.bounding_box_oriented
        volume = 1e-6 * mesh.volume  # in L
        bounding_vol = 1e-6 * bounding_box.volume  # in L
        extents = bounding_box.primitive.extents.tolist()
        size = "x".join(f"{dim:.1f}" for dim in extents) + " mm" # bounding box dimensions, in mm
        surface = 1e-6 * mesh.area  # in m^2
        density = DEFAULT_DENSITY
        weight = density * volume  # in kg, as g/cm3 = kg/L
        return {"size": size, "bounding_vol": bounding_vol, "surface": surface, "weight": weight, "density": density}
    except Exception as e:
        raise ValueError(f"Failed to process mesh {mesh_url}: {str(e)}")


