"""
Static 3D visualisation for SOFC microstructure validation volumes.

Phases:
    1 = Ni
    2 = YSZ
    3 = pore

Pore is treated as transparent by not rendering a pore mesh.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv
import scipy.io as sio
from PIL import Image
from skimage import measure


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = Path(
    r"C:\Users\r43341mm\OneDrive - The University of Manchester"
    r"\Research\SharedData\PhaseFieldResults\Validation3D\mat"
)
OUTPUT_DIR = PROJECT_DIR / "output"

# Set this to one file name from INPUT_DIR, for example "t000000.mat".
# Safer option: leave this unchanged and run with: --file t000010.mat
# Leave as None to use the first supported file found in INPUT_DIR.
# Supported formats: .mat, .npy, .npz
INPUT_FILE: str | None = "t000100.mat"
TARGET_KEY = "C"

# "full" renders the whole volume.
# "region" renders the rectangular crop below.
RENDER_MODE = "full"

# Coordinates are in array order: (z, y, x). End coordinates are exclusive.
# Example: (20, 30, 40) to (90, 120, 110) keeps:
# z = 20..89, y = 30..119, x = 40..109
REGION_START_ZYX = (0, 0, 0)
REGION_END_ZYX = (80, 80, 80)

# Export settings. Use ("pdf",) or ("svg",) if you only want one format.
OUTPUT_FORMATS = ("pdf", )
OUTPUT_BASENAME: str | None = None
FIGURE_SIZE_PIXELS = (2200, 2200)
TIGHT_SQUARE_OUTPUT = True
TIGHT_PADDING_FRACTION = 0.035
TIGHT_BACKGROUND_THRESHOLD = 248
PDF_DPI = 300

# Save the 3D scene as a high-resolution raster image inside the PDF/SVG file.
# This keeps transparency and lighting robust for complex meshes.
RASTERIZE_3D_IN_VECTOR_FILE = True

# Phase IDs
PHASE_NI = 1
PHASE_YSZ = 2
PHASE_PORE = 3

# Initial-geometry visual style.
USE_GRAYSCALE = False
NI_COLOR = "#A82B2F"
YSZ_COLOR = "#D8D0C2"
YSZ_OPACITY = 1.0
BACKGROUND_COLOR = "white"

# Material settings.
NI_SPECULAR = 0.12
NI_SPECULAR_POWER = 18.0
NI_DIFFUSE = 0.76
NI_AMBIENT = 0.24

YSZ_SPECULAR = 0.01
YSZ_SPECULAR_POWER = 8.0
YSZ_DIFFUSE = 0.62
YSZ_AMBIENT = 0.30

# Camera settings.
# "manual" uses the location/focal-point values below.
# Built-ins such as "iso", "xy", "xz", and "yz" are also supported.
CAMERA_POSITION = "manual"
CAMERA_PARALLEL_PROJECTION = True
CAMERA_ZOOM = 1.75

# Manual camera controls. Values are offsets from the volume center, multiplied
# by the largest plotted volume dimension. For example, with a 100^3 volume:
# camera x = center_x + CAMERA_LOCATION_RELATIVE[0] * 100.
# Increase z to see more top surface. Make y more negative to move farther away.
CAMERA_LOCATION_RELATIVE = (0.00, -4.00, 0.24)
CAMERA_FOCAL_POINT_RELATIVE = (0.00, 0.00, 0.0) #(0.00, 0.00, 0.02)
CAMERA_VIEW_UP = (0.0, 0.0, 1.0)

# Marching cubes reads the array in (z, y, x) order. These settings choose
# which array axis is drawn as the final plot (x, y, z).
# Use ("x", "y", "z") for physical xyz, or ("y", "x", "z") to swap x and y.
PLOT_AXES = ("z", "x", "y")

# The current validation volume is 100^3, so full-volume rendering can stay
# high quality without decimation. For much larger volumes, set DECIMATE=True.
FULL_VOLUME_SMOOTH_NI = 45
FULL_VOLUME_SMOOTH_YSZ = 25
FULL_VOLUME_DECIMATE = False

# Regions are smaller, so use smoother surfaces.
REGION_SMOOTH_NI = 60
REGION_SMOOTH_YSZ = 25
REGION_DECIMATE = False

DECIMATE_TARGET_REDUCTION = 0.50


# ============================================================
# LOADING
# ============================================================

SUPPORTED_EXTENSIONS = (".mat", ".npy", ".npz")
ACTIVE_COLOR_SCHEME = "grayscale" if USE_GRAYSCALE else "color"


def apply_color_scheme(scheme: str | None) -> None:
    global ACTIVE_COLOR_SCHEME, NI_COLOR, YSZ_COLOR

    selected = ("grayscale" if USE_GRAYSCALE else "color") if scheme is None else scheme
    palettes = {
        "color": {
            "ni": "#A82B2F",
            "ysz": "#D8D0C2",
        },
        "grayscale": {
            "ni": "#4A4A4A",
            "ysz": "#BEB8AD",
        },
    }

    if selected not in palettes:
        raise ValueError("Color scheme must be either 'color' or 'grayscale'.")

    ACTIVE_COLOR_SCHEME = selected
    NI_COLOR = palettes[selected]["ni"]
    YSZ_COLOR = palettes[selected]["ysz"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Static 3D visualisation for validation microstructure volumes."
    )
    parser.add_argument(
        "--file",
        dest="input_file",
        default=None,
        help="One input file name from INPUT_DIR, for example t000000.mat.",
    )
    parser.add_argument(
        "--scheme",
        choices=("color", "grayscale"),
        default=None,
        help="Render color or grayscale output. Default uses COLOR_SCHEME.",
    )
    return parser.parse_args()


def resolve_input_file(input_file: str | None = None) -> Path:
    selected_input = INPUT_FILE if input_file is None else input_file

    if selected_input is not None:
        path = Path(selected_input)
        if not path.is_absolute():
            path = INPUT_DIR / path
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")
        return path

    candidates: list[Path] = []
    for extension in SUPPORTED_EXTENSIONS:
        candidates.extend(INPUT_DIR.glob(f"*{extension}"))

    if not candidates:
        suffixes = ", ".join(SUPPORTED_EXTENSIONS)
        raise FileNotFoundError(
            f"No supported volume file found in {INPUT_DIR}. "
            f"Add a {suffixes} file or set INPUT_FILE."
        )

    return sorted(candidates)[0]


def load_volume(path: Path, key: str) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix == ".mat":
        try:
            data = sio.loadmat(path)
            if key not in data:
                visible_keys = [k for k in data if not k.startswith("__")]
                raise KeyError(f"Key {key!r} not found. Available keys: {visible_keys}")
            volume = np.asarray(data[key])
        except NotImplementedError:
            with h5py.File(path, "r") as handle:
                if key not in handle:
                    raise KeyError(f"Key {key!r} not found. Available keys: {list(handle.keys())}")
                volume = np.asarray(handle[key]).T

    elif suffix == ".npy":
        volume = np.load(path)

    elif suffix == ".npz":
        data = np.load(path)
        if key not in data:
            raise KeyError(f"Key {key!r} not found. Available keys: {list(data.keys())}")
        volume = data[key]

    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")

    return volume.astype(np.uint8, copy=False)


# ============================================================
# VOLUME PREPARATION
# ============================================================


def crop_region(volume: np.ndarray) -> np.ndarray:
    start = np.asarray(REGION_START_ZYX, dtype=int)
    end = np.asarray(REGION_END_ZYX, dtype=int)
    shape = np.asarray(volume.shape, dtype=int)

    if start.shape != (3,) or end.shape != (3,):
        raise ValueError("REGION_START_ZYX and REGION_END_ZYX must contain three coordinates.")
    if np.any(start < 0):
        raise ValueError(f"REGION_START_ZYX must be non-negative, got {tuple(start)}")
    if np.any(end > shape):
        raise ValueError(
            f"REGION_END_ZYX {tuple(end)} is outside volume shape {tuple(shape)}"
        )
    if np.any(end <= start):
        raise ValueError(
            f"Each end coordinate must be larger than start. "
            f"Start={tuple(start)}, end={tuple(end)}"
        )

    slices = tuple(slice(int(start[i]), int(end[i])) for i in range(3))
    return volume[slices].copy()


def prepare_volume(volume: np.ndarray) -> tuple[np.ndarray, int, int, bool]:
    mode = RENDER_MODE.lower().strip()

    if mode == "full":
        return (
            volume.copy(),
            FULL_VOLUME_SMOOTH_NI,
            FULL_VOLUME_SMOOTH_YSZ,
            FULL_VOLUME_DECIMATE,
        )

    if mode == "region":
        return (
            crop_region(volume),
            REGION_SMOOTH_NI,
            REGION_SMOOTH_YSZ,
            REGION_DECIMATE,
        )

    raise ValueError("RENDER_MODE must be either 'full' or 'region'.")


# ============================================================
# MESHING
# ============================================================


def volume_to_mesh(
    volume: np.ndarray,
    phase_id: int,
    *,
    smooth_iter: int,
    decimate: bool,
) -> pv.PolyData | None:
    binary = (volume == phase_id).astype(np.float32)
    if not np.any(binary):
        return None

    padded = np.pad(binary, 1, mode="constant", constant_values=0)

    try:
        vertices, faces, _, _ = measure.marching_cubes(padded, level=0.5)
    except (RuntimeError, ValueError):
        return None

    faces_vtk = np.column_stack(
        [np.full(len(faces), 3, dtype=np.int32), faces.astype(np.int32)]
    )

    mesh = pv.PolyData(orient_vertices(vertices), faces_vtk)

    if decimate and DECIMATE_TARGET_REDUCTION > 0:
        mesh = mesh.decimate(DECIMATE_TARGET_REDUCTION)

    if smooth_iter > 0:
        mesh = mesh.smooth(n_iter=smooth_iter, relaxation_factor=0.1)

    return mesh.compute_normals(
        auto_orient_normals=True,
        consistent_normals=True,
        inplace=False,
    )


def orient_vertices(vertices: np.ndarray) -> np.ndarray:
    array_axis_to_column = {"z": 0, "y": 1, "x": 2}
    try:
        columns = [array_axis_to_column[axis] for axis in PLOT_AXES]
    except KeyError as exc:
        raise ValueError("PLOT_AXES must contain only 'x', 'y', and 'z'.") from exc

    if len(columns) != 3 or len(set(columns)) != 3:
        raise ValueError("PLOT_AXES must contain each of 'x', 'y', and 'z' exactly once.")

    # Undo the one-voxel pad after axis remapping.
    return vertices[:, columns] - 1.0


def oriented_dimensions(array_shape: tuple[int, int, int]) -> np.ndarray:
    array_axis_to_index = {"z": 0, "y": 1, "x": 2}
    return np.asarray([array_shape[array_axis_to_index[axis]] for axis in PLOT_AXES], dtype=float)


# ============================================================
# RENDERING
# ============================================================


def build_plotter() -> pv.Plotter:
    plotter = pv.Plotter(
        off_screen=True,
        window_size=FIGURE_SIZE_PIXELS,
    )
    plotter.set_background(BACKGROUND_COLOR)
    plotter.enable_depth_peeling(number_of_peels=10)
    plotter.enable_anti_aliasing("ssaa")

    plotter.remove_all_lights()
    plotter.add_light(
        pv.Light(
            position=(0, -300, 60),
            focal_point=(0, 0, 0),
            color="white",
            intensity=0.65,
        )
    )
    plotter.add_light(
        pv.Light(
            position=(-180, -240, 120),
            focal_point=(0, 0, 0),
            color="white",
            intensity=0.18,
        )
    )
    plotter.add_light(
        pv.Light(
            position=(180, -240, 80),
            focal_point=(0, 0, 0),
            color="white",
            intensity=0.12,
        )
    )

    return plotter


def apply_camera(plotter: pv.Plotter, array_shape: tuple[int, int, int]) -> None:
    if CAMERA_POSITION == "manual":
        dims = oriented_dimensions(array_shape)
        center = (dims - 1.0) / 2.0
        span = float(np.max(dims))

        camera_position = tuple(
            center[i] + CAMERA_LOCATION_RELATIVE[i] * span for i in range(3)
        )
        focal_point = tuple(
            center[i] + CAMERA_FOCAL_POINT_RELATIVE[i] * span for i in range(3)
        )
        plotter.camera_position = [camera_position, focal_point, CAMERA_VIEW_UP]
        if CAMERA_PARALLEL_PROJECTION:
            plotter.enable_parallel_projection()
        plotter.reset_camera_clipping_range()
        plotter.camera.zoom(CAMERA_ZOOM)
        return

    plotter.camera_position = CAMERA_POSITION
    plotter.reset_camera()
    if CAMERA_PARALLEL_PROJECTION:
        plotter.enable_parallel_projection()
    plotter.camera.zoom(CAMERA_ZOOM)


def output_stem(input_path: Path) -> str:
    scheme_suffix = "" if ACTIVE_COLOR_SCHEME == "color" else f"_{ACTIVE_COLOR_SCHEME}"

    if OUTPUT_BASENAME:
        return f"{OUTPUT_BASENAME}{scheme_suffix}"

    mode = RENDER_MODE.lower().strip()
    if mode == "region":
        z0, y0, x0 = REGION_START_ZYX
        z1, y1, x1 = REGION_END_ZYX
        return f"{input_path.stem}_region_z{z0}-{z1}_y{y0}-{y1}_x{x0}-{x1}{scheme_suffix}"

    return f"{input_path.stem}_full{scheme_suffix}"


def make_tight_square_image(image_array: np.ndarray) -> Image.Image:
    rgb = image_array[:, :, :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    content_mask = np.any(rgb < TIGHT_BACKGROUND_THRESHOLD, axis=2)
    if not np.any(content_mask):
        return Image.fromarray(rgb)

    rows, cols = np.where(content_mask)
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(cols.min()), int(cols.max()) + 1

    height, width = rgb.shape[:2]
    content_size = max(y1 - y0, x1 - x0)
    padding = max(2, int(round(content_size * TIGHT_PADDING_FRACTION)))

    y0 = max(0, y0 - padding)
    y1 = min(height, y1 + padding)
    x0 = max(0, x0 - padding)
    x1 = min(width, x1 + padding)

    cropped = rgb[y0:y1, x0:x1]
    crop_h, crop_w = cropped.shape[:2]
    square_size = max(crop_h, crop_w)

    square = np.full((square_size, square_size, 3), 255, dtype=np.uint8)
    top = (square_size - crop_h) // 2
    left = (square_size - crop_w) // 2
    square[top:top + crop_h, left:left + crop_w] = cropped

    return Image.fromarray(square)


def save_tight_square_pdf(plotter: pv.Plotter, path: Path) -> None:
    image_array = plotter.screenshot(return_img=True)
    image = make_tight_square_image(image_array)
    image.save(path, "PDF", resolution=PDF_DPI)


def save_scene(plotter: pv.Plotter, input_path: Path) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = output_stem(input_path)
    saved_paths: list[Path] = []

    plotter.show(auto_close=False)

    for fmt in OUTPUT_FORMATS:
        clean_fmt = fmt.lower().lstrip(".")
        if clean_fmt not in {"pdf", "svg"}:
            raise ValueError("OUTPUT_FORMATS currently supports only 'pdf' and 'svg'.")

        path = OUTPUT_DIR / f"{stem}.{clean_fmt}"
        if clean_fmt == "pdf" and TIGHT_SQUARE_OUTPUT:
            save_tight_square_pdf(plotter, path)
        else:
            plotter.save_graphic(
                str(path),
                raster=RASTERIZE_3D_IN_VECTOR_FILE,
            )
        saved_paths.append(path)

    plotter.close()
    return saved_paths


def render_static(volume: np.ndarray, input_path: Path) -> list[Path]:
    sub_volume, smooth_ni, smooth_ysz, decimate = prepare_volume(volume)

    print(f"Input: {input_path}")
    print(f"Render mode: {RENDER_MODE}")
    print(f"Rendered volume shape (z, y, x): {sub_volume.shape}")
    print(f"Labels present: {np.unique(sub_volume)}")

    print("Extracting Ni mesh...")
    ni_mesh = volume_to_mesh(
        sub_volume,
        PHASE_NI,
        smooth_iter=smooth_ni,
        decimate=decimate,
    )

    print("Extracting YSZ mesh...")
    ysz_mesh = volume_to_mesh(
        sub_volume,
        PHASE_YSZ,
        smooth_iter=smooth_ysz,
        decimate=decimate,
    )

    if ni_mesh is None and ysz_mesh is None:
        raise ValueError("The selected volume contains no Ni or YSZ to render.")

    plotter = build_plotter()

    if ysz_mesh is not None:
        plotter.add_mesh(
            ysz_mesh,
            color=YSZ_COLOR,
            opacity=YSZ_OPACITY,
            smooth_shading=True,
            specular=YSZ_SPECULAR,
            specular_power=YSZ_SPECULAR_POWER,
            diffuse=YSZ_DIFFUSE,
            ambient=YSZ_AMBIENT,
            name="YSZ",
        )

    if ni_mesh is not None:
        plotter.add_mesh(
            ni_mesh,
            color=NI_COLOR,
            smooth_shading=True,
            specular=NI_SPECULAR,
            specular_power=NI_SPECULAR_POWER,
            diffuse=NI_DIFFUSE,
            ambient=NI_AMBIENT,
            name="Ni",
        )

    apply_camera(plotter, sub_volume.shape)

    return save_scene(plotter, input_path)


def main() -> None:
    args = parse_args()
    apply_color_scheme(args.scheme)
    input_path = resolve_input_file(args.input_file)
    volume = load_volume(input_path, TARGET_KEY)
    saved_paths = render_static(volume, input_path)

    print("Saved:")
    for path in saved_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
