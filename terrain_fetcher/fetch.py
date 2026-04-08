"""Fetch elevation and imagery data from USGS REST APIs."""

import io

import numpy as np
import requests
from PIL import Image

_TIMEOUT = 60

_3DEP_URL = (
    "https://elevation.nationalmap.gov/arcgis/rest/services"
    "/3DEPElevation/ImageServer/exportImage"
)
_NAIP_URL = (
    "https://imagery.nationalmap.gov/arcgis/rest/services"
    "/USGSNAIPImagery/ImageServer/exportImage"
)

_MAX_PIXELS = 8000


def fetch_elevation(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> tuple[np.ndarray, tuple[float, float, float, float], tuple[int, int]]:
    """Fetch elevation grid from USGS 3DEP.

    Returns (elevations, bbox, (rows, cols)) where elevations is a float32
    array of shape (rows, cols) in meters (NAVD88), and bbox is the actual
    (lon_min, lat_min, lon_max, lat_max) of the returned raster.
    """
    bbox = f"{lon_min},{lat_min},{lon_max},{lat_max}"

    # First, request with auto size to see what resolution 3DEP gives us
    params = {
        "bbox": bbox,
        "bboxSR": "4326",
        "imageSR": "4326",
        "format": "tiff",
        "pixelType": "F32",
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }

    resp = requests.get(_3DEP_URL, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()

    if resp.headers.get("Content-Type", "").startswith("application/json"):
        raise RuntimeError(f"3DEP API error: {resp.text}")

    img = Image.open(io.BytesIO(resp.content))
    rows, cols = img.size[1], img.size[0]

    if rows > _MAX_PIXELS or cols > _MAX_PIXELS:
        raise RuntimeError(
            f"Requested area too large: {cols}x{rows} pixels "
            f"(max {_MAX_PIXELS}x{_MAX_PIXELS}). Reduce bounding box."
        )

    elevations = np.array(img, dtype=np.float32)
    actual_bbox = (lon_min, lat_min, lon_max, lat_max)
    return elevations, actual_bbox, (rows, cols)


def fetch_imagery(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> np.ndarray:
    """Fetch RGB imagery from USGS NAIP.

    Returns uint8 RGB array of shape (H, W, 3).
    """
    bbox = f"{lon_min},{lat_min},{lon_max},{lat_max}"

    params = {
        "bbox": bbox,
        "bboxSR": "4326",
        "imageSR": "4326",
        "format": "png",
        "f": "image",
    }

    resp = requests.get(_NAIP_URL, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()

    if resp.headers.get("Content-Type", "").startswith("application/json"):
        raise RuntimeError(f"NAIP API error: {resp.text}")

    img = Image.open(io.BytesIO(resp.content))
    arr = np.array(img)

    # NAIP returns 4-band (RGB + NIR), keep only RGB
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    return arr.astype(np.uint8)
