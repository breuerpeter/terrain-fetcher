"""Fetch elevation and imagery data from USGS REST APIs."""

import io
import logging

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

_TIMEOUT = 60

_3DEP_BASE = (
    "https://elevation.nationalmap.gov/arcgis/rest/services"
    "/3DEPElevation/ImageServer"
)
_NAIP_BASE = (
    "https://imagery.nationalmap.gov/arcgis/rest/services"
    "/USGSNAIPImagery/ImageServer"
)

_3DEP_URL = _3DEP_BASE + "/exportImage"
_NAIP_URL = _NAIP_BASE + "/exportImage"

_MAX_PIXELS = 8000


def _query_source_info(base_url: str, lat: float, lon: float) -> dict:
    """Query the identify endpoint to get source catalog attributes."""
    import json as _json

    geom = _json.dumps({
        "x": lon, "y": lat,
        "spatialReference": {"wkid": 4326},
    })
    resp = requests.get(
        base_url + "/identify",
        params={
            "geometry": geom,
            "geometryType": "esriGeometryPoint",
            "returnCatalogItems": "true",
            "f": "json",
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()

    features = data.get("catalogItems", {}).get("features", [])
    if features:
        return features[0].get("attributes", {})
    return {}


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

    content_type = resp.headers.get("Content-Type", "")
    if content_type.startswith("application/json"):
        raise RuntimeError(f"3DEP API error: {resp.text}")

    if len(resp.content) < 1024:
        raise RuntimeError(
            f"3DEP returned only {len(resp.content)} bytes — "
            f"area likely outside US coverage "
            f"(bbox: {lat_min:.4f}-{lat_max:.4f}N, {lon_min:.4f}-{lon_max:.4f}E)"
        )

    try:
        img = Image.open(io.BytesIO(resp.content))
        img.load()
    except OSError as e:
        raise RuntimeError(
            f"Failed to decode 3DEP elevation TIFF: {e}. "
            f"Area may be outside US coverage."
        ) from e

    rows, cols = img.size[1], img.size[0]

    if rows > _MAX_PIXELS or cols > _MAX_PIXELS:
        raise RuntimeError(
            f"Requested area too large: {cols}x{rows} pixels "
            f"(max {_MAX_PIXELS}x{_MAX_PIXELS}). Reduce bounding box."
        )

    elevations = np.array(img, dtype=np.float32)
    actual_bbox = (lon_min, lat_min, lon_max, lat_max)

    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    info = _query_source_info(_3DEP_BASE, center_lat, center_lon)
    src = info.get("Source") or info.get("Name") or "unknown"
    date = info.get("AcquisitionDate") or info.get("pubdate") or info.get("Cdate")
    logger.info(f"3DEP elevation: {cols}x{rows} px, source: {src}, date: {date or 'unknown'}")

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

    content_type = resp.headers.get("Content-Type", "")
    if content_type.startswith("application/json"):
        raise RuntimeError(f"NAIP API error: {resp.text}")

    if len(resp.content) < 1024:
        raise RuntimeError(
            f"NAIP returned only {len(resp.content)} bytes — "
            f"area likely outside US coverage "
            f"(bbox: {lat_min:.4f}-{lat_max:.4f}N, {lon_min:.4f}-{lon_max:.4f}E)"
        )

    try:
        img = Image.open(io.BytesIO(resp.content))
        img.load()
    except OSError as e:
        raise RuntimeError(
            f"Failed to decode NAIP imagery: {e}. "
            f"Area may be outside US coverage."
        ) from e
    arr = np.array(img)

    # NAIP returns 4-band (RGB + NIR), keep only RGB
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    info = _query_source_info(_NAIP_BASE, center_lat, center_lon)
    year = info.get("Year")
    state = info.get("State")
    res = info.get("resolution_value")
    res_unit = info.get("resolution_units", "")
    logger.info(
        f"NAIP imagery: {arr.shape[1]}x{arr.shape[0]} px, "
        f"year: {year or 'unknown'}, state: {state or '?'}, "
        f"resolution: {res} {res_unit}" if res else
        f"NAIP imagery: {arr.shape[1]}x{arr.shape[0]} px, "
        f"year: {year or 'unknown'}, state: {state or '?'}"
    )

    return arr.astype(np.uint8)
