"""Coordinate conversion: NAD83+NAVD88 → WGS84 → local meters."""

import numpy as np
import pyproj


def grid_positions(
    bbox: tuple[float, float, float, float],
    rows: int,
    cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate lon/lat arrays for each grid cell center.

    bbox is (lon_min, lat_min, lon_max, lat_max).
    Returns (lons, lats) each of shape (rows, cols).
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    # Pixel centers: half-pixel inset from edges
    lon = np.linspace(lon_min, lon_max, cols)
    # Rows go top-to-bottom in the raster (north to south)
    lat = np.linspace(lat_max, lat_min, rows)
    lons, lats = np.meshgrid(lon, lat)
    return lons, lats


def nad83_navd88_to_wgs84(
    lats: np.ndarray,
    lons: np.ndarray,
    elevations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert NAD83+NAVD88 coordinates to WGS84 (lat, lon, ellipsoid height).

    Returns (wgs84_lats, wgs84_lons, wgs84_heights).
    """
    transformer = pyproj.Transformer.from_crs(
        "EPSG:5498",  # NAD83 + NAVD88
        "EPSG:4979",  # WGS84 3D (ellipsoid height)
        always_xy=False,
    )
    wgs84_lats, wgs84_lons, wgs84_heights = transformer.transform(
        lats.ravel(), lons.ravel(), elevations.ravel()
    )
    shape = lats.shape
    return (
        np.array(wgs84_lats).reshape(shape),
        np.array(wgs84_lons).reshape(shape),
        np.array(wgs84_heights).reshape(shape),
    )


def wgs84_to_local(
    lats: np.ndarray,
    lons: np.ndarray,
    heights: np.ndarray,
    ref_lat: float,
    ref_lon: float,
    ref_alt: float,
) -> np.ndarray:
    """Convert WGS84 positions to local ENU meters relative to a reference point.

    Returns vertices array of shape (N, 3) as (east, north, up) which maps
    to (x=east, y=north, z=up) in a Z-up coordinate frame.
    """
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4979",  # WGS84 3D
        f"+proj=topocentric +lat_0={ref_lat} +lon_0={ref_lon} +h_0={ref_alt} +ellps=WGS84",
        always_xy=False,
    )
    e, n, u = transformer.transform(
        lats.ravel(), lons.ravel(), heights.ravel()
    )
    vertices = np.column_stack([e, n, u]).astype(np.float32)
    return vertices
