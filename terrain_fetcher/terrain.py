"""Terrain: fetch, convert, triangulate, and export textured terrain."""

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from terrain_fetcher.convert import (
    grid_positions,
    nad83_navd88_to_wgs84,
    wgs84_to_local,
)
from terrain_fetcher.fetch import fetch_elevation, fetch_imagery
from terrain_fetcher.mesh import compute_uvs, triangulate_grid


class Terrain:
    """Textured terrain mesh from USGS data.

    Runs the full pipeline on construction: fetch elevation + imagery,
    coordinate conversion, triangulation, UV mapping.

    Attributes:
        vertices: (N, 3) float32 array — (east, north, up) in local meters.
        indices: (M, 3) uint32 array — triangle vertex indices.
        uvs: (N, 2) float32 array — normalized texture coordinates.
        texture_rgb: (H, W, 3) uint8 array — RGB imagery texture.
    """

    def __init__(
        self,
        ref_lat: float,
        ref_lon: float,
        ref_alt: float,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> None:
        # Step 1: Fetch elevation
        elevations, bbox, (rows, cols) = fetch_elevation(
            lat_min, lat_max, lon_min, lon_max
        )

        # Step 2: Fetch imagery
        self.texture_rgb = fetch_imagery(lat_min, lat_max, lon_min, lon_max)

        # Step 3: Convert coordinates
        lons, lats = grid_positions(bbox, rows, cols)
        wgs84_lats, wgs84_lons, wgs84_heights = nad83_navd88_to_wgs84(
            lats, lons, elevations
        )
        self.vertices = wgs84_to_local(
            wgs84_lats, wgs84_lons, wgs84_heights, ref_lat, ref_lon, ref_alt
        )

        # Step 4: Triangulate
        self.indices = triangulate_grid(rows, cols)

        # Step 5: Compute UVs
        self.uvs = compute_uvs(rows, cols)

    def export_glb(self, path: str | Path) -> Path:
        """Export terrain as a textured GLB file.

        Returns the resolved output path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        texture_img = Image.fromarray(self.texture_rgb)

        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=texture_img,
        )
        visual = trimesh.visual.TextureVisuals(
            uv=self.uvs,
            material=material,
        )

        mesh = trimesh.Trimesh(
            vertices=self.vertices,
            faces=self.indices,
            visual=visual,
            process=False,
        )

        mesh.export(str(path), file_type="glb")
        return path.resolve()
