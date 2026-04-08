"""CLI script to export terrain as GLB."""

import argparse
import logging
from pathlib import Path

from terrain_fetcher.terrain import Terrain


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Fetch USGS terrain data and export as textured GLB."
    )
    parser.add_argument("--ref-lat", type=float, required=True, help="Reference latitude (WGS84 deg)")
    parser.add_argument("--ref-lon", type=float, required=True, help="Reference longitude (WGS84 deg)")
    parser.add_argument("--ref-alt", type=float, required=True, help="Reference altitude (WGS84 ellipsoid height, m)")
    parser.add_argument("--lat-min", type=float, required=True, help="South bound (WGS84 deg)")
    parser.add_argument("--lat-max", type=float, required=True, help="North bound (WGS84 deg)")
    parser.add_argument("--lon-min", type=float, required=True, help="West bound (WGS84 deg)")
    parser.add_argument("--lon-max", type=float, required=True, help="East bound (WGS84 deg)")
    parser.add_argument("-o", "--output", type=str, default="exports/terrain.glb", help="Output GLB path")

    args = parser.parse_args()

    terrain = Terrain(
        ref_lat=args.ref_lat,
        ref_lon=args.ref_lon,
        ref_alt=args.ref_alt,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
    )

    out = terrain.export_glb(args.output)
    print(f"Exported: {out}")


if __name__ == "__main__":
    main()
