"""Grid triangulation and UV computation."""

import numpy as np


def triangulate_grid(rows: int, cols: int) -> np.ndarray:
    """Convert a regular grid to triangle indices.

    Each grid cell produces 2 triangles. Vertices are indexed row-major:
    vertex(r, c) = r * cols + c.

    Returns indices array of shape (M, 3) with uint32 dtype.
    """
    # For each cell (r, c) where r < rows-1 and c < cols-1:
    r, c = np.meshgrid(
        np.arange(rows - 1, dtype=np.uint32),
        np.arange(cols - 1, dtype=np.uint32),
        indexing="ij",
    )
    r = r.ravel()
    c = c.ravel()

    # Corner indices for each cell
    tl = r * cols + c           # top-left
    tr = r * cols + (c + 1)     # top-right
    bl = (r + 1) * cols + c     # bottom-left
    br = (r + 1) * cols + (c + 1)  # bottom-right

    # Two triangles per cell
    tri1 = np.column_stack([tl, bl, tr])
    tri2 = np.column_stack([tr, bl, br])

    return np.vstack([tri1, tri2]).astype(np.uint32)


def compute_uvs(rows: int, cols: int) -> np.ndarray:
    """Compute normalized UV coordinates for a grid of vertices.

    UV (0,0) = top-left, (1,1) = bottom-right.
    Returns array of shape (rows*cols, 2) with float32 dtype.
    """
    u = np.linspace(0.0, 1.0, cols)
    v = np.linspace(0.0, 1.0, rows)
    uu, vv = np.meshgrid(u, v)
    return np.column_stack([uu.ravel(), vv.ravel()]).astype(np.float32)
