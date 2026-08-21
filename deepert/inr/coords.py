"""Coordinate handling for INR-reparameterized ERT inversion.

The implicit neural representation consumes cell-center coordinates, so the
mesh geometry has to be mapped into the input range the network expects.
"""

from __future__ import annotations

import numpy as np


def cell_centers(mesh) -> np.ndarray:
    """Return the (n_cells, 2) centroid of every mesh cell."""

    nodes = np.asarray(mesh.nodes, dtype=float)
    cells = np.asarray(mesh.cells, dtype=np.int32)
    return nodes[cells].mean(axis=1)


def normalized_coords(mesh) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(centers, coords)`` with each axis of ``coords`` scaled to [-1, 1].

    Each axis is normalized independently by its own extent. On a domain that
    is not square this is anisotropic: one metre of x and one metre of z map to
    different coordinate distances, so a network frequency means different
    things horizontally and vertically. That is the behaviour the original
    SIREN demo had, and it is preserved here deliberately. It matters when
    tuning ``w0`` (SIREN) or ``sigma`` (Fourier features), and it matters more
    on real profiles, which are far wider than they are deep.
    """

    centers = cell_centers(mesh)
    coords = np.empty_like(centers)
    for axis in range(centers.shape[1]):
        lower = centers[:, axis].min()
        span = centers[:, axis].max() - lower
        if span <= 0.0:
            raise ValueError(f"mesh has zero extent along axis {axis}; cannot normalize")
        coords[:, axis] = 2.0 * (centers[:, axis] - lower) / span - 1.0
    return centers, coords
