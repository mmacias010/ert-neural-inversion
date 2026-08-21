"""Synthetic test cases for INR-reparameterized inversion.

Mesh and survey builders moved verbatim from ``test_siren_inversion.py`` so
that the benchmark harness no longer has to import from a test script.
"""

from __future__ import annotations

import numpy as np

from deepert.mesh import Mesh
from deepert.survey import Survey


def build_mesh(nx: int = 25, nz: int = 20, dx: float = 5.0, dz: float = 5.0) -> Mesh:
    """Rectangular grid split into triangles (default 25 x 20 -> 1000 cells)."""

    xs = np.linspace(0.0, nx * dx, nx + 1)
    zs = np.linspace(0.0, -(nz * dz), nz + 1)   # negative z = depth below surface
    XX, ZZ = np.meshgrid(xs, zs)
    nodes = np.column_stack([XX.ravel(), ZZ.ravel()])

    cells = []
    for row in range(nz):
        for col in range(nx):
            tl = row * (nx + 1) + col
            tr = tl + 1
            bl = tl + (nx + 1)
            br = bl + 1
            cells.append([tl, tr, bl])
            cells.append([tr, br, bl])
    cells = np.array(cells, dtype=np.int32)
    surface_node_ids = np.arange(nx + 1, dtype=np.int32)
    return Mesh.from_arrays(nodes, cells, surface_node_ids=surface_node_ids)


def build_survey(n_elec: int = 16, x0: float = 2.5, spacing: float = 8.0) -> Survey:
    """Surface electrodes with a Wenner-alpha measurement sequence."""

    electrode_x = x0 + spacing * np.arange(n_elec)
    electrode_positions = np.column_stack([electrode_x, np.zeros(n_elec)])
    measurements = []
    for a in range(1, n_elec // 3 + 1):           # Wenner spacings
        for i in range(n_elec - 3 * a):
            measurements.append([i, i + 3 * a, i + a, i + 2 * a])  # A B M N
    measurements = np.array(measurements, dtype=np.int32)
    return Survey.from_arrays(electrode_positions, measurements)


def two_layer_resistivity(
    centers: np.ndarray,
    *,
    boundary: float = -15.0,
    top: float = 50.0,
    bottom: float = 500.0,
) -> np.ndarray:
    """Two-layer true model, boundary shallow enough for a Wenner array to sense."""

    return np.where(centers[:, 1] >= boundary, float(top), float(bottom))


def synthetic_observations(
    forward,
    true_resistivity: np.ndarray,
    *,
    relative_noise: float = 0.015,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-model the true field and add multiplicative Gaussian noise.

    Noise is drawn from an explicit ``RandomState`` rather than NumPy's global
    RNG so the noise realization is controlled independently of network
    initialization. ``RandomState(seed)`` reproduces the exact stream that
    ``np.random.seed(seed)`` followed by ``np.random.randn`` would produce.

    Caveat worth stating in any writeup: the data are generated with the same
    mesh and the same solver used for the inversion. That is an *inverse
    crime* and it makes absolute recovery look better than it would on
    independent data. It is tolerable for comparing architectures against each
    other, since all of them inherit the identical bias, but it is not
    evidence about absolute inversion quality.
    """

    rng = np.random.RandomState(int(seed))
    obs_rhoa = np.asarray(forward.response(true_resistivity), dtype=float)
    obs_rhoa = obs_rhoa * (1.0 + float(relative_noise) * rng.randn(obs_rhoa.size))
    return obs_rhoa, np.log(obs_rhoa)
