"""True resistivity fields used as inversion targets.

The two-layer model is the development target: hyperparameters are tuned on it
and it is then set aside. The block and ParFlow targets are held out -- they
are only ever inverted with frozen hyperparameters, so reported numbers come
from settings chosen without seeing these problems.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def two_layer(centers: np.ndarray, *, boundary: float = -15.0, top: float = 50.0, bottom: float = 500.0) -> np.ndarray:
    """Development target: one horizontal interface."""

    return np.where(centers[:, 1] >= boundary, float(top), float(bottom))


def block_anomaly(
    centers: np.ndarray,
    *,
    background: float = 100.0,
    conductive: tuple[float, float, float, float, float] = (30.0, 55.0, -5.0, -18.0, 20.0),
    resistive: tuple[float, float, float, float, float] = (75.0, 105.0, -6.0, -20.0, 800.0),
) -> np.ndarray:
    """Held-out target: two sharp-edged blocks in a uniform background.

    Blocks are kept above about 20 m depth because that is roughly the limit of
    what a 16-electrode Wenner array senses; structure placed deeper cannot be
    recovered by any parameterization and would only add noise to the metrics.

    Sharp rectangular edges are the point: step discontinuities are exactly the
    high-frequency content a plain ReLU MLP is expected to over-smooth.
    """

    x, z = centers[:, 0], centers[:, 1]
    values = np.full(centers.shape[0], float(background))
    for x0, x1, z_top, z_bottom, rho in (conductive, resistive):
        inside = (x >= x0) & (x <= x1) & (z <= z_top) & (z >= z_bottom)
        values[inside] = float(rho)
    return values


def mixed_calibration(
    centers: np.ndarray,
    *,
    surface: float = 80.0,
    at_depth: float = 250.0,
    depth_scale: float = 25.0,
    wedge_top: float = -6.0,
    wedge_dip: float = 0.22,
    wedge_factor: float = 0.45,
    conductive: tuple[float, float, float, float, float] = (58.0, 82.0, -7.0, -17.0, 25.0),
    resistive: tuple[float, float, float, float, float] = (14.0, 34.0, -10.0, -20.0, 900.0),
) -> np.ndarray:
    """Calibration target containing smooth, intermediate, and sharp structure.

    The original development target (``two_layer``) is smooth, and tuning on it
    systematically selected low-frequency configurations -- SIREN's w0 was
    driven to 1.0, which then failed badly on sharp targets. A calibration
    target must contain the range of spatial frequencies the method will be
    asked to represent, or hyperparameter selection collapses toward whichever
    end the target happens to occupy.

    Three components, deliberately spanning the band:
      * smooth exponential increase with depth (low frequency)
      * a dipping interface (intermediate)
      * one sharp rectangular block (high frequency)

    All structure sits above ~20 m, within what a 16-electrode Wenner array
    senses. The gradient saturates below the sensed zone so the unresolvable
    deep half does not carry an absurd dynamic range.

    Deliberately NOT a near-duplicate of the ``blocks`` evaluation target: the
    anomalies differ in position, size and contrast, and are embedded in a
    gradient and a dipping interface that ``blocks`` does not have. A
    calibration target that resembles the test target too closely leaks the
    test problem into hyperparameter selection.

    Used ONLY for calibration; never evaluated on.
    """

    x, z = centers[:, 0], centers[:, 1]
    depth = np.clip(-z, 0.0, float(depth_scale))          # saturate below the sensed zone
    values = float(surface) * (float(at_depth) / float(surface)) ** (depth / float(depth_scale))
    wedge = z > (float(wedge_top) - float(wedge_dip) * x)
    values = np.where(wedge, values * float(wedge_factor), values)
    for x0, x1, z_top, z_bottom, rho in (conductive, resistive):
        inside = (x >= x0) & (x <= x1) & (z <= z_top) & (z >= z_bottom)
        values = np.where(inside, float(rho), values)
    return values


def parflow_slice(
    path: str | Path,
    centers: np.ndarray,
    *,
    depth_extent: float | None = None,
) -> np.ndarray:
    """Held-out target: a ParFlow-derived resistivity slice resampled onto the mesh.

    The stored arrays are ``(nz, nx)`` with **index 0 at the bottom** of the
    domain (verified empirically: the shallowest rows are the ones that vary
    between timesteps as the hillslope wets and dries, the deepest rows are
    static bedrock). Axis 0 is flipped here so row 0 is the surface.

    The ParFlow domain and the ERT mesh do not share physical extents, so this
    resamples the field as a *pattern* onto the mesh's normalized coordinates
    rather than co-locating it physically. That is fine for a benchmark target
    -- it supplies realistic heterogeneity and a wide dynamic range -- but it
    is not a physically registered model and should not be described as one.

    ``depth_extent`` restricts the resampling to the top N metres of the mesh,
    stretching the slice over the sensed zone instead of the full mesh depth.
    """

    values = np.asarray(np.load(Path(path)), dtype=float)
    if values.ndim != 2:
        raise ValueError(f"expected a 2D (nz, nx) slice, got shape {values.shape}")
    values = values[::-1, :]                      # row 0 becomes the surface
    n_depth, n_x = values.shape

    interpolator = RegularGridInterpolator(
        (np.linspace(0.0, 1.0, n_depth), np.linspace(0.0, 1.0, n_x)),
        values,
        bounds_error=False,
        fill_value=None,
    )

    x, z = centers[:, 0], centers[:, 1]
    x_norm = (x - x.min()) / (x.max() - x.min())
    surface = z.max()
    span = float(depth_extent) if depth_extent is not None else (surface - z.min())
    depth_norm = np.clip((surface - z) / span, 0.0, 1.0)

    sampled = interpolator(np.column_stack([depth_norm, x_norm]))
    if np.any(sampled <= 0.0) or not np.all(np.isfinite(sampled)):
        raise ValueError("resampled ParFlow slice contains non-positive or non-finite resistivity")
    return sampled


def coverage_mask(jacobian: np.ndarray, *, quantile: float = 0.5) -> np.ndarray:
    """Cells the survey actually senses, from the column magnitude of the Jacobian.

    The mesh extends to 100 m depth but a 16-electrode Wenner array senses only
    the top ~20 m, so most cells are unconstrained by the data. Model error
    computed over the whole mesh is therefore dominated by regions no method
    could recover, which compresses exactly the differences the benchmark is
    trying to measure. Metrics are reported both ways.

    The mask is computed once from the Jacobian at the *true* model, so it is
    identical for every architecture and cannot favour any of them.
    """

    sensitivity = np.abs(np.asarray(jacobian, dtype=float)).sum(axis=0)
    threshold = np.quantile(sensitivity, float(quantile))
    return sensitivity >= threshold
