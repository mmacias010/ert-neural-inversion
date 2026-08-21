"""Inverse cosine-transform quadrature rules for 2.5D ERT."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import roots_laguerre, roots_legendre

from deepert.utils.torch_runtime import Array
from deepert.utils.torch_runtime import torch_np

from deepert.survey import Survey
from deepert.utils.dtypes import FLOAT_DTYPE


@dataclass(frozen=True)
class CosineTransformWeights:
    """Wavenumbers and weights used in the 2.5D inverse cosine transform."""

    wavenumbers: Array
    weights: Array


def survey_wavenumber_bounds(survey: Survey) -> tuple[float, float]:
    """Infer integration bounds from the electrode layout."""

    positions = np.asarray(survey.electrode_positions, dtype=float)
    pairwise = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(pairwise, axis=-1)
    positive = distances[distances > 0.0]
    r_min = float(np.min(positive) / 2.0)
    r_max = float(np.max(positive) * 2.0)
    return r_min, r_max


def build_inverse_cosine_weights(r_min: float, r_max: float) -> CosineTransformWeights:
    """Gauss-Legendre and Gauss-Laguerre rule."""

    n_legendre = max(int(6.0 * np.log10(r_max / r_min)), 4)
    n_laguerre = 4
    k0 = 1.0 / (2.0 * r_min)

    legendre_points, legendre_weights = roots_legendre(n_legendre)
    legendre_points = 0.5 * (legendre_points + 1.0)
    legendre_weights = 0.5 * legendre_weights
    k_leg = k0 * legendre_points * legendre_points
    w_leg = 2.0 * k0 * legendre_points * legendre_weights / np.pi

    laguerre_points, laguerre_weights = roots_laguerre(n_laguerre)
    k_lag = k0 * (laguerre_points + 1.0)
    w_lag = k0 * np.exp(laguerre_points) * laguerre_weights / np.pi

    return CosineTransformWeights(
        wavenumbers=torch_np.asarray(np.concatenate((k_leg, k_lag)), dtype=FLOAT_DTYPE),
        weights=torch_np.asarray(np.concatenate((w_leg, w_lag)), dtype=FLOAT_DTYPE),
    )
