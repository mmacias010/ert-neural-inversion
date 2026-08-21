"""Data-misfit interfaces for Deepert inversions.

The first implementation intentionally mirrors the historical Deepert
weighted L2 objective in transformed-data space.  In the current inversion
core, the transformed data are log apparent resistivities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class DataMisfit(Protocol):
    """Interface for data-misfit terms used by linearized inversions."""

    name: str

    def residual(self, predicted: np.ndarray, observed: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Return the weighted residual used for objective reporting."""

    def linearized_rhs(self, predicted: np.ndarray, observed: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Return the right-hand side for the linearized data equation."""

    def weighted_jacobian(self, jacobian: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Apply data weights to a Jacobian block."""

    def linearized_system(
        self,
        predicted: np.ndarray,
        observed: np.ndarray,
        weights: np.ndarray,
        jacobian: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(A, b)`` for the current linearized data term."""

    def phi(self, predicted: np.ndarray, observed: np.ndarray, weights: np.ndarray) -> float:
        """Return the unnormalized data objective."""

    def chi2(self, predicted: np.ndarray, observed: np.ndarray, weights: np.ndarray) -> float:
        """Return the mean squared weighted residual."""


@dataclass(frozen=True)
class WeightedLogL2Misfit:
    """Weighted L2 misfit in the already-log-transformed data domain."""

    name: str = "weighted_log_l2"

    def residual(self, predicted: np.ndarray, observed: np.ndarray, weights: np.ndarray) -> np.ndarray:
        predicted_array = np.asarray(predicted, dtype=float)
        observed_array = np.asarray(observed, dtype=float)
        weight_array = np.asarray(weights, dtype=float)
        return (predicted_array - observed_array) * weight_array

    def linearized_rhs(self, predicted: np.ndarray, observed: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return -self.residual(predicted, observed, weights)

    def weighted_jacobian(self, jacobian: np.ndarray, weights: np.ndarray) -> np.ndarray:
        matrix = np.asarray(jacobian, dtype=float)
        weight_array = np.asarray(weights, dtype=float).reshape(-1)
        if matrix.ndim != 2:
            raise ValueError("jacobian must be a 2D array")
        if weight_array.shape != (matrix.shape[0],):
            raise ValueError(
                "weights must have one value per Jacobian row "
                f"({weight_array.shape} != ({matrix.shape[0]},))"
            )
        return matrix * weight_array[:, None]

    def linearized_system(
        self,
        predicted: np.ndarray,
        observed: np.ndarray,
        weights: np.ndarray,
        jacobian: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.weighted_jacobian(jacobian, weights), self.linearized_rhs(predicted, observed, weights)

    def phi(self, predicted: np.ndarray, observed: np.ndarray, weights: np.ndarray) -> float:
        residual = self.residual(predicted, observed, weights).reshape(-1)
        return float(np.dot(residual, residual))

    def chi2(self, predicted: np.ndarray, observed: np.ndarray, weights: np.ndarray) -> float:
        residual = self.residual(predicted, observed, weights)
        return float(np.mean(residual**2))


@dataclass(frozen=True)
class WeightedLogL1Misfit:
    """IRLS-smoothed L1 misfit in the already-log-transformed data domain."""

    epsilon: float = 1.0e-3
    name: str = "weighted_log_l1"

    def residual(self, predicted: np.ndarray, observed: np.ndarray, weights: np.ndarray) -> np.ndarray:
        predicted_array = np.asarray(predicted, dtype=float)
        observed_array = np.asarray(observed, dtype=float)
        weight_array = np.asarray(weights, dtype=float)
        return (predicted_array - observed_array) * weight_array

    def _sqrt_irls_weight(self, residual: np.ndarray) -> np.ndarray:
        eps = max(float(self.epsilon), np.finfo(float).eps)
        return np.power(residual**2 + eps**2, -0.25)

    def linearized_rhs(self, predicted: np.ndarray, observed: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return -self.residual(predicted, observed, weights)

    def weighted_jacobian(self, jacobian: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return WeightedLogL2Misfit().weighted_jacobian(jacobian, weights)

    def linearized_system(
        self,
        predicted: np.ndarray,
        observed: np.ndarray,
        weights: np.ndarray,
        jacobian: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        residual = self.residual(predicted, observed, weights)
        sqrt_weight = self._sqrt_irls_weight(residual)
        return (
            self.weighted_jacobian(jacobian, weights) * sqrt_weight.reshape(-1)[:, None],
            -sqrt_weight * residual,
        )

    def phi(self, predicted: np.ndarray, observed: np.ndarray, weights: np.ndarray) -> float:
        residual = self.residual(predicted, observed, weights).reshape(-1)
        eps = max(float(self.epsilon), np.finfo(float).eps)
        return float(np.sum(2.0 * (np.sqrt(residual**2 + eps**2) - eps)))

    def chi2(self, predicted: np.ndarray, observed: np.ndarray, weights: np.ndarray) -> float:
        residual = self.residual(predicted, observed, weights)
        return float(np.mean(residual**2))


@dataclass(frozen=True)
class WeightedLogHuberMisfit:
    """Huber robust misfit in the already-log-transformed data domain."""

    delta: float = 1.0
    epsilon: float = 1.0e-12
    name: str = "weighted_log_huber"

    def residual(self, predicted: np.ndarray, observed: np.ndarray, weights: np.ndarray) -> np.ndarray:
        predicted_array = np.asarray(predicted, dtype=float)
        observed_array = np.asarray(observed, dtype=float)
        weight_array = np.asarray(weights, dtype=float)
        return (predicted_array - observed_array) * weight_array

    def _sqrt_irls_weight(self, residual: np.ndarray) -> np.ndarray:
        delta = max(float(self.delta), np.finfo(float).eps)
        eps = max(float(self.epsilon), np.finfo(float).eps)
        abs_residual = np.abs(residual)
        weight_sq = np.ones_like(abs_residual, dtype=float)
        mask = abs_residual > delta
        weight_sq[mask] = delta / np.maximum(abs_residual[mask], eps)
        return np.sqrt(weight_sq)

    def linearized_rhs(self, predicted: np.ndarray, observed: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return -self.residual(predicted, observed, weights)

    def weighted_jacobian(self, jacobian: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return WeightedLogL2Misfit().weighted_jacobian(jacobian, weights)

    def linearized_system(
        self,
        predicted: np.ndarray,
        observed: np.ndarray,
        weights: np.ndarray,
        jacobian: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        residual = self.residual(predicted, observed, weights)
        sqrt_weight = self._sqrt_irls_weight(residual)
        return (
            self.weighted_jacobian(jacobian, weights) * sqrt_weight.reshape(-1)[:, None],
            -sqrt_weight * residual,
        )

    def phi(self, predicted: np.ndarray, observed: np.ndarray, weights: np.ndarray) -> float:
        residual = self.residual(predicted, observed, weights).reshape(-1)
        delta = max(float(self.delta), np.finfo(float).eps)
        abs_residual = np.abs(residual)
        values = np.where(abs_residual <= delta, residual**2, 2.0 * delta * abs_residual - delta**2)
        return float(np.sum(values))

    def chi2(self, predicted: np.ndarray, observed: np.ndarray, weights: np.ndarray) -> float:
        residual = self.residual(predicted, observed, weights)
        return float(np.mean(residual**2))


@dataclass(frozen=True)
class LogDataDifferenceL2Misfit(WeightedLogL2Misfit):
    """L2 data-difference misfit for time-lapse log apparent resistivities.

    In a time-lapse inversion this is evaluated as differences to a reference
    survey, i.e. ``log(d_t) - log(d_ref)``.  The vector methods inherit the
    weighted L2 behavior so the name can still be resolved by the generic
    registry; the coupled time-lapse assembly lives in ``core.py``.
    """

    name: str = "log_data_difference_l2"


_DATA_MISFITS: dict[str, DataMisfit] = {
    "weighted_log_l2": WeightedLogL2Misfit(),
    "weighted_l2": WeightedLogL2Misfit(),
    "l2": WeightedLogL2Misfit(),
    "log_l2": WeightedLogL2Misfit(),
    "weighted_log_l1": WeightedLogL1Misfit(),
    "log_l1": WeightedLogL1Misfit(),
    "l1": WeightedLogL1Misfit(),
    "weighted_log_huber": WeightedLogHuberMisfit(),
    "log_huber": WeightedLogHuberMisfit(),
    "huber": WeightedLogHuberMisfit(),
    "smooth_l1": WeightedLogHuberMisfit(),
    "log_data_difference_l2": LogDataDifferenceL2Misfit(),
    "data_difference_l2": LogDataDifferenceL2Misfit(),
    "difference_log_l2": LogDataDifferenceL2Misfit(),
    "ratio_log_l2": LogDataDifferenceL2Misfit(),
    "time_lapse_difference_l2": LogDataDifferenceL2Misfit(),
}

_PUBLIC_DATA_MISFITS = (
    "weighted_log_l2",
    "weighted_log_l1",
    "weighted_log_huber",
    "log_data_difference_l2",
)


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def available_data_misfits() -> tuple[str, ...]:
    """Return canonical data-misfit names for user-facing configuration."""

    return tuple(_PUBLIC_DATA_MISFITS)


def build_data_misfit(name: str | DataMisfit) -> DataMisfit:
    """Resolve a data-misfit object from a registered name."""

    if hasattr(name, "residual") and hasattr(name, "linearized_rhs"):
        return name  # type: ignore[return-value]
    key = _normalize_name(str(name))
    try:
        return _DATA_MISFITS[key]
    except KeyError as exc:
        choices = ", ".join(available_data_misfits())
        raise ValueError(f"unknown data_misfit={name!r}; available choices: {choices}") from exc
