"""Petrophysical transforms for inversion model parameters.

The inversion core only needs a mapping from an optimization state to
log-resistivity plus the local derivative of that mapping.  This module keeps
the physical parameterization separate from the ERT solver, so future moisture,
conductivity, or rock-physics relations can be added without changing the
optimizer and regularization code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


ArrayLike = object


class PetrophysicalTransform(Protocol):
    """Interface used by the inversion core."""

    name: str
    parameter_name: str

    def state_from_log_resistivity(self, log_resistivity: np.ndarray) -> np.ndarray:
        """Convert a log-resistivity starting model to optimizer state."""

    def log_resistivity_from_state(self, state: np.ndarray) -> np.ndarray:
        """Map optimizer state to cell log-resistivity."""

    def d_log_resistivity_d_state(self, state: np.ndarray) -> np.ndarray:
        """Return the diagonal chain-rule factor d log(rho) / d state."""

    def clip_state(self, state: np.ndarray) -> np.ndarray:
        """Project state back to admissible values when needed."""

    def parameter_from_state(self, state: np.ndarray) -> np.ndarray:
        """Return the physical parameter represented by the state."""

    def d_parameter_d_state(self, state: np.ndarray) -> np.ndarray:
        """Return the diagonal chain-rule factor d(parameter) / d(state)."""


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def _log_bounds(bounds: tuple[float, float] | None) -> tuple[float, float] | None:
    if bounds is None:
        return None
    lo, hi = bounds
    if not (0.0 < lo < hi):
        raise ValueError("model_bounds must be positive and ordered as (min, max)")
    return float(np.log(lo)), float(np.log(hi))


def _clip_log_resistivity(log_resistivity: np.ndarray, bounds: tuple[float, float] | None) -> np.ndarray:
    log_bounds = _log_bounds(bounds)
    values = np.asarray(log_resistivity, dtype=float)
    if log_bounds is None:
        return values
    lo, hi = log_bounds
    return np.clip(values, lo, hi)


def _sigmoid(state: np.ndarray) -> np.ndarray:
    state_array = np.asarray(state, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(state_array, -60.0, 60.0)))


def _logit(probability: np.ndarray) -> np.ndarray:
    eps = 1.0e-12
    p = np.clip(np.asarray(probability, dtype=float), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def _parameter_array(value: ArrayLike, *, n_cells: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full((int(n_cells),), float(array), dtype=float)
    else:
        array = np.asarray(array, dtype=float).reshape(-1)
    if array.shape != (int(n_cells),):
        raise ValueError(f"{name} must be scalar or have shape ({int(n_cells)},)")
    return array


def _parameter_array_or_matrix(value: ArrayLike, *, n_cells: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full((int(n_cells),), float(array), dtype=float)
    if array.ndim == 1:
        result = np.asarray(array, dtype=float).reshape(-1)
        if result.shape != (int(n_cells),):
            raise ValueError(f"{name} must be scalar, cell-wise, or have first dimension {int(n_cells)}")
        return result
    if array.shape[0] != int(n_cells):
        raise ValueError(f"{name} first dimension must be {int(n_cells)}, got {array.shape}")
    return np.asarray(array, dtype=float)


def _as_parameter_shape(parameter: np.ndarray, state: np.ndarray) -> np.ndarray:
    if np.asarray(state).ndim == 2:
        return parameter.reshape(-1, 1)
    return parameter


@dataclass(frozen=True)
class LogResistivityTransform:
    """Historical Deepert parameterization: state is log-resistivity."""

    model_transform: str = "log"
    model_bounds: tuple[float, float] | None = None
    name: str = "log_resistivity"
    parameter_name: str = "resistivity"

    def state_from_log_resistivity(self, log_resistivity: np.ndarray) -> np.ndarray:
        values = np.asarray(log_resistivity, dtype=float)
        if self.model_transform == "log":
            return _clip_log_resistivity(values, self.model_bounds)
        if self.model_transform != "log_lu":
            raise ValueError("model_transform must be 'log' or 'log_lu'")
        if self.model_bounds is None:
            raise ValueError("model_bounds are required for model_transform='log_lu'")
        lo, hi = self.model_bounds
        span = hi - lo
        rho = np.exp(values)
        rho = np.clip(rho, lo + span * 1.0e-12, hi - span * 1.0e-12)
        return np.log(rho - lo) - np.log(hi - rho)

    def log_resistivity_from_state(self, state: np.ndarray) -> np.ndarray:
        state_array = np.asarray(state, dtype=float)
        if self.model_transform == "log":
            return _clip_log_resistivity(state_array, self.model_bounds)
        if self.model_transform != "log_lu":
            raise ValueError("model_transform must be 'log' or 'log_lu'")
        if self.model_bounds is None:
            raise ValueError("model_bounds are required for model_transform='log_lu'")
        lo, hi = self.model_bounds
        exp_state = np.exp(np.clip(state_array, -50.0, 50.0))
        rho = (exp_state * hi + lo) / (exp_state + 1.0)
        return np.log(rho)

    def d_log_resistivity_d_state(self, state: np.ndarray) -> np.ndarray:
        state_array = np.asarray(state, dtype=float)
        if self.model_transform == "log":
            return np.ones_like(state_array, dtype=float)
        if self.model_bounds is None:
            raise ValueError("model_bounds are required for model_transform='log_lu'")
        lo, hi = self.model_bounds
        rho = np.exp(self.log_resistivity_from_state(state_array))
        return ((rho - lo) * (hi - rho)) / ((hi - lo) * rho)

    def clip_state(self, state: np.ndarray) -> np.ndarray:
        state_array = np.asarray(state, dtype=float)
        if self.model_transform == "log":
            return _clip_log_resistivity(state_array, self.model_bounds)
        return state_array

    def parameter_from_state(self, state: np.ndarray) -> np.ndarray:
        return np.exp(self.log_resistivity_from_state(state))

    def d_parameter_d_state(self, state: np.ndarray) -> np.ndarray:
        rho = self.parameter_from_state(state)
        return rho * self.d_log_resistivity_d_state(state)


@dataclass(frozen=True)
class LogConductivityTransform:
    """Conductivity parameterization: state is log-conductivity."""

    model_bounds: tuple[float, float] | None = None
    name: str = "log_conductivity"
    parameter_name: str = "conductivity"

    def state_from_log_resistivity(self, log_resistivity: np.ndarray) -> np.ndarray:
        return self.clip_state(-np.asarray(log_resistivity, dtype=float))

    def log_resistivity_from_state(self, state: np.ndarray) -> np.ndarray:
        return -self.clip_state(np.asarray(state, dtype=float))

    def d_log_resistivity_d_state(self, state: np.ndarray) -> np.ndarray:
        return -np.ones_like(np.asarray(state, dtype=float), dtype=float)

    def clip_state(self, state: np.ndarray) -> np.ndarray:
        state_array = np.asarray(state, dtype=float)
        log_bounds = _log_bounds(self.model_bounds)
        if log_bounds is None:
            return state_array
        rho_lo, rho_hi = log_bounds
        return np.clip(state_array, -rho_hi, -rho_lo)

    def parameter_from_state(self, state: np.ndarray) -> np.ndarray:
        return np.exp(self.clip_state(state))

    def d_parameter_d_state(self, state: np.ndarray) -> np.ndarray:
        return self.parameter_from_state(state)


@dataclass(frozen=True)
class SaturationTransform:
    """Saturation parameterization with a bounded sigmoid state."""

    rho_sat: np.ndarray
    n: np.ndarray
    rho_sat_s: np.ndarray | None = None
    saturation_floor: float = 1.0e-4
    name: str = "saturation"
    parameter_name: str = "saturation"

    def __post_init__(self) -> None:
        if not (0.0 < float(self.saturation_floor) < 1.0):
            raise ValueError("saturation_floor must be in (0, 1)")
        if np.any(np.asarray(self.rho_sat, dtype=float) <= 0.0):
            raise ValueError("rho_sat must be positive")
        if np.any(np.asarray(self.n, dtype=float) <= 0.0):
            raise ValueError("n must be positive")

    def _surface_sigma(self) -> tuple[np.ndarray, np.ndarray]:
        sigma_sat = 1.0 / np.asarray(self.rho_sat, dtype=float)
        if self.rho_sat_s is None:
            return sigma_sat, np.zeros_like(sigma_sat)
        rho_sat_s = np.asarray(self.rho_sat_s, dtype=float)
        has_surface = np.isfinite(rho_sat_s) & (rho_sat_s > 0.0)
        sigma_s = np.zeros_like(sigma_sat)
        sigma_s[has_surface] = 1.0 / rho_sat_s[has_surface]
        sigma_p = sigma_sat - sigma_s
        if np.any(sigma_p <= 0.0):
            raise ValueError("rho_sat_s must be larger than rho_sat where surface conduction is used")
        return sigma_p, sigma_s

    def _saturation_from_state(self, state: np.ndarray) -> np.ndarray:
        normalized = _sigmoid(np.asarray(state, dtype=float))
        floor = float(self.saturation_floor)
        return floor + (1.0 - floor) * normalized

    def _sigma_from_saturation(self, saturation: np.ndarray) -> np.ndarray:
        sat = np.asarray(saturation, dtype=float)
        rho_sat = _as_parameter_shape(np.asarray(self.rho_sat, dtype=float), sat)
        n_values = _as_parameter_shape(np.asarray(self.n, dtype=float), sat)
        sigma_p, sigma_s = self._surface_sigma()
        sigma_p = _as_parameter_shape(sigma_p, sat)
        sigma_s = _as_parameter_shape(sigma_s, sat)
        if rho_sat.shape[0] != sat.shape[0]:
            raise ValueError("saturation state first dimension does not match petrophysical parameter count")
        return sigma_p * np.power(sat, n_values) + sigma_s * np.power(sat, n_values - 1.0)

    def state_from_log_resistivity(self, log_resistivity: np.ndarray) -> np.ndarray:
        target_sigma = np.exp(-np.asarray(log_resistivity, dtype=float))
        floor = float(self.saturation_floor)
        lo = np.full_like(target_sigma, floor, dtype=float)
        hi = np.ones_like(target_sigma, dtype=float)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            sigma_mid = self._sigma_from_saturation(mid)
            lo = np.where(sigma_mid < target_sigma, mid, lo)
            hi = np.where(sigma_mid >= target_sigma, mid, hi)
        saturation = np.clip(0.5 * (lo + hi), floor, 1.0)
        normalized = (saturation - floor) / (1.0 - floor)
        return _logit(normalized)

    def log_resistivity_from_state(self, state: np.ndarray) -> np.ndarray:
        sigma = self._sigma_from_saturation(self._saturation_from_state(state))
        return -np.log(np.maximum(sigma, np.finfo(float).tiny))

    def d_log_resistivity_d_state(self, state: np.ndarray) -> np.ndarray:
        state_array = np.asarray(state, dtype=float)
        saturation = self._saturation_from_state(state_array)
        n_values = _as_parameter_shape(np.asarray(self.n, dtype=float), saturation)
        sigma_p, sigma_s = self._surface_sigma()
        sigma_p = _as_parameter_shape(sigma_p, saturation)
        sigma_s = _as_parameter_shape(sigma_s, saturation)
        sigma = self._sigma_from_saturation(saturation)
        d_sigma_d_s = sigma_p * n_values * np.power(saturation, n_values - 1.0)
        has_surface_term = sigma_s != 0.0
        if np.any(has_surface_term):
            d_sigma_d_s = d_sigma_d_s + sigma_s * (n_values - 1.0) * np.power(saturation, n_values - 2.0)
        floor = float(self.saturation_floor)
        d_s_d_state = (saturation - floor) * (1.0 - saturation) / (1.0 - floor)
        return -(d_sigma_d_s / np.maximum(sigma, np.finfo(float).tiny)) * d_s_d_state

    def clip_state(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=float)

    def parameter_from_state(self, state: np.ndarray) -> np.ndarray:
        return self._saturation_from_state(state)

    def d_parameter_d_state(self, state: np.ndarray) -> np.ndarray:
        saturation = self._saturation_from_state(state)
        floor = float(self.saturation_floor)
        return (saturation - floor) * (1.0 - saturation) / (1.0 - floor)


@dataclass(frozen=True)
class WaterContentTransform:
    """Volumetric water-content parameterization using the saturation relation.

    The optimizer state is mapped to bounded saturation first, then converted to
    water content as ``theta = phi * S``.  The ERT model still uses the same
    unit-specific petrophysical relation as ``SaturationTransform``:

        sigma = sigma_p * S**n + sigma_s * S**(n - 1)

    This keeps the physical chain differentiable while making ``theta`` the
    reported and regularized inversion parameter.
    """

    rho_sat: np.ndarray
    n: np.ndarray
    phi: np.ndarray
    rho_sat_s: np.ndarray | None = None
    saturation_floor: float = 1.0e-4
    name: str = "water_content"
    parameter_name: str = "water_content"

    def __post_init__(self) -> None:
        if not (0.0 < float(self.saturation_floor) < 1.0):
            raise ValueError("saturation_floor must be in (0, 1)")
        if np.any(np.asarray(self.rho_sat, dtype=float) <= 0.0):
            raise ValueError("rho_sat must be positive")
        if np.any(np.asarray(self.n, dtype=float) <= 0.0):
            raise ValueError("n must be positive")
        phi = np.asarray(self.phi, dtype=float)
        if np.any(phi <= 0.0) or not np.all(np.isfinite(phi)):
            raise ValueError("phi must be positive and finite")

    def _surface_sigma(self) -> tuple[np.ndarray, np.ndarray]:
        sigma_sat = 1.0 / np.asarray(self.rho_sat, dtype=float)
        if self.rho_sat_s is None:
            return sigma_sat, np.zeros_like(sigma_sat)
        rho_sat_s = np.asarray(self.rho_sat_s, dtype=float)
        has_surface = np.isfinite(rho_sat_s) & (rho_sat_s > 0.0)
        sigma_s = np.zeros_like(sigma_sat)
        sigma_s[has_surface] = 1.0 / rho_sat_s[has_surface]
        sigma_p = sigma_sat - sigma_s
        if np.any(sigma_p <= 0.0):
            raise ValueError("rho_sat_s must be larger than rho_sat where surface conduction is used")
        return sigma_p, sigma_s

    def _saturation_from_state(self, state: np.ndarray) -> np.ndarray:
        normalized = _sigmoid(np.asarray(state, dtype=float))
        floor = float(self.saturation_floor)
        return floor + (1.0 - floor) * normalized

    def _sigma_from_saturation(self, saturation: np.ndarray) -> np.ndarray:
        sat = np.asarray(saturation, dtype=float)
        rho_sat = _as_parameter_shape(np.asarray(self.rho_sat, dtype=float), sat)
        n_values = _as_parameter_shape(np.asarray(self.n, dtype=float), sat)
        sigma_p, sigma_s = self._surface_sigma()
        sigma_p = _as_parameter_shape(sigma_p, sat)
        sigma_s = _as_parameter_shape(sigma_s, sat)
        if rho_sat.shape[0] != sat.shape[0]:
            raise ValueError("water-content state first dimension does not match petrophysical parameter count")
        return sigma_p * np.power(sat, n_values) + sigma_s * np.power(sat, n_values - 1.0)

    def state_from_log_resistivity(self, log_resistivity: np.ndarray) -> np.ndarray:
        target_sigma = np.exp(-np.asarray(log_resistivity, dtype=float))
        floor = float(self.saturation_floor)
        lo = np.full_like(target_sigma, floor, dtype=float)
        hi = np.ones_like(target_sigma, dtype=float)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            sigma_mid = self._sigma_from_saturation(mid)
            lo = np.where(sigma_mid < target_sigma, mid, lo)
            hi = np.where(sigma_mid >= target_sigma, mid, hi)
        saturation = np.clip(0.5 * (lo + hi), floor, 1.0)
        normalized = (saturation - floor) / (1.0 - floor)
        return _logit(normalized)

    def log_resistivity_from_state(self, state: np.ndarray) -> np.ndarray:
        sigma = self._sigma_from_saturation(self._saturation_from_state(state))
        return -np.log(np.maximum(sigma, np.finfo(float).tiny))

    def d_log_resistivity_d_state(self, state: np.ndarray) -> np.ndarray:
        state_array = np.asarray(state, dtype=float)
        saturation = self._saturation_from_state(state_array)
        n_values = _as_parameter_shape(np.asarray(self.n, dtype=float), saturation)
        sigma_p, sigma_s = self._surface_sigma()
        sigma_p = _as_parameter_shape(sigma_p, saturation)
        sigma_s = _as_parameter_shape(sigma_s, saturation)
        sigma = self._sigma_from_saturation(saturation)
        d_sigma_d_s = sigma_p * n_values * np.power(saturation, n_values - 1.0)
        has_surface_term = sigma_s != 0.0
        if np.any(has_surface_term):
            d_sigma_d_s = d_sigma_d_s + sigma_s * (n_values - 1.0) * np.power(saturation, n_values - 2.0)
        floor = float(self.saturation_floor)
        d_s_d_state = (saturation - floor) * (1.0 - saturation) / (1.0 - floor)
        return -(d_sigma_d_s / np.maximum(sigma, np.finfo(float).tiny)) * d_s_d_state

    def clip_state(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=float)

    def parameter_from_state(self, state: np.ndarray) -> np.ndarray:
        saturation = self._saturation_from_state(state)
        phi = _as_parameter_shape(np.asarray(self.phi, dtype=float), saturation)
        return phi * saturation

    def d_parameter_d_state(self, state: np.ndarray) -> np.ndarray:
        saturation = self._saturation_from_state(state)
        phi = _as_parameter_shape(np.asarray(self.phi, dtype=float), saturation)
        floor = float(self.saturation_floor)
        d_s_d_state = (saturation - floor) * (1.0 - saturation) / (1.0 - floor)
        return phi * d_s_d_state


@dataclass(frozen=True)
class RelativeArchieWaterContentTransform:
    """Relative Archie parameterization with bounded volumetric water content.

    The physical mapping is

        log(rho_t) = log(rho0) - n * log(theta_t / theta0) - log(C_T)

    where ``rho0`` is the baseline model at the reference temperature, ``theta0``
    is the baseline water-content model, and ``C_T`` is the temperature
    correction factor used to map field-temperature resistivity to the reference
    temperature.  If no temperature factor is provided, ``C_T = 1``.

    The optimizer state is an unconstrained variable mapped to ``theta`` through
    a sigmoid, keeping water content inside ``[theta_min, theta_max]``.
    """

    rho0: np.ndarray
    theta0: np.ndarray
    n: np.ndarray
    theta_min: np.ndarray
    theta_max: np.ndarray
    temperature_correction_factor: np.ndarray | None = None
    name: str = "relative_archie_water_content"
    parameter_name: str = "water_content"

    def __post_init__(self) -> None:
        rho0 = np.asarray(self.rho0, dtype=float)
        theta0 = np.asarray(self.theta0, dtype=float)
        n_values = np.asarray(self.n, dtype=float)
        theta_min = np.asarray(self.theta_min, dtype=float)
        theta_max = np.asarray(self.theta_max, dtype=float)
        if np.any(rho0 <= 0.0) or not np.all(np.isfinite(rho0)):
            raise ValueError("rho0 must be positive and finite")
        if np.any(theta0 <= 0.0) or not np.all(np.isfinite(theta0)):
            raise ValueError("theta0 must be positive and finite")
        if np.any(n_values <= 0.0) or not np.all(np.isfinite(n_values)):
            raise ValueError("n must be positive and finite")
        if np.any(theta_min <= 0.0) or np.any(theta_max <= theta_min):
            raise ValueError("theta bounds must satisfy 0 < theta_min < theta_max")
        if np.any(theta0 <= theta_min) or np.any(theta0 >= theta_max):
            raise ValueError("theta0 must lie strictly inside [theta_min, theta_max]")
        if self.temperature_correction_factor is not None:
            factor = np.asarray(self.temperature_correction_factor, dtype=float)
            if np.any(factor <= 0.0) or not np.all(np.isfinite(factor)):
                raise ValueError("temperature_correction_factor must be positive and finite")

    def _temperature_factor(self, state: np.ndarray) -> np.ndarray:
        state_array = np.asarray(state, dtype=float)
        if self.temperature_correction_factor is None:
            return np.ones_like(state_array, dtype=float)
        factor = np.asarray(self.temperature_correction_factor, dtype=float)
        if factor.ndim == 0:
            return np.full_like(state_array, float(factor), dtype=float)
        if factor.ndim == 1:
            return _as_parameter_shape(factor, state_array)
        if factor.shape != state_array.shape:
            raise ValueError(
                "temperature_correction_factor must be scalar, cell-wise, or match the state shape "
                f"({factor.shape} != {state_array.shape})"
            )
        return factor

    def _theta_from_state(self, state: np.ndarray) -> np.ndarray:
        state_array = np.asarray(state, dtype=float)
        theta_min = _as_parameter_shape(np.asarray(self.theta_min, dtype=float), state_array)
        theta_max = _as_parameter_shape(np.asarray(self.theta_max, dtype=float), state_array)
        return theta_min + (theta_max - theta_min) * _sigmoid(state_array)

    def state_from_log_resistivity(self, log_resistivity: np.ndarray) -> np.ndarray:
        log_rho = np.asarray(log_resistivity, dtype=float)
        rho0 = _as_parameter_shape(np.asarray(self.rho0, dtype=float), log_rho)
        theta0 = _as_parameter_shape(np.asarray(self.theta0, dtype=float), log_rho)
        n_values = _as_parameter_shape(np.asarray(self.n, dtype=float), log_rho)
        theta_min = _as_parameter_shape(np.asarray(self.theta_min, dtype=float), log_rho)
        theta_max = _as_parameter_shape(np.asarray(self.theta_max, dtype=float), log_rho)
        temperature_factor = self._temperature_factor(log_rho)
        theta = theta0 * np.exp(-(log_rho + np.log(temperature_factor) - np.log(rho0)) / n_values)
        span = theta_max - theta_min
        theta = np.clip(theta, theta_min + span * 1.0e-12, theta_max - span * 1.0e-12)
        return _logit((theta - theta_min) / span)

    def log_resistivity_from_state(self, state: np.ndarray) -> np.ndarray:
        state_array = np.asarray(state, dtype=float)
        theta = self._theta_from_state(state_array)
        rho0 = _as_parameter_shape(np.asarray(self.rho0, dtype=float), state_array)
        theta0 = _as_parameter_shape(np.asarray(self.theta0, dtype=float), state_array)
        n_values = _as_parameter_shape(np.asarray(self.n, dtype=float), state_array)
        temperature_factor = self._temperature_factor(state_array)
        return np.log(rho0) - n_values * (np.log(theta) - np.log(theta0)) - np.log(temperature_factor)

    def d_log_resistivity_d_state(self, state: np.ndarray) -> np.ndarray:
        state_array = np.asarray(state, dtype=float)
        theta = self._theta_from_state(state_array)
        theta_min = _as_parameter_shape(np.asarray(self.theta_min, dtype=float), state_array)
        theta_max = _as_parameter_shape(np.asarray(self.theta_max, dtype=float), state_array)
        n_values = _as_parameter_shape(np.asarray(self.n, dtype=float), state_array)
        d_theta_d_state = (theta - theta_min) * (theta_max - theta) / (theta_max - theta_min)
        return -n_values * d_theta_d_state / theta

    def clip_state(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=float)

    def parameter_from_state(self, state: np.ndarray) -> np.ndarray:
        return self._theta_from_state(state)

    def d_parameter_d_state(self, state: np.ndarray) -> np.ndarray:
        state_array = np.asarray(state, dtype=float)
        theta = self._theta_from_state(state_array)
        theta_min = _as_parameter_shape(np.asarray(self.theta_min, dtype=float), state_array)
        theta_max = _as_parameter_shape(np.asarray(self.theta_max, dtype=float), state_array)
        return (theta - theta_min) * (theta_max - theta) / (theta_max - theta_min)


def available_petrophysical_transforms() -> tuple[str, ...]:
    """Return user-facing petrophysical transform names."""

    return (
        "log_resistivity",
        "log_conductivity",
        "saturation",
        "water_content",
        "relative_archie_water_content",
    )


def build_petrophysical_transform(
    name: str | PetrophysicalTransform,
    *,
    n_cells: int,
    model_transform: str = "log",
    model_bounds: tuple[float, float] | None = None,
    saturation_floor: float = 1.0e-4,
    parameters: dict[str, ArrayLike] | None = None,
) -> PetrophysicalTransform:
    """Resolve and instantiate a petrophysical transform."""

    if hasattr(name, "log_resistivity_from_state") and hasattr(name, "d_log_resistivity_d_state"):
        return name  # type: ignore[return-value]

    key = _normalize_name(str(name))
    if key in ("log_resistivity", "resistivity", "rho"):
        return LogResistivityTransform(model_transform=model_transform, model_bounds=model_bounds)
    if key in ("log_conductivity", "conductivity", "sigma"):
        if model_transform != "log":
            raise ValueError("log_conductivity currently supports model_transform='log' only")
        return LogConductivityTransform(model_bounds=model_bounds)
    if key in ("saturation", "water_saturation"):
        if model_transform != "log":
            raise ValueError("saturation currently supports model_transform='log' only")
        params = parameters or {}
        missing = [param for param in ("rho_sat", "n") if param not in params]
        if missing:
            raise ValueError(f"saturation transform requires petrophysical parameters: {missing}")
        rho_sat = _parameter_array(params["rho_sat"], n_cells=n_cells, name="rho_sat")
        n_values = _parameter_array(params["n"], n_cells=n_cells, name="n")
        rho_sat_s = None
        if "rho_sat_s" in params and params["rho_sat_s"] is not None:
            rho_sat_s = _parameter_array(params["rho_sat_s"], n_cells=n_cells, name="rho_sat_s")
        return SaturationTransform(
            rho_sat=rho_sat,
            rho_sat_s=rho_sat_s,
            n=n_values,
            saturation_floor=saturation_floor,
        )
    if key in ("water_content", "theta", "archie_water_content", "absolute_archie_water_content"):
        if model_transform != "log":
            raise ValueError("water_content currently supports model_transform='log' only")
        params = parameters or {}
        missing = [param for param in ("rho_sat", "n", "phi") if param not in params]
        if missing:
            raise ValueError(f"water_content transform requires petrophysical parameters: {missing}")
        rho_sat = _parameter_array(params["rho_sat"], n_cells=n_cells, name="rho_sat")
        n_values = _parameter_array(params["n"], n_cells=n_cells, name="n")
        phi = _parameter_array(params["phi"], n_cells=n_cells, name="phi")
        rho_sat_s = None
        if "rho_sat_s" in params and params["rho_sat_s"] is not None:
            rho_sat_s = _parameter_array(params["rho_sat_s"], n_cells=n_cells, name="rho_sat_s")
        return WaterContentTransform(
            rho_sat=rho_sat,
            rho_sat_s=rho_sat_s,
            n=n_values,
            phi=phi,
            saturation_floor=saturation_floor,
        )
    if key in ("relative_archie_water_content", "relative_archie"):
        if model_transform != "log":
            raise ValueError("relative_archie_water_content currently supports model_transform='log' only")
        params = parameters or {}
        missing = [param for param in ("rho0", "theta0", "n") if param not in params]
        if missing:
            raise ValueError(f"relative_archie_water_content requires petrophysical parameters: {missing}")
        rho0 = _parameter_array(params["rho0"], n_cells=n_cells, name="rho0")
        theta0 = _parameter_array(params["theta0"], n_cells=n_cells, name="theta0")
        n_values = _parameter_array(params["n"], n_cells=n_cells, name="n")
        theta_min = _parameter_array(params.get("theta_min", 0.02), n_cells=n_cells, name="theta_min")
        theta_max = _parameter_array(params.get("theta_max", 0.5), n_cells=n_cells, name="theta_max")
        temperature_correction_factor = None
        if "temperature_correction_factor" in params and params["temperature_correction_factor"] is not None:
            temperature_correction_factor = _parameter_array_or_matrix(
                params["temperature_correction_factor"],
                n_cells=n_cells,
                name="temperature_correction_factor",
            )
        return RelativeArchieWaterContentTransform(
            rho0=rho0,
            theta0=theta0,
            n=n_values,
            theta_min=theta_min,
            theta_max=theta_max,
            temperature_correction_factor=temperature_correction_factor,
        )

    choices = ", ".join(available_petrophysical_transforms())
    raise ValueError(f"unknown petrophysical_transform={name!r}; available choices: {choices}")
