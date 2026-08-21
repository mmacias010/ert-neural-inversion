"""Optimizer registries for Deepert inversions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LinearizedOptimizer:
    """Metadata for a linearized inversion update backend."""

    name: str
    description: str


@dataclass(frozen=True)
class OptimizationAlgorithm:
    """Metadata for an outer nonlinear inversion optimizer."""

    name: str
    description: str
    uses_linearized_solver: bool = False
    gradient_based: bool = False


_LINEARIZED_OPTIMIZERS: dict[str, LinearizedOptimizer] = {
    "lsqr": LinearizedOptimizer("lsqr", "SciPy LSQR on the assembled linearized system."),
    "gpu_cgls": LinearizedOptimizer("gpu_cgls", "CuPy CGLS on the assembled linearized system."),
    "normal_cg": LinearizedOptimizer("normal_cg", "SciPy conjugate-gradient solve on normal equations."),
    "pyhydro_cgls": LinearizedOptimizer("pyhydro_cgls", "PyHydroGeophysX-style CGLS on normal equations."),
}

_OPTIMIZATION_ALGORITHMS: dict[str, OptimizationAlgorithm] = {
    "gauss_newton_cgls": OptimizationAlgorithm(
        "gauss_newton_cgls",
        "Damped Gauss-Newton update solved with the configured CGLS/LSQR backend.",
        uses_linearized_solver=True,
    ),
    "levenberg_marquardt": OptimizationAlgorithm(
        "levenberg_marquardt",
        "Levenberg-Marquardt damped Gauss-Newton update.",
        uses_linearized_solver=True,
    ),
    "lbfgs": OptimizationAlgorithm(
        "lbfgs",
        "Limited-memory BFGS using Deepert Jacobian gradients and line search.",
        gradient_based=True,
    ),
    "lbfgs_b": OptimizationAlgorithm(
        "lbfgs_b",
        "Projected limited-memory BFGS with model-bound clipping.",
        gradient_based=True,
    ),
    "nonlinear_cg": OptimizationAlgorithm(
        "nonlinear_cg",
        "Nonlinear conjugate-gradient descent using Polak-Ribiere updates.",
        gradient_based=True,
    ),
    "adam": OptimizationAlgorithm(
        "adam",
        "Adam first-order optimizer using Deepert Jacobian gradients.",
        gradient_based=True,
    ),
}


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def available_linearized_optimizers() -> tuple[str, ...]:
    """Return canonical linearized solver names."""

    return tuple(sorted(_LINEARIZED_OPTIMIZERS))


def available_optimization_algorithms() -> tuple[str, ...]:
    """Return canonical outer optimization algorithm names."""

    return tuple(sorted(_OPTIMIZATION_ALGORITHMS))


def build_linearized_optimizer(name: str | LinearizedOptimizer) -> LinearizedOptimizer:
    """Resolve a linearized optimizer from a registered name."""

    if isinstance(name, LinearizedOptimizer):
        return name
    key = _normalize_name(str(name))
    try:
        return _LINEARIZED_OPTIMIZERS[key]
    except KeyError as exc:
        choices = ", ".join(available_linearized_optimizers())
        raise ValueError(f"unknown linearized_solver={name!r}; available choices: {choices}") from exc


def build_optimization_algorithm(name: str | OptimizationAlgorithm) -> OptimizationAlgorithm:
    """Resolve an outer optimization algorithm from a registered name."""

    if isinstance(name, OptimizationAlgorithm):
        return name
    key = _normalize_name(str(name))
    try:
        return _OPTIMIZATION_ALGORITHMS[key]
    except KeyError as exc:
        choices = ", ".join(available_optimization_algorithms())
        raise ValueError(f"unknown optimizer={name!r}; available choices: {choices}") from exc
