from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from deepert.inversion.core import (
    InversionConfig,
    _linearized_objective_gradient,
    _optimizer_increment,
    _solve_increment,
)


def test_gpu_cgls_matches_lsqr_when_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    try:
        import cupy  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"CuPy is not available: {exc}")

    rng = np.random.default_rng(1234)
    dense = rng.normal(size=(18, 7))
    dense[::3, 2] = 0.0
    matrix = sp.csr_matrix(dense)
    expected = rng.normal(size=7)
    rhs = dense @ expected

    lsqr_solution = _solve_increment(
        matrix,
        rhs,
        InversionConfig(linearized_solver="lsqr", lsqr_atol=1e-10, lsqr_btol=1e-10),
    )
    gpu_solution = _solve_increment(
        matrix,
        rhs,
        InversionConfig(linearized_solver="gpu_cgls", cgls_max_iterations=200, cgls_tolerance=1e-12),
    )

    np.testing.assert_allclose(gpu_solution, lsqr_solution, rtol=1.0e-6, atol=1.0e-7)


def test_outer_optimizers_return_finite_descent_steps() -> None:
    matrix = sp.csr_matrix(
        np.asarray(
            [
                [2.0, 0.0],
                [0.0, 1.5],
                [1.0, -0.5],
            ]
        )
    )
    rhs = np.asarray([1.0, -0.5, 0.25])
    current = np.asarray([0.2, -0.1])
    gradient = _linearized_objective_gradient(matrix, rhs)

    gn = _optimizer_increment(
        matrix,
        rhs,
        current_state=current,
        optimizer_state={},
        config=InversionConfig(
            optimization_algorithm="gauss_newton_cgls",
            linearized_solver="lsqr",
            max_log_step=None,
        ),
    )
    expected = _solve_increment(matrix, rhs, InversionConfig(linearized_solver="lsqr"))
    np.testing.assert_allclose(gn, expected)

    for optimizer in ("levenberg_marquardt", "lbfgs", "lbfgs_b", "nonlinear_cg", "adam"):
        step = _optimizer_increment(
            matrix,
            rhs,
            current_state=current,
            optimizer_state={},
            config=InversionConfig(
                optimization_algorithm=optimizer,
                linearized_solver="lsqr",
                max_log_step=None,
                optimizer_max_step=1.0,
            ),
        )
        assert np.all(np.isfinite(step))
        assert step.shape == current.shape
        if optimizer != "levenberg_marquardt":
            assert float(np.dot(step, gradient)) < 0.0
