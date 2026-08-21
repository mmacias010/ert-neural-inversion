from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from deepert.inversion import (
    build_data_misfit,
    build_linearized_optimizer,
    build_optimization_algorithm,
    build_petrophysical_transform,
    build_spatial_regularization,
    build_temporal_regularization,
)
from deepert.inversion.core import (
    InversionConfig,
    _log_data_difference_chi2,
    _regularization_domain_value_and_derivative,
    _timelapse_data_system,
)
from deepert.inversion.regularization import build_spatial_regularization_matrix
from deepert.mesh import Mesh


def test_weighted_log_l2_misfit_matches_historical_formulas() -> None:
    misfit = build_data_misfit("weighted-log-l2")
    predicted = np.asarray([1.0, 2.0])
    observed = np.asarray([0.5, 2.5])
    weights = np.asarray([2.0, 0.5])
    jacobian = np.asarray([[1.0, 2.0], [3.0, 4.0]])

    residual = np.asarray([1.0, -0.25])
    np.testing.assert_allclose(misfit.residual(predicted, observed, weights), residual)
    np.testing.assert_allclose(misfit.linearized_rhs(predicted, observed, weights), -residual)
    np.testing.assert_allclose(misfit.weighted_jacobian(jacobian, weights), jacobian * weights[:, None])
    system_matrix, system_rhs = misfit.linearized_system(predicted, observed, weights, jacobian)
    np.testing.assert_allclose(system_matrix, jacobian * weights[:, None])
    np.testing.assert_allclose(system_rhs, -residual)
    assert misfit.phi(predicted, observed, weights) == pytest.approx(float(np.dot(residual, residual)))
    assert misfit.chi2(predicted, observed, weights) == pytest.approx(float(np.mean(residual**2)))


def test_robust_misfit_interfaces_build_irls_systems() -> None:
    predicted = np.asarray([0.0, 3.0])
    observed = np.asarray([0.0, 0.0])
    weights = np.ones(2)
    jacobian = np.eye(2)

    huber = build_data_misfit("huber")
    huber_matrix, huber_rhs = huber.linearized_system(predicted, observed, weights, jacobian)
    np.testing.assert_allclose(huber_matrix, np.diag([1.0, np.sqrt(1.0 / 3.0)]))
    np.testing.assert_allclose(huber_rhs, np.asarray([-0.0, -np.sqrt(1.0 / 3.0) * 3.0]))
    assert huber.phi(predicted, observed, weights) == pytest.approx(5.0)

    l1 = build_data_misfit("l1")
    l1_matrix, l1_rhs = l1.linearized_system(predicted, observed, weights, jacobian)
    assert np.all(np.isfinite(l1_matrix))
    assert np.all(np.isfinite(l1_rhs))
    assert l1.phi(predicted, observed, weights) > 0.0


def test_log_data_difference_l2_builds_coupled_timelapse_system() -> None:
    misfit = build_data_misfit("difference-log-l2")
    predicted = np.asarray([[1.0], [1.4], [1.7]])
    observed = np.asarray([[0.9], [1.2], [1.8]])
    weights = np.ones_like(predicted)
    jacobians = [np.asarray([[2.0]]), np.asarray([[3.0]]), np.asarray([[5.0]])]

    matrix, rhs = _timelapse_data_system(misfit, predicted, observed, weights, jacobians)
    scale = 1.0 / np.sqrt(2.0)
    expected_matrix = np.asarray(
        [
            [2.0, 0.0, 0.0],
            [-2.0 * scale, 3.0 * scale, 0.0],
            [-2.0 * scale, 0.0, 5.0 * scale],
        ]
    )
    expected_rhs = -np.asarray([0.1, 0.1 * scale, -0.2 * scale])

    np.testing.assert_allclose(matrix.toarray(), expected_matrix)
    np.testing.assert_allclose(rhs, expected_rhs)
    assert _log_data_difference_chi2(predicted, observed, weights) == pytest.approx(np.mean(expected_rhs**2))


def test_spatial_regularization_registry_builds_current_matrices() -> None:
    nodes = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, -1.0],
            [0.0, -1.0],
        ],
        dtype=float,
    )
    cells = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh = Mesh.from_arrays(nodes, cells)
    forward = SimpleNamespace(mesh=mesh)

    identity = build_spatial_regularization("identity").matrix(forward, 2).toarray()
    first_order_reg = build_spatial_regularization("first-order")
    first_order = build_spatial_regularization_matrix("first-order", forward, 2).toarray()
    system_matrix, system_rhs = first_order_reg.linearized_system(
        forward,
        np.asarray([1.0, 4.0]),
        2,
        reference_roughness=np.asarray([0.0]),
        scale=2.0,
    )

    np.testing.assert_allclose(identity, np.eye(2))
    np.testing.assert_allclose(first_order, np.asarray([[1.0, -1.0]]))
    np.testing.assert_allclose(system_matrix.toarray(), np.asarray([[2.0, -2.0]]))
    np.testing.assert_allclose(system_rhs, np.asarray([6.0]))


def test_structural_prior_spatial_regularization_weakens_cross_unit_edges() -> None:
    nodes = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, -1.0],
            [0.0, -1.0],
        ],
        dtype=float,
    )
    cells = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh = Mesh.from_arrays(nodes, cells)
    forward = SimpleNamespace(mesh=mesh, structural_prior_cell_ids=np.asarray([1, 2]), structural_cross_weight=0.25)

    structural = build_spatial_regularization("structural-prior").matrix(forward, 2).toarray()
    np.testing.assert_allclose(structural, np.asarray([[0.25, -0.25]]))


def test_robust_spatial_regularization_builds_irls_systems() -> None:
    nodes = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, -1.0],
            [0.0, -1.0],
        ],
        dtype=float,
    )
    cells = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh = Mesh.from_arrays(nodes, cells)
    forward = SimpleNamespace(mesh=mesh)
    current = np.asarray([0.0, 3.0])

    huber = build_spatial_regularization("spatial-huber")
    huber_matrix, huber_rhs = huber.linearized_system(
        forward,
        current,
        2,
        reference_roughness=np.asarray([0.0]),
        scale=2.0,
    )
    huber_weight = np.sqrt(1.0 / 3.0)
    np.testing.assert_allclose(huber_matrix.toarray(), np.asarray([[2.0 * huber_weight, -2.0 * huber_weight]]))
    np.testing.assert_allclose(huber_rhs, np.asarray([2.0 * huber_weight * 3.0]))

    tv = build_spatial_regularization("spatial-tv")
    tv_matrix, tv_rhs = tv.linearized_system(
        forward,
        current,
        2,
        reference_roughness=np.asarray([0.0]),
        scale=2.0,
    )
    assert np.all(np.isfinite(tv_matrix.toarray()))
    assert np.all(np.isfinite(tv_rhs))


def test_temporal_regularization_registry_builds_current_first_difference() -> None:
    temporal = build_temporal_regularization("first-order-l2")
    actual = temporal.matrix(n_cells=2, n_times=3, scale=1.7).toarray()
    expected = np.asarray(
        [
            [-1.7, 0.0, 1.7, 0.0, 0.0, 0.0],
            [0.0, -1.7, 0.0, 1.7, 0.0, 0.0],
            [0.0, 0.0, -1.7, 0.0, 1.7, 0.0],
            [0.0, 0.0, 0.0, -1.7, 0.0, 1.7],
        ]
    )

    np.testing.assert_allclose(actual, expected)
    system_matrix, system_rhs = temporal.linearized_system(
        np.asarray([1.0, 2.0, 4.0, 8.0, 16.0, 32.0]),
        n_cells=2,
        n_times=3,
        scale=1.7,
    )
    np.testing.assert_allclose(system_matrix.toarray(), expected)
    np.testing.assert_allclose(system_rhs, -expected @ np.asarray([1.0, 2.0, 4.0, 8.0, 16.0, 32.0]))


def test_second_order_temporal_regularization_matrix() -> None:
    temporal = build_temporal_regularization("second-order-l2")
    actual = temporal.matrix(n_cells=2, n_times=4, scale=2.0).toarray()
    expected = np.asarray(
        [
            [2.0, 0.0, -4.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, -4.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, -4.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 0.0, -4.0, 0.0, 2.0],
        ]
    )

    np.testing.assert_allclose(actual, expected)


def test_baseline_reference_temporal_regularization_matrix() -> None:
    temporal = build_temporal_regularization("baseline-reference")
    actual = temporal.matrix(n_cells=2, n_times=3, scale=1.5).toarray()
    expected = np.asarray(
        [
            [-1.5, 0.0, 1.5, 0.0, 0.0, 0.0],
            [0.0, -1.5, 0.0, 1.5, 0.0, 0.0],
            [-1.5, 0.0, 0.0, 0.0, 1.5, 0.0],
            [0.0, -1.5, 0.0, 0.0, 0.0, 1.5],
        ]
    )

    np.testing.assert_allclose(actual, expected)


def test_active_time_constraint_downweights_large_temporal_changes() -> None:
    temporal = build_temporal_regularization("active-time-constraint")
    current = np.asarray([0.0, 0.2, 0.21])
    matrix, rhs = temporal.linearized_system(
        current,
        n_cells=1,
        n_times=3,
        scale=2.0,
        threshold=0.05,
        minimum_weight=0.1,
    )

    base = np.asarray([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]])
    weights = 0.1 + 0.9 / (1.0 + (np.asarray([0.2, 0.01]) / 0.05) ** 2)
    np.testing.assert_allclose(matrix.toarray(), np.diag(2.0 * weights) @ base)
    np.testing.assert_allclose(rhs, -2.0 * weights * np.asarray([0.2, 0.01]))


def test_robust_temporal_regularization_builds_irls_systems() -> None:
    current = np.asarray([0.0, 3.0, 4.0])
    base = np.asarray([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]])

    huber = build_temporal_regularization("temporal-huber")
    huber_matrix, huber_rhs = huber.linearized_system(current, n_cells=1, n_times=3, scale=2.0)
    huber_weight = np.asarray([np.sqrt(1.0 / 3.0), 1.0])
    np.testing.assert_allclose(huber_matrix.toarray(), np.diag(2.0 * huber_weight) @ base)
    np.testing.assert_allclose(huber_rhs, -2.0 * huber_weight * np.asarray([3.0, 1.0]))

    tv = build_temporal_regularization("temporal-tv")
    tv_matrix, tv_rhs = tv.linearized_system(current, n_cells=1, n_times=3, scale=2.0)
    assert np.all(np.isfinite(tv_matrix.toarray()))
    assert np.all(np.isfinite(tv_rhs))
    assert np.linalg.norm(tv_matrix.toarray()[0]) < np.linalg.norm(tv_matrix.toarray()[1])


def test_linearized_optimizer_registry_keeps_existing_solver_names() -> None:
    assert build_linearized_optimizer("gpu_cgls").name == "gpu_cgls"
    with pytest.raises(ValueError, match="unknown linearized_solver"):
        build_linearized_optimizer("not-a-solver")
    with pytest.raises(ValueError, match="unknown linearized_solver"):
        build_linearized_optimizer("gpu-timelapse-cgls")


def test_optimization_algorithm_registry_resolves_public_names() -> None:
    assert build_optimization_algorithm("gauss-newton-cgls").uses_linearized_solver
    assert build_optimization_algorithm("levenberg_marquardt").uses_linearized_solver
    assert build_optimization_algorithm("lbfgs").gradient_based
    assert build_optimization_algorithm("lbfgs-b").gradient_based
    assert build_optimization_algorithm("nonlinear-cg").gradient_based
    assert build_optimization_algorithm("adam").gradient_based
    with pytest.raises(ValueError, match="unknown optimizer"):
        build_optimization_algorithm("not-an-optimizer")


def test_log_conductivity_petrophysical_transform_maps_back_to_resistivity() -> None:
    transform = build_petrophysical_transform("log-conductivity", n_cells=2, model_bounds=(10.0, 1000.0))
    log_rho = np.log(np.asarray([20.0, 200.0]))
    state = transform.state_from_log_resistivity(log_rho)

    np.testing.assert_allclose(transform.log_resistivity_from_state(state), log_rho)
    np.testing.assert_allclose(transform.parameter_from_state(state), 1.0 / np.exp(log_rho))
    np.testing.assert_allclose(transform.d_log_resistivity_d_state(state), -np.ones(2))


def test_saturation_petrophysical_transform_roundtrip_and_derivative() -> None:
    transform = build_petrophysical_transform(
        "saturation",
        n_cells=3,
        saturation_floor=1.0e-3,
        parameters={
            "rho_sat": np.asarray([170.0, 1100.0, 2400.0]),
            "rho_sat_s": np.asarray([510.0, np.nan, np.nan]),
            "n": np.asarray([2.2, 1.8, 2.5]),
        },
    )
    saturation = np.asarray([0.35, 0.45, 0.65])
    floor = 1.0e-3
    state = np.log((saturation - floor) / (1.0 - saturation))
    log_rho = transform.log_resistivity_from_state(state)
    roundtrip_state = transform.state_from_log_resistivity(log_rho)

    np.testing.assert_allclose(transform.parameter_from_state(roundtrip_state), saturation, rtol=1.0e-8, atol=1.0e-8)

    eps = 1.0e-6
    finite_difference = (
        transform.log_resistivity_from_state(state + eps)
        - transform.log_resistivity_from_state(state - eps)
    ) / (2.0 * eps)
    np.testing.assert_allclose(transform.d_log_resistivity_d_state(state), finite_difference, rtol=1.0e-5, atol=1.0e-6)


def test_physical_regularization_domain_parameter_and_theta() -> None:
    floor = 1.0e-3
    saturation = np.asarray([0.35, 0.45, 0.65])
    state = np.log((saturation - floor) / (1.0 - saturation))
    parameters = {
        "rho_sat": np.asarray([170.0, 1100.0, 2400.0]),
        "rho_sat_s": np.asarray([510.0, np.nan, np.nan]),
        "n": np.asarray([2.2, 1.8, 2.5]),
        "phi": np.asarray([0.40, 0.18, 0.05]),
    }
    config_s = InversionConfig(
        regularization_domain="physical",
        physical_regularization_quantity="parameter",
        petrophysical_transform="saturation",
        petrophysical_parameters=parameters,
        saturation_floor=floor,
    )
    regularization_model_s, regularization_derivative_s = _regularization_domain_value_and_derivative(state, config_s)
    np.testing.assert_allclose(regularization_model_s, saturation, rtol=1.0e-8, atol=1.0e-8)

    config_theta = InversionConfig(
        regularization_domain="physical",
        physical_regularization_quantity="theta",
        petrophysical_transform="saturation",
        petrophysical_parameters=parameters,
        saturation_floor=floor,
    )
    regularization_model_theta, regularization_derivative_theta = _regularization_domain_value_and_derivative(
        state,
        config_theta,
    )
    np.testing.assert_allclose(regularization_model_theta, saturation * parameters["phi"], rtol=1.0e-8, atol=1.0e-8)
    np.testing.assert_allclose(
        regularization_derivative_theta,
        regularization_derivative_s * parameters["phi"],
        rtol=1.0e-8,
        atol=1.0e-8,
    )


def test_theta_regularization_requires_saturation_transform() -> None:
    config = InversionConfig(
        regularization_domain="physical",
        physical_regularization_quantity="theta",
        petrophysical_transform="log_resistivity",
    )
    with pytest.raises(ValueError, match="requires petrophysical_transform='saturation'"):
        _regularization_domain_value_and_derivative(np.asarray([1.0]), config)
