from __future__ import annotations

import numpy as np
import pytest
import torch

from deepert.forward import ERTForward2p5D, ERTForwardModeling
from deepert.mesh import Mesh
from deepert.survey import Survey


def _flat_four_electrode_case() -> tuple[Mesh, Survey, np.ndarray]:
    nodes = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [0.0, -1.0],
            [1.0, -1.0],
            [2.0, -1.0],
            [3.0, -1.0],
        ],
        dtype=float,
    )
    cells = np.asarray(
        [
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
        ],
        dtype=np.int32,
    )
    electrodes = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=float,
    )
    measurements = np.asarray([[0, 3, 1, 2]], dtype=np.int32)
    mesh = Mesh.from_arrays(nodes, cells, surface_node_ids=np.asarray([0, 1, 2, 3], dtype=np.int32))
    survey = Survey.from_arrays(electrodes, measurements)
    conductivity = np.ones((mesh.cell_count,), dtype=np.float32)
    return mesh, survey, conductivity


def test_flat_forward_scipy_regression_values() -> None:
    mesh, survey, conductivity = _flat_four_electrode_case()
    forward = ERTForward2p5D.from_mesh_survey(mesh, survey, linear_solver_backend="scipy")

    response = forward.solve(conductivity)
    jacobian = forward.jacobian(conductivity, batch_size=1)

    np.testing.assert_allclose(
        np.asarray(forward.wavenumbers),
        np.asarray(
            [
                1.1400917e-03,
                2.8694769e-02,
                1.4492519e-01,
                3.8354439e-01,
                6.8990415e-01,
                9.3360960e-01,
                1.3225477e00,
                2.7457612e00,
                5.5366201e00,
                1.0395071e01,
            ],
            dtype=np.float32,
        ),
        rtol=1.0e-6,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        np.asarray(forward.weights),
        np.asarray(
            [
                1.8413631e-03,
                1.9452337e-02,
                5.6700651e-02,
                9.2240982e-02,
                9.5381640e-02,
                5.2692916e-02,
                2.6506910e-01,
                6.5193123e-01,
                1.1558298e00,
                2.0649223e00,
            ],
            dtype=np.float32,
        ),
        rtol=1.0e-6,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        np.asarray(response.apparent_resistivity),
        np.asarray([1.0000906], dtype=np.float32),
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        np.asarray(response.resistance),
        np.asarray([0.15916936], dtype=np.float32),
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        np.asarray(jacobian),
        np.asarray(
            [[0.00112671, -0.01648985, -0.0889006, -0.01795806, -0.00099201, -0.0153421]],
            dtype=np.float32,
        ),
        rtol=1.0e-5,
        atol=1.0e-7,
    )


def test_flat_forward_cudss_regression_values_when_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    try:
        import cupy  # noqa: F401
        import nvmath.sparse.advanced  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"cuDSS stack is not available: {exc}")

    mesh, survey, conductivity = _flat_four_electrode_case()
    forward = ERTForward2p5D.from_mesh_survey(mesh, survey, linear_solver_backend="cudss")
    try:
        response = forward.solve(conductivity)
    finally:
        forward.close()

    np.testing.assert_allclose(
        np.asarray(response.apparent_resistivity),
        np.asarray([1.0000906], dtype=np.float32),
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        np.asarray(response.resistance),
        np.asarray([0.15916936], dtype=np.float32),
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def test_forward_modeling_facade_regression_values() -> None:
    mesh, survey, conductivity = _flat_four_electrode_case()
    resistivity = 1.0 / conductivity
    modeling = ERTForwardModeling(mesh=mesh, data=survey, linear_solver_backend="scipy")

    log_response, log_jacobian = modeling.forward_and_jacobian(np.log(resistivity), log_transform=True)

    np.testing.assert_allclose(log_response, np.asarray([9.059497e-05], dtype=np.float32), rtol=1.0e-5, atol=1.0e-7)
    np.testing.assert_allclose(
        log_jacobian,
        np.asarray(
            [[-0.00707874, 0.10359942, 0.55852836, 0.11282383, 0.00623242, 0.09638854]],
            dtype=np.float32,
        ),
        rtol=1.0e-5,
        atol=1.0e-7,
    )
