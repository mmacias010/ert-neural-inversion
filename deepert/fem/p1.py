"""P1 triangle basis functions and batched quadrature data."""

from __future__ import annotations

from dataclasses import dataclass

from deepert.utils.torch_runtime import Array
from deepert.utils.torch_runtime import torch_np
import numpy as np

from deepert.mesh import Mesh
from deepert.utils.dtypes import FLOAT_DTYPE, NP_FLOAT_DTYPE


@dataclass(frozen=True)
class TriangleQuadrature:
    """Reference triangle quadrature points and weights."""

    points: Array
    weights: Array


@dataclass(frozen=True)
class P1ElementData:
    """Batched P1 element tensors for tensorized assembly."""

    quadrature_points: Array
    quadrature_weights: Array
    shape_values: Array
    reference_gradients: Array
    gradients: Array
    cell_quadrature_points: Array
    cell_areas: Array


def triangle_quadrature(order: int = 2) -> TriangleQuadrature:
    """Return a low-order Gauss rule on the reference triangle."""

    if order == 1:
        points = torch_np.asarray([[1.0 / 3.0, 1.0 / 3.0]], dtype=FLOAT_DTYPE)
        weights = torch_np.asarray([0.5], dtype=FLOAT_DTYPE)
        return TriangleQuadrature(points=points, weights=weights)

    if order == 2:
        points = torch_np.asarray(
            [
                [1.0 / 6.0, 1.0 / 6.0],
                [2.0 / 3.0, 1.0 / 6.0],
                [1.0 / 6.0, 2.0 / 3.0],
            ],
            dtype=FLOAT_DTYPE,
        )
        weights = torch_np.asarray([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=FLOAT_DTYPE)
        return TriangleQuadrature(points=points, weights=weights)

    raise ValueError(f"unsupported triangle quadrature order: {order}")


def p1_shape_functions(points: Array) -> Array:
    """Evaluate P1 shape functions at reference coordinates."""

    point_array = torch_np.asarray(points, dtype=FLOAT_DTYPE)
    xi = point_array[..., 0]
    eta = point_array[..., 1]
    return torch_np.stack((1.0 - xi - eta, xi, eta), axis=-1)


def reference_shape_gradients() -> Array:
    """Return the constant reference gradients of the P1 basis."""

    return torch_np.asarray(
        [
            [-1.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=FLOAT_DTYPE,
    )


def build_p1_element_data(mesh: Mesh, quadrature_order: int = 2) -> P1ElementData:
    """Prepare batched element tensors for later FEM assembly."""

    dtype = NP_FLOAT_DTYPE
    if quadrature_order == 1:
        quadrature_points = np.asarray([[1.0 / 3.0, 1.0 / 3.0]], dtype=dtype)
        quadrature_weights = np.asarray([0.5], dtype=dtype)
    elif quadrature_order == 2:
        quadrature_points = np.asarray(
            [
                [1.0 / 6.0, 1.0 / 6.0],
                [2.0 / 3.0, 1.0 / 6.0],
                [1.0 / 6.0, 2.0 / 3.0],
            ],
            dtype=dtype,
        )
        quadrature_weights = np.asarray([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=dtype)
    else:
        raise ValueError(f"unsupported triangle quadrature order: {quadrature_order}")

    xi = quadrature_points[..., 0]
    eta = quadrature_points[..., 1]
    shape_values = np.stack((1.0 - xi - eta, xi, eta), axis=-1).astype(dtype)
    ref_gradients = np.asarray(
        [
            [-1.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=dtype,
    )

    cell_nodes = np.asarray(mesh.nodes, dtype=dtype)[np.asarray(mesh.cells, dtype=np.int32)]
    edge_1 = cell_nodes[:, 1] - cell_nodes[:, 0]
    edge_2 = cell_nodes[:, 2] - cell_nodes[:, 0]
    jacobians = np.stack((edge_1, edge_2), axis=-1)
    inverse_jacobian_t = np.linalg.inv(jacobians).transpose((0, 2, 1))
    cell_gradients = np.einsum("eij,nj->eni", inverse_jacobian_t, ref_gradients)
    gradients = np.broadcast_to(
        cell_gradients[:, None, :, :],
        (mesh.cell_count, quadrature_points.shape[0], 3, 2),
    )
    cell_quadrature_points = np.einsum("qi,eid->eqd", shape_values, cell_nodes)
    cell_areas = 0.5 * np.abs(np.linalg.det(jacobians))

    return P1ElementData(
        quadrature_points=torch_np.asarray(quadrature_points, dtype=FLOAT_DTYPE),
        quadrature_weights=torch_np.asarray(quadrature_weights, dtype=FLOAT_DTYPE),
        shape_values=torch_np.asarray(shape_values, dtype=FLOAT_DTYPE),
        reference_gradients=torch_np.asarray(ref_gradients, dtype=FLOAT_DTYPE),
        gradients=torch_np.asarray(gradients, dtype=FLOAT_DTYPE),
        cell_quadrature_points=torch_np.asarray(cell_quadrature_points, dtype=FLOAT_DTYPE),
        cell_areas=torch_np.asarray(cell_areas, dtype=FLOAT_DTYPE),
    )
