"""
 triangle basis functions and batched quadrature data."""

from __future__ import annotations

from dataclasses import dataclass

from deepert.utils.torch_runtime import Array
from deepert.utils.torch_runtime import torch_np

from deepert.mesh import Mesh
from deepert.utils.dtypes import FLOAT_DTYPE


@dataclass(frozen=True)
class P2ElementData:
    """Batched P2 element tensors for tensorized assembly."""

    quadrature_points: Array
    quadrature_weights: Array
    shape_values: Array
    reference_gradients: Array
    gradients: Array
    cell_areas: Array


def triangle_quadrature_p2() -> tuple[Array, Array]:
    """Return a degree-4 triangle rule exact for P2 mass matrices."""

    points = torch_np.asarray(
        [
            [0.445948490915965, 0.445948490915965],
            [0.445948490915965, 0.108103018168070],
            [0.108103018168070, 0.445948490915965],
            [0.091576213509771, 0.091576213509771],
            [0.091576213509771, 0.816847572980459],
            [0.816847572980459, 0.091576213509771],
        ],
        dtype=FLOAT_DTYPE,
    )
    weights = torch_np.asarray(
        [
            0.1116907948390055,
            0.1116907948390055,
            0.1116907948390055,
            0.0549758718276610,
            0.0549758718276610,
            0.0549758718276610,
        ],
        dtype=FLOAT_DTYPE,
    )
    return points, weights


def p2_shape_functions(points: Array) -> Array:
    """Evaluate P2 shape functions at reference coordinates."""

    point_array = torch_np.asarray(points, dtype=FLOAT_DTYPE)
    l2 = point_array[..., 0]
    l3 = point_array[..., 1]
    l1 = 1.0 - l2 - l3
    return torch_np.stack(
        (
            l1 * (2.0 * l1 - 1.0),
            l2 * (2.0 * l2 - 1.0),
            l3 * (2.0 * l3 - 1.0),
            4.0 * l1 * l2,
            4.0 * l2 * l3,
            4.0 * l3 * l1,
        ),
        axis=-1,
    )


def reference_shape_gradients_p2(points: Array) -> Array:
    """Evaluate P2 reference gradients at reference coordinates."""

    point_array = torch_np.asarray(points, dtype=FLOAT_DTYPE)
    l2 = point_array[..., 0]
    l3 = point_array[..., 1]
    l1 = 1.0 - l2 - l3
    grad_l1 = torch_np.asarray([-1.0, -1.0], dtype=FLOAT_DTYPE)
    grad_l2 = torch_np.asarray([1.0, 0.0], dtype=FLOAT_DTYPE)
    grad_l3 = torch_np.asarray([0.0, 1.0], dtype=FLOAT_DTYPE)

    return torch_np.stack(
        (
            (4.0 * l1 - 1.0)[..., None] * grad_l1,
            (4.0 * l2 - 1.0)[..., None] * grad_l2,
            (4.0 * l3 - 1.0)[..., None] * grad_l3,
            4.0 * (l1[..., None] * grad_l2 + l2[..., None] * grad_l1),
            4.0 * (l2[..., None] * grad_l3 + l3[..., None] * grad_l2),
            4.0 * (l3[..., None] * grad_l1 + l1[..., None] * grad_l3),
        ),
        axis=-2,
    )


def build_p2_element_data(mesh: Mesh) -> P2ElementData:
    """Prepare batched P2 element tensors for later FEM assembly."""

    quadrature_points, quadrature_weights = triangle_quadrature_p2()
    shape_values = p2_shape_functions(quadrature_points)
    reference_gradients = reference_shape_gradients_p2(quadrature_points)

    cell_nodes = mesh.nodes[mesh.cells]
    edge_1 = cell_nodes[:, 1] - cell_nodes[:, 0]
    edge_2 = cell_nodes[:, 2] - cell_nodes[:, 0]
    jacobians = torch_np.stack((edge_1, edge_2), axis=-1)
    inverse_jacobian_t = torch_np.linalg.inv(jacobians).permute(0, 2, 1)
    gradients = torch_np.einsum("eij,qnj->eqni", inverse_jacobian_t, reference_gradients)

    return P2ElementData(
        quadrature_points=quadrature_points,
        quadrature_weights=quadrature_weights,
        shape_values=shape_values,
        reference_gradients=reference_gradients,
        gradients=gradients,
        cell_areas=mesh.cell_areas,
    )
