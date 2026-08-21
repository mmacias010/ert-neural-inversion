"""Tensorized assembly utilities for finite-element sparse operators."""

from __future__ import annotations

from dataclasses import dataclass

from deepert.utils.torch_runtime import Array
from deepert.utils.torch_runtime import torch_np
from deepert.utils.torch_runtime import BCOO
import numpy as np

from deepert.fem.p1 import P1ElementData
from deepert.fem.p2 import P2ElementData
from deepert.mesh import Mesh
from deepert.utils.dtypes import FLOAT_DTYPE, INT_DTYPE, NP_FLOAT_DTYPE


@dataclass(frozen=True)
class COORouting:
    """Reusable COO routing indices for sparse element assembly."""

    indices: Array
    shape: tuple[int, int]


def _expand_coefficient(coefficients: Array | float, element_data: P1ElementData) -> Array:
    coefficient_array = torch_np.asarray(coefficients, dtype=FLOAT_DTYPE)
    cell_count = element_data.cell_areas.shape[0]
    quadrature_count = element_data.quadrature_weights.shape[0]

    if coefficient_array.ndim == 0:
        return torch_np.broadcast_to(coefficient_array, (cell_count, quadrature_count))
    if coefficient_array.shape == (cell_count,):
        return torch_np.broadcast_to(coefficient_array[:, None], (cell_count, quadrature_count))
    if coefficient_array.shape == (cell_count, quadrature_count):
        return coefficient_array

    raise ValueError(
        "coefficients must be scalar, shape (num_cells,), or shape (num_cells, num_quadrature)"
    )


def _static_scalar_value(value: Array | float) -> float | None:
    try:
        value_array = np.asarray(value)
    except Exception:
        return None
    if value_array.ndim != 0:
        return None
    return float(value_array)


def _expand_coefficient_generic(
    coefficients: Array | float,
    *,
    cell_count: int,
    quadrature_count: int,
) -> Array:
    coefficient_array = torch_np.asarray(coefficients, dtype=FLOAT_DTYPE)

    if coefficient_array.ndim == 0:
        return torch_np.broadcast_to(coefficient_array, (cell_count, quadrature_count))
    if coefficient_array.shape == (cell_count,):
        return torch_np.broadcast_to(coefficient_array[:, None], (cell_count, quadrature_count))
    if coefficient_array.shape == (cell_count, quadrature_count):
        return coefficient_array

    raise ValueError(
        "coefficients must be scalar, shape (num_cells,), or shape (num_cells, num_quadrature)"
    )


def build_coo_routing(mesh: Mesh) -> COORouting:
    """Precompute the COO routing pattern for cellwise assembly."""

    cells = np.asarray(mesh.cells, dtype=np.int32)
    local_dof = int(cells.shape[1])
    row_indices = np.broadcast_to(cells[:, :, None], (mesh.cell_count, local_dof, local_dof)).reshape(-1)
    col_indices = np.broadcast_to(cells[:, None, :], (mesh.cell_count, local_dof, local_dof)).reshape(-1)
    indices = np.stack((row_indices, col_indices), axis=1).astype(np.int32)
    return COORouting(indices=torch_np.asarray(indices, dtype=INT_DTYPE), shape=(mesh.node_count, mesh.node_count))


def build_coo_routing_from_connectivity(connectivity: Array, node_count: int) -> COORouting:
    """Precompute COO routing for arbitrary local connectivity."""

    connectivity_np = np.asarray(connectivity, dtype=np.int32)
    local_dof = int(connectivity_np.shape[1])
    row_indices = np.broadcast_to(
        connectivity_np[:, :, None],
        (connectivity_np.shape[0], local_dof, local_dof),
    ).reshape(-1)
    col_indices = np.broadcast_to(
        connectivity_np[:, None, :],
        (connectivity_np.shape[0], local_dof, local_dof),
    ).reshape(-1)
    indices = np.stack((row_indices, col_indices), axis=1).astype(np.int32)
    return COORouting(indices=torch_np.asarray(indices, dtype=INT_DTYPE), shape=(node_count, node_count))


def assemble_local_stiffness(element_data: P1ElementData, conductivity: Array | float) -> Array:
    """Assemble elementwise diffusion matrices with batched einsums."""

    static_conductivity = _static_scalar_value(conductivity)
    if static_conductivity is not None:
        dtype = NP_FLOAT_DTYPE
        gradients = np.asarray(element_data.gradients, dtype=dtype)
        weights = (
            2.0
            * np.asarray(element_data.cell_areas, dtype=dtype)[:, None]
            * np.asarray(element_data.quadrature_weights, dtype=dtype)[None, :]
        )
        local = np.einsum("eqid,eqjd,eq->eij", gradients, gradients, static_conductivity * weights)
        return torch_np.asarray(local, dtype=FLOAT_DTYPE)

    conductivity_values = _expand_coefficient(conductivity, element_data)
    weights = 2.0 * element_data.cell_areas[:, None] * element_data.quadrature_weights[None, :]
    return torch_np.einsum(
        "eqid,eqjd,eq->eij",
        element_data.gradients,
        element_data.gradients,
        conductivity_values * weights,
    )


def assemble_local_mass(
    element_data: P1ElementData, coefficients: Array | float = 1.0
) -> Array:
    """Assemble elementwise mass matrices with optional scalar coefficients."""

    static_coefficient = _static_scalar_value(coefficients)
    if static_coefficient is not None:
        dtype = NP_FLOAT_DTYPE
        shape_values = np.asarray(element_data.shape_values, dtype=dtype)
        weights = (
            2.0
            * np.asarray(element_data.cell_areas, dtype=dtype)[:, None]
            * np.asarray(element_data.quadrature_weights, dtype=dtype)[None, :]
        )
        local = np.einsum("qi,qj,eq->eij", shape_values, shape_values, static_coefficient * weights)
        return torch_np.asarray(local, dtype=FLOAT_DTYPE)

    coefficient_values = _expand_coefficient(coefficients, element_data)
    weights = 2.0 * element_data.cell_areas[:, None] * element_data.quadrature_weights[None, :]
    return torch_np.einsum(
        "qi,qj,eq->eij",
        element_data.shape_values,
        element_data.shape_values,
        coefficient_values * weights,
    )


def assemble_local_stiffness_p2(
    element_data: P2ElementData,
    conductivity: Array | float,
) -> Array:
    """Assemble quadratic elementwise diffusion matrices."""

    conductivity_values = _expand_coefficient_generic(
        conductivity,
        cell_count=element_data.cell_areas.shape[0],
        quadrature_count=element_data.quadrature_weights.shape[0],
    )
    weights = 2.0 * element_data.cell_areas[:, None] * element_data.quadrature_weights[None, :]
    return torch_np.einsum(
        "eqid,eqjd,eq->eij",
        element_data.gradients,
        element_data.gradients,
        conductivity_values * weights,
    )


def assemble_local_mass_p2(
    element_data: P2ElementData,
    coefficients: Array | float = 1.0,
) -> Array:
    """Assemble quadratic elementwise mass matrices."""

    coefficient_values = _expand_coefficient_generic(
        coefficients,
        cell_count=element_data.cell_areas.shape[0],
        quadrature_count=element_data.quadrature_weights.shape[0],
    )
    weights = 2.0 * element_data.cell_areas[:, None] * element_data.quadrature_weights[None, :]
    return torch_np.einsum(
        "qi,qj,eq->eij",
        element_data.shape_values,
        element_data.shape_values,
        coefficient_values * weights,
    )


def assemble_global_bcoo(local_matrices: Array, routing: COORouting) -> BCOO:
    """Reduce local element matrices into a global sparse operator."""

    sparse_matrix = BCOO(
        (local_matrices.reshape(-1), routing.indices),
        shape=routing.shape,
        unique_indices=False,
    )
    return sparse_matrix.sum_duplicates()


def assemble_helmholtz_operator(
    element_data: P1ElementData,
    routing: COORouting,
    conductivity: Array | float,
    wavenumber_sq: Array | float,
) -> BCOO:
    """Assemble a sparse Helmholtz-type operator K + k^2 M."""

    conductivity_values = torch_np.asarray(conductivity, dtype=FLOAT_DTYPE)
    local_operator = assemble_local_stiffness(
        element_data=element_data,
        conductivity=conductivity,
    ) + assemble_local_mass(
        element_data=element_data,
        coefficients=conductivity_values * wavenumber_sq,
    )
    return assemble_global_bcoo(local_operator, routing)
