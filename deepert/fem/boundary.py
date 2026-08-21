"""Boundary assembly utilities for 2.5D ERT."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import k0 as besselk0
from scipy.special import k1 as besselk1

from deepert.utils.torch_runtime import Array
from deepert.utils.torch_runtime import torch_np
from deepert.utils.torch_runtime import BCOO

from deepert.mesh import Mesh
from deepert.utils.dtypes import FLOAT_DTYPE, INT_DTYPE, NP_FLOAT_DTYPE


@dataclass(frozen=True)
class BoundaryRouting:
    """Reusable COO routing indices for boundary-edge assembly."""

    indices: Array
    shape: tuple[int, int]


def build_boundary_routing(mesh: Mesh) -> BoundaryRouting:
    """Precompute the COO routing pattern for boundary edge assembly."""

    return build_boundary_routing_from_connectivity(mesh.boundary_edges, mesh.node_count)


def build_boundary_routing_from_connectivity(boundary_connectivity: Array, node_count: int) -> BoundaryRouting:
    """Precompute COO routing for arbitrary boundary connectivity."""

    connectivity_np = np.asarray(boundary_connectivity, dtype=np.int32)
    edge_count = connectivity_np.shape[0]
    local_dof = int(connectivity_np.shape[1])
    row_indices = np.broadcast_to(connectivity_np[:, :, None], (edge_count, local_dof, local_dof)).reshape(-1)
    col_indices = np.broadcast_to(connectivity_np[:, None, :], (edge_count, local_dof, local_dof)).reshape(-1)
    indices = np.stack((row_indices, col_indices), axis=1).astype(np.int32)
    return BoundaryRouting(indices=torch_np.asarray(indices, dtype=INT_DTYPE), shape=(node_count, node_count))


def _static_scalar_value(value: Array | float) -> float | None:
    try:
        value_array = np.asarray(value)
    except Exception:
        return None
    if value_array.ndim != 0:
        return None
    return float(value_array)


def assemble_local_boundary_mass(mesh: Mesh, coefficients: Array | float) -> Array:
    """Assemble midpoint-based Robin edge matrices."""

    static_coefficient = _static_scalar_value(coefficients)
    if static_coefficient is not None:
        dtype = NP_FLOAT_DTYPE
        reference = np.asarray([[2.0, 1.0], [1.0, 2.0]], dtype=dtype) / 6.0
        lengths = np.asarray(mesh.boundary_edge_lengths, dtype=dtype)
        local = static_coefficient * lengths[:, None, None] * reference[None, :, :]
        return torch_np.asarray(local, dtype=FLOAT_DTYPE)

    coefficient_array = torch_np.asarray(coefficients, dtype=FLOAT_DTYPE)
    if coefficient_array.ndim == 0:
        coefficient_array = torch_np.broadcast_to(coefficient_array, (mesh.boundary_edges.shape[0],))
    elif coefficient_array.shape != (mesh.boundary_edges.shape[0],):
        raise ValueError("boundary coefficients must be scalar or shape (num_boundary_edges,)")

    reference = torch_np.asarray([[2.0, 1.0], [1.0, 2.0]], dtype=FLOAT_DTYPE) / 6.0
    return coefficient_array[:, None, None] * mesh.boundary_edge_lengths[:, None, None] * reference


def assemble_local_boundary_mass_p2(lengths: Array, coefficients: Array | float) -> Array:
    """Assemble quadratic Robin edge matrices."""

    static_coefficient = _static_scalar_value(coefficients)
    if static_coefficient is not None:
        dtype = NP_FLOAT_DTYPE
        length_array_np = np.asarray(lengths, dtype=dtype)
        reference = np.asarray(
            [
                [4.0, 2.0, -1.0],
                [2.0, 16.0, 2.0],
                [-1.0, 2.0, 4.0],
            ],
            dtype=dtype,
        ) / 30.0
        local = static_coefficient * length_array_np[:, None, None] * reference[None, :, :]
        return torch_np.asarray(local, dtype=FLOAT_DTYPE)

    length_array = torch_np.asarray(lengths, dtype=FLOAT_DTYPE)
    coefficient_array = torch_np.asarray(coefficients, dtype=FLOAT_DTYPE)
    if coefficient_array.ndim == 0:
        coefficient_array = torch_np.broadcast_to(coefficient_array, (length_array.shape[0],))
    elif coefficient_array.shape != (length_array.shape[0],):
        raise ValueError("boundary coefficients must be scalar or shape (num_boundary_edges,)")

    reference = torch_np.asarray(
        [
            [4.0, 2.0, -1.0],
            [2.0, 16.0, 2.0],
            [-1.0, 2.0, 4.0],
        ],
        dtype=FLOAT_DTYPE,
    ) / 30.0
    return coefficient_array[:, None, None] * length_array[:, None, None] * reference


def assemble_boundary_bcoo(local_matrices: Array, routing: BoundaryRouting) -> BCOO:
    """Reduce local boundary edge matrices into a global sparse operator."""

    sparse_matrix = BCOO(
        (local_matrices.reshape(-1), routing.indices),
        shape=routing.shape,
        unique_indices=False,
    )
    return sparse_matrix.sum_duplicates()


def _expand_cell_coefficient(coefficients: Array | float, mesh: Mesh) -> Array:
    coefficient_array = torch_np.asarray(coefficients, dtype=FLOAT_DTYPE)
    if coefficient_array.ndim == 0:
        return torch_np.broadcast_to(coefficient_array, (mesh.cell_count,))
    if coefficient_array.shape == (mesh.cell_count,):
        return coefficient_array
    raise ValueError("cell coefficients must be scalar or shape (num_cells,)")


def robin_boundary_coefficients(
    mesh: Mesh,
    conductivity: Array | float,
    source_center: Array,
    wavenumber: float,
) -> Array:
    """Compute 2.5D Robin coefficients for the mixed boundary condition."""

    static_conductivity = _static_scalar_value(conductivity)
    if static_conductivity is None:
        conductivity_values = _expand_cell_coefficient(conductivity, mesh)
        boundary_sigma = conductivity_values[mesh.boundary_edge_cells]
    else:
        boundary_sigma = np.full(mesh.boundary_edges.shape[0], static_conductivity, dtype=NP_FLOAT_DTYPE)

    centers = np.asarray(mesh.boundary_edge_centers, dtype=float)
    normals = np.asarray(mesh.boundary_edge_normals, dtype=float)
    source = np.asarray(source_center, dtype=float)
    mirrored_source = np.asarray([source[0], -source[1]], dtype=float)
    r1 = centers - source
    r2 = centers - mirrored_source
    r1_abs = np.linalg.norm(r1, axis=1)
    r2_abs = np.linalg.norm(r2, axis=1)
    top_boundary = np.asarray(mesh.surface_edge_mask, dtype=bool)

    geometry = np.zeros(centers.shape[0], dtype=float)
    valid = (r1_abs > 1e-12) & (r2_abs > 1e-12) & (~top_boundary)
    if np.any(valid):
        denominator = besselk0(r1_abs[valid] * wavenumber) + besselk0(r2_abs[valid] * wavenumber)
        stable = denominator > 1e-12
        valid_indices = np.flatnonzero(valid)
        if np.any(stable):
            stable_indices = valid_indices[stable]
            numerator = (
                np.sum(r1[stable_indices] * normals[stable_indices], axis=1)
                / r1_abs[stable_indices]
                * besselk1(r1_abs[stable_indices] * wavenumber)
                + np.sum(r2[stable_indices] * normals[stable_indices], axis=1)
                / r2_abs[stable_indices]
                * besselk1(r2_abs[stable_indices] * wavenumber)
            )
            geometry[stable_indices] = wavenumber * numerator / denominator[stable]

    if static_conductivity is not None:
        return torch_np.asarray(boundary_sigma * geometry, dtype=FLOAT_DTYPE)
    return boundary_sigma * torch_np.asarray(geometry, dtype=FLOAT_DTYPE)
