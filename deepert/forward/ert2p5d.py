"""2.5D multi-wavenumber ERT forward operator."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import logging
import os
from pathlib import Path

from deepert.utils.torch_runtime import Array
from deepert.utils.torch_runtime import torch_runtime
from deepert.utils.torch_runtime import torch_np
from deepert.utils.torch_runtime import CSR
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import k0 as besselk0
import torch

from deepert.fem import (
    build_boundary_routing,
    build_boundary_routing_from_connectivity,
    build_coo_routing,
    build_coo_routing_from_connectivity,
    build_p1_element_data,
    build_p2_element_data,
    assemble_local_boundary_mass,
    assemble_local_boundary_mass_p2,
    assemble_local_mass,
    assemble_local_mass_p2,
    assemble_local_stiffness,
    assemble_local_stiffness_p2,
    robin_boundary_coefficients,
)
from deepert.forward.integration import build_inverse_cosine_weights, survey_wavenumber_bounds
from deepert.mesh import Mesh
from deepert.survey import Survey
from deepert.utils.dtypes import FLOAT_DTYPE, INT_DTYPE, NP_FLOAT_DTYPE

_SOURCE_INSET_FACTOR = 0.69
_VALID_LINEAR_SOLVER_BACKENDS = frozenset({"auto", "cudss", "scipy"})
_VALID_TOPOGRAPHIC_GEOMETRIC_FACTOR_MODES = frozenset({"analytic", "numerical"})
_TERRAIN_AUXILIARY_CACHE_VERSION = "terrain_auxiliary_v1"
_CUDSS_LOGGER = logging.getLogger("deepert.cudss")
_CUDSS_LOGGER.setLevel(logging.ERROR)


def _enable_torch_float64_for_terrain_auxiliary() -> None:
    """Enable local terrain auxiliary float64 arrays without changing FLOAT_DTYPE."""

    if not torch_runtime.config.torch_enable_float64:
        torch_runtime.config.update("torch_enable_float64", True)


def _normalize_cache_dir(cache_dir: str | Path | None) -> Path | None:
    if cache_dir is None:
        env_cache_dir = os.environ.get("DEEPERT_TERRAIN_CACHE_DIR")
        if env_cache_dir is None or env_cache_dir.strip() == "":
            return None
        cache_dir = env_cache_dir
    return Path(cache_dir).expanduser()


def _update_digest_value(digest: "hashlib._Hash", label: str, value: object) -> None:
    digest.update(label.encode("utf-8"))
    digest.update(repr(value).encode("utf-8"))


def _update_digest_array(digest: "hashlib._Hash", label: str, values: Array | np.ndarray) -> None:
    array = np.ascontiguousarray(np.asarray(values))
    digest.update(label.encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def _numpy_dtype(dtype) -> np.dtype:
    if dtype == torch_np.float64:
        return np.dtype(np.float64)
    if dtype == torch_np.float32:
        return np.dtype(np.float32)
    if dtype == torch_np.int32:
        return np.dtype(np.int32)
    return np.dtype(dtype)


@dataclass(frozen=True)
class ForwardResponse:
    """Result of a 2.5D ERT forward solve."""

    apparent_resistivity: Array
    resistance: Array
    electrode_potentials: Array
    integrated_potentials: Array
    wavenumbers: Array
    weights: Array


@dataclass(frozen=True)
class SingleWavenumberFields:
    """Unintegrated field components for one 2.5D wavenumber."""

    wavenumber: float
    rhs: Array
    sub_potentials: Array
    unit_primary: Array
    primary: Array
    source_resistivities: Array


@dataclass(frozen=True)
class SparseOperatorPattern:
    """Fixed sparse structure shared by all assembled operators."""

    shape: tuple[int, int]
    unique_indices: Array
    volume_inverse: Array
    boundary_inverse: Array
    row_indices: np.ndarray
    col_indices: np.ndarray
    csr_indptr: np.ndarray
    csr_indices: np.ndarray


@dataclass(frozen=True)
class OperatorTemplates:
    """Local element templates reused across conductivity models and wavenumbers."""

    stiffness: Array
    mass: Array
    boundary_mass: Array


@dataclass(frozen=True)
class SourceResistivityData:
    """Precomputed source-to-cell adjacency for nodal source resistivities."""

    node_cell_weights: Array
    node_cell_counts: Array


@dataclass(frozen=True)
class AuxiliaryDiscretization:
    """Auxiliary h2/p2 discretization used for topographic numerical solves."""

    geometry_mesh: Mesh
    parent_cell_ids: Array
    dof_nodes: Array
    cell_connectivity: Array
    boundary_connectivity: Array
    routing: object
    boundary_routing: object
    operator_pattern: SparseOperatorPattern
    operator_templates: OperatorTemplates
    source_positions: Array
    source_matrix: Array
    electrode_matrix: Array
    original_node_matrix: Array
    boundary_geometries: Array


@dataclass(frozen=True)
class StructuredQuadMesh:
    """Structured quadrilateral mesh used for terrain-following auxiliary solves."""

    nodes: Array
    cells: Array
    boundary_edges: Array
    boundary_edge_cells: Array
    boundary_edge_centers: Array
    boundary_edge_lengths: Array
    boundary_edge_normals: Array
    surface_node_ids: Array
    surface_edge_mask: Array
    surface_nodes: Array
    surface_reference_level: Array
    flat_surface: Array

    @property
    def node_count(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def cell_count(self) -> int:
        return int(self.cells.shape[0])


def _find_source_node_ids(mesh: Mesh, positions: Array, tol: float = 1e-8) -> Array:
    """Map source positions to mesh node ids when they coincide with nodes."""

    nodes = np.asarray(mesh.nodes, dtype=float)
    source_positions = np.asarray(positions, dtype=float)
    node_ids = np.full(source_positions.shape[0], -1, dtype=np.int32)
    for idx, source in enumerate(source_positions):
        distances = np.linalg.norm(nodes - source, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= tol:
            node_ids[idx] = nearest
    return torch_np.asarray(node_ids, dtype=INT_DTYPE)


def _find_nearest_node_electrode_ids(mesh: Mesh, positions: Array, tol: float = 1e-2) -> Array:
    """Match source electrodes to nearby mesh nodes within the default tolerance."""

    return _find_source_node_ids(mesh, positions, tol=tol)


def _find_entity_node_ids(mesh: Mesh, positions: Array, cell_ids: Array, tol: float = 1e-4) -> Array:
    """Match points to vertices of their containing cell."""

    nodes = np.asarray(mesh.nodes, dtype=float)
    cells = np.asarray(mesh.cells, dtype=np.int32)
    entity_positions = np.asarray(positions, dtype=float)
    containing_cells = np.asarray(cell_ids, dtype=np.int32)
    node_ids = np.full(entity_positions.shape[0], -1, dtype=np.int32)

    for idx, (point, cell_id) in enumerate(zip(entity_positions, containing_cells, strict=False)):
        cell_nodes = cells[cell_id]
        distances = np.linalg.norm(nodes[cell_nodes] - point, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= tol:
            node_ids[idx] = int(cell_nodes[nearest])

    return torch_np.asarray(node_ids, dtype=INT_DTYPE)


def _locate_point_cells(mesh: Mesh, points: Array) -> Array:
    cell_ids, _ = mesh.locate_points(points)
    return cell_ids.astype(INT_DTYPE)


def _build_interpolation_matrix(mesh: Mesh, points: Array) -> Array:
    cell_ids, barycentric_weights = mesh.locate_points(points)
    cell_ids_np = np.asarray(cell_ids, dtype=np.int32)
    weights_np = np.asarray(barycentric_weights, dtype=NP_FLOAT_DTYPE)
    cell_nodes = np.asarray(mesh.cells, dtype=np.int32)[cell_ids_np]
    matrix = np.zeros((points.shape[0], mesh.node_count), dtype=NP_FLOAT_DTYPE)
    row_indices = np.repeat(np.arange(points.shape[0], dtype=np.int32), cell_nodes.shape[1])
    np.add.at(matrix, (row_indices, cell_nodes.reshape(-1)), weights_np.reshape(-1))
    return torch_np.asarray(matrix, dtype=FLOAT_DTYPE)


def _build_quadratic_interpolation_matrix(
    geometry_mesh: Mesh,
    cell_connectivity: Array,
    node_count: int,
    points: Array,
) -> Array:
    """Interpolate points into quadratic triangle DOFs."""

    cell_ids, barycentric_weights = geometry_mesh.locate_points(points)
    barycentric_np = np.asarray(barycentric_weights, dtype=NP_FLOAT_DTYPE)
    l1 = barycentric_np[:, 0]
    l2 = barycentric_np[:, 1]
    l3 = barycentric_np[:, 2]
    shape_values = np.stack(
        (
            l1 * (2.0 * l1 - 1.0),
            l2 * (2.0 * l2 - 1.0),
            l3 * (2.0 * l3 - 1.0),
            4.0 * l1 * l2,
            4.0 * l2 * l3,
            4.0 * l3 * l1,
        ),
        axis=-1,
    ).astype(NP_FLOAT_DTYPE)
    cell_dofs = np.asarray(cell_connectivity, dtype=np.int32)[np.asarray(cell_ids, dtype=np.int32)]
    matrix = np.zeros((points.shape[0], node_count), dtype=NP_FLOAT_DTYPE)
    row_indices = np.repeat(np.arange(points.shape[0], dtype=np.int32), cell_connectivity.shape[1])
    np.add.at(matrix, (row_indices, cell_dofs.reshape(-1)), shape_values.reshape(-1))
    return torch_np.asarray(matrix, dtype=FLOAT_DTYPE)


def _build_discretization_interpolation_matrix(
    geometry_mesh: Mesh,
    dof_nodes: Array,
    cell_connectivity: Array,
    points: Array,
) -> Array:
    """Interpolate points into either linear or quadratic discretization DOFs."""

    if cell_connectivity.shape[1] == 3:
        return _build_interpolation_matrix(geometry_mesh, points)
    if cell_connectivity.shape[1] == 4:
        return _build_interpolation_matrix(geometry_mesh, points)
    if cell_connectivity.shape[1] == 6:
        return _build_quadratic_interpolation_matrix(geometry_mesh, cell_connectivity, int(dof_nodes.shape[0]), points)
    raise ValueError("unsupported cell connectivity width")


def _same_point_locations(left: Array, right: Array, tol: float = 1e-8) -> bool:
    left_np = np.asarray(left, dtype=float)
    right_np = np.asarray(right, dtype=float)
    return left_np.shape == right_np.shape and bool(np.all(np.abs(left_np - right_np) <= tol))


def _points_on_segments(points: np.ndarray, segment_nodes: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Return a point-to-segment incidence mask."""

    start = segment_nodes[:, 0]
    stop = segment_nodes[:, 1]
    edge_vectors = stop - start
    edge_lengths_sq = np.sum(edge_vectors * edge_vectors, axis=1)

    deltas = points[:, None, :] - start[None, :, :]
    numerators = np.sum(deltas * edge_vectors[None, :, :], axis=2)
    safe_lengths_sq = np.maximum(edge_lengths_sq, tol)
    projection = numerators / safe_lengths_sq[None, :]
    closest = start[None, :, :] + projection[:, :, None] * edge_vectors[None, :, :]
    distance = np.linalg.norm(points[:, None, :] - closest, axis=2)

    return (
        (projection >= -tol)
        & (projection <= 1.0 + tol)
        & (distance <= tol)
        & (edge_lengths_sq[None, :] > tol)
    )


def _surface_inward_normals(mesh: Mesh, points: Array, tol: float = 1e-8) -> tuple[Array, Array]:
    """Average inward surface normals for points that lie on the top boundary."""

    points_np = np.asarray(points, dtype=float)
    boundary_edges = np.asarray(mesh.boundary_edges, dtype=np.int32)
    surface_edge_mask = np.asarray(mesh.surface_edge_mask, dtype=bool)
    if not np.any(surface_edge_mask):
        normals = np.zeros((points_np.shape[0], 2), dtype=float)
        return torch_np.asarray(normals, dtype=FLOAT_DTYPE), torch_np.zeros((points_np.shape[0],), dtype=bool)

    surface_edges = boundary_edges[surface_edge_mask]
    surface_edge_nodes = np.asarray(mesh.nodes[surface_edges], dtype=float)
    inward_normals = -np.asarray(mesh.boundary_edge_normals[surface_edge_mask], dtype=float)
    hits = _points_on_segments(points_np, surface_edge_nodes, tol=tol)

    averaged = np.zeros((points_np.shape[0], 2), dtype=float)
    on_surface = np.any(hits, axis=1)
    for point_id in np.flatnonzero(on_surface):
        normal = inward_normals[hits[point_id]].sum(axis=0)
        norm = np.linalg.norm(normal)
        if norm > tol:
            averaged[point_id] = normal / norm

    return torch_np.asarray(averaged, dtype=FLOAT_DTYPE), torch_np.asarray(on_surface, dtype=bool)


def _build_source_positions(mesh: Mesh, survey: Survey, source_cell_ids: Array) -> Array:
    """Regularize surface point sources by moving them slightly into the domain."""

    positions = torch_np.asarray(survey.electrode_positions, dtype=FLOAT_DTYPE)
    if np.all(np.asarray(source_cell_ids, dtype=np.int32) >= 0):
        return positions

    inward_normals, on_surface = _surface_inward_normals(mesh, positions)
    if not bool(torch_np.any(on_surface)):
        return positions

    cell_centers = torch_np.mean(mesh.nodes[mesh.cells[source_cell_ids]], axis=1)
    center_offsets = cell_centers - positions
    projected_depth = torch_np.sum(center_offsets * inward_normals, axis=1)
    fallback_depth = torch_np.linalg.norm(center_offsets, axis=1)
    source_depth = torch_np.where(projected_depth > 1e-8, projected_depth, fallback_depth)
    source_inset = _SOURCE_INSET_FACTOR * source_depth
    normal_candidate = positions + inward_normals * source_inset[:, None]
    center_candidate = positions + _SOURCE_INSET_FACTOR * center_offsets

    source_positions_np = np.asarray(positions, dtype=float)
    normal_candidate_np = np.asarray(normal_candidate, dtype=float)
    center_candidate_np = np.asarray(center_candidate, dtype=float)
    on_surface_np = np.asarray(on_surface, dtype=bool)

    for point_id in np.flatnonzero(on_surface_np):
        try:
            mesh.locate_points(torch_np.asarray(source_positions_np[point_id][None, :], dtype=FLOAT_DTYPE))
            continue
        except ValueError:
            pass

        try:
            mesh.locate_points(torch_np.asarray(normal_candidate_np[point_id][None, :], dtype=FLOAT_DTYPE))
            source_positions_np[point_id] = normal_candidate_np[point_id]
        except ValueError:
            source_positions_np[point_id] = center_candidate_np[point_id]

    return torch_np.asarray(source_positions_np, dtype=FLOAT_DTYPE)


def _exact_dcsolution_on_nodes(mesh: Mesh, source: Array, wavenumber: float) -> Array:
    """Evaluate the half-space analytical primary field for unit resistivity."""

    nodes = np.asarray(mesh.nodes, dtype=float)
    source_np = np.asarray(source, dtype=float)
    distances = np.linalg.norm(nodes - source_np, axis=1)
    surface_level = float(np.max(nodes[:, 1]))
    mirrored_source = np.asarray([source_np[0], 2.0 * surface_level - source_np[1]], dtype=float)

    values = np.zeros(nodes.shape[0], dtype=float)
    if abs(source_np[1] - surface_level) <= 1e-8:
        valid = distances > 1e-12
        values[valid] = besselk0(distances[valid] * wavenumber) / np.pi
        return torch_np.asarray(values, dtype=FLOAT_DTYPE)

    mirrored_distances = np.linalg.norm(nodes - mirrored_source, axis=1)
    valid = (distances > 1e-12) & (mirrored_distances > 1e-12)
    values[valid] = (
        besselk0(distances[valid] * wavenumber) + besselk0(mirrored_distances[valid] * wavenumber)
    ) / (2.0 * np.pi)
    return torch_np.asarray(values, dtype=FLOAT_DTYPE)


def _node_singularity_value(mesh: Mesh, node_id: int, wavenumber: float) -> float:
    """Compute the singular value replacement used at source nodes."""

    node = np.asarray(mesh.nodes[node_id], dtype=float)
    cells = np.asarray(mesh.cells, dtype=np.int32)
    mask = np.any(cells == node_id, axis=1)
    neighboring_cells = cells[mask]
    neighboring_nodes = np.unique(neighboring_cells.reshape(-1))
    neighboring_nodes = neighboring_nodes[neighboring_nodes != node_id]
    if neighboring_nodes.size == 0:
        return 0.0
    min_radius = np.min(np.linalg.norm(np.asarray(mesh.nodes[neighboring_nodes], dtype=float) - node, axis=1))
    return float(besselk0((min_radius / 6.0) * wavenumber) / np.pi)


def _expand_conductivity(conductivity: Array | float, cell_count: int) -> Array:
    conductivity_array = torch_np.asarray(conductivity, dtype=FLOAT_DTYPE)
    if conductivity_array.ndim == 0:
        return torch_np.broadcast_to(conductivity_array, (cell_count,))
    if conductivity_array.shape != (cell_count,):
        raise ValueError(f"conductivity must be scalar or shape ({cell_count},)")
    return conductivity_array


def _expand_measurement_vector(values: Array | float, measurement_count: int, *, name: str) -> Array:
    measurement_array = torch_np.asarray(values, dtype=FLOAT_DTYPE)
    if measurement_array.ndim == 0:
        return torch_np.broadcast_to(measurement_array, (measurement_count,))
    if measurement_array.shape != (measurement_count,):
        raise ValueError(f"{name} must be scalar or shape ({measurement_count},)")
    return measurement_array


def _expand_measurement_matrix(values: Array | float, measurement_count: int, *, name: str) -> Array:
    measurement_array = torch_np.asarray(values, dtype=FLOAT_DTYPE)
    if measurement_array.ndim == 1:
        measurement_array = measurement_array[None, :]
    if measurement_array.ndim != 2 or measurement_array.shape[1] != measurement_count:
        raise ValueError(f"{name} must have shape ({measurement_count},) or (batch, {measurement_count})")
    return measurement_array


def _normalize_linear_solver_backend(backend: str) -> str:
    normalized = str(backend).strip().lower()
    if normalized not in _VALID_LINEAR_SOLVER_BACKENDS:
        valid = ", ".join(sorted(_VALID_LINEAR_SOLVER_BACKENDS))
        raise ValueError(f"linear_solver_backend must be one of: {valid}")
    if normalized == "auto":
        if torch.cuda.is_available():
            try:
                import cupy  # noqa: F401
                from nvmath.sparse.advanced import DirectSolver  # noqa: F401
                return "cudss"
            except ImportError:
                pass
        return "scipy"
    return normalized


def _normalize_topographic_geometric_factor_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in _VALID_TOPOGRAPHIC_GEOMETRIC_FACTOR_MODES:
        valid = ", ".join(sorted(_VALID_TOPOGRAPHIC_GEOMETRIC_FACTOR_MODES))
        raise ValueError(f"topographic_geometric_factor_mode must be one of: {valid}")
    return normalized


def _array_is_on_cuda(values: Array) -> bool:
    if bool(getattr(values, "is_cuda", False)):
        return True
    try:
        devices = values.devices()
    except AttributeError:
        return False

    for device in devices:
        platform = str(getattr(device, "platform", "")).lower()
        description = str(device).lower()
        if platform in {"cuda", "gpu"} or "cuda" in description or "gpu" in description:
            return True
    return False


def _build_sparse_operator_pattern(routing, boundary_routing) -> SparseOperatorPattern:
    volume_indices = np.asarray(routing.indices, dtype=np.int32)
    boundary_indices = np.asarray(boundary_routing.indices, dtype=np.int32)
    combined_indices = np.concatenate((volume_indices, boundary_indices), axis=0)
    unique_indices, inverse = np.unique(combined_indices, axis=0, return_inverse=True)

    row_indices = np.asarray(unique_indices[:, 0], dtype=np.int32)
    col_indices = np.asarray(unique_indices[:, 1], dtype=np.int32)
    indptr = np.zeros(routing.shape[0] + 1, dtype=np.int32)
    np.add.at(indptr, row_indices + 1, 1)
    indptr = np.cumsum(indptr, dtype=np.int32)

    volume_size = volume_indices.shape[0]
    return SparseOperatorPattern(
        shape=routing.shape,
        unique_indices=torch_np.asarray(unique_indices, dtype=INT_DTYPE),
        volume_inverse=torch_np.asarray(inverse[:volume_size], dtype=INT_DTYPE),
        boundary_inverse=torch_np.asarray(inverse[volume_size:], dtype=INT_DTYPE),
        row_indices=row_indices,
        col_indices=col_indices,
        csr_indptr=indptr,
        csr_indices=col_indices,
    )


def _quad_boundary_topology(cells: Array) -> tuple[Array, Array]:
    """Return lexicographically sorted quadrilateral boundary edges and adjacent cells."""

    edge_to_cells: dict[tuple[int, int], list[int]] = {}
    cells_np = np.asarray(cells, dtype=np.int32)
    for cell_id, cell in enumerate(cells_np):
        for start, stop in ((0, 1), (1, 2), (2, 3), (3, 0)):
            edge = tuple(sorted((int(cell[start]), int(cell[stop]))))
            edge_to_cells.setdefault(edge, []).append(cell_id)

    boundary_edges: list[tuple[int, int]] = []
    boundary_cells: list[int] = []
    for edge, adjacent in edge_to_cells.items():
        if len(adjacent) == 1:
            boundary_edges.append(edge)
            boundary_cells.append(adjacent[0])

    order = sorted(range(len(boundary_edges)), key=lambda idx: boundary_edges[idx])
    return (
        torch_np.asarray([boundary_edges[idx] for idx in order], dtype=INT_DTYPE),
        torch_np.asarray([boundary_cells[idx] for idx in order], dtype=INT_DTYPE),
    )


def _quad_boundary_geometry(
    nodes: Array,
    cells: Array,
    boundary_edges: Array,
    boundary_edge_cells: Array,
) -> tuple[Array, Array, Array]:
    """Compute centers, lengths, and outward normals for quadrilateral boundary edges."""

    nodes_np = np.asarray(nodes, dtype=NP_FLOAT_DTYPE)
    cells_np = np.asarray(cells, dtype=np.int32)
    boundary_edges_np = np.asarray(boundary_edges, dtype=np.int32)
    boundary_edge_cells_np = np.asarray(boundary_edge_cells, dtype=np.int32)

    edge_nodes = nodes_np[boundary_edges_np]
    centers = np.mean(edge_nodes, axis=1)
    edge_vectors = edge_nodes[:, 1] - edge_nodes[:, 0]
    lengths = np.linalg.norm(edge_vectors, axis=1)
    candidate_normals = np.stack((edge_vectors[:, 1], -edge_vectors[:, 0]), axis=1) / lengths[:, None]
    cell_centers = np.mean(nodes_np[cells_np[boundary_edge_cells_np]], axis=1)
    direction = centers - cell_centers
    orientation = np.sum(candidate_normals * direction, axis=1)
    signs = np.where(orientation >= 0.0, 1.0, -1.0).astype(NP_FLOAT_DTYPE)
    normals = candidate_normals * signs[:, None]
    return (
        torch_np.asarray(centers, dtype=FLOAT_DTYPE),
        torch_np.asarray(lengths, dtype=FLOAT_DTYPE),
        torch_np.asarray(normals, dtype=FLOAT_DTYPE),
    )


def _quad_surface_metadata(
    nodes: Array,
    boundary_edges: Array,
    surface_node_ids: Array,
    tol: float = 1e-8,
) -> tuple[Array, Array, Array, Array, Array]:
    """Build surface metadata from an explicit ordered quadrilateral boundary path."""

    nodes_np = np.asarray(nodes, dtype=float)
    boundary_edges_np = np.asarray(boundary_edges, dtype=np.int32)
    surface_node_ids_np = np.asarray(surface_node_ids, dtype=np.int32)
    surface_edges = {
        tuple(sorted((int(surface_node_ids_np[idx]), int(surface_node_ids_np[idx + 1]))))
        for idx in range(len(surface_node_ids_np) - 1)
    }
    surface_edge_mask = np.asarray(
        [tuple(sorted((int(edge[0]), int(edge[1])))) in surface_edges for edge in boundary_edges_np],
        dtype=bool,
    )
    surface_nodes = nodes_np[surface_node_ids_np]
    if surface_nodes.size == 0:
        flat_surface = True
        reference_level = 0.0
    else:
        flat_surface = bool(np.max(np.abs(surface_nodes[:, 1] - surface_nodes[0, 1])) <= tol)
        reference_level = float(np.mean(surface_nodes[:, 1]))
    return (
        torch_np.asarray(surface_node_ids_np, dtype=INT_DTYPE),
        torch_np.asarray(surface_edge_mask, dtype=bool),
        torch_np.asarray(surface_nodes, dtype=FLOAT_DTYPE),
        torch_np.asarray(reference_level, dtype=FLOAT_DTYPE),
        torch_np.asarray(flat_surface, dtype=bool),
    )


def _structured_quad_mesh_from_arrays(
    nodes: Array,
    cells: Array,
    surface_node_ids: Array,
) -> StructuredQuadMesh:
    boundary_edges, boundary_edge_cells = _quad_boundary_topology(cells)
    boundary_edge_centers, boundary_edge_lengths, boundary_edge_normals = _quad_boundary_geometry(
        nodes,
        cells,
        boundary_edges,
        boundary_edge_cells,
    )
    surface_node_ids_out, surface_edge_mask, surface_nodes, surface_reference_level, flat_surface = _quad_surface_metadata(
        nodes,
        boundary_edges,
        surface_node_ids,
    )
    return StructuredQuadMesh(
        nodes=torch_np.asarray(nodes, dtype=FLOAT_DTYPE),
        cells=torch_np.asarray(cells, dtype=INT_DTYPE),
        boundary_edges=boundary_edges,
        boundary_edge_cells=boundary_edge_cells,
        boundary_edge_centers=boundary_edge_centers,
        boundary_edge_lengths=boundary_edge_lengths,
        boundary_edge_normals=boundary_edge_normals,
        surface_node_ids=surface_node_ids_out,
        surface_edge_mask=surface_edge_mask,
        surface_nodes=surface_nodes,
        surface_reference_level=surface_reference_level,
        flat_surface=flat_surface,
    )


def _build_structured_quad_mesh(mesh: Mesh) -> tuple[StructuredQuadMesh, Array] | None:
    """Build a structured quadrilateral mesh from native quads or legacy half-cells."""

    if mesh.is_quadrilateral_mesh:
        return (
            _structured_quad_mesh_from_arrays(
                mesh.nodes,
                mesh.cells,
                mesh.surface_node_ids,
            ),
            torch_np.arange(mesh.cell_count, dtype=INT_DTYPE),
        )

    expanded = mesh.expand_columnar_cells()
    if expanded is None:
        return None

    expanded_mesh, expanded_parent_cells = expanded
    cells_np = np.asarray(expanded_mesh.cells, dtype=np.int32)
    parent_np = np.asarray(expanded_parent_cells, dtype=np.int32)
    nodes_np = np.asarray(expanded_mesh.nodes, dtype=float)

    grouped: dict[int, list[np.ndarray]] = {}
    for cell, parent_id in zip(cells_np, parent_np, strict=False):
        grouped.setdefault(int(parent_id), []).append(cell)

    quad_cells: list[list[int]] = []
    quad_parents: list[int] = []
    for parent_id in range(mesh.cell_count):
        pair = grouped.get(parent_id)
        if pair is None or len(pair) != 2:
            return None
        node_ids = np.unique(np.concatenate(pair))
        if node_ids.shape[0] != 4:
            return None

        x_coords = nodes_np[node_ids, 0]
        left_ids = node_ids[np.argsort(x_coords)[:2]]
        right_ids = node_ids[np.argsort(x_coords)[2:]]
        if np.max(nodes_np[left_ids, 0]) > np.min(nodes_np[right_ids, 0]) + 1e-6:
            return None

        top_left = int(left_ids[np.argmax(nodes_np[left_ids, 1])])
        bottom_left = int(left_ids[np.argmin(nodes_np[left_ids, 1])])
        top_right = int(right_ids[np.argmax(nodes_np[right_ids, 1])])
        bottom_right = int(right_ids[np.argmin(nodes_np[right_ids, 1])])
        quad_cells.append([bottom_left, bottom_right, top_right, top_left])
        quad_parents.append(parent_id)

    quad_mesh = _structured_quad_mesh_from_arrays(
        expanded_mesh.nodes,
        torch_np.asarray(quad_cells, dtype=INT_DTYPE),
        expanded_mesh.surface_node_ids,
    )
    return quad_mesh, torch_np.asarray(quad_parents, dtype=INT_DTYPE)


def _refine_structured_quad_mesh(
    mesh: StructuredQuadMesh,
    parent_cell_ids: Array,
) -> tuple[StructuredQuadMesh, Array]:
    """Uniformly refine a structured quadrilateral mesh into four Q1 subcells per quad."""

    nodes_np = np.asarray(mesh.nodes, dtype=float)
    cells_np = np.asarray(mesh.cells, dtype=np.int32)
    parent_np = np.asarray(parent_cell_ids, dtype=np.int32)

    refined_nodes = nodes_np.tolist()
    edge_midpoints: dict[tuple[int, int], int] = {}
    refined_cells: list[list[int]] = []
    refined_parents: list[int] = []

    def midpoint(node_a: int, node_b: int) -> int:
        edge = (min(node_a, node_b), max(node_a, node_b))
        if edge in edge_midpoints:
            return edge_midpoints[edge]
        node_id = len(refined_nodes)
        refined_nodes.append((0.5 * (nodes_np[edge[0]] + nodes_np[edge[1]])).tolist())
        edge_midpoints[edge] = node_id
        return node_id

    for cell_id, (bottom_left, bottom_right, top_right, top_left) in enumerate(cells_np.tolist()):
        mid_bottom = midpoint(bottom_left, bottom_right)
        mid_right = midpoint(bottom_right, top_right)
        mid_top = midpoint(top_right, top_left)
        mid_left = midpoint(top_left, bottom_left)
        center = len(refined_nodes)
        refined_nodes.append(
            (0.25 * (nodes_np[bottom_left] + nodes_np[bottom_right] + nodes_np[top_right] + nodes_np[top_left])).tolist()
        )
        refined_cells.extend(
            [
                [bottom_left, mid_bottom, center, mid_left],
                [mid_bottom, bottom_right, mid_right, center],
                [center, mid_right, top_right, mid_top],
                [mid_left, center, mid_top, top_left],
            ]
        )
        refined_parents.extend([int(parent_np[cell_id])] * 4)

    surface_node_ids_np = np.asarray(mesh.surface_node_ids, dtype=np.int32)
    refined_surface_node_ids = [int(surface_node_ids_np[0])]
    for start_node, stop_node in zip(surface_node_ids_np[:-1], surface_node_ids_np[1:], strict=False):
        midpoint_id = edge_midpoints[(min(int(start_node), int(stop_node)), max(int(start_node), int(stop_node)))]
        refined_surface_node_ids.extend((midpoint_id, int(stop_node)))

    refined_mesh = _structured_quad_mesh_from_arrays(
        torch_np.asarray(refined_nodes, dtype=FLOAT_DTYPE),
        torch_np.asarray(refined_cells, dtype=INT_DTYPE),
        torch_np.asarray(refined_surface_node_ids, dtype=INT_DTYPE),
    )
    return refined_mesh, torch_np.asarray(refined_parents, dtype=INT_DTYPE)


def _q1_shape_functions(points: Array) -> Array:
    """Evaluate bilinear Q1 shape functions on [-1, 1]^2."""

    point_array = torch_np.asarray(points, dtype=FLOAT_DTYPE)
    xi = point_array[..., 0]
    eta = point_array[..., 1]
    return 0.25 * torch_np.stack(
        (
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ),
        axis=-1,
    )


def _q1_reference_shape_gradients(points: Array) -> Array:
    """Return bilinear Q1 reference gradients on [-1, 1]^2."""

    point_array = torch_np.asarray(points, dtype=FLOAT_DTYPE)
    xi = point_array[:, 0]
    eta = point_array[:, 1]
    dxi = 0.25 * torch_np.stack((-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)), axis=1)
    deta = 0.25 * torch_np.stack((-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)), axis=1)
    return torch_np.stack((dxi, deta), axis=-1)


def _build_q1_operator_templates(mesh: StructuredQuadMesh) -> OperatorTemplates:
    """Build bilinear quadrilateral operator templates for a structured mesh."""

    gauss = float(1.0 / np.sqrt(3.0))
    quadrature_points = np.asarray(
        [
            [-gauss, -gauss],
            [gauss, -gauss],
            [gauss, gauss],
            [-gauss, gauss],
        ],
        dtype=NP_FLOAT_DTYPE,
    )
    quadrature_weights = np.ones((4,), dtype=NP_FLOAT_DTYPE)

    xi = quadrature_points[:, 0]
    eta = quadrature_points[:, 1]
    shape_values = 0.25 * np.stack(
        (
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ),
        axis=-1,
    ).astype(NP_FLOAT_DTYPE)
    dxi = 0.25 * np.stack((-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)), axis=1)
    deta = 0.25 * np.stack((-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)), axis=1)
    reference_gradients = np.stack((dxi, deta), axis=-1).astype(NP_FLOAT_DTYPE)

    cell_nodes = np.asarray(mesh.nodes, dtype=NP_FLOAT_DTYPE)[np.asarray(mesh.cells, dtype=np.int32)]
    jacobians = np.einsum("cid,qia->cqda", cell_nodes, reference_gradients)
    determinant = np.linalg.det(jacobians)
    inverse_jacobian = np.linalg.inv(jacobians)
    gradients = np.einsum("qia,cqab->cqib", reference_gradients, inverse_jacobian)
    weighted_det = np.abs(determinant) * quadrature_weights[None, :]

    stiffness = np.einsum("cqid,cqjd,cq->cij", gradients, gradients, weighted_det)
    mass = np.einsum("qi,qj,cq->cij", shape_values, shape_values, weighted_det)
    boundary_reference = np.asarray([[2.0, 1.0], [1.0, 2.0]], dtype=NP_FLOAT_DTYPE) / 6.0
    boundary_lengths = np.asarray(mesh.boundary_edge_lengths, dtype=NP_FLOAT_DTYPE)
    boundary_mass = boundary_lengths[:, None, None] * boundary_reference[None, :, :]
    return OperatorTemplates(
        stiffness=torch_np.asarray(stiffness, dtype=FLOAT_DTYPE),
        mass=torch_np.asarray(mass, dtype=FLOAT_DTYPE),
        boundary_mass=torch_np.asarray(boundary_mass, dtype=FLOAT_DTYPE),
    )


def _q2_shape_functions_np(points: np.ndarray) -> np.ndarray:
    """Evaluate 8-node serendipity quadrilateral shape functions."""

    xi = points[:, 0]
    eta = points[:, 1]
    return np.stack(
        (
            -0.25 * (1.0 - xi) * (1.0 - eta) * (1.0 + xi + eta),
            -0.25 * (1.0 + xi) * (1.0 - eta) * (1.0 - xi + eta),
            -0.25 * (1.0 + xi) * (1.0 + eta) * (1.0 - xi - eta),
            -0.25 * (1.0 - xi) * (1.0 + eta) * (1.0 + xi - eta),
            0.5 * (1.0 - xi * xi) * (1.0 - eta),
            0.5 * (1.0 + xi) * (1.0 - eta * eta),
            0.5 * (1.0 - xi * xi) * (1.0 + eta),
            0.5 * (1.0 - xi) * (1.0 - eta * eta),
        ),
        axis=-1,
    )


def _q2_reference_shape_gradients_np(points: np.ndarray) -> np.ndarray:
    """Return reference gradients for 8-node serendipity quadrilateral basis."""

    xi = points[:, 0]
    eta = points[:, 1]
    dxi = np.stack(
        (
            0.25 * (1.0 - eta) * (2.0 * xi + eta),
            0.25 * (1.0 - eta) * (2.0 * xi - eta),
            0.25 * (1.0 + eta) * (2.0 * xi + eta),
            0.25 * (1.0 + eta) * (2.0 * xi - eta),
            -xi * (1.0 - eta),
            0.5 * (1.0 - eta * eta),
            -xi * (1.0 + eta),
            -0.5 * (1.0 - eta * eta),
        ),
        axis=-1,
    )
    deta = np.stack(
        (
            0.25 * (1.0 - xi) * (xi + 2.0 * eta),
            0.25 * (1.0 + xi) * (-xi + 2.0 * eta),
            0.25 * (1.0 + xi) * (xi + 2.0 * eta),
            0.25 * (1.0 - xi) * (-xi + 2.0 * eta),
            -0.5 * (1.0 - xi * xi),
            -(1.0 + xi) * eta,
            0.5 * (1.0 - xi * xi),
            -(1.0 - xi) * eta,
        ),
        axis=-1,
    )
    return np.stack((dxi, deta), axis=-1)


def _build_q2_operator_templates(mesh: StructuredQuadMesh) -> OperatorTemplates:
    """Build serendipity-Q2 field templates on a bilinear quadrilateral geometry."""

    dtype = np.float64 if torch_runtime.config.torch_enable_float64 else np.float32
    gauss_points_1d, gauss_weights_1d = np.polynomial.legendre.leggauss(3)
    quadrature_points = np.asarray(
        [[xi, eta] for eta in gauss_points_1d for xi in gauss_points_1d],
        dtype=dtype,
    )
    quadrature_weights = np.asarray(
        [wx * wy for wy in gauss_weights_1d for wx in gauss_weights_1d],
        dtype=dtype,
    )

    shape_values = _q2_shape_functions_np(quadrature_points).astype(dtype)
    reference_gradients = _q2_reference_shape_gradients_np(quadrature_points).astype(dtype)
    xi = quadrature_points[:, 0]
    eta = quadrature_points[:, 1]
    geometry_dxi = 0.25 * np.stack((-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)), axis=1)
    geometry_deta = 0.25 * np.stack((-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)), axis=1)
    geometry_gradients = np.stack((geometry_dxi, geometry_deta), axis=-1).astype(dtype)

    geometry_cells = np.asarray(mesh.nodes, dtype=dtype)[np.asarray(mesh.cells, dtype=np.int32)]
    jacobians = np.einsum("cid,qia->cqda", geometry_cells, geometry_gradients)
    determinant = np.linalg.det(jacobians)
    inverse_jacobian = np.linalg.inv(jacobians)
    gradients = np.einsum("qia,cqab->cqib", reference_gradients, inverse_jacobian)
    weighted_det = np.abs(determinant) * quadrature_weights[None, :]

    stiffness = np.einsum("cqid,cqjd,cq->cij", gradients, gradients, weighted_det)
    mass = np.einsum("qi,qj,cq->cij", shape_values, shape_values, weighted_det)
    boundary_reference = np.asarray(
        [
            [4.0, 2.0, -1.0],
            [2.0, 16.0, 2.0],
            [-1.0, 2.0, 4.0],
        ],
        dtype=dtype,
    ) / 30.0
    boundary_lengths = np.asarray(mesh.boundary_edge_lengths, dtype=dtype)
    boundary_mass = boundary_lengths[:, None, None] * boundary_reference[None, :, :]
    torch_dtype = torch_np.float64 if torch_runtime.config.torch_enable_float64 else FLOAT_DTYPE
    return OperatorTemplates(
        stiffness=torch_np.asarray(stiffness, dtype=torch_dtype),
        mass=torch_np.asarray(mass, dtype=torch_dtype),
        boundary_mass=torch_np.asarray(boundary_mass, dtype=torch_dtype),
    )


def _build_structured_quad_q2_topology(mesh: StructuredQuadMesh) -> tuple[Array, Array, Array]:
    """Build shared-edge serendipity-Q2 topology on a structured quadrilateral mesh."""

    nodes_np = np.asarray(mesh.nodes, dtype=float)
    cells_np = np.asarray(mesh.cells, dtype=np.int32)
    boundary_edges_np = np.asarray(mesh.boundary_edges, dtype=np.int32)

    q2_nodes = nodes_np.tolist()
    edge_midpoints: dict[tuple[int, int], int] = {}

    def midpoint_node(node_a: int, node_b: int) -> int:
        edge = (min(node_a, node_b), max(node_a, node_b))
        if edge in edge_midpoints:
            return edge_midpoints[edge]
        node_id = len(q2_nodes)
        q2_nodes.append((0.5 * (nodes_np[edge[0]] + nodes_np[edge[1]])).tolist())
        edge_midpoints[edge] = node_id
        return node_id

    q2_cells: list[list[int]] = []
    for bottom_left, bottom_right, top_right, top_left in cells_np.tolist():
        mid_bottom = midpoint_node(bottom_left, bottom_right)
        mid_right = midpoint_node(bottom_right, top_right)
        mid_top = midpoint_node(top_right, top_left)
        mid_left = midpoint_node(top_left, bottom_left)
        q2_cells.append(
            [
                int(bottom_left),
                int(bottom_right),
                int(top_right),
                int(top_left),
                mid_bottom,
                mid_right,
                mid_top,
                mid_left,
            ]
        )

    q2_boundary_edges = [
        [int(node_a), midpoint_node(int(node_a), int(node_b)), int(node_b)]
        for node_a, node_b in boundary_edges_np.tolist()
    ]
    return (
        torch_np.asarray(q2_nodes, dtype=FLOAT_DTYPE),
        torch_np.asarray(q2_cells, dtype=INT_DTYPE),
        torch_np.asarray(q2_boundary_edges, dtype=INT_DTYPE),
    )


def _exact_node_selection_indices(
    dof_nodes: Array,
    points: Array,
    tol: float = 1e-6,
) -> np.ndarray:
    """Return source node indices for points that coincide with mesh nodes."""

    dof_nodes_np = np.asarray(dof_nodes, dtype=float)
    points_np = np.asarray(points, dtype=float)
    indices = np.empty((points_np.shape[0],), dtype=np.int32)
    scale = 1.0 / tol
    node_lookup: dict[tuple[int, int], int] = {}
    for node_id, node in enumerate(dof_nodes_np):
        key = tuple(np.rint(node * scale).astype(np.int64).tolist())
        node_lookup.setdefault(key, node_id)

    for row_id, point in enumerate(points_np):
        key = tuple(np.rint(point * scale).astype(np.int64).tolist())
        nearest = node_lookup.get(key)
        if nearest is None:
            distances = np.linalg.norm(dof_nodes_np - point, axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] > tol:
                raise ValueError("point does not coincide with a quadrilateral auxiliary node")
        indices[row_id] = int(nearest)
    return indices


def _build_exact_node_selection_matrix(
    dof_nodes: Array,
    points: Array,
    tol: float = 1e-6,
) -> Array:
    """Build a one-hot interpolation matrix for points that coincide with mesh nodes."""

    indices = _exact_node_selection_indices(dof_nodes, points, tol=tol)
    matrix = np.zeros((indices.shape[0], np.asarray(dof_nodes).shape[0]), dtype=NP_FLOAT_DTYPE)
    matrix[np.arange(indices.shape[0]), indices] = 1.0
    return torch_np.asarray(matrix, dtype=FLOAT_DTYPE)


def _build_surface_interpolation_matrix(
    surface_nodes: Array,
    surface_node_ids: Array,
    node_count: int,
    points: Array,
    tol: float = 1e-5,
) -> Array:
    """Interpolate points that lie on the top surface polyline into nodal DOFs."""

    surface_nodes_np = np.asarray(surface_nodes, dtype=float)
    surface_node_ids_np = np.asarray(surface_node_ids, dtype=np.int32)
    points_np = np.asarray(points, dtype=float)
    matrix = np.zeros((points_np.shape[0], node_count), dtype=NP_FLOAT_DTYPE)

    segments = np.stack((surface_nodes_np[:-1], surface_nodes_np[1:]), axis=1)
    hits = _points_on_segments(points_np, segments, tol=tol)

    for row_id, point in enumerate(points_np):
        distances = np.linalg.norm(surface_nodes_np - point, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= tol:
            matrix[row_id, surface_node_ids_np[nearest]] = 1.0
            continue

        segment_ids = np.flatnonzero(hits[row_id])
        if segment_ids.size == 0:
            raise ValueError("surface point could not be matched to an auxiliary surface edge")
        segment_id = int(segment_ids[0])
        start = segments[segment_id, 0]
        stop = segments[segment_id, 1]
        edge = stop - start
        edge_length_sq = float(np.dot(edge, edge))
        if edge_length_sq <= tol:
            raise ValueError("degenerate surface edge in auxiliary quadrilateral mesh")
        weight_stop = float(np.dot(point - start, edge) / edge_length_sq)
        weight_stop = min(max(weight_stop, 0.0), 1.0)
        weight_start = 1.0 - weight_stop
        matrix[row_id, surface_node_ids_np[segment_id]] = weight_start
        matrix[row_id, surface_node_ids_np[segment_id + 1]] = weight_stop

    return torch_np.asarray(matrix, dtype=FLOAT_DTYPE)


def _build_surface_q2_interpolation_matrix(
    surface_nodes: Array,
    surface_node_ids: Array,
    boundary_connectivity: Array,
    node_count: int,
    points: Array,
    tol: float = 1e-5,
) -> Array:
    """Interpolate top-surface points into serendipity-Q2 edge DOFs."""

    surface_nodes_np = np.asarray(surface_nodes, dtype=float)
    surface_node_ids_np = np.asarray(surface_node_ids, dtype=np.int32)
    boundary_np = np.asarray(boundary_connectivity, dtype=np.int32)
    points_np = np.asarray(points, dtype=float)
    matrix = np.zeros((points_np.shape[0], node_count), dtype=NP_FLOAT_DTYPE)

    edge_midpoints = {
        tuple(sorted((int(start), int(stop)))): int(midpoint)
        for start, midpoint, stop in boundary_np.tolist()
    }
    segments = np.stack((surface_nodes_np[:-1], surface_nodes_np[1:]), axis=1)
    hits = _points_on_segments(points_np, segments, tol=tol)

    for row_id, point in enumerate(points_np):
        distances = np.linalg.norm(surface_nodes_np - point, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= tol:
            matrix[row_id, surface_node_ids_np[nearest]] = 1.0
            continue

        segment_ids = np.flatnonzero(hits[row_id])
        if segment_ids.size == 0:
            raise ValueError("surface point could not be matched to an auxiliary surface edge")
        segment_id = int(segment_ids[0])
        start = segments[segment_id, 0]
        stop = segments[segment_id, 1]
        edge = stop - start
        edge_length_sq = float(np.dot(edge, edge))
        if edge_length_sq <= tol:
            raise ValueError("degenerate surface edge in auxiliary quadrilateral mesh")
        t = float(np.dot(point - start, edge) / edge_length_sq)
        t = min(max(t, 0.0), 1.0)

        start_id = int(surface_node_ids_np[segment_id])
        stop_id = int(surface_node_ids_np[segment_id + 1])
        midpoint_id = edge_midpoints[tuple(sorted((start_id, stop_id)))]
        matrix[row_id, start_id] = 2.0 * (t - 0.5) * (t - 1.0)
        matrix[row_id, midpoint_id] = 4.0 * t * (1.0 - t)
        matrix[row_id, stop_id] = 2.0 * t * (t - 0.5)

    return torch_np.asarray(matrix, dtype=FLOAT_DTYPE)


def _build_structured_quad_auxiliary_discretization(
    mesh: Mesh,
    survey: Survey,
    wavenumbers: Array,
    *,
    use_h2: bool,
    use_p2: bool,
) -> AuxiliaryDiscretization | None:
    """Build a structured quadrilateral auxiliary mesh for terrain-following strip meshes."""

    structured = _build_structured_quad_mesh(mesh)
    if structured is None:
        return None

    geometry_mesh, parent_cell_ids = structured
    if use_h2:
        geometry_mesh, parent_cell_ids = _refine_structured_quad_mesh(geometry_mesh, parent_cell_ids)

    source_positions = torch_np.asarray(survey.electrode_positions, dtype=FLOAT_DTYPE)
    if use_p2:
        dof_nodes, cell_connectivity, boundary_connectivity = _build_structured_quad_q2_topology(geometry_mesh)
        operator_templates = _build_q2_operator_templates(geometry_mesh)
        electrode_matrix = _build_surface_q2_interpolation_matrix(
            geometry_mesh.surface_nodes,
            geometry_mesh.surface_node_ids,
            boundary_connectivity,
            int(dof_nodes.shape[0]),
            survey.electrode_positions,
        )
    else:
        dof_nodes = geometry_mesh.nodes
        cell_connectivity = geometry_mesh.cells
        boundary_connectivity = geometry_mesh.boundary_edges
        operator_templates = _build_q1_operator_templates(geometry_mesh)
        electrode_matrix = _build_surface_interpolation_matrix(
            geometry_mesh.surface_nodes,
            geometry_mesh.surface_node_ids,
            int(dof_nodes.shape[0]),
            survey.electrode_positions,
        )

    routing = build_coo_routing_from_connectivity(cell_connectivity, int(dof_nodes.shape[0]))
    boundary_routing = build_boundary_routing_from_connectivity(boundary_connectivity, int(dof_nodes.shape[0]))
    operator_pattern = _build_sparse_operator_pattern(routing, boundary_routing)
    source_matrix = electrode_matrix
    original_node_matrix = _build_exact_node_selection_matrix(dof_nodes, mesh.nodes)
    # The terrain-strip H2 mesh keeps these outer boundaries as
    # homogeneous natural boundaries, not mixed/Robin boundaries.
    boundary_geometries = torch_np.asarray(
        np.zeros((wavenumbers.shape[0], geometry_mesh.boundary_edges.shape[0]), dtype=NP_FLOAT_DTYPE),
        dtype=FLOAT_DTYPE,
    )

    return AuxiliaryDiscretization(
        geometry_mesh=geometry_mesh,
        parent_cell_ids=parent_cell_ids,
        dof_nodes=dof_nodes,
        cell_connectivity=cell_connectivity,
        boundary_connectivity=boundary_connectivity,
        routing=routing,
        boundary_routing=boundary_routing,
        operator_pattern=operator_pattern,
        operator_templates=operator_templates,
        source_positions=source_positions,
        source_matrix=source_matrix,
        electrode_matrix=electrode_matrix,
        original_node_matrix=original_node_matrix,
        boundary_geometries=boundary_geometries,
    )


def _build_auxiliary_discretization(
    mesh: Mesh,
    survey: Survey,
    wavenumbers: Array,
    *,
    quadrature_order: int,
    use_h2: bool,
    use_p2: bool,
) -> AuxiliaryDiscretization:
    """Build the auxiliary discretization used for topographic numerical solves."""

    structured_quad = _build_structured_quad_auxiliary_discretization(
        mesh,
        survey,
        wavenumbers,
        use_h2=use_h2,
        use_p2=use_p2,
    )
    if structured_quad is not None:
        return structured_quad

    geometry_mesh = mesh
    parent_cell_ids = torch_np.arange(mesh.cell_count, dtype=INT_DTYPE)
    if not use_p2:
        expanded = geometry_mesh.expand_columnar_cells()
        if expanded is not None:
            geometry_mesh, expanded_parent_cells = expanded
            parent_cell_ids = parent_cell_ids[expanded_parent_cells]
    if use_h2:
        geometry_mesh, refined_parent_cells = geometry_mesh.refine_uniform()
        parent_cell_ids = parent_cell_ids[refined_parent_cells]

    if use_p2:
        dof_nodes, cell_connectivity, boundary_connectivity = geometry_mesh.build_quadratic_topology()
        element_data = build_p2_element_data(geometry_mesh)
        routing = build_coo_routing_from_connectivity(cell_connectivity, int(dof_nodes.shape[0]))
        boundary_routing = build_boundary_routing_from_connectivity(boundary_connectivity, int(dof_nodes.shape[0]))
        operator_templates = OperatorTemplates(
            stiffness=assemble_local_stiffness_p2(element_data, conductivity=1.0),
            mass=assemble_local_mass_p2(element_data, coefficients=1.0),
            boundary_mass=assemble_local_boundary_mass_p2(geometry_mesh.boundary_edge_lengths, 1.0),
        )
    else:
        dof_nodes = geometry_mesh.nodes
        cell_connectivity = geometry_mesh.cells
        boundary_connectivity = geometry_mesh.boundary_edges
        element_data = build_p1_element_data(geometry_mesh, quadrature_order=quadrature_order)
        routing = build_coo_routing(geometry_mesh)
        boundary_routing = build_boundary_routing(geometry_mesh)
        operator_templates = OperatorTemplates(
            stiffness=assemble_local_stiffness(element_data, conductivity=1.0),
            mass=assemble_local_mass(element_data, coefficients=1.0),
            boundary_mass=assemble_local_boundary_mass(geometry_mesh, 1.0),
        )

    source_cell_ids = _locate_point_cells(geometry_mesh, survey.electrode_positions)
    source_positions = _build_source_positions(geometry_mesh, survey, source_cell_ids)
    electrode_matrix = _build_discretization_interpolation_matrix(
        geometry_mesh,
        dof_nodes,
        cell_connectivity,
        survey.electrode_positions,
    )
    if _same_point_locations(source_positions, survey.electrode_positions):
        source_matrix = electrode_matrix
    else:
        source_matrix = _build_discretization_interpolation_matrix(
            geometry_mesh,
            dof_nodes,
            cell_connectivity,
            source_positions,
        )
    original_node_matrix = _build_discretization_interpolation_matrix(
        geometry_mesh,
        dof_nodes,
        cell_connectivity,
        mesh.nodes,
    )
    operator_pattern = _build_sparse_operator_pattern(routing, boundary_routing)
    source_center = torch_np.asarray(np.mean(np.asarray(survey.electrode_positions, dtype=float), axis=0), dtype=FLOAT_DTYPE)
    boundary_geometries = torch_np.stack(
        [
            robin_boundary_coefficients(
                mesh=geometry_mesh,
                conductivity=1.0,
                source_center=source_center,
                wavenumber=float(wavenumber),
            )
            for wavenumber in wavenumbers.tolist()
        ],
        axis=0,
    )

    return AuxiliaryDiscretization(
        geometry_mesh=geometry_mesh,
        parent_cell_ids=parent_cell_ids,
        dof_nodes=dof_nodes,
        cell_connectivity=cell_connectivity,
        boundary_connectivity=boundary_connectivity,
        routing=routing,
        boundary_routing=boundary_routing,
        operator_pattern=operator_pattern,
        operator_templates=operator_templates,
        source_positions=source_positions,
        source_matrix=source_matrix,
        electrode_matrix=electrode_matrix,
        original_node_matrix=original_node_matrix,
        boundary_geometries=boundary_geometries,
    )


def _build_source_resistivity_data(mesh: Mesh, source_node_ids: Array) -> SourceResistivityData:
    source_node_ids_np = np.asarray(source_node_ids, dtype=np.int32)
    cells_np = np.asarray(mesh.cells, dtype=np.int32)
    weights = np.zeros((source_node_ids_np.shape[0], mesh.cell_count), dtype=NP_FLOAT_DTYPE)
    counts = np.ones((source_node_ids_np.shape[0],), dtype=NP_FLOAT_DTYPE)

    for source_idx, node_id in enumerate(source_node_ids_np):
        if node_id < 0:
            continue
        mask = np.any(cells_np == node_id, axis=1)
        weights[source_idx, mask] = 1.0
        counts[source_idx] = float(np.count_nonzero(mask))

    return SourceResistivityData(
        node_cell_weights=torch_np.asarray(weights, dtype=FLOAT_DTYPE),
        node_cell_counts=torch_np.asarray(counts, dtype=FLOAT_DTYPE),
    )


@dataclass(frozen=True)
class ERTForward2p5D:
    """Pinned 2.5D forward operator using a secondary-field terrain path."""

    mesh: Mesh
    survey: Survey
    element_data: object
    routing: object
    boundary_routing: object
    source_positions: Array
    source_cell_ids: Array
    matched_node_ids: Array
    source_node_ids: Array
    source_matrix: Array
    electrode_matrix: Array
    wavenumbers: Array
    weights: Array
    source_center: Array
    operator_pattern: SparseOperatorPattern
    operator_templates: OperatorTemplates
    source_resistivity_data: SourceResistivityData
    boundary_geometries: Array
    wavenumber_to_index: dict[float, int]
    use_numerical_primary: bool
    use_numerical_geometric_factors: bool
    numerical_h2_refined: bool
    numerical_p2_refined: bool
    topographic_geometric_factor_mode: str
    primary_auxiliary_discretization: AuxiliaryDiscretization | None
    primary_potential_discretization: AuxiliaryDiscretization | None
    geometric_auxiliary_discretization: AuxiliaryDiscretization | None
    auxiliary_discretization: AuxiliaryDiscretization | None
    linear_solver_backend: str
    terrain_cache_dir: Path | None
    _unit_primary_cache: dict[float, Array] = field(default_factory=dict, init=False, repr=False, compare=False)
    _reference_rhs_cache: dict[float, Array] = field(default_factory=dict, init=False, repr=False, compare=False)
    _cudss_state: dict[str, object] = field(default_factory=dict, init=False, repr=False, compare=False)
    _kernel_cache: dict[str, object] = field(default_factory=dict, init=False, repr=False, compare=False)
    _derived_cache: dict[str, Array] = field(default_factory=dict, init=False, repr=False, compare=False)

    @classmethod
    def from_mesh_survey(
        cls,
        mesh: Mesh,
        survey: Survey,
        quadrature_order: int = 2,
        numerical_h2_refined: bool = True,
        numerical_p2_refined: bool = True,
        topographic_geometric_factor_mode: str = "analytic",
        linear_solver_backend: str = "auto",
        terrain_cache_dir: str | Path | None = None,
    ) -> "ERTForward2p5D":
        """Build the single supported forward configuration."""

        routing = build_coo_routing(mesh)
        boundary_routing = build_boundary_routing(mesh)
        source_cell_ids = _locate_point_cells(mesh, survey.electrode_positions)
        source_positions = _build_source_positions(mesh, survey, source_cell_ids)
        matched_node_ids = _find_nearest_node_electrode_ids(mesh, survey.electrode_positions)
        entity_node_ids = _find_entity_node_ids(mesh, survey.electrode_positions, source_cell_ids)
        source_node_ids = torch_np.where(matched_node_ids >= 0, matched_node_ids, entity_node_ids)
        electrode_matrix = _build_interpolation_matrix(mesh, survey.electrode_positions)
        if _same_point_locations(source_positions, survey.electrode_positions):
            source_matrix = electrode_matrix
        else:
            source_matrix = _build_interpolation_matrix(mesh, source_positions)
        r_min, r_max = survey_wavenumber_bounds(survey)
        cosine_weights = build_inverse_cosine_weights(r_min, r_max)
        source_center = torch_np.asarray(
            np.mean(np.asarray(survey.electrode_positions, dtype=float), axis=0),
            dtype=FLOAT_DTYPE,
        )
        operator_pattern = _build_sparse_operator_pattern(routing, boundary_routing)
        if mesh.is_triangle_mesh:
            element_data = build_p1_element_data(mesh, quadrature_order=quadrature_order)
            operator_templates = OperatorTemplates(
                stiffness=assemble_local_stiffness(element_data, conductivity=1.0),
                mass=assemble_local_mass(element_data, coefficients=1.0),
                boundary_mass=assemble_local_boundary_mass(mesh, 1.0),
            )
        else:
            element_data = None
            operator_templates = _build_q1_operator_templates(
                _structured_quad_mesh_from_arrays(
                    mesh.nodes,
                    mesh.cells,
                    mesh.surface_node_ids,
                )
            )
        source_resistivity_data = _build_source_resistivity_data(mesh, source_node_ids)
        boundary_geometries = torch_np.stack(
            [
                robin_boundary_coefficients(
                    mesh=mesh,
                    conductivity=1.0,
                    source_center=source_center,
                    wavenumber=float(wavenumber),
                )
                for wavenumber in cosine_weights.wavenumbers.tolist()
            ],
            axis=0,
        )
        wavenumber_to_index = {
            float(wavenumber): idx for idx, wavenumber in enumerate(cosine_weights.wavenumbers.tolist())
        }
        normalized_topographic_geometric_factor_mode = _normalize_topographic_geometric_factor_mode(
            topographic_geometric_factor_mode
        )
        use_numerical_primary = not mesh.is_flat_surface
        use_numerical_geometric_factors = (
            not mesh.is_flat_surface and normalized_topographic_geometric_factor_mode == "numerical"
        )
        if use_numerical_primary:
            _enable_torch_float64_for_terrain_auxiliary()

        primary_auxiliary_discretization = None
        primary_potential_discretization = None
        if use_numerical_primary:
            # The secondary-field solve stays on an H2/P1 mesh and
            # computes terrain primary potentials separately on mesh.createP2().
            primary_auxiliary_discretization = _build_auxiliary_discretization(
                mesh,
                survey,
                cosine_weights.wavenumbers,
                quadrature_order=quadrature_order,
                use_h2=numerical_h2_refined,
                use_p2=False,
            )
            primary_potential_discretization = _build_auxiliary_discretization(
                mesh,
                survey,
                cosine_weights.wavenumbers,
                quadrature_order=quadrature_order,
                use_h2=numerical_h2_refined,
                use_p2=True,
            )

        geometric_auxiliary_discretization = None
        if use_numerical_geometric_factors:
            geometric_auxiliary_discretization = _build_auxiliary_discretization(
                mesh,
                survey,
                cosine_weights.wavenumbers,
                quadrature_order=quadrature_order,
                use_h2=numerical_h2_refined,
                use_p2=numerical_p2_refined,
            )

        auxiliary_discretization = geometric_auxiliary_discretization
        if auxiliary_discretization is None:
            auxiliary_discretization = primary_auxiliary_discretization

        return cls(
            mesh=mesh,
            survey=survey,
            element_data=element_data,
            routing=routing,
            boundary_routing=boundary_routing,
            source_positions=source_positions,
            source_cell_ids=source_cell_ids,
            matched_node_ids=matched_node_ids,
            source_node_ids=source_node_ids,
            source_matrix=source_matrix,
            electrode_matrix=electrode_matrix,
            wavenumbers=cosine_weights.wavenumbers,
            weights=cosine_weights.weights,
            source_center=source_center,
            operator_pattern=operator_pattern,
            operator_templates=operator_templates,
            source_resistivity_data=source_resistivity_data,
            boundary_geometries=boundary_geometries,
            wavenumber_to_index=wavenumber_to_index,
            use_numerical_primary=use_numerical_primary,
            use_numerical_geometric_factors=use_numerical_geometric_factors,
            numerical_h2_refined=numerical_h2_refined,
            numerical_p2_refined=numerical_p2_refined,
            topographic_geometric_factor_mode=normalized_topographic_geometric_factor_mode,
            primary_auxiliary_discretization=primary_auxiliary_discretization,
            primary_potential_discretization=primary_potential_discretization,
            geometric_auxiliary_discretization=geometric_auxiliary_discretization,
            auxiliary_discretization=auxiliary_discretization,
            linear_solver_backend=_normalize_linear_solver_backend(linear_solver_backend),
            terrain_cache_dir=_normalize_cache_dir(terrain_cache_dir),
        )

    def _wavenumber_index(self, wavenumber: float) -> int:
        return self.wavenumber_to_index[float(wavenumber)]

    def _assemble_operator_values_kernel(self):
        kernel = self._kernel_cache.get("assemble_operator_values")
        if kernel is not None:
            return kernel

        stiffness = self.operator_templates.stiffness
        mass = self.operator_templates.mass
        boundary_mass = self.operator_templates.boundary_mass
        boundary_edge_cells = self.mesh.boundary_edge_cells
        volume_inverse = self.operator_pattern.volume_inverse
        boundary_inverse = self.operator_pattern.boundary_inverse
        nnz = self.operator_pattern.unique_indices.shape[0]

        def kernel(conductivity: Array, wavenumber_sq: Array, boundary_geometry: Array) -> Array:
            volume_values = conductivity[:, None, None] * (stiffness + wavenumber_sq * mass)
            boundary_values = (
                conductivity[boundary_edge_cells] * boundary_geometry
            )[:, None, None] * boundary_mass
            data = torch_np.zeros((nnz,), dtype=FLOAT_DTYPE)
            data = data.at[volume_inverse].add(volume_values.reshape(-1))
            data = data.at[boundary_inverse].add(boundary_values.reshape(-1))
            return data

        self._kernel_cache["assemble_operator_values"] = kernel
        return kernel

    def _assemble_operator_values_batch_kernel(self):
        kernel = self._kernel_cache.get("assemble_operator_values_batch")
        if kernel is not None:
            return kernel

        stiffness = self.operator_templates.stiffness
        mass = self.operator_templates.mass
        boundary_mass = self.operator_templates.boundary_mass
        boundary_edge_cells = self.mesh.boundary_edge_cells
        boundary_geometries = self.boundary_geometries
        wavenumber_sq = torch_np.square(self.wavenumbers).astype(FLOAT_DTYPE)
        volume_inverse = self.operator_pattern.volume_inverse
        boundary_inverse = self.operator_pattern.boundary_inverse
        nnz = self.operator_pattern.unique_indices.shape[0]
        wave_count = int(self.wavenumbers.shape[0])

        def kernel(conductivity: Array) -> Array:
            conductivity_boundary = conductivity[boundary_edge_cells]
            volume_values = conductivity[None, :, None, None] * (
                stiffness[None, :, :, :] + wavenumber_sq[:, None, None, None] * mass[None, :, :, :]
            )
            boundary_values = (
                conductivity_boundary[None, :] * boundary_geometries
            )[:, :, None, None] * boundary_mass[None, :, :, :]
            data = torch_np.zeros((wave_count, nnz), dtype=FLOAT_DTYPE)
            data = data.at[:, volume_inverse].add(volume_values.reshape(wave_count, -1))
            data = data.at[:, boundary_inverse].add(boundary_values.reshape(wave_count, -1))
            return data

        self._kernel_cache["assemble_operator_values_batch"] = kernel
        return kernel

    def _apply_operator_values_kernel(self):
        kernel = self._kernel_cache.get("apply_operator_values")
        if kernel is not None:
            return kernel

        shape = self.operator_pattern.shape
        csr_indices = torch_np.asarray(self.operator_pattern.csr_indices, dtype=INT_DTYPE)
        csr_indptr = torch_np.asarray(self.operator_pattern.csr_indptr, dtype=INT_DTYPE)

        def kernel(values: Array, vectors: Array) -> Array:
            operator = CSR((values, csr_indices, csr_indptr), shape=shape)
            return (operator @ vectors.T).T

        self._kernel_cache["apply_operator_values"] = kernel
        return kernel

    def _apply_operator_values_batch_kernel(self):
        kernel = self._kernel_cache.get("apply_operator_values_batch")
        if kernel is not None:
            return kernel

        shape = self.operator_pattern.shape
        csr_indices = torch_np.asarray(self.operator_pattern.csr_indices, dtype=INT_DTYPE)
        csr_indptr = torch_np.asarray(self.operator_pattern.csr_indptr, dtype=INT_DTYPE)

        def apply_single(args: tuple[Array, Array]) -> Array:
            values, vectors = args
            operator = CSR((values, csr_indices, csr_indptr), shape=shape)
            return (operator @ vectors.T).T

        def kernel(values: Array, vectors: Array) -> Array:
            return torch_runtime.map(apply_single, (values, vectors))

        self._kernel_cache["apply_operator_values_batch"] = kernel
        return kernel

    def _cupy_sparse_modules(self):
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cupy_sparse
        except ImportError as exc:
            raise ImportError("ERTForward2p5D requires CuPy and cuDSS on a CUDA-capable system") from exc
        return cp, cupy_sparse

    def _cupy_csr_fixed(self):
        fixed = self._cudss_state.get("gpu_fixed")
        if fixed is not None:
            return fixed

        cp, _ = self._cupy_sparse_modules()
        fixed = (
            cp.asarray(self.operator_pattern.csr_indptr, dtype=cp.int32),
            cp.asarray(self.operator_pattern.csr_indices, dtype=cp.int32),
        )
        self._cudss_state["gpu_fixed"] = fixed
        self._cudss_state["gpu_enabled"] = True
        return fixed

    def _cupy_csr_fixed_for(self, operator_pattern: SparseOperatorPattern, state_key: str):
        fixed = self._cudss_state.get(state_key)
        if fixed is not None:
            return fixed

        cp, _ = self._cupy_sparse_modules()
        fixed = (
            cp.asarray(operator_pattern.csr_indptr, dtype=cp.int32),
            cp.asarray(operator_pattern.csr_indices, dtype=cp.int32),
        )
        self._cudss_state[state_key] = fixed
        self._cudss_state["gpu_enabled"] = True
        return fixed

    def _auxiliary_float_dtype(self):
        if self.use_numerical_primary and torch_runtime.config.torch_enable_float64:
            return torch_np.float64
        return FLOAT_DTYPE

    def _cupy_from_torch(self, values: Array, *, transpose: bool = False, fortran: bool = False, dtype=FLOAT_DTYPE):
        cp, _ = self._cupy_sparse_modules()
        array = values.astype(dtype)
        if transpose:
            array = array.T

        if not _array_is_on_cuda(array):
            numpy_array = np.asarray(array)
            cupy_array = cp.asarray(numpy_array)
            if fortran:
                cupy_array = cp.asfortranarray(cupy_array)
            self._cudss_state["gpu_zero_copy"] = False
            return cupy_array

        try:
            cupy_array = cp.from_dlpack(array.__dlpack__())
            if fortran:
                cupy_array = cp.asfortranarray(cupy_array)
            self._cudss_state["gpu_zero_copy"] = True
            return cupy_array
        except RuntimeError:
            numpy_array = np.asarray(array)
            cupy_array = cp.asarray(numpy_array)
            if fortran:
                cupy_array = cp.asfortranarray(cupy_array)
            self._cudss_state["gpu_zero_copy"] = False
            return cupy_array

    def _torch_from_cupy(self, values, *, dtype=FLOAT_DTYPE):
        cp, _ = self._cupy_sparse_modules()
        values = cp.ascontiguousarray(values)
        if self._cudss_state.get("gpu_zero_copy", False):
            return torch_runtime.dlpack.from_dlpack(values).astype(dtype).copy()
        return torch_np.asarray(cp.asnumpy(values), dtype=dtype)

    def _torch_batch_rhs_from_cupy(self, values, *, dtype=FLOAT_DTYPE):
        cp, _ = self._cupy_sparse_modules()
        transposed = cp.ascontiguousarray(values.transpose((0, 2, 1)))
        if self._cudss_state.get("gpu_zero_copy", False):
            return torch_runtime.dlpack.from_dlpack(transposed).astype(dtype).copy()
        return torch_np.asarray(cp.asnumpy(transposed), dtype=dtype)

    def _terrain_cache_key(
        self,
        cache_name: str,
        discretization: AuxiliaryDiscretization,
        *extra_arrays: Array,
    ) -> str | None:
        if self.terrain_cache_dir is None:
            return None
        digest = hashlib.sha256()
        _update_digest_value(digest, "version", _TERRAIN_AUXILIARY_CACHE_VERSION)
        _update_digest_value(digest, "cache_name", cache_name)
        _update_digest_value(digest, "torch_enable_float64", bool(torch_runtime.config.torch_enable_float64))
        _update_digest_value(digest, "float_dtype", str(NP_FLOAT_DTYPE))
        _update_digest_value(digest, "auxiliary_dtype", str(_numpy_dtype(self._auxiliary_float_dtype())))
        _update_digest_value(digest, "numerical_h2_refined", self.numerical_h2_refined)
        _update_digest_value(digest, "numerical_p2_refined", self.numerical_p2_refined)
        _update_digest_value(digest, "topographic_geometric_factor_mode", self.topographic_geometric_factor_mode)
        _update_digest_array(digest, "mesh.nodes", self.mesh.nodes)
        _update_digest_array(digest, "mesh.cells", self.mesh.cells)
        _update_digest_array(digest, "mesh.surface_node_ids", self.mesh.surface_node_ids)
        _update_digest_array(digest, "survey.electrode_positions", self.survey.electrode_positions)
        _update_digest_array(digest, "survey.measurements", self.survey.measurements)
        _update_digest_array(digest, "wavenumbers", self.wavenumbers)
        _update_digest_array(digest, "weights", self.weights)
        _update_digest_array(digest, "discretization.parent_cell_ids", discretization.parent_cell_ids)
        _update_digest_array(digest, "discretization.dof_nodes", discretization.dof_nodes)
        _update_digest_array(digest, "discretization.cell_connectivity", discretization.cell_connectivity)
        _update_digest_array(digest, "discretization.boundary_connectivity", discretization.boundary_connectivity)
        _update_digest_array(digest, "discretization.source_matrix", discretization.source_matrix)
        _update_digest_array(digest, "discretization.electrode_matrix", discretization.electrode_matrix)
        _update_digest_array(digest, "discretization.original_node_matrix", discretization.original_node_matrix)
        _update_digest_array(digest, "discretization.operator_pattern.csr_indptr", discretization.operator_pattern.csr_indptr)
        _update_digest_array(digest, "discretization.operator_pattern.csr_indices", discretization.operator_pattern.csr_indices)
        _update_digest_array(digest, "discretization.operator_templates.stiffness", discretization.operator_templates.stiffness)
        _update_digest_array(digest, "discretization.operator_templates.mass", discretization.operator_templates.mass)
        _update_digest_array(
            digest,
            "discretization.operator_templates.boundary_mass",
            discretization.operator_templates.boundary_mass,
        )
        _update_digest_array(digest, "discretization.boundary_geometries", discretization.boundary_geometries)
        if self.primary_auxiliary_discretization is not None:
            _update_digest_array(
                digest,
                "primary_auxiliary.dof_nodes",
                self.primary_auxiliary_discretization.dof_nodes,
            )
        for index, values in enumerate(extra_arrays):
            _update_digest_array(digest, f"extra.{index}", values)
        return digest.hexdigest()

    def _terrain_cache_path(self, cache_key: str | None) -> Path | None:
        if self.terrain_cache_dir is None or cache_key is None:
            return None
        return self.terrain_cache_dir / f"{cache_key}.npz"

    def _load_terrain_cached_array(self, cache_key: str | None, *, shape: tuple[int, ...], dtype) -> Array | None:
        cache_path = self._terrain_cache_path(cache_key)
        if cache_path is None or not cache_path.exists():
            return None
        expected_dtype = _numpy_dtype(dtype)
        try:
            with np.load(cache_path, allow_pickle=False) as payload:
                cached = np.asarray(payload["value"])
        except Exception:
            return None
        if cached.shape != tuple(shape):
            return None
        if cached.dtype != expected_dtype:
            cached = cached.astype(expected_dtype, copy=False)
        return torch_np.asarray(cached, dtype=dtype)

    def _store_terrain_cached_array(self, cache_key: str | None, values: Array) -> None:
        cache_path = self._terrain_cache_path(cache_key)
        if cache_path is None:
            return
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            array = np.asarray(torch_runtime.block_until_ready(values))
            tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp.npz")
            np.savez(tmp_path, value=array)
            tmp_path.replace(cache_path)
        except Exception:
            return

    def _assemble_discretization_operator_values_batch(
        self,
        discretization: AuxiliaryDiscretization,
        conductivity: Array | float | None = None,
        *,
        cache_key: str,
    ) -> Array:
        kernel = self._kernel_cache.get(cache_key)
        if kernel is None:
            dtype = self._auxiliary_float_dtype()
            stiffness = discretization.operator_templates.stiffness.astype(dtype)
            mass = discretization.operator_templates.mass.astype(dtype)
            boundary_mass = discretization.operator_templates.boundary_mass.astype(dtype)
            parent_cell_ids = discretization.parent_cell_ids
            boundary_parent_cells = parent_cell_ids[discretization.geometry_mesh.boundary_edge_cells]
            boundary_geometries = discretization.boundary_geometries.astype(dtype)
            wavenumber_sq = torch_np.square(self.wavenumbers.astype(dtype)).astype(dtype)
            volume_inverse = discretization.operator_pattern.volume_inverse
            boundary_inverse = discretization.operator_pattern.boundary_inverse
            nnz = discretization.operator_pattern.unique_indices.shape[0]
            wave_count = int(self.wavenumbers.shape[0])

            def kernel(model_conductivity: Array) -> Array:
                model_conductivity = model_conductivity.astype(dtype)
                discretization_conductivity = model_conductivity[parent_cell_ids]
                conductivity_boundary = model_conductivity[boundary_parent_cells]
                volume_values = discretization_conductivity[None, :, None, None] * (
                    stiffness[None, :, :, :] + wavenumber_sq[:, None, None, None] * mass[None, :, :, :]
                )
                boundary_values = (
                    conductivity_boundary[None, :] * boundary_geometries
                )[:, :, None, None] * boundary_mass[None, :, :, :]
                data = torch_np.zeros((wave_count, nnz), dtype=dtype)
                data = data.at[:, volume_inverse].add(volume_values.reshape(wave_count, -1))
                data = data.at[:, boundary_inverse].add(boundary_values.reshape(wave_count, -1))
                return data

            self._kernel_cache[cache_key] = kernel

        if conductivity is None:
            conductivity_array = torch_np.ones((self.mesh.cell_count,), dtype=FLOAT_DTYPE)
        else:
            conductivity_array = _expand_conductivity(conductivity, self.mesh.cell_count)
        return self._kernel_cache[cache_key](conductivity_array)

    def _apply_operator_values_batch_with_pattern(
        self,
        operator_pattern: SparseOperatorPattern,
        values: Array,
        vectors: Array,
        *,
        cache_key: str,
    ) -> Array:
        kernel = self._kernel_cache.get(cache_key)
        if kernel is None:
            shape = operator_pattern.shape
            csr_indices = torch_np.asarray(operator_pattern.csr_indices, dtype=INT_DTYPE)
            csr_indptr = torch_np.asarray(operator_pattern.csr_indptr, dtype=INT_DTYPE)

            def apply_single(args: tuple[Array, Array]) -> Array:
                single_values, single_vectors = args
                operator = CSR((single_values, csr_indices, csr_indptr), shape=shape)
                return (operator @ single_vectors.T).T

            def kernel(values_batch: Array, vectors_batch: Array) -> Array:
                return torch_runtime.map(apply_single, (values_batch, vectors_batch))

            self._kernel_cache[cache_key] = kernel

        return self._kernel_cache[cache_key](values, vectors)

    def _solve_cudss_batch_with_pattern(
        self,
        operator_pattern: SparseOperatorPattern,
        operator_values: Array,
        rhs: Array,
        *,
        state_prefix: str,
        refactorize: bool = True,
    ) -> Array:
        """Solve a batched family of sparse symmetric systems for an arbitrary operator pattern."""

        try:
            from nvmath.sparse.advanced import (
                DirectSolver,
                DirectSolverAlgType,
                DirectSolverMatrixType,
                DirectSolverOptions,
            )
        except ImportError as exc:
            raise ImportError("ERTForward2p5D requires nvmath-python to use the cuDSS backend") from exc

        cp, cupy_sparse = self._cupy_sparse_modules()
        solve_dtype = self._auxiliary_float_dtype()
        fixed = self._cupy_csr_fixed_for(operator_pattern, f"{state_prefix}_gpu_fixed")
        values_cp = self._cupy_from_torch(operator_values, dtype=solve_dtype)
        rhs_raw_cp = self._cupy_from_torch(rhs, dtype=solve_dtype)

        matrix_data_key = f"{state_prefix}_matrix_data_batch_gpu"
        matrix_data = self._cudss_state.get(matrix_data_key)
        if matrix_data is None:
            matrix_data = [
                cp.array(values_cp[wavenumber_index], copy=True)
                for wavenumber_index in range(operator_values.shape[0])
            ]
            self._cudss_state[matrix_data_key] = matrix_data
        else:
            for wavenumber_index, data in enumerate(matrix_data):
                data[...] = values_cp[wavenumber_index]

        matrices_key = f"{state_prefix}_matrices_batch_gpu"
        matrices = self._cudss_state.get(matrices_key)
        if matrices is None:
            matrices = [
                cupy_sparse.csr_matrix((matrix_data[wavenumber_index], fixed[1], fixed[0]), shape=operator_pattern.shape)
                for wavenumber_index in range(operator_values.shape[0])
            ]
            self._cudss_state[matrices_key] = matrices

        rhs_batch_base_key = f"{state_prefix}_rhs_batch_base_gpu"
        rhs_batch_key = f"{state_prefix}_rhs_batch_gpu"
        rhs_batch_base_gpu = self._cudss_state.get(rhs_batch_base_key)
        rhs_batch_gpu = self._cudss_state.get(rhs_batch_key)
        if rhs_batch_base_gpu is None or rhs_batch_gpu is None:
            rhs_batch_base_gpu = rhs_raw_cp.copy()
            rhs_batch_gpu = rhs_batch_base_gpu.transpose((0, 2, 1))
            self._cudss_state[rhs_batch_base_key] = rhs_batch_base_gpu
            self._cudss_state[rhs_batch_key] = rhs_batch_gpu
        else:
            rhs_batch_base_gpu[...] = rhs_raw_cp

        solver_key = f"{state_prefix}_solver_batch_gpu"
        solver = self._cudss_state.get(solver_key)
        factorize = refactorize
        if solver is None:
            options = DirectSolverOptions(
                sparse_system_type=DirectSolverMatrixType.SYMMETRIC,
                logger=_CUDSS_LOGGER,
                blocking=True,
            )
            solver = DirectSolver(matrices, rhs_batch_gpu, options=options)
            solver.plan_config.algorithm = DirectSolverAlgType.ALG_1
            solver.plan()
            self._cudss_state[solver_key] = solver
            factorize = True
        else:
            # Keep the planned matrix operands stable: nvmath invalidates the
            # plan when `a` is reset. The matrix objects are reused and their
            # data buffers were updated above, so only RHS needs resetting.
            solver.reset_operands(b=rhs_batch_gpu)

        # Reuse only symbolic/plan/buffer state. The numeric factors depend on
        # conductivity through matrix.data, so every changed model must
        # factorize again before solve. For repeated RHS blocks of the same
        # Jacobian, callers may set refactorize=False after the first block.
        if factorize:
            solver.factorize()
        solution = solver.solve()
        return self._torch_batch_rhs_from_cupy(solution, dtype=solve_dtype)

    def _solve_scipy_batch_with_pattern(
        self,
        operator_pattern: SparseOperatorPattern,
        operator_values: Array,
        rhs: Array,
        *,
        state_prefix: str,
        refactorize: bool = True,
    ) -> Array:
        """Solve batched sparse systems with SciPy for CPU-only environments."""

        del state_prefix, refactorize
        values_np = np.asarray(torch_runtime.device_get(operator_values), dtype=NP_FLOAT_DTYPE)
        rhs_np = np.asarray(torch_runtime.device_get(rhs), dtype=NP_FLOAT_DTYPE)
        if values_np.ndim == 1:
            values_np = values_np[None, :]
        if rhs_np.ndim == 2:
            rhs_np = rhs_np[None, :, :]
        indices = np.asarray(operator_pattern.csr_indices, dtype=np.int32)
        indptr = np.asarray(operator_pattern.csr_indptr, dtype=np.int32)
        solutions: list[np.ndarray] = []
        for batch_index in range(values_np.shape[0]):
            matrix = sp.csr_matrix(
                (values_np[batch_index], indices, indptr),
                shape=operator_pattern.shape,
            ).tocsc()
            factor = spla.splu(matrix)
            solution = factor.solve(np.asfortranarray(rhs_np[batch_index].T))
            if solution.ndim == 1:
                solution = solution[:, None]
            solutions.append(np.asarray(solution.T, dtype=NP_FLOAT_DTYPE))
        return torch_np.asarray(np.stack(solutions, axis=0), dtype=FLOAT_DTYPE)

    def _solve_batch_with_pattern(
        self,
        operator_pattern: SparseOperatorPattern,
        operator_values: Array,
        rhs: Array,
        *,
        state_prefix: str,
        refactorize: bool = True,
    ) -> Array:
        if self.linear_solver_backend == "scipy":
            return self._solve_scipy_batch_with_pattern(
                operator_pattern,
                operator_values,
                rhs,
                state_prefix=state_prefix,
                refactorize=refactorize,
            )
        return self._solve_cudss_batch_with_pattern(
            operator_pattern,
            operator_values,
            rhs,
            state_prefix=state_prefix,
            refactorize=refactorize,
        )

    def _assemble_operator_values(self, conductivity: Array, wavenumber_index: int) -> Array:
        kernel = self._assemble_operator_values_kernel()
        return kernel(
            conductivity,
            torch_np.asarray(self.wavenumbers[wavenumber_index] ** 2, dtype=FLOAT_DTYPE),
            self.boundary_geometries[wavenumber_index],
        )

    def _assemble_operator_values_batch(self, conductivity: Array) -> Array:
        kernel = self._assemble_operator_values_batch_kernel()
        return kernel(conductivity)

    def _apply_operator_values(self, values: Array, vectors: Array) -> Array:
        kernel = self._apply_operator_values_kernel()
        return kernel(values, vectors)

    def _apply_operator_values_batch(self, values: Array, vectors: Array) -> Array:
        kernel = self._apply_operator_values_batch_kernel()
        return kernel(values, vectors)

    def _measurement_receiver_matrix_for(self, electrode_matrix: Array, *, cache_key: str) -> Array:
        cached = self._derived_cache.get(cache_key)
        if cached is not None:
            return cached

        measurements = self.survey.measurements
        cached = electrode_matrix[measurements[:, 2]] - electrode_matrix[measurements[:, 3]]
        self._derived_cache[cache_key] = cached
        return cached

    def _current_receiver_matrix_for(self, electrode_matrix: Array, *, cache_key: str) -> Array:
        cached = self._derived_cache.get(cache_key)
        if cached is not None:
            return cached

        measurements = self.survey.measurements
        cached = electrode_matrix[measurements[:, 0]] - electrode_matrix[measurements[:, 1]]
        self._derived_cache[cache_key] = cached
        return cached

    def _measurement_receiver_matrix(self) -> Array:
        cached = self._measurement_receiver_matrix_for(
            self.electrode_matrix,
            cache_key="measurement_receiver_matrix",
        )
        return cached

    def _current_receiver_matrix(self) -> Array:
        cached = self._current_receiver_matrix_for(
            self.electrode_matrix,
            cache_key="current_receiver_matrix",
        )
        return cached

    def _integrate_potentials(self, sub_potentials: Array) -> Array:
        return torch_np.tensordot(self.weights.astype(sub_potentials.dtype), sub_potentials, axes=(0, 0))

    def _project_from_integrated(self, integrated_potentials: Array, projection_matrix: Array) -> Array:
        """Project integrated potentials onto electrodes/receivers with dtype alignment."""

        matrix = projection_matrix
        if getattr(matrix, "dtype", None) != getattr(integrated_potentials, "dtype", None):
            matrix = matrix.astype(integrated_potentials.dtype)
        return integrated_potentials @ matrix.T

    def _apply_source_difference(
        self,
        source_potentials: Array,
        positive_sources: Array,
        negative_sources: Array,
    ) -> Array:
        measurement_ids = torch_np.arange(self.survey.measurement_count, dtype=INT_DTYPE)
        safe_positive = torch_np.maximum(positive_sources, 0)
        safe_negative = torch_np.maximum(negative_sources, 0)
        positive = torch_np.where(
            positive_sources >= 0,
            source_potentials[safe_positive, measurement_ids],
            0.0,
        )
        negative = torch_np.where(
            negative_sources >= 0,
            source_potentials[safe_negative, measurement_ids],
            0.0,
        )
        return positive - negative

    def _apply_measurement_map_from_integrated_with_receiver_and_sources(
        self,
        integrated_potentials: Array,
        receiver_matrix: Array,
        positive_sources: Array,
        negative_sources: Array,
    ) -> Array:
        receiver_potentials = self._project_from_integrated(integrated_potentials, receiver_matrix)
        return self._apply_source_difference(receiver_potentials, positive_sources, negative_sources)

    def _apply_measurement_map_from_integrated_with_receiver(
        self,
        integrated_potentials: Array,
        receiver_matrix: Array,
    ) -> Array:
        measurements = self.survey.measurements
        return self._apply_measurement_map_from_integrated_with_receiver_and_sources(
            integrated_potentials,
            receiver_matrix,
            measurements[:, 0],
            measurements[:, 1],
        )

    def _apply_reciprocal_measurement_map_from_integrated_with_receiver(
        self,
        integrated_potentials: Array,
        receiver_matrix: Array,
    ) -> Array:
        measurements = self.survey.measurements
        return self._apply_measurement_map_from_integrated_with_receiver_and_sources(
            integrated_potentials,
            receiver_matrix,
            measurements[:, 2],
            measurements[:, 3],
        )

    def _apply_measurement_map_from_integrated(self, integrated_potentials: Array) -> Array:
        return self._apply_measurement_map_from_integrated_with_receiver(
            integrated_potentials,
            self._measurement_receiver_matrix(),
        )

    def _apply_reciprocal_measurement_map_from_integrated(self, integrated_potentials: Array) -> Array:
        return self._apply_reciprocal_measurement_map_from_integrated_with_receiver(
            integrated_potentials,
            self._current_receiver_matrix(),
        )

    def _apply_measurement_map(self, phi_stack: Array) -> Array:
        return self._apply_measurement_map_from_integrated(self._integrate_potentials(phi_stack))

    def _combine_reciprocal_resistances(self, normal: Array, reciprocal: Array) -> Array:
        return torch_np.sqrt(torch_np.abs(normal * reciprocal))

    def _combine_reciprocal_resistance_jvp(
        self,
        normal: Array,
        reciprocal: Array,
        delta_normal: Array,
        delta_reciprocal: Array,
    ) -> Array:
        product = normal * reciprocal
        combined = self._combine_reciprocal_resistances(normal, reciprocal)
        safe_combined = torch_np.maximum(combined, torch_np.asarray(1e-30, dtype=combined.dtype))
        return 0.5 * torch_np.sign(product) * (delta_normal * reciprocal + normal * delta_reciprocal) / safe_combined

    def _apply_measurement_map_transpose_with_receiver_and_sources(
        self,
        cotangent: Array | float,
        receiver_matrix: Array,
        *,
        node_count: int,
        positive_sources: Array,
        negative_sources: Array,
    ) -> Array:
        rhs_dtype = receiver_matrix.dtype
        cotangent_array = _expand_measurement_vector(
            cotangent,
            self.survey.measurement_count,
            name="cotangent",
        ).astype(rhs_dtype)
        source_rhs = torch_np.zeros((self.survey.electrode_count, node_count), dtype=rhs_dtype)
        weighted_receivers = cotangent_array[:, None] * receiver_matrix
        positive_mask = positive_sources >= 0
        negative_mask = negative_sources >= 0
        source_rhs = source_rhs.at[torch_np.maximum(positive_sources, 0)].add(weighted_receivers * positive_mask[:, None])
        source_rhs = source_rhs.at[torch_np.maximum(negative_sources, 0)].add(-weighted_receivers * negative_mask[:, None])
        return self.weights.astype(rhs_dtype)[:, None, None] * source_rhs[None, :, :]

    def _apply_measurement_map_transpose_batch_with_receiver_and_sources(
        self,
        cotangent: Array | float,
        receiver_matrix: Array,
        *,
        node_count: int,
        positive_sources: Array,
        negative_sources: Array,
    ) -> Array:
        rhs_dtype = receiver_matrix.dtype
        cotangent_matrix = _expand_measurement_matrix(
            cotangent,
            self.survey.measurement_count,
            name="cotangent",
        ).astype(rhs_dtype)
        batch_size = int(cotangent_matrix.shape[0])
        source_count = int(self.survey.electrode_count)
        source_rhs = torch_np.zeros((batch_size, source_count, node_count), dtype=rhs_dtype)
        weighted_receivers = cotangent_matrix[:, :, None] * receiver_matrix[None, :, :]
        positive_mask = positive_sources >= 0
        negative_mask = negative_sources >= 0
        source_rhs = source_rhs.at[:, torch_np.maximum(positive_sources, 0), :].add(
            weighted_receivers * positive_mask[None, :, None]
        )
        source_rhs = source_rhs.at[:, torch_np.maximum(negative_sources, 0), :].add(
            -weighted_receivers * negative_mask[None, :, None]
        )
        rhs = self.weights.astype(rhs_dtype)[:, None, None, None] * source_rhs[None, :, :, :]
        return rhs.reshape((self.wavenumbers.shape[0], batch_size * source_count, node_count))

    def _apply_measurement_map_transpose_batch_with_receiver(
        self,
        cotangent: Array | float,
        receiver_matrix: Array,
        *,
        node_count: int,
    ) -> Array:
        measurements = self.survey.measurements
        return self._apply_measurement_map_transpose_batch_with_receiver_and_sources(
            cotangent,
            receiver_matrix,
            node_count=node_count,
            positive_sources=measurements[:, 0],
            negative_sources=measurements[:, 1],
        )

    def _apply_reciprocal_measurement_map_transpose_batch_with_receiver(
        self,
        cotangent: Array | float,
        receiver_matrix: Array,
        *,
        node_count: int,
    ) -> Array:
        measurements = self.survey.measurements
        return self._apply_measurement_map_transpose_batch_with_receiver_and_sources(
            cotangent,
            receiver_matrix,
            node_count=node_count,
            positive_sources=measurements[:, 2],
            negative_sources=measurements[:, 3],
        )

    def _apply_measurement_map_transpose_with_receiver(
        self,
        cotangent: Array | float,
        receiver_matrix: Array,
        *,
        node_count: int,
    ) -> Array:
        measurements = self.survey.measurements
        return self._apply_measurement_map_transpose_with_receiver_and_sources(
            cotangent,
            receiver_matrix,
            node_count=node_count,
            positive_sources=measurements[:, 0],
            negative_sources=measurements[:, 1],
        )

    def _apply_reciprocal_measurement_map_transpose_with_receiver(
        self,
        cotangent: Array | float,
        receiver_matrix: Array,
        *,
        node_count: int,
    ) -> Array:
        measurements = self.survey.measurements
        return self._apply_measurement_map_transpose_with_receiver_and_sources(
            cotangent,
            receiver_matrix,
            node_count=node_count,
            positive_sources=measurements[:, 2],
            negative_sources=measurements[:, 3],
        )

    def _apply_measurement_map_transpose(self, cotangent: Array | float) -> Array:
        return self._apply_measurement_map_transpose_with_receiver(
            cotangent,
            self._measurement_receiver_matrix(),
            node_count=self.mesh.node_count,
        )

    def _apply_reciprocal_measurement_map_transpose(self, cotangent: Array | float) -> Array:
        return self._apply_reciprocal_measurement_map_transpose_with_receiver(
            cotangent,
            self._current_receiver_matrix(),
            node_count=self.mesh.node_count,
        )

    def _source_resistivities_kernel(self):
        kernel = self._kernel_cache.get("source_resistivities")
        if kernel is not None:
            return kernel

        node_cell_weights = self.source_resistivity_data.node_cell_weights
        node_cell_counts = self.source_resistivity_data.node_cell_counts
        source_node_ids = self.source_node_ids
        source_cell_ids = self.source_cell_ids

        def kernel(conductivity: Array) -> Array:
            log_resistivity = -torch_np.log(conductivity)
            node_log_rho = torch_np.einsum("ec,c->e", node_cell_weights, log_resistivity) / node_cell_counts
            node_rho = torch_np.exp(node_log_rho)
            entity_rho = 1.0 / conductivity[source_cell_ids]
            return torch_np.where(source_node_ids >= 0, node_rho, entity_rho)

        self._kernel_cache["source_resistivities"] = kernel
        return kernel

    def _build_rhs_kernel(self):
        kernel = self._kernel_cache.get("build_rhs")
        if kernel is not None:
            return kernel

        shape = self.operator_pattern.shape
        csr_indices = torch_np.asarray(self.operator_pattern.csr_indices, dtype=INT_DTYPE)
        csr_indptr = torch_np.asarray(self.operator_pattern.csr_indptr, dtype=INT_DTYPE)

        def kernel(operator_values: Array, unit_primary: Array, source_resistivities: Array, reference_rhs: Array):
            operator = CSR((operator_values, csr_indices, csr_indptr), shape=shape)
            primary = unit_primary * source_resistivities[:, None]
            op_primary = (operator @ primary.T).T
            rhs = reference_rhs - op_primary
            return rhs, primary

        self._kernel_cache["build_rhs"] = kernel
        return kernel

    def _build_rhs_batch_kernel(self):
        kernel = self._kernel_cache.get("build_rhs_batch")
        if kernel is not None:
            return kernel

        apply_batch = self._apply_operator_values_batch_kernel()

        def kernel(
            operator_values: Array,
            unit_primary: Array,
            source_resistivities: Array,
            reference_rhs: Array,
        ):
            primary = unit_primary * source_resistivities[None, :, None]
            op_primary = apply_batch(operator_values, primary)
            rhs = reference_rhs - op_primary
            return rhs, primary

        self._kernel_cache["build_rhs_batch"] = kernel
        return kernel

    def _solve_cudss(self, operator_values: Array, rhs: Array) -> Array:
        """Solve one sparse SPD system with NVIDIA cuDSS through nvmath-python."""

        try:
            from nvmath.sparse.advanced import DirectSolver, DirectSolverMatrixType, DirectSolverOptions
        except ImportError as exc:
            raise ImportError("ERTForward2p5D requires nvmath-python to use the cuDSS backend") from exc

        cp, cupy_sparse = self._cupy_sparse_modules()
        fixed = self._cupy_csr_fixed()
        values_cp = self._cupy_from_torch(operator_values)
        rhs_cp = self._cupy_from_torch(rhs, transpose=True, fortran=True)

        matrix_data = self._cudss_state.get("matrix_data_gpu")
        if matrix_data is None:
            matrix_data = cp.array(values_cp, copy=True)
            self._cudss_state["matrix_data_gpu"] = matrix_data
        else:
            matrix_data[...] = values_cp

        csr_operator = self._cudss_state.get("matrix_gpu")
        if csr_operator is None:
            csr_operator = cupy_sparse.csr_matrix(
                (matrix_data, fixed[1], fixed[0]),
                shape=self.operator_pattern.shape,
            )
            self._cudss_state["matrix_gpu"] = csr_operator

        solver = self._cudss_state.get("solver_gpu")
        if solver is None:
            options = DirectSolverOptions(
                sparse_system_type=DirectSolverMatrixType.SPD,
                logger=_CUDSS_LOGGER,
                blocking=True,
            )
            solver = DirectSolver(csr_operator, rhs_cp, options=options)
            solver.plan()
            self._cudss_state["solver_gpu"] = solver
        else:
            solver.reset_operands(b=rhs_cp)

        solver.factorize()
        solution = solver.solve()
        if self._cudss_state.get("gpu_zero_copy", False):
            solution_transposed = cp.ascontiguousarray(solution.T)
            cp.cuda.get_current_stream().synchronize()
            return torch_runtime.dlpack.from_dlpack(solution_transposed).astype(FLOAT_DTYPE).copy()
        return torch_np.asarray(cp.asnumpy(solution.T), dtype=FLOAT_DTYPE)

    def _solve_cudss_batch(self, operator_values: Array, rhs: Array) -> Array:
        """Solve a batched family of sparse SPD systems with NVIDIA cuDSS."""

        try:
            from nvmath.sparse.advanced import DirectSolver, DirectSolverMatrixType, DirectSolverOptions
        except ImportError as exc:
            raise ImportError("ERTForward2p5D requires nvmath-python to use the cuDSS backend") from exc

        cp, cupy_sparse = self._cupy_sparse_modules()
        fixed = self._cupy_csr_fixed()
        values_cp = self._cupy_from_torch(operator_values)
        rhs_raw_cp = self._cupy_from_torch(rhs)

        matrix_data = self._cudss_state.get("matrix_data_batch_gpu")
        if matrix_data is None:
            matrix_data = [
                cp.array(values_cp[wavenumber_index], copy=True)
                for wavenumber_index in range(operator_values.shape[0])
            ]
            self._cudss_state["matrix_data_batch_gpu"] = matrix_data
        else:
            for wavenumber_index, data in enumerate(matrix_data):
                data[...] = values_cp[wavenumber_index]

        matrices = self._cudss_state.get("matrices_batch_gpu")
        if matrices is None:
            matrices = [
                cupy_sparse.csr_matrix((matrix_data[wavenumber_index], fixed[1], fixed[0]), shape=self.operator_pattern.shape)
                for wavenumber_index in range(operator_values.shape[0])
            ]
            self._cudss_state["matrices_batch_gpu"] = matrices

        rhs_batch_base_gpu = self._cudss_state.get("rhs_batch_base_gpu")
        rhs_batch_gpu = self._cudss_state.get("rhs_batch_gpu")
        if rhs_batch_base_gpu is None or rhs_batch_gpu is None:
            rhs_batch_base_gpu = rhs_raw_cp.copy()
            rhs_batch_gpu = rhs_batch_base_gpu.transpose((0, 2, 1))
            self._cudss_state["rhs_batch_base_gpu"] = rhs_batch_base_gpu
            self._cudss_state["rhs_batch_gpu"] = rhs_batch_gpu
        else:
            rhs_batch_base_gpu[...] = rhs_raw_cp

        solver = self._cudss_state.get("solver_batch_gpu")
        if solver is None:
            options = DirectSolverOptions(
                sparse_system_type=DirectSolverMatrixType.SPD,
                logger=_CUDSS_LOGGER,
                blocking=True,
            )
            solver = DirectSolver(matrices, rhs_batch_gpu, options=options)
            solver.plan()
            self._cudss_state["solver_batch_gpu"] = solver
        else:
            solver.reset_operands(b=rhs_batch_gpu)

        solver.factorize()
        solution = solver.solve()
        return self._torch_batch_rhs_from_cupy(solution)

    def _solve_linear_system(self, operator_values: Array, rhs: Array) -> Array:
        if self.linear_solver_backend == "scipy":
            return self._solve_scipy_batch_with_pattern(
                self.operator_pattern,
                operator_values,
                rhs,
                state_prefix="main",
            )[0]
        return self._solve_cudss(operator_values, rhs)

    def _solve_linear_system_batch(self, operator_values: Array, rhs: Array) -> Array:
        if self.linear_solver_backend == "scipy":
            return self._solve_scipy_batch_with_pattern(
                self.operator_pattern,
                operator_values,
                rhs,
                state_prefix="main",
            )
        return self._solve_cudss_batch(operator_values, rhs)

    def _check_finite(self, values: Array, *, context: str, wavenumber: float | None = None) -> None:
        if bool(torch_np.all(torch_np.isfinite(values))):
            return

        details = [f"{context} produced non-finite values"]
        if wavenumber is not None:
            details.append(f"at wavenumber={wavenumber:.6e}")
        raise FloatingPointError(", ".join(details))

    def _electrode_source_resistivities(self, conductivity: Array | float) -> Array:
        conductivity_array = _expand_conductivity(conductivity, self.mesh.cell_count)
        kernel = self._source_resistivities_kernel()
        return kernel(conductivity_array)

    def _auxiliary_reference_rhs_stack(self, discretization: AuxiliaryDiscretization, unit_primary: Array) -> Array:
        cached = self._reference_rhs_cache.get("auxiliary__stack__")
        if cached is not None:
            return cached

        disk_cache_key = self._terrain_cache_key("auxiliary_reference_rhs_stack", discretization, unit_primary)
        cached = self._load_terrain_cached_array(
            disk_cache_key,
            shape=tuple(int(dim) for dim in unit_primary.shape),
            dtype=unit_primary.dtype,
        )
        if cached is not None:
            self._reference_rhs_cache["auxiliary__stack__"] = cached
            return cached

        reference_operator_values = self._assemble_discretization_operator_values_batch(
            discretization,
            cache_key="assemble_primary_auxiliary_reference_operator_values_batch",
        )
        # The auxiliary correction uses S1 * prim / rhoSource - S * prim.
        # Since prim is rhoSource * unit_primary, the fixed part is
        # S1 * unit_primary.
        cached = self._apply_operator_values_batch_with_pattern(
            discretization.operator_pattern,
            reference_operator_values,
            unit_primary,
            cache_key="apply_primary_auxiliary_reference_operator_values_batch",
        )
        self._reference_rhs_cache["auxiliary__stack__"] = cached
        self._store_terrain_cached_array(disk_cache_key, cached)
        return cached

    def _solve_secondary_fields_auxiliary(self, conductivity: Array | float) -> tuple[AuxiliaryDiscretization, Array, Array]:
        if self.primary_auxiliary_discretization is None:
            raise ValueError("primary auxiliary discretization is not available")

        solve_conductivity = _expand_conductivity(conductivity, self.mesh.cell_count)
        discretization = self.primary_auxiliary_discretization
        cached_fields = self._consume_prepared_total_fields(solve_conductivity)
        if cached_fields is not None:
            operator_values, total_fields = cached_fields
            return discretization, operator_values, total_fields

        operator_values = self._assemble_discretization_operator_values_batch(
            discretization,
            solve_conductivity,
            cache_key="assemble_primary_auxiliary_operator_values_batch",
        )
        unit_primary = self._auxiliary_sub_potential_stack()
        source_resistivities = self._electrode_source_resistivities(solve_conductivity)
        primary = unit_primary * source_resistivities[None, :, None]
        op_primary = self._apply_operator_values_batch_with_pattern(
            discretization.operator_pattern,
            operator_values,
            primary,
            cache_key="apply_primary_auxiliary_operator_values_batch",
        )
        rhs = self._auxiliary_reference_rhs_stack(discretization, unit_primary) - op_primary
        secondary_fields = self._solve_batch_with_pattern(
            discretization.operator_pattern,
            operator_values,
            rhs,
            state_prefix="primary_secondary_fields",
        )
        self._check_finite(secondary_fields, context="auxiliary secondary fields")
        total_fields = secondary_fields + primary
        self._check_finite(total_fields, context="auxiliary total fields")
        return discretization, operator_values, total_fields

    def _solve_total_fields(self, conductivity: Array | float) -> tuple[Array, Array]:
        if self.use_numerical_primary:
            _, operator_values, total_fields = self._solve_secondary_fields_auxiliary(conductivity)
            return operator_values, total_fields

        solve_conductivity = _expand_conductivity(conductivity, self.mesh.cell_count)
        cached_fields = self._consume_prepared_total_fields(solve_conductivity)
        if cached_fields is not None:
            return cached_fields

        operator_values = self._assemble_operator_values_batch(solve_conductivity)
        source_resistivities = self._electrode_source_resistivities(solve_conductivity)
        rhs, primary = self._build_rhs_batch_kernel()(
            operator_values,
            self._unit_primary_stack(),
            source_resistivities,
            self._reference_rhs_stack(),
        )
        total_fields = self._solve_linear_system_batch(operator_values, rhs) + primary
        self._check_finite(total_fields, context="total fields")
        return operator_values, total_fields

    def _operator_tangent_apply(self, delta_conductivity: Array | float, phi_stack: Array) -> Array:
        if self.use_numerical_primary:
            if self.primary_auxiliary_discretization is None:
                raise ValueError("primary auxiliary discretization is not available")
            tangent_values = self._assemble_discretization_operator_values_batch(
                self.primary_auxiliary_discretization,
                delta_conductivity,
                cache_key="assemble_primary_auxiliary_operator_values_batch",
            )
            return self._apply_operator_values_batch_with_pattern(
                self.primary_auxiliary_discretization.operator_pattern,
                tangent_values,
                phi_stack,
                cache_key="apply_primary_auxiliary_operator_values_batch",
            )

        tangent_conductivity = _expand_conductivity(delta_conductivity, self.mesh.cell_count)
        tangent_values = self._assemble_operator_values_batch(tangent_conductivity)
        return self._apply_operator_values_batch(tangent_values, phi_stack)

    def _accumulate_adjoint_cell_gradient(
        self,
        phi_stack: Array,
        lambda_stack: Array,
        *,
        include_robin_boundary_derivative: bool = True,
    ) -> Array:
        if self.use_numerical_primary:
            if self.primary_auxiliary_discretization is None:
                raise ValueError("primary auxiliary discretization is not available")

            discretization = self.primary_auxiliary_discretization
            wavenumber_sq = torch_np.square(self.wavenumbers).astype(FLOAT_DTYPE)
            volume_templates = discretization.operator_templates.stiffness[None, :, :, :] + wavenumber_sq[
                :, None, None, None
            ] * discretization.operator_templates.mass[None, :, :, :]
            phi_local = phi_stack[:, :, discretization.cell_connectivity]
            lambda_local = lambda_stack[:, :, discretization.cell_connectivity]
            auxiliary_gradient = -torch_np.einsum("wsci,wcij,wscj->c", lambda_local, volume_templates, phi_local)

            if include_robin_boundary_derivative:
                boundary_templates = (
                    discretization.boundary_geometries[:, :, None, None]
                    * discretization.operator_templates.boundary_mass[None, :, :, :]
                )
                phi_boundary = phi_stack[:, :, discretization.boundary_connectivity]
                lambda_boundary = lambda_stack[:, :, discretization.boundary_connectivity]
                boundary_gradient = -torch_np.einsum("wsbi,wbij,wsbj->b", lambda_boundary, boundary_templates, phi_boundary)
                auxiliary_gradient = auxiliary_gradient.at[discretization.geometry_mesh.boundary_edge_cells].add(
                    boundary_gradient
                )

            gradient = torch_np.zeros((self.mesh.cell_count,), dtype=auxiliary_gradient.dtype)
            return gradient.at[discretization.parent_cell_ids].add(auxiliary_gradient)

        wavenumber_sq = torch_np.square(self.wavenumbers).astype(FLOAT_DTYPE)
        volume_templates = self.operator_templates.stiffness[None, :, :, :] + wavenumber_sq[:, None, None, None] * (
            self.operator_templates.mass[None, :, :, :]
        )
        phi_local = phi_stack[:, :, self.mesh.cells]
        lambda_local = lambda_stack[:, :, self.mesh.cells]
        gradient = -torch_np.einsum("wsci,wcij,wscj->c", lambda_local, volume_templates, phi_local)

        if not include_robin_boundary_derivative:
            return gradient

        boundary_templates = self.boundary_geometries[:, :, None, None] * self.operator_templates.boundary_mass[None, :, :, :]
        phi_boundary = phi_stack[:, :, self.mesh.boundary_edges]
        lambda_boundary = lambda_stack[:, :, self.mesh.boundary_edges]
        boundary_gradient = -torch_np.einsum("wsbi,wbij,wsbj->b", lambda_boundary, boundary_templates, phi_boundary)
        return gradient.at[self.mesh.boundary_edge_cells].add(boundary_gradient)

    def _accumulate_adjoint_cell_gradient_batch(
        self,
        phi_stack: Array,
        lambda_stack: Array,
        *,
        include_robin_boundary_derivative: bool = True,
    ) -> Array:
        """Accumulate one cell-gradient row per batched adjoint cotangent."""

        if self.use_numerical_primary:
            if self.primary_auxiliary_discretization is None:
                raise ValueError("primary auxiliary discretization is not available")

            discretization = self.primary_auxiliary_discretization
            wavenumber_sq = torch_np.square(self.wavenumbers).astype(FLOAT_DTYPE)
            volume_templates = discretization.operator_templates.stiffness[None, :, :, :] + wavenumber_sq[
                :, None, None, None
            ] * discretization.operator_templates.mass[None, :, :, :]
            phi_local = phi_stack[:, :, discretization.cell_connectivity]
            lambda_local = lambda_stack[:, :, :, discretization.cell_connectivity]
            auxiliary_gradient = -torch_np.einsum("wqeci,wcij,wecj->qc", lambda_local, volume_templates, phi_local)

            if include_robin_boundary_derivative:
                boundary_templates = (
                    discretization.boundary_geometries[:, :, None, None]
                    * discretization.operator_templates.boundary_mass[None, :, :, :]
                )
                phi_boundary = phi_stack[:, :, discretization.boundary_connectivity]
                lambda_boundary = lambda_stack[:, :, :, discretization.boundary_connectivity]
                boundary_gradient = -torch_np.einsum("wqeri,wrij,werj->qr", lambda_boundary, boundary_templates, phi_boundary)
                auxiliary_gradient = auxiliary_gradient.at[:, discretization.geometry_mesh.boundary_edge_cells].add(
                    boundary_gradient
                )

            gradient = torch_np.zeros((lambda_stack.shape[1], self.mesh.cell_count), dtype=auxiliary_gradient.dtype)
            return gradient.at[:, discretization.parent_cell_ids].add(auxiliary_gradient)

        wavenumber_sq = torch_np.square(self.wavenumbers).astype(FLOAT_DTYPE)
        volume_templates = self.operator_templates.stiffness[None, :, :, :] + wavenumber_sq[:, None, None, None] * (
            self.operator_templates.mass[None, :, :, :]
        )
        phi_local = phi_stack[:, :, self.mesh.cells]
        lambda_local = lambda_stack[:, :, :, self.mesh.cells]
        gradient = -torch_np.einsum("wqeci,wcij,wecj->qc", lambda_local, volume_templates, phi_local)

        if not include_robin_boundary_derivative:
            return gradient

        boundary_templates = self.boundary_geometries[:, :, None, None] * self.operator_templates.boundary_mass[None, :, :, :]
        phi_boundary = phi_stack[:, :, self.mesh.boundary_edges]
        lambda_boundary = lambda_stack[:, :, :, self.mesh.boundary_edges]
        boundary_gradient = -torch_np.einsum("wqeri,wrij,werj->qr", lambda_boundary, boundary_templates, phi_boundary)
        return gradient.at[:, self.mesh.boundary_edge_cells].add(boundary_gradient)

    def _normal_sensitivity_direct_kernel(
        self,
        *,
        cache_key: str,
        cell_connectivity: Array,
        parent_cell_ids: Array,
        parameter_count: int,
        volume_templates: Array,
    ):
        kernel = self._kernel_cache.get(cache_key)
        if kernel is not None:
            return kernel

        cell_connectivity = torch_np.asarray(cell_connectivity, dtype=INT_DTYPE)
        parent_cell_ids = torch_np.asarray(parent_cell_ids, dtype=INT_DTYPE)
        weighted_templates = (self.weights.astype(volume_templates.dtype)[:, None, None, None] * volume_templates).astype(
            volume_templates.dtype
        )
        cell_count = int(cell_connectivity.shape[0])
        nodes_per_cell = int(cell_connectivity.shape[1])
        tensor_cache: dict[str, tuple[Array, Array, Array]] = {}
        cupy_tensor_cache: dict[str, tuple[object, object, object]] = {}
        cupy_kernel_cache: dict[str, object] = {}
        cupy_disabled = False

        def cached_tensors(device: torch.device) -> tuple[Array, Array, Array]:
            key = str(device)
            cached = tensor_cache.get(key)
            if cached is None:
                cached = (
                    cell_connectivity.reshape(-1).to(device=device, dtype=torch.long),
                    parent_cell_ids.to(device=device, dtype=torch.long),
                    weighted_templates.to(device=device),
                )
                tensor_cache[key] = cached
            return cached

        def torch_to_cupy(array: Array):
            cp, _ = self._cupy_sparse_modules()
            contiguous = array.contiguous()
            if contiguous.is_cuda:
                return cp.from_dlpack(contiguous.__dlpack__())
            return cp.asarray(np.asarray(contiguous))

        def cached_cupy_tensors(dtype: torch.dtype):
            cp, _ = self._cupy_sparse_modules()
            key = str(dtype)
            cached = cupy_tensor_cache.get(key)
            if cached is None:
                cp_dtype = cp.float64 if dtype == torch.float64 else cp.float32
                cached = (
                    cp.asarray(np.asarray(cell_connectivity.reshape(-1), dtype=np.int32)),
                    cp.asarray(np.asarray(parent_cell_ids, dtype=np.int32)),
                    cp.asarray(
                        np.asarray(
                            weighted_templates,
                            dtype=np.float64 if dtype == torch.float64 else np.float32,
                        ),
                        dtype=cp_dtype,
                    ),
                )
                cupy_tensor_cache[key] = cached
            return cached

        def cupy_raw_kernel(dtype: torch.dtype):
            cp, _ = self._cupy_sparse_modules()
            key = "f64" if dtype == torch.float64 else "f32"
            kernel = cupy_kernel_cache.get(key)
            if kernel is not None:
                return kernel
            scalar_type = "double" if dtype == torch.float64 else "float"
            kernel_name = f"normal_sensitivity_{key}"
            source = f"""
            extern "C" __global__
            void {kernel_name}(
                const {scalar_type}* __restrict__ phi,
                const int* __restrict__ current_positive,
                const int* __restrict__ current_negative,
                const int* __restrict__ receiver_positive,
                const int* __restrict__ receiver_negative,
                const int* __restrict__ cell_connectivity,
                const int* __restrict__ parent_cell_ids,
                const {scalar_type}* __restrict__ weighted_templates,
                {scalar_type}* __restrict__ gradient,
                const int wavenumber_count,
                const int source_count,
                const int node_count,
                const int measurement_count,
                const int cell_count,
                const int parameter_count
            ) {{
                const int linear_id = blockIdx.x * blockDim.x + threadIdx.x;
                const int total = measurement_count * cell_count;
                if (linear_id >= total) {{
                    return;
                }}

                const int cell_id = linear_id % cell_count;
                const int measurement_id = linear_id / cell_count;
                const int parameter_id = parent_cell_ids[cell_id];
                if (parameter_id < 0 || parameter_id >= parameter_count) {{
                    return;
                }}

                const int current_p = current_positive[measurement_id];
                const int current_n = current_negative[measurement_id];
                const int receiver_p = receiver_positive[measurement_id];
                const int receiver_n = receiver_negative[measurement_id];
                const int n0 = cell_connectivity[cell_id * 3 + 0];
                const int n1 = cell_connectivity[cell_id * 3 + 1];
                const int n2 = cell_connectivity[cell_id * 3 + 2];
                const int nodes[3] = {{n0, n1, n2}};

                {scalar_type} total_value = ({scalar_type})0;
                for (int wavenumber_id = 0; wavenumber_id < wavenumber_count; ++wavenumber_id) {{
                    const long long phi_base = (long long)wavenumber_id * source_count * node_count;
                    const long long template_base = ((long long)wavenumber_id * cell_count + cell_id) * 9;
                    {scalar_type} current_values[3];
                    {scalar_type} receiver_values[3];
                    for (int local_id = 0; local_id < 3; ++local_id) {{
                        const int node_id = nodes[local_id];
                        const {scalar_type} current_pos_value =
                            current_p >= 0 ? phi[phi_base + (long long)current_p * node_count + node_id] : ({scalar_type})0;
                        const {scalar_type} current_neg_value =
                            current_n >= 0 ? phi[phi_base + (long long)current_n * node_count + node_id] : ({scalar_type})0;
                        const {scalar_type} receiver_pos_value =
                            receiver_p >= 0 ? phi[phi_base + (long long)receiver_p * node_count + node_id] : ({scalar_type})0;
                        const {scalar_type} receiver_neg_value =
                            receiver_n >= 0 ? phi[phi_base + (long long)receiver_n * node_count + node_id] : ({scalar_type})0;
                        current_values[local_id] = current_pos_value - current_neg_value;
                        receiver_values[local_id] = receiver_pos_value - receiver_neg_value;
                    }}
                    for (int i = 0; i < 3; ++i) {{
                        for (int j = 0; j < 3; ++j) {{
                            total_value += receiver_values[i] * weighted_templates[template_base + i * 3 + j] * current_values[j];
                        }}
                    }}
                }}

                atomicAdd(gradient + (long long)measurement_id * parameter_count + parameter_id, -total_value);
            }}
            """
            kernel = cp.RawKernel(source, kernel_name)
            cupy_kernel_cache[key] = kernel
            return kernel

        def compute_cupy(
            phi_stack: Array,
            current_positive: Array,
            current_negative: Array,
            receiver_positive: Array,
            receiver_negative: Array,
        ) -> Array | None:
            # Avoid Torch eager materializing W x B x C x 3 field blocks for the common triangular-cell path.
            nonlocal cupy_disabled
            if cupy_disabled or nodes_per_cell != 3 or not torch.cuda.is_available():
                return None
            try:
                cp, _ = self._cupy_sparse_modules()
                solve_dtype = phi_stack.dtype
                if solve_dtype not in (torch.float32, torch.float64):
                    return None
                compute_device = phi_stack.device if phi_stack.is_cuda else torch.device("cuda")
                phi_work = phi_stack.to(device=compute_device, dtype=solve_dtype).contiguous()
                current_positive_work = current_positive.to(device=compute_device, dtype=torch.int32).contiguous()
                current_negative_work = current_negative.to(device=compute_device, dtype=torch.int32).contiguous()
                receiver_positive_work = receiver_positive.to(device=compute_device, dtype=torch.int32).contiguous()
                receiver_negative_work = receiver_negative.to(device=compute_device, dtype=torch.int32).contiguous()
                phi_cp = torch_to_cupy(phi_work)
                current_positive_cp = torch_to_cupy(current_positive_work)
                current_negative_cp = torch_to_cupy(current_negative_work)
                receiver_positive_cp = torch_to_cupy(receiver_positive_work)
                receiver_negative_cp = torch_to_cupy(receiver_negative_work)
                cell_connectivity_cp, parent_cell_ids_cp, weighted_templates_cp = cached_cupy_tensors(solve_dtype)
                cp_dtype = cp.float64 if solve_dtype == torch.float64 else cp.float32
                gradient_cp = cp.zeros((int(current_positive.shape[0]), parameter_count), dtype=cp_dtype)
                kernel = cupy_raw_kernel(solve_dtype)
                total_threads = int(current_positive.shape[0]) * cell_count
                threads_per_block = 256
                blocks = (total_threads + threads_per_block - 1) // threads_per_block
                kernel(
                    (blocks,),
                    (threads_per_block,),
                    (
                        phi_cp,
                        current_positive_cp,
                        current_negative_cp,
                        receiver_positive_cp,
                        receiver_negative_cp,
                        cell_connectivity_cp,
                        parent_cell_ids_cp,
                        weighted_templates_cp,
                        gradient_cp,
                        int(phi_stack.shape[0]),
                        int(phi_stack.shape[1]),
                        int(phi_stack.shape[2]),
                        int(current_positive.shape[0]),
                        cell_count,
                        parameter_count,
                    ),
                )
                if phi_stack.is_cuda:
                    return torch_runtime.dlpack.from_dlpack(gradient_cp).to(dtype=solve_dtype).clone()
                return torch_np.asarray(cp.asnumpy(gradient_cp), dtype=solve_dtype)
            except Exception:
                cupy_disabled = True
                return None

        def compute(
            phi_stack: Array,
            current_positive: Array,
            current_negative: Array,
            receiver_positive: Array,
            receiver_negative: Array,
            flat_cell_connectivity_work: Array,
            parent_cell_ids_work: Array,
            weighted_templates_work: Array,
        ) -> Array:
            def source_difference(positive: Array, negative: Array) -> Array:
                positive_values = torch.index_select(phi_stack, 1, positive.to(device=phi_stack.device, dtype=torch.long))
                negative_values = torch.index_select(phi_stack, 1, negative.to(device=phi_stack.device, dtype=torch.long))
                return positive_values - negative_values

            current_fields = source_difference(current_positive, current_negative)
            receiver_fields = source_difference(receiver_positive, receiver_negative)
            local_shape = (int(phi_stack.shape[0]), int(current_fields.shape[1]), cell_count, nodes_per_cell)
            current_local = torch.index_select(current_fields, 2, flat_cell_connectivity_work).reshape(local_shape)
            receiver_local = torch.index_select(receiver_fields, 2, flat_cell_connectivity_work).reshape(local_shape)
            auxiliary_gradient = -torch_np.einsum(
                "wbci,wcij,wbcj->bc",
                receiver_local,
                weighted_templates_work,
                current_local,
            )
            gradient = torch.zeros(
                (current_positive.shape[0], parameter_count),
                dtype=auxiliary_gradient.dtype,
                device=phi_stack.device,
            )
            return gradient.index_add_(1, parent_cell_ids_work, auxiliary_gradient)

        def kernel(
            phi_stack: Array,
            current_positive: Array,
            current_negative: Array,
            receiver_positive: Array,
            receiver_negative: Array,
        ) -> Array:
            cupy_result = compute_cupy(
                phi_stack,
                current_positive,
                current_negative,
                receiver_positive,
                receiver_negative,
            )
            if cupy_result is not None:
                return cupy_result
            compute_device = phi_stack.device
            if not phi_stack.is_cuda and torch.cuda.is_available():
                compute_device = torch.device("cuda")
                phi_stack_work = phi_stack.to(device=compute_device)
            else:
                phi_stack_work = phi_stack
            flat_cell_connectivity_work, parent_cell_ids_work, weighted_templates_work = cached_tensors(compute_device)
            result = compute(
                phi_stack_work,
                current_positive.to(device=compute_device),
                current_negative.to(device=compute_device),
                receiver_positive.to(device=compute_device),
                receiver_negative.to(device=compute_device),
                flat_cell_connectivity_work,
                parent_cell_ids_work,
                weighted_templates_work,
            )
            if result.device != phi_stack.device:
                return result.cpu()
            return result

        self._kernel_cache[cache_key] = kernel
        return kernel

    def _normal_sensitivity_from_fields_direct(
        self,
        phi_stack: Array,
        *,
        batch_size: int,
        cell_connectivity: Array,
        parent_cell_ids: Array,
        parameter_count: int,
        volume_templates: Array,
        cache_key: str,
    ) -> Array:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        kernel = self._normal_sensitivity_direct_kernel(
            cache_key=cache_key,
            cell_connectivity=cell_connectivity,
            parent_cell_ids=parent_cell_ids,
            parameter_count=parameter_count,
            volume_templates=volume_templates,
        )
        measurements = np.asarray(self.survey.measurements, dtype=np.int32)
        rows = []
        for start in range(0, int(self.survey.measurement_count), batch_size):
            stop = min(start + batch_size, int(self.survey.measurement_count))
            chunk = measurements[start:stop]
            gradient_rows = kernel(
                phi_stack,
                torch_np.asarray(chunk[:, 0], dtype=INT_DTYPE),
                torch_np.asarray(chunk[:, 1], dtype=INT_DTYPE),
                torch_np.asarray(chunk[:, 2], dtype=INT_DTYPE),
                torch_np.asarray(chunk[:, 3], dtype=INT_DTYPE),
            )
            rows.append(gradient_rows)

        return torch_np.concatenate(rows, axis=0)

    def _direct_sensitivity_inputs(
        self,
        *,
        cell_connectivity: Array,
        parent_cell_ids: Array,
        parameter_count: int,
        volume_templates: Array,
        cache_key: str,
        sensitivity_cell_parameter_ids: Array | None,
        sensitivity_parameter_count: int | None,
    ) -> tuple[Array, Array, int, Array, str]:
        if sensitivity_cell_parameter_ids is None:
            return cell_connectivity, parent_cell_ids, parameter_count, volume_templates, cache_key

        cell_parameter_ids = np.asarray(sensitivity_cell_parameter_ids, dtype=np.int32).ravel()
        cell_count = int(cell_connectivity.shape[0])
        if cell_parameter_ids.shape != (cell_count,):
            raise ValueError(
                "sensitivity_cell_parameter_ids must have one entry per sensitivity cell "
                f"({cell_parameter_ids.shape} != ({cell_count},))"
            )

        active_cell_ids = np.flatnonzero(cell_parameter_ids >= 0).astype(np.int32)
        if active_cell_ids.size == 0:
            raise ValueError("sensitivity_cell_parameter_ids must contain at least one active cell")

        active_parameter_ids = cell_parameter_ids[active_cell_ids]
        resolved_parameter_count = (
            int(sensitivity_parameter_count)
            if sensitivity_parameter_count is not None
            else int(np.max(active_parameter_ids)) + 1
        )
        if resolved_parameter_count < 1:
            raise ValueError("sensitivity_parameter_count must be positive")
        if np.any(active_parameter_ids >= resolved_parameter_count):
            raise ValueError("sensitivity_cell_parameter_ids contain ids outside sensitivity_parameter_count")

        active_cell_ids_torch = torch_np.asarray(active_cell_ids, dtype=INT_DTYPE)
        digest = hashlib.sha256()
        _update_digest_value(digest, "parameter_count", resolved_parameter_count)
        _update_digest_array(digest, "cell_parameter_ids", cell_parameter_ids)
        parameterized_cache_key = (
            f"{cache_key}_param_{resolved_parameter_count}_{active_cell_ids.size}_{digest.hexdigest()[:16]}"
        )
        return (
            torch_np.take(cell_connectivity, active_cell_ids_torch, axis=0),
            torch_np.asarray(active_parameter_ids, dtype=INT_DTYPE),
            resolved_parameter_count,
            torch_np.take(volume_templates, active_cell_ids_torch, axis=1),
            parameterized_cache_key,
        )

    def _discretization_sub_potential_stack(
        self,
        discretization: AuxiliaryDiscretization,
        *,
        cache_key: str,
        state_prefix: str,
    ) -> Array:
        cached = self._derived_cache.get(cache_key)
        if cached is not None:
            return cached

        solve_dtype = self._auxiliary_float_dtype()
        disk_cache_key = self._terrain_cache_key(cache_key, discretization)
        cached = self._load_terrain_cached_array(
            disk_cache_key,
            shape=(
                int(self.wavenumbers.shape[0]),
                int(discretization.source_matrix.shape[0]),
                int(discretization.dof_nodes.shape[0]),
            ),
            dtype=solve_dtype,
        )
        if cached is not None:
            self._derived_cache[cache_key] = cached
            return cached

        operator_values = self._assemble_discretization_operator_values_batch(
            discretization,
            cache_key=f"{cache_key}_operator_values",
        )
        rhs = torch_np.broadcast_to(
            discretization.source_matrix[None, :, :],
            (self.wavenumbers.shape[0], discretization.source_matrix.shape[0], discretization.source_matrix.shape[1]),
        ).astype(operator_values.dtype)
        cached = self._solve_batch_with_pattern(
            discretization.operator_pattern,
            operator_values,
            rhs,
            state_prefix=state_prefix,
        )
        self._derived_cache[cache_key] = cached
        self._store_terrain_cached_array(disk_cache_key, cached)
        return cached

    def _auxiliary_sub_potential_stack(self) -> Array:
        if self.primary_auxiliary_discretization is None:
            raise ValueError("primary auxiliary discretization is not available")

        cached = self._derived_cache.get("auxiliary_sub_potentials")
        if cached is not None:
            return cached

        disk_cache_key = self._terrain_cache_key(
            "auxiliary_sub_potentials",
            self.primary_auxiliary_discretization,
        )
        cached = self._load_terrain_cached_array(
            disk_cache_key,
            shape=(
                int(self.wavenumbers.shape[0]),
                int(self.primary_auxiliary_discretization.source_matrix.shape[0]),
                int(self.primary_auxiliary_discretization.dof_nodes.shape[0]),
            ),
            dtype=self._auxiliary_float_dtype(),
        )
        if cached is not None:
            self._derived_cache["auxiliary_sub_potentials"] = cached
            return cached

        if self.primary_potential_discretization is None:
            raise ValueError("primary-potential discretization is not available")

        primary_potentials = self._discretization_sub_potential_stack(
            self.primary_potential_discretization,
            cache_key="primary_potential_sub_potentials",
            state_prefix="primary_potential",
        )
        selection_indices = _exact_node_selection_indices(
            self.primary_potential_discretization.dof_nodes,
            self.primary_auxiliary_discretization.dof_nodes,
        )
        cached = torch_np.take(primary_potentials, torch_np.asarray(selection_indices, dtype=INT_DTYPE), axis=-1)
        self._derived_cache["auxiliary_sub_potentials"] = cached
        self._store_terrain_cached_array(disk_cache_key, cached)
        return cached

    def _projected_auxiliary_unit_primary_stack(self) -> Array:
        if self.primary_auxiliary_discretization is None:
            raise ValueError("primary auxiliary discretization is not available")

        cached = self._derived_cache.get("projected_auxiliary_unit_primary")
        if cached is not None:
            return cached

        cached = self._auxiliary_sub_potential_stack() @ self.primary_auxiliary_discretization.original_node_matrix.T
        self._derived_cache["projected_auxiliary_unit_primary"] = cached
        return cached

    def _auxiliary_geometric_factors(self) -> Array:
        if self.geometric_auxiliary_discretization is None:
            raise ValueError("geometric-factor auxiliary discretization is not available")

        cached = self._derived_cache.get("auxiliary_geometric_factors")
        if cached is not None:
            return cached

        discretization = self.geometric_auxiliary_discretization
        sub_potentials = self._discretization_sub_potential_stack(
            discretization,
            cache_key="auxiliary_geometric_sub_potentials",
            state_prefix="auxiliary_geometric",
        )
        integrated_potentials = torch_np.tensordot(self.weights, sub_potentials, axes=(0, 0))
        electrode_potentials = self._project_from_integrated(integrated_potentials, discretization.electrode_matrix)
        source_potentials = electrode_potentials[self.survey.measurements[:, 0]] - electrode_potentials[
            self.survey.measurements[:, 1]
        ]
        resistance = source_potentials[torch_np.arange(self.survey.measurement_count), self.survey.measurements[:, 2]] - source_potentials[
            torch_np.arange(self.survey.measurement_count), self.survey.measurements[:, 3]
        ]
        cached = 1.0 / resistance
        self._derived_cache["auxiliary_geometric_factors"] = cached
        return cached

    def _unit_primary_potentials(self, wavenumber_index: int) -> Array:
        if self.use_numerical_primary:
            return self._unit_primary_stack()[wavenumber_index]

        wavenumber = float(self.wavenumbers[wavenumber_index])
        cached = self._unit_primary_cache.get(wavenumber)
        if cached is not None:
            return cached

        primaries = []
        for source_id, source in enumerate(np.asarray(self.survey.electrode_positions, dtype=float)):
            primary = _exact_dcsolution_on_nodes(self.mesh, source, wavenumber)
            node_id = int(self.source_node_ids[source_id])
            if node_id >= 0:
                primary = primary.at[node_id].set(_node_singularity_value(self.mesh, node_id, wavenumber))
            primaries.append(primary)
        cached = torch_np.stack(primaries, axis=0)
        self._unit_primary_cache[wavenumber] = cached
        return cached

    def _unit_primary_stack(self) -> Array:
        cached = self._unit_primary_cache.get("__stack__")
        if cached is not None:
            return cached

        if self.use_numerical_primary:
            cached = self._projected_auxiliary_unit_primary_stack()
        else:
            cached = torch_np.stack(
                [self._unit_primary_potentials(wavenumber_index) for wavenumber_index in range(self.wavenumbers.shape[0])],
                axis=0,
            )
        self._unit_primary_cache["__stack__"] = cached
        return cached

    def _reference_rhs(self, wavenumber_index: int, unit_primary: Array) -> Array:
        wavenumber = float(self.wavenumbers[wavenumber_index])
        cached = self._reference_rhs_cache.get(wavenumber)
        if cached is not None:
            return cached

        cached = self._reference_rhs_stack()[wavenumber_index]
        self._reference_rhs_cache[wavenumber] = cached
        return cached

    def _reference_rhs_stack(self) -> Array:
        cached = self._reference_rhs_cache.get("__stack__")
        if cached is not None:
            return cached

        if self.use_numerical_primary:
            reference_values = self._assemble_operator_values_batch(
                torch_np.ones((self.mesh.cell_count,), dtype=FLOAT_DTYPE)
            )
            cached = self._apply_operator_values_batch(reference_values, self._unit_primary_stack())
        else:
            reference_values = self._assemble_operator_values_batch(
                torch_np.ones((self.mesh.cell_count,), dtype=FLOAT_DTYPE)
            )
            cached = self._apply_operator_values_batch(reference_values, self._unit_primary_stack())
        self._reference_rhs_cache["__stack__"] = cached
        return cached

    def _compute_measurement_response(self, conductivity: Array | float) -> tuple[Array, Array, Array]:
        solve_conductivity = _expand_conductivity(conductivity, self.mesh.cell_count)
        cached = self._consume_prepared_measurement_response(solve_conductivity)
        if cached is not None:
            return cached

        if self.use_numerical_primary:
            discretization, _, sub_potentials = self._solve_secondary_fields_auxiliary(solve_conductivity)
            integrated_potentials = self._integrate_potentials(sub_potentials)
            self._check_finite(integrated_potentials, context="integrated auxiliary potentials")
            electrode_potentials = self._project_from_integrated(integrated_potentials, discretization.electrode_matrix)
            resistance = self._auxiliary_resistance_from_integrated(discretization, integrated_potentials)
            return integrated_potentials, electrode_potentials, resistance

        _, sub_potentials = self._solve_total_fields(solve_conductivity)
        integrated_potentials = self._integrate_potentials(sub_potentials)

        self._check_finite(integrated_potentials, context="integrated potentials")

        electrode_potentials = self._project_from_integrated(integrated_potentials, self.electrode_matrix)
        resistance = self._native_resistance_from_integrated(integrated_potentials)
        return integrated_potentials, electrode_potentials, resistance

    def _auxiliary_resistance_from_integrated(
        self,
        discretization: AuxiliaryDiscretization,
        integrated_potentials: Array,
    ) -> Array:
        normal_resistance = self._apply_measurement_map_from_integrated_with_receiver(
            integrated_potentials,
            self._measurement_receiver_matrix_for(
                discretization.electrode_matrix,
                cache_key="primary_auxiliary_measurement_receiver_matrix",
            ),
        )
        reciprocal_resistance = self._apply_reciprocal_measurement_map_from_integrated_with_receiver(
            integrated_potentials,
            self._current_receiver_matrix_for(
                discretization.electrode_matrix,
                cache_key="primary_auxiliary_current_receiver_matrix",
            ),
        )
        return self._combine_reciprocal_resistances(normal_resistance, reciprocal_resistance)

    def _native_resistance_from_integrated(self, integrated_potentials: Array) -> Array:
        normal_resistance = self._apply_measurement_map_from_integrated(integrated_potentials)
        reciprocal_resistance = self._apply_reciprocal_measurement_map_from_integrated(integrated_potentials)
        return self._combine_reciprocal_resistances(normal_resistance, reciprocal_resistance)

    def _compute_resistance(self, conductivity: Array | float) -> Array:
        solve_conductivity = _expand_conductivity(conductivity, self.mesh.cell_count)
        cached = self._consume_prepared_measurement_response(solve_conductivity)
        if cached is not None:
            return cached[2]

        if self.use_numerical_primary:
            discretization, _, sub_potentials = self._solve_secondary_fields_auxiliary(solve_conductivity)
            integrated_potentials = self._integrate_potentials(sub_potentials)
            self._check_finite(integrated_potentials, context="integrated auxiliary potentials")
            return self._auxiliary_resistance_from_integrated(discretization, integrated_potentials)

        _, sub_potentials = self._solve_total_fields(solve_conductivity)
        integrated_potentials = self._integrate_potentials(sub_potentials)
        self._check_finite(integrated_potentials, context="integrated potentials")
        return self._native_resistance_from_integrated(integrated_potentials)

    def _store_prepared_measurement_response(
        self,
        conductivity: Array,
        integrated_potentials: Array,
        electrode_potentials: Array,
        resistance: Array,
    ) -> None:
        try:
            conductivity_host = np.asarray(torch_runtime.block_until_ready(conductivity))
        except Exception:
            return
        self._derived_cache["prepared_response_conductivity_host"] = conductivity_host
        self._derived_cache["prepared_response_integrated_potentials"] = integrated_potentials
        self._derived_cache["prepared_response_electrode_potentials"] = electrode_potentials
        self._derived_cache["prepared_response_resistance"] = resistance

    def _consume_prepared_measurement_response(self, conductivity: Array) -> tuple[Array, Array, Array] | None:
        conductivity_host = self._derived_cache.pop("prepared_response_conductivity_host", None)
        integrated_potentials = self._derived_cache.pop("prepared_response_integrated_potentials", None)
        electrode_potentials = self._derived_cache.pop("prepared_response_electrode_potentials", None)
        resistance = self._derived_cache.pop("prepared_response_resistance", None)
        if (
            conductivity_host is None
            or integrated_potentials is None
            or electrode_potentials is None
            or resistance is None
        ):
            return None

        try:
            candidate = np.asarray(torch_runtime.block_until_ready(conductivity))
        except Exception:
            return None
        if candidate.shape != conductivity_host.shape or not np.array_equal(candidate, conductivity_host):
            return None
        return integrated_potentials, electrode_potentials, resistance

    def _store_prepared_total_fields(
        self,
        conductivity: Array,
        operator_values: Array,
        total_fields: Array,
    ) -> None:
        try:
            conductivity_host = np.asarray(torch_runtime.block_until_ready(conductivity))
        except Exception:
            return
        self._derived_cache["prepared_total_fields_conductivity_host"] = conductivity_host
        self._derived_cache["prepared_total_fields_operator_values"] = operator_values
        self._derived_cache["prepared_total_fields"] = total_fields

    def _consume_prepared_total_fields(self, conductivity: Array) -> tuple[Array, Array] | None:
        conductivity_host = self._derived_cache.pop("prepared_total_fields_conductivity_host", None)
        operator_values = self._derived_cache.pop("prepared_total_fields_operator_values", None)
        total_fields = self._derived_cache.pop("prepared_total_fields", None)
        if conductivity_host is None or operator_values is None or total_fields is None:
            return None

        try:
            candidate = np.asarray(torch_runtime.block_until_ready(conductivity))
        except Exception:
            return None
        if candidate.shape != conductivity_host.shape or not np.array_equal(candidate, conductivity_host):
            return None
        return operator_values, total_fields

    def _geometric_factors(self) -> Array:
        if not self.use_numerical_geometric_factors:
            return self.survey.geometric_factors()

        cached = self._derived_cache.get("geometric_factors")
        if cached is not None:
            return cached

        cached = self._auxiliary_geometric_factors()
        self._derived_cache["geometric_factors"] = cached
        return cached

    def _solve_wavenumber_fields_indexed(
        self,
        conductivity: Array | float,
        wavenumber_index: int,
        source_resistivities: Array | None = None,
    ) -> SingleWavenumberFields:
        solve_conductivity = _expand_conductivity(conductivity, self.mesh.cell_count)
        wavenumber = float(self.wavenumbers[wavenumber_index])
        operator_values = self._assemble_operator_values(solve_conductivity, wavenumber_index)
        if source_resistivities is None:
            source_resistivities = self._electrode_source_resistivities(solve_conductivity)
        unit_primary = self._unit_primary_potentials(wavenumber_index)
        rhs, primary = self._build_rhs_kernel()(
            operator_values,
            unit_primary,
            source_resistivities,
            self._reference_rhs(wavenumber_index, unit_primary),
        )
        sub_potentials = self._solve_linear_system(operator_values, rhs) + primary

        self._check_finite(rhs, context="rhs", wavenumber=wavenumber)
        self._check_finite(sub_potentials, context="solution", wavenumber=wavenumber)

        return SingleWavenumberFields(
            wavenumber=wavenumber,
            rhs=rhs,
            sub_potentials=sub_potentials,
            unit_primary=unit_primary,
            primary=primary,
            source_resistivities=source_resistivities,
        )

    def _solve_wavenumber_fields(self, conductivity: Array | float, wavenumber: float) -> SingleWavenumberFields:
        """Solve one unintegrated 2.5D subproblem."""

        return self._solve_wavenumber_fields_indexed(conductivity, self._wavenumber_index(wavenumber))

    def solve(self, conductivity: Array | float, currents: Array | float = 1.0) -> ForwardResponse:
        """Solve the multi-wavenumber 2.5D forward problem for a conductivity model."""

        integrated_potentials, electrode_potentials, resistance = self._compute_measurement_response(conductivity)
        current_array = torch_np.asarray(currents, dtype=FLOAT_DTYPE)
        apparent_resistivity = torch_np.abs(self._geometric_factors()) * resistance / current_array

        return ForwardResponse(
            apparent_resistivity=apparent_resistivity,
            resistance=resistance,
            electrode_potentials=electrode_potentials,
            integrated_potentials=integrated_potentials,
            wavenumbers=self.wavenumbers,
            weights=self.weights,
        )

    def _resolve_jacobian_batch_size(
        self,
        batch_size: int | None,
        *,
        include_robin_boundary_derivative: bool,
        normal_sensitivity: bool,
    ) -> int:
        if batch_size is None:
            if normal_sensitivity and not include_robin_boundary_derivative:
                measurement_count = int(self.survey.measurement_count)
                if int(self.mesh.cell_count) >= 4096:
                    return min(measurement_count, 64)
                return measurement_count
            return 8
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        return int(batch_size)

    def solve_with_jacobian(
        self,
        conductivity: Array | float,
        currents: Array | float = 1.0,
        *,
        batch_size: int | None = None,
        include_robin_boundary_derivative: bool = False,
        normal_sensitivity: bool = True,
        jacobian_cell_parameter_ids: Array | None = None,
        jacobian_parameter_count: int | None = None,
    ) -> tuple[ForwardResponse, Array]:
        """Solve forward response and materialize the resistance Jacobian.

        The returned Jacobian is ``d resistance / d conductivity``. Callers can
        apply geometric factors and model-space chain rules without recomputing
        the forward fields. By default this uses normal-quadrupole
        sensitivity and omits the mixed Robin boundary derivative. Pass
        ``include_robin_boundary_derivative=True, normal_sensitivity=False``
        for the exact derivative of the deepert reciprocal-averaged response.
        ``jacobian_cell_parameter_ids`` optionally accumulates the direct
        normal sensitivity into marker/parameter columns without materializing
        a full cell-by-cell Jacobian.
        """

        batch_size = self._resolve_jacobian_batch_size(
            batch_size,
            include_robin_boundary_derivative=include_robin_boundary_derivative,
            normal_sensitivity=normal_sensitivity,
        )

        if self.use_numerical_primary:
            discretization, operator_values, sub_potentials = self._solve_secondary_fields_auxiliary(conductivity)
            integrated_potentials = self._integrate_potentials(sub_potentials)
            self._check_finite(integrated_potentials, context="integrated auxiliary potentials")
            electrode_potentials = self._project_from_integrated(integrated_potentials, discretization.electrode_matrix)
            measurement_receiver = self._measurement_receiver_matrix_for(
                discretization.electrode_matrix,
                cache_key="primary_auxiliary_measurement_receiver_matrix",
            )
            current_receiver = self._current_receiver_matrix_for(
                discretization.electrode_matrix,
                cache_key="primary_auxiliary_current_receiver_matrix",
            )
            normal_resistance = self._apply_measurement_map_from_integrated_with_receiver(
                integrated_potentials,
                measurement_receiver,
            )
            reciprocal_resistance = self._apply_reciprocal_measurement_map_from_integrated_with_receiver(
                integrated_potentials,
                current_receiver,
            )
            resistance = self._combine_reciprocal_resistances(normal_resistance, reciprocal_resistance)
            if normal_sensitivity and not include_robin_boundary_derivative:
                wavenumber_sq = torch_np.square(self.wavenumbers).astype(sub_potentials.dtype)
                volume_templates = discretization.operator_templates.stiffness[None, :, :, :] + wavenumber_sq[
                    :, None, None, None
                ] * discretization.operator_templates.mass[None, :, :, :]
                sensitivity_cell_parameter_ids = None
                if jacobian_cell_parameter_ids is not None:
                    mesh_parameter_ids = np.asarray(jacobian_cell_parameter_ids, dtype=np.int32).ravel()
                    if mesh_parameter_ids.shape != (int(self.mesh.cell_count),):
                        raise ValueError(
                            "jacobian_cell_parameter_ids must have one entry per forward mesh cell "
                            f"({mesh_parameter_ids.shape} != ({int(self.mesh.cell_count)},))"
                        )
                    sensitivity_cell_parameter_ids = mesh_parameter_ids[
                        np.asarray(discretization.parent_cell_ids, dtype=np.int32)
                    ]
                (
                    sensitivity_cell_connectivity,
                    sensitivity_parent_ids,
                    sensitivity_parameter_count,
                    sensitivity_volume_templates,
                    sensitivity_cache_key,
                ) = self._direct_sensitivity_inputs(
                    cell_connectivity=discretization.cell_connectivity,
                    parent_cell_ids=discretization.parent_cell_ids,
                    parameter_count=int(self.mesh.cell_count),
                    volume_templates=volume_templates,
                    cache_key="primary_normal_sensitivity_direct",
                    sensitivity_cell_parameter_ids=sensitivity_cell_parameter_ids,
                    sensitivity_parameter_count=jacobian_parameter_count,
                )
                resistance_jacobian = self._normal_sensitivity_from_fields_direct(
                    sub_potentials,
                    batch_size=batch_size,
                    cell_connectivity=sensitivity_cell_connectivity,
                    parent_cell_ids=sensitivity_parent_ids,
                    parameter_count=sensitivity_parameter_count,
                    volume_templates=sensitivity_volume_templates,
                    cache_key=sensitivity_cache_key,
                )
            else:
                resistance_jacobian = self._jacobian_from_fields_adjoint_batch(
                    operator_values,
                    sub_potentials,
                    measurement_receiver=measurement_receiver,
                    current_receiver=current_receiver,
                    normal=normal_resistance,
                    reciprocal=reciprocal_resistance,
                    node_count=int(discretization.dof_nodes.shape[0]),
                    operator_pattern=discretization.operator_pattern,
                    state_prefix=f"primary_jacobian_adjoint_fields_b{batch_size}",
                    batch_size=batch_size,
                    include_robin_boundary_derivative=include_robin_boundary_derivative,
                    normal_sensitivity=normal_sensitivity,
                )
        else:
            operator_values, sub_potentials = self._solve_total_fields(conductivity)
            integrated_potentials = self._integrate_potentials(sub_potentials)
            self._check_finite(integrated_potentials, context="integrated potentials")
            electrode_potentials = self._project_from_integrated(integrated_potentials, self.electrode_matrix)
            normal_resistance = self._apply_measurement_map_from_integrated(integrated_potentials)
            reciprocal_resistance = self._apply_reciprocal_measurement_map_from_integrated(integrated_potentials)
            resistance = self._combine_reciprocal_resistances(normal_resistance, reciprocal_resistance)
            if normal_sensitivity and not include_robin_boundary_derivative:
                wavenumber_sq = torch_np.square(self.wavenumbers).astype(sub_potentials.dtype)
                volume_templates = self.operator_templates.stiffness[None, :, :, :] + wavenumber_sq[:, None, None, None] * (
                    self.operator_templates.mass[None, :, :, :]
                )
                (
                    sensitivity_cell_connectivity,
                    sensitivity_parent_ids,
                    sensitivity_parameter_count,
                    sensitivity_volume_templates,
                    sensitivity_cache_key,
                ) = self._direct_sensitivity_inputs(
                    cell_connectivity=self.mesh.cells,
                    parent_cell_ids=torch_np.arange(self.mesh.cell_count, dtype=INT_DTYPE),
                    parameter_count=int(self.mesh.cell_count),
                    volume_templates=volume_templates,
                    cache_key="normal_sensitivity_direct",
                    sensitivity_cell_parameter_ids=jacobian_cell_parameter_ids,
                    sensitivity_parameter_count=jacobian_parameter_count,
                )
                resistance_jacobian = self._normal_sensitivity_from_fields_direct(
                    sub_potentials,
                    batch_size=batch_size,
                    cell_connectivity=sensitivity_cell_connectivity,
                    parent_cell_ids=sensitivity_parent_ids,
                    parameter_count=sensitivity_parameter_count,
                    volume_templates=sensitivity_volume_templates,
                    cache_key=sensitivity_cache_key,
                )
            else:
                resistance_jacobian = self._jacobian_from_fields_adjoint_batch(
                    operator_values,
                    sub_potentials,
                    measurement_receiver=self._measurement_receiver_matrix(),
                    current_receiver=self._current_receiver_matrix(),
                    normal=normal_resistance,
                    reciprocal=reciprocal_resistance,
                    node_count=int(self.mesh.node_count),
                    operator_pattern=self.operator_pattern,
                    state_prefix=f"jacobian_adjoint_fields_b{batch_size}",
                    batch_size=batch_size,
                    include_robin_boundary_derivative=include_robin_boundary_derivative,
                    normal_sensitivity=normal_sensitivity,
                )

        current_array = torch_np.asarray(currents, dtype=FLOAT_DTYPE)
        apparent_resistivity = torch_np.abs(self._geometric_factors()) * resistance / current_array
        response = ForwardResponse(
            apparent_resistivity=apparent_resistivity,
            resistance=resistance,
            electrode_potentials=electrode_potentials,
            integrated_potentials=integrated_potentials,
            wavenumbers=self.wavenumbers,
            weights=self.weights,
        )
        return response, resistance_jacobian

    def resistance(self, conductivity: Array | float) -> Array:
        """Return the measurement resistance vector for a conductivity model."""

        return self._compute_resistance(conductivity)

    def apparent_resistivity_values(self, conductivity: Array | float, currents: Array | float = 1.0) -> Array:
        """Return apparent resistivity values without allocating a ForwardResponse."""

        current_array = torch_np.asarray(currents, dtype=FLOAT_DTYPE)
        return torch_np.abs(self._geometric_factors()) * self._compute_resistance(conductivity) / current_array

    def apparent_resistivity_series(self, conductivities: Array | float, currents: Array | float = 1.0) -> Array:
        """Return apparent resistivity for a sequence of conductivity models.

        The forward operator, geometric factors, cuDSS plan, sparse
        matrix buffers, and RHS buffers are reused across all timesteps. This is
        the preferred path for time-lapse forward prediction when electrode
        potentials and unintegrated fields are not needed.
        """

        conductivity_array = torch_np.asarray(conductivities, dtype=self._auxiliary_float_dtype())
        if conductivity_array.ndim != 2 or conductivity_array.shape[1] != self.mesh.cell_count:
            raise ValueError(f"conductivities must have shape (n_steps, {self.mesh.cell_count})")

        currents_array = torch_np.asarray(currents, dtype=FLOAT_DTYPE)
        rows = []
        for step_index in range(int(conductivity_array.shape[0])):
            step_currents = currents_array[step_index] if currents_array.ndim == 2 else currents_array
            rows.append(self.apparent_resistivity_values(conductivity_array[step_index], currents=step_currents))
        return torch_np.stack(rows, axis=0)

    def prepare(self, conductivity: Array | float | None = None, *, include_solver_state: bool = True) -> None:
        """Populate geometry, cache, and optional cuDSS plan state before timed solves.

        The preparation conductivity is only a representative model used to
        size cuDSS buffers. Later solves still update matrix
        values and run a fresh numeric factorization whenever conductivity
        changes.
        """

        if conductivity is None:
            solve_conductivity = torch_np.ones((self.mesh.cell_count,), dtype=FLOAT_DTYPE)
        else:
            solve_conductivity = _expand_conductivity(conductivity, self.mesh.cell_count)

        if self.use_numerical_primary:
            if self.primary_auxiliary_discretization is None:
                raise ValueError("primary auxiliary discretization is not available")

            discretization = self.primary_auxiliary_discretization
            operator_values = self._assemble_discretization_operator_values_batch(
                discretization,
                solve_conductivity,
                cache_key="assemble_primary_auxiliary_operator_values_batch",
            )
            unit_primary = self._auxiliary_sub_potential_stack()
            source_resistivities = self._electrode_source_resistivities(solve_conductivity)
            primary = unit_primary * source_resistivities[None, :, None]
            op_primary = self._apply_operator_values_batch_with_pattern(
                discretization.operator_pattern,
                operator_values,
                primary,
                cache_key="apply_primary_auxiliary_operator_values_batch",
            )
            rhs = self._auxiliary_reference_rhs_stack(discretization, unit_primary) - op_primary
            self._measurement_receiver_matrix_for(
                discretization.electrode_matrix,
                cache_key="primary_auxiliary_measurement_receiver_matrix",
            )
            self._current_receiver_matrix_for(
                discretization.electrode_matrix,
                cache_key="primary_auxiliary_current_receiver_matrix",
            )
            geometric_factors = self._geometric_factors()
            torch_runtime.block_until_ready((operator_values, unit_primary, source_resistivities, op_primary, rhs))
            torch_runtime.block_until_ready(geometric_factors)
            if include_solver_state:
                prepared = self._solve_batch_with_pattern(
                    discretization.operator_pattern,
                    operator_values,
                    rhs,
                    state_prefix="primary_secondary_fields",
                )
                total_fields = prepared + primary
                integrated_potentials = self._integrate_potentials(total_fields)
                electrode_potentials = self._project_from_integrated(integrated_potentials, discretization.electrode_matrix)
                resistance = self._auxiliary_resistance_from_integrated(discretization, integrated_potentials)
                apparent_resistivity = torch_np.abs(geometric_factors) * resistance
                torch_runtime.block_until_ready((prepared, integrated_potentials, electrode_potentials, resistance, apparent_resistivity))
                total_fields = torch_runtime.block_until_ready(total_fields)
                self._store_prepared_total_fields(solve_conductivity, operator_values, total_fields)
                self._store_prepared_measurement_response(
                    solve_conductivity,
                    integrated_potentials,
                    electrode_potentials,
                    resistance,
                )
            return

        operator_values = self._assemble_operator_values_batch(solve_conductivity)
        source_resistivities = self._electrode_source_resistivities(solve_conductivity)
        rhs, primary = self._build_rhs_batch_kernel()(
            operator_values,
            self._unit_primary_stack(),
            source_resistivities,
            self._reference_rhs_stack(),
        )
        self._measurement_receiver_matrix()
        self._current_receiver_matrix()
        geometric_factors = self._geometric_factors()
        torch_runtime.block_until_ready((operator_values, source_resistivities, rhs, primary))
        torch_runtime.block_until_ready(geometric_factors)
        if include_solver_state:
            prepared = self._solve_linear_system_batch(operator_values, rhs)
            total_fields = prepared + primary
            integrated_potentials = self._integrate_potentials(total_fields)
            electrode_potentials = self._project_from_integrated(integrated_potentials, self.electrode_matrix)
            resistance = self._native_resistance_from_integrated(integrated_potentials)
            apparent_resistivity = torch_np.abs(geometric_factors) * resistance
            torch_runtime.block_until_ready((prepared, integrated_potentials, electrode_potentials, resistance, apparent_resistivity))
            total_fields = torch_runtime.block_until_ready(total_fields)
            self._store_prepared_total_fields(solve_conductivity, operator_values, total_fields)
            self._store_prepared_measurement_response(
                solve_conductivity,
                integrated_potentials,
                electrode_potentials,
                resistance,
            )

    def jvp(self, conductivity: Array | float, delta_conductivity: Array | float) -> Array:
        """Apply the resistance Jacobian to a conductivity perturbation."""

        if self.use_numerical_primary:
            discretization, operator_values, phi_stack = self._solve_secondary_fields_auxiliary(conductivity)
            tangent_rhs = -self._operator_tangent_apply(delta_conductivity, phi_stack)
            delta_phi = self._solve_batch_with_pattern(
                discretization.operator_pattern,
                operator_values,
                tangent_rhs,
                state_prefix="primary_tangent_fields",
            )
            self._check_finite(delta_phi, context="auxiliary tangent fields")
            measurement_receiver = self._measurement_receiver_matrix_for(
                discretization.electrode_matrix,
                cache_key="primary_auxiliary_measurement_receiver_matrix",
            )
            current_receiver = self._current_receiver_matrix_for(
                discretization.electrode_matrix,
                cache_key="primary_auxiliary_current_receiver_matrix",
            )
            integrated_phi = self._integrate_potentials(phi_stack)
            integrated_delta_phi = self._integrate_potentials(delta_phi)
            normal = self._apply_measurement_map_from_integrated_with_receiver(
                integrated_phi,
                measurement_receiver,
            )
            reciprocal = self._apply_reciprocal_measurement_map_from_integrated_with_receiver(
                integrated_phi,
                current_receiver,
            )
            delta_normal = self._apply_measurement_map_from_integrated_with_receiver(
                integrated_delta_phi,
                measurement_receiver,
            )
            delta_reciprocal = self._apply_reciprocal_measurement_map_from_integrated_with_receiver(
                integrated_delta_phi,
                current_receiver,
            )
            return self._combine_reciprocal_resistance_jvp(normal, reciprocal, delta_normal, delta_reciprocal)

        operator_values, phi_stack = self._solve_total_fields(conductivity)
        tangent_rhs = -self._operator_tangent_apply(delta_conductivity, phi_stack)
        delta_phi = self._solve_linear_system_batch(operator_values, tangent_rhs)
        self._check_finite(delta_phi, context="tangent fields")
        integrated_phi = self._integrate_potentials(phi_stack)
        integrated_delta_phi = self._integrate_potentials(delta_phi)
        normal = self._apply_measurement_map_from_integrated(integrated_phi)
        reciprocal = self._apply_reciprocal_measurement_map_from_integrated(integrated_phi)
        delta_normal = self._apply_measurement_map_from_integrated(integrated_delta_phi)
        delta_reciprocal = self._apply_reciprocal_measurement_map_from_integrated(integrated_delta_phi)
        return self._combine_reciprocal_resistance_jvp(normal, reciprocal, delta_normal, delta_reciprocal)

    def vjp(self, conductivity: Array | float, cotangent: Array | float) -> Array:
        """Apply the transposed resistance Jacobian to a measurement cotangent."""

        if self.use_numerical_primary:
            discretization, operator_values, phi_stack = self._solve_secondary_fields_auxiliary(conductivity)
            measurement_receiver = self._measurement_receiver_matrix_for(
                discretization.electrode_matrix,
                cache_key="primary_auxiliary_measurement_receiver_matrix",
            )
            current_receiver = self._current_receiver_matrix_for(
                discretization.electrode_matrix,
                cache_key="primary_auxiliary_current_receiver_matrix",
            )
            integrated_phi = self._integrate_potentials(phi_stack)
            normal = self._apply_measurement_map_from_integrated_with_receiver(
                integrated_phi,
                measurement_receiver,
            )
            reciprocal = self._apply_reciprocal_measurement_map_from_integrated_with_receiver(
                integrated_phi,
                current_receiver,
            )
            product = normal * reciprocal
            combined = self._combine_reciprocal_resistances(normal, reciprocal)
            safe_combined = torch_np.maximum(combined, torch_np.asarray(1e-30, dtype=combined.dtype))
            cotangent_array = _expand_measurement_vector(
                cotangent,
                self.survey.measurement_count,
                name="cotangent",
            ).astype(combined.dtype)
            common = 0.5 * cotangent_array * torch_np.sign(product) / safe_combined
            normal_cotangent = common * reciprocal
            reciprocal_cotangent = common * normal
            adjoint_rhs = self._apply_measurement_map_transpose_with_receiver(
                normal_cotangent,
                measurement_receiver,
                node_count=int(discretization.dof_nodes.shape[0]),
            ) + self._apply_reciprocal_measurement_map_transpose_with_receiver(
                reciprocal_cotangent,
                current_receiver,
                node_count=int(discretization.dof_nodes.shape[0]),
            )
            lambda_stack = self._solve_batch_with_pattern(
                discretization.operator_pattern,
                operator_values,
                adjoint_rhs,
                state_prefix="primary_adjoint_fields",
            )
            self._check_finite(lambda_stack, context="auxiliary adjoint fields")
            gradient = self._accumulate_adjoint_cell_gradient(phi_stack, lambda_stack)
            self._check_finite(gradient, context="conductivity gradient")
            return gradient

        operator_values, phi_stack = self._solve_total_fields(conductivity)
        integrated_phi = self._integrate_potentials(phi_stack)
        normal = self._apply_measurement_map_from_integrated(integrated_phi)
        reciprocal = self._apply_reciprocal_measurement_map_from_integrated(integrated_phi)
        product = normal * reciprocal
        combined = self._combine_reciprocal_resistances(normal, reciprocal)
        safe_combined = torch_np.maximum(combined, torch_np.asarray(1e-30, dtype=combined.dtype))
        cotangent_array = _expand_measurement_vector(
            cotangent,
            self.survey.measurement_count,
            name="cotangent",
        ).astype(combined.dtype)
        common = 0.5 * cotangent_array * torch_np.sign(product) / safe_combined
        normal_cotangent = common * reciprocal
        reciprocal_cotangent = common * normal
        adjoint_rhs = self._apply_measurement_map_transpose(normal_cotangent) + self._apply_reciprocal_measurement_map_transpose(
            reciprocal_cotangent
        )
        lambda_stack = self._solve_linear_system_batch(operator_values, adjoint_rhs)
        self._check_finite(lambda_stack, context="adjoint fields")
        gradient = self._accumulate_adjoint_cell_gradient(phi_stack, lambda_stack)
        self._check_finite(gradient, context="conductivity gradient")
        return gradient

    def _jacobian_from_fields_adjoint_batch(
        self,
        operator_values: Array,
        phi_stack: Array,
        *,
        measurement_receiver: Array,
        current_receiver: Array,
        normal: Array,
        reciprocal: Array,
        node_count: int,
        operator_pattern: SparseOperatorPattern,
        state_prefix: str,
        batch_size: int,
        include_robin_boundary_derivative: bool,
        normal_sensitivity: bool,
    ) -> Array:
        measurement_count = int(self.survey.measurement_count)
        source_count = int(self.survey.electrode_count)
        product = normal * reciprocal
        combined = self._combine_reciprocal_resistances(normal, reciprocal)
        safe_combined = torch_np.maximum(combined, torch_np.asarray(1e-30, dtype=combined.dtype))
        rows = []
        refactorize = True

        for start in range(0, measurement_count, batch_size):
            stop = min(start + batch_size, measurement_count)
            chunk_size = stop - start
            cotangent = torch_np.eye(measurement_count, dtype=combined.dtype)[start:stop]
            if chunk_size < batch_size:
                padding = torch_np.zeros((batch_size - chunk_size, measurement_count), dtype=combined.dtype)
                cotangent = torch_np.concatenate((cotangent, padding), axis=0)

            if normal_sensitivity:
                adjoint_rhs = self._apply_measurement_map_transpose_batch_with_receiver(
                    cotangent,
                    measurement_receiver,
                    node_count=node_count,
                )
            else:
                common = 0.5 * cotangent * torch_np.sign(product)[None, :] / safe_combined[None, :]
                normal_cotangent = common * reciprocal[None, :]
                reciprocal_cotangent = common * normal[None, :]
                adjoint_rhs = self._apply_measurement_map_transpose_batch_with_receiver(
                    normal_cotangent,
                    measurement_receiver,
                    node_count=node_count,
                ) + self._apply_reciprocal_measurement_map_transpose_batch_with_receiver(
                    reciprocal_cotangent,
                    current_receiver,
                    node_count=node_count,
                )
            lambda_flat = self._solve_batch_with_pattern(
                operator_pattern,
                operator_values,
                adjoint_rhs,
                state_prefix=state_prefix,
                refactorize=refactorize,
            )
            refactorize = False
            lambda_stack = lambda_flat.reshape(
                (
                    int(self.wavenumbers.shape[0]),
                    batch_size,
                    source_count,
                    node_count,
                )
            )
            gradient_rows = self._accumulate_adjoint_cell_gradient_batch(
                phi_stack,
                lambda_stack,
                include_robin_boundary_derivative=include_robin_boundary_derivative,
            )
            rows.append(gradient_rows[:chunk_size])

        return torch_np.concatenate(rows, axis=0)

    def jacobian(
        self,
        conductivity: Array | float,
        *,
        batch_size: int | None = None,
        include_robin_boundary_derivative: bool = False,
        normal_sensitivity: bool = True,
    ) -> Array:
        """Materialize the explicit resistance Jacobian with batched adjoint solves.

        By default this uses the normal-quadrupole sensitivity convention.
        Pass ``include_robin_boundary_derivative=True,
        normal_sensitivity=False`` for the exact derivative of the deepert
        reciprocal-averaged response.
        """

        batch_size = self._resolve_jacobian_batch_size(
            batch_size,
            include_robin_boundary_derivative=include_robin_boundary_derivative,
            normal_sensitivity=normal_sensitivity,
        )

        if self.use_numerical_primary:
            discretization, operator_values, phi_stack = self._solve_secondary_fields_auxiliary(conductivity)
            measurement_receiver = self._measurement_receiver_matrix_for(
                discretization.electrode_matrix,
                cache_key="primary_auxiliary_measurement_receiver_matrix",
            )
            current_receiver = self._current_receiver_matrix_for(
                discretization.electrode_matrix,
                cache_key="primary_auxiliary_current_receiver_matrix",
            )
            integrated_phi = self._integrate_potentials(phi_stack)
            normal = self._apply_measurement_map_from_integrated_with_receiver(
                integrated_phi,
                measurement_receiver,
            )
            reciprocal = self._apply_reciprocal_measurement_map_from_integrated_with_receiver(
                integrated_phi,
                current_receiver,
            )
            if normal_sensitivity and not include_robin_boundary_derivative:
                wavenumber_sq = torch_np.square(self.wavenumbers).astype(phi_stack.dtype)
                volume_templates = discretization.operator_templates.stiffness[None, :, :, :] + wavenumber_sq[
                    :, None, None, None
                ] * discretization.operator_templates.mass[None, :, :, :]
                return self._normal_sensitivity_from_fields_direct(
                    phi_stack,
                    batch_size=batch_size,
                    cell_connectivity=discretization.cell_connectivity,
                    parent_cell_ids=discretization.parent_cell_ids,
                    parameter_count=int(self.mesh.cell_count),
                    volume_templates=volume_templates,
                    cache_key="primary_normal_sensitivity_direct",
                )
            return self._jacobian_from_fields_adjoint_batch(
                operator_values,
                phi_stack,
                measurement_receiver=measurement_receiver,
                current_receiver=current_receiver,
                normal=normal,
                reciprocal=reciprocal,
                node_count=int(discretization.dof_nodes.shape[0]),
                operator_pattern=discretization.operator_pattern,
                state_prefix=f"primary_jacobian_adjoint_fields_b{batch_size}",
                batch_size=batch_size,
                include_robin_boundary_derivative=include_robin_boundary_derivative,
                normal_sensitivity=normal_sensitivity,
            )

        operator_values, phi_stack = self._solve_total_fields(conductivity)
        integrated_phi = self._integrate_potentials(phi_stack)
        normal = self._apply_measurement_map_from_integrated(integrated_phi)
        reciprocal = self._apply_reciprocal_measurement_map_from_integrated(integrated_phi)
        if normal_sensitivity and not include_robin_boundary_derivative:
            wavenumber_sq = torch_np.square(self.wavenumbers).astype(phi_stack.dtype)
            volume_templates = self.operator_templates.stiffness[None, :, :, :] + wavenumber_sq[:, None, None, None] * (
                self.operator_templates.mass[None, :, :, :]
            )
            return self._normal_sensitivity_from_fields_direct(
                phi_stack,
                batch_size=batch_size,
                cell_connectivity=self.mesh.cells,
                parent_cell_ids=torch_np.arange(self.mesh.cell_count, dtype=INT_DTYPE),
                parameter_count=int(self.mesh.cell_count),
                volume_templates=volume_templates,
                cache_key="normal_sensitivity_direct",
            )
        return self._jacobian_from_fields_adjoint_batch(
            operator_values,
            phi_stack,
            measurement_receiver=self._measurement_receiver_matrix(),
            current_receiver=self._current_receiver_matrix(),
            normal=normal,
            reciprocal=reciprocal,
            node_count=int(self.mesh.node_count),
            operator_pattern=self.operator_pattern,
            state_prefix=f"jacobian_adjoint_fields_b{batch_size}",
            batch_size=batch_size,
            include_robin_boundary_derivative=include_robin_boundary_derivative,
            normal_sensitivity=normal_sensitivity,
        )

    def jacobian_columnwise(self, conductivity: Array | float) -> Array:
        """Materialize the explicit resistance Jacobian with one JVP per cell."""

        basis = torch_np.eye(self.mesh.cell_count, dtype=FLOAT_DTYPE)
        columns = [self.jvp(conductivity, basis[cell_index]) for cell_index in range(self.mesh.cell_count)]
        return torch_np.stack(columns, axis=1)

    def close(self) -> None:
        solver_keys = [
            key
            for key in list(self._cudss_state)
            if key == "solver_gpu" or key.endswith("_solver_gpu") or key.endswith("_solver_batch_gpu")
        ]
        for key in solver_keys:
            solver = self._cudss_state.pop(key, None)
            if solver is None:
                continue
            try:
                solver.free()
            except Exception:
                pass
        for key in list(self._cudss_state):
            if (
                key == "matrix_gpu"
                or key == "matrix_data_gpu"
                or key == "matrix_data_batch_gpu"
                or key.endswith("_matrix_gpu")
                or key.endswith("_matrix_data_gpu")
                or key.endswith("_matrix_data_batch_gpu")
                or key.endswith("_matrices_batch_gpu")
                or key.endswith("_rhs_batch_base_gpu")
                or key.endswith("_rhs_batch_gpu")
                or key.endswith("_gpu_fixed")
            ):
                self._cudss_state.pop(key, None)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
