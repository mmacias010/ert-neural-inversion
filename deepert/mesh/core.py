"""2D mesh data structures used by the ERT forward solver."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from pathlib import Path
import numpy as np

from deepert.utils.torch_runtime import Array
from deepert.utils.torch_runtime import torch_np
import meshio

from deepert.utils.dtypes import FLOAT_DTYPE, INT_DTYPE, NP_FLOAT_DTYPE


def triangle_areas(nodes: Array, cells: Array) -> Array:
    """Compute the area of each triangle cell."""

    nodes_np = np.asarray(nodes, dtype=NP_FLOAT_DTYPE)
    cells_np = np.asarray(cells, dtype=np.int32)
    cell_nodes = nodes_np[cells_np]
    edge_1 = cell_nodes[:, 1] - cell_nodes[:, 0]
    edge_2 = cell_nodes[:, 2] - cell_nodes[:, 0]
    cross = edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0]
    return torch_np.asarray(0.5 * np.abs(cross), dtype=FLOAT_DTYPE)


def cell_areas_2d(nodes: Array, cells: Array) -> Array:
    """Compute polygonal cell areas for triangle or quadrilateral cells."""

    nodes_np = np.asarray(nodes, dtype=NP_FLOAT_DTYPE)
    cells_np = np.asarray(cells, dtype=np.int32)
    if cells_np.ndim != 2 or cells_np.shape[1] not in (3, 4):
        raise ValueError("cells must have shape (num_cells, 3) or (num_cells, 4)")
    cell_nodes = nodes_np[cells_np]
    x_values = cell_nodes[:, :, 0]
    y_values = cell_nodes[:, :, 1]
    cross_sum = np.sum(
        x_values * np.roll(y_values, -1, axis=1) - np.roll(x_values, -1, axis=1) * y_values,
        axis=1,
    )
    return torch_np.asarray(0.5 * np.abs(cross_sum), dtype=FLOAT_DTYPE)


def _cell_edge_pairs(cell_width: int) -> tuple[tuple[int, int], ...]:
    if cell_width == 3:
        return ((0, 1), (1, 2), (2, 0))
    if cell_width == 4:
        return ((0, 1), (1, 2), (2, 3), (3, 0))
    raise ValueError("cells must have shape (num_cells, 3) or (num_cells, 4)")


def extract_boundary_edges(cells: Array) -> Array:
    """Return edges that belong to exactly one 2D cell."""

    cell_array = torch_np.asarray(cells, dtype=INT_DTYPE)
    edge_slices = [cell_array[:, [start, stop]] for start, stop in _cell_edge_pairs(int(cell_array.shape[1]))]
    edges = torch_np.concatenate(edge_slices, axis=0)
    edges = torch_np.sort(edges, axis=1)
    unique_edges, counts = torch_np.unique(edges, axis=0, return_counts=True)
    return unique_edges[counts == 1]


def _boundary_topology(cells: Array) -> tuple[Array, Array]:
    """Return lexicographically sorted boundary edges and their adjacent cells."""

    edge_to_cells: dict[tuple[int, int], list[int]] = {}
    cells_np = np.asarray(cells, dtype=np.int32)
    edge_pairs = _cell_edge_pairs(int(cells_np.shape[1]))
    for cell_id, cell in enumerate(cells_np):
        for start, stop in edge_pairs:
            edge = tuple(sorted((int(cell[start]), int(cell[stop]))))
            edge_to_cells.setdefault(edge, []).append(cell_id)

    boundary_edges: list[tuple[int, int]] = []
    boundary_cells: list[int] = []
    for edge, adjacent in edge_to_cells.items():
        if len(adjacent) == 1:
            boundary_edges.append(edge)
            boundary_cells.append(adjacent[0])

    order = sorted(range(len(boundary_edges)), key=lambda idx: boundary_edges[idx])
    boundary_edges = [boundary_edges[idx] for idx in order]
    boundary_cells = [boundary_cells[idx] for idx in order]
    return (
        torch_np.asarray(boundary_edges, dtype=INT_DTYPE),
        torch_np.asarray(boundary_cells, dtype=INT_DTYPE),
    )


def _boundary_geometry(
    nodes: Array,
    cells: Array,
    boundary_edges: Array,
    boundary_edge_cells: Array,
) -> tuple[Array, Array, Array]:
    """Compute centers, lengths, and outward normals for boundary edges."""

    nodes_np = np.asarray(nodes, dtype=NP_FLOAT_DTYPE)
    cells_np = np.asarray(cells, dtype=np.int32)
    boundary_edges_np = np.asarray(boundary_edges, dtype=np.int32)
    boundary_edge_cells_np = np.asarray(boundary_edge_cells, dtype=np.int32)

    edge_nodes = nodes_np[boundary_edges_np]
    centers = np.mean(edge_nodes, axis=1)
    edge_vectors = edge_nodes[:, 1] - edge_nodes[:, 0]
    lengths = np.linalg.norm(edge_vectors, axis=1)

    candidate_normals = np.stack(
        (edge_vectors[:, 1], -edge_vectors[:, 0]),
        axis=1,
    ) / lengths[:, None]
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


def _surface_topology(
    nodes: Array,
    boundary_edges: Array,
    tol: float = 1e-8,
) -> tuple[Array, Array, Array, Array, Array]:
    """Infer the upper boundary path between the left and right corners."""

    nodes_np = np.asarray(nodes, dtype=float)
    boundary_edges_np = np.asarray(boundary_edges, dtype=np.int32)
    boundary_node_ids = np.unique(boundary_edges_np.reshape(-1))
    x_coords = nodes_np[boundary_node_ids, 0]
    left_x = float(np.min(x_coords))
    right_x = float(np.max(x_coords))
    left_candidates = boundary_node_ids[np.abs(x_coords - left_x) <= tol]
    right_candidates = boundary_node_ids[np.abs(x_coords - right_x) <= tol]
    start_node = int(max(left_candidates.tolist(), key=lambda idx: (nodes_np[idx, 1], -nodes_np[idx, 0])))
    stop_node = int(max(right_candidates.tolist(), key=lambda idx: (nodes_np[idx, 1], -nodes_np[idx, 0])))

    edge_nodes = nodes_np[boundary_edges_np]
    edge_centers = np.mean(edge_nodes, axis=1)
    edge_lengths = np.linalg.norm(edge_nodes[:, 1] - edge_nodes[:, 0], axis=1)
    edge_y_max = float(np.max(edge_centers[:, 1]))
    edge_y_span = max(float(np.max(edge_centers[:, 1]) - np.min(edge_centers[:, 1])), tol)

    adjacency: dict[int, list[tuple[int, float]]] = {int(node_id): [] for node_id in boundary_node_ids.tolist()}
    for edge_id, (node_a, node_b) in enumerate(boundary_edges_np.tolist()):
        center_y = edge_centers[edge_id, 1]
        cost = float(edge_lengths[edge_id] * (1.0 + 100.0 * (edge_y_max - center_y) / edge_y_span))
        adjacency[int(node_a)].append((int(node_b), cost))
        adjacency[int(node_b)].append((int(node_a), cost))

    distances: dict[int, float] = {start_node: 0.0}
    previous_nodes: dict[int, int] = {}
    heap: list[tuple[float, int]] = [(0.0, start_node)]
    visited: set[int] = set()

    while heap:
        distance, node_id = heapq.heappop(heap)
        if node_id in visited:
            continue
        visited.add(node_id)
        if node_id == stop_node:
            break

        for neighbor, cost in adjacency[node_id]:
            new_distance = distance + cost
            if new_distance >= distances.get(neighbor, float("inf")):
                continue
            distances[neighbor] = new_distance
            previous_nodes[neighbor] = node_id
            heapq.heappush(heap, (new_distance, neighbor))

    if start_node == stop_node:
        surface_node_ids = [start_node]
    elif stop_node not in previous_nodes:
        surface_node_ids = [start_node, stop_node]
    else:
        surface_node_ids = [stop_node]
        current = stop_node
        while current != start_node:
            current = previous_nodes[current]
            surface_node_ids.append(current)
        surface_node_ids.reverse()

    surface_edges = {
        tuple(sorted((int(surface_node_ids[idx]), int(surface_node_ids[idx + 1]))))
        for idx in range(len(surface_node_ids) - 1)
    }
    surface_edge_mask = np.asarray(
        [tuple(sorted((int(edge[0]), int(edge[1])))) in surface_edges for edge in boundary_edges_np],
        dtype=bool,
    )

    surface_nodes = nodes_np[np.asarray(surface_node_ids, dtype=np.int32)]
    if surface_nodes.size == 0:
        flat_surface = True
        reference_level = 0.0
    else:
        flat_surface = bool(np.max(np.abs(surface_nodes[:, 1] - surface_nodes[0, 1])) <= tol)
        reference_level = float(np.mean(surface_nodes[:, 1]))

    return (
        torch_np.asarray(surface_node_ids, dtype=INT_DTYPE),
        torch_np.asarray(surface_edge_mask, dtype=bool),
        torch_np.asarray(surface_nodes, dtype=FLOAT_DTYPE),
        torch_np.asarray(reference_level, dtype=FLOAT_DTYPE),
        torch_np.asarray(flat_surface, dtype=bool),
    )


def _surface_topology_from_node_path(
    nodes: Array,
    boundary_edges: Array,
    surface_node_ids: Array,
    tol: float = 1e-8,
) -> tuple[Array, Array, Array, Array, Array]:
    """Build surface metadata from an explicit ordered boundary-node path."""

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


def locate_points_in_triangles(
    nodes: Array,
    cells: Array,
    points: Array,
    tol: float = 1e-4,
) -> tuple[Array, Array]:
    """Locate points in a triangle mesh and return containing cell ids and barycentric weights."""

    points_np = np.asarray(points, dtype=float)
    cells_np = np.asarray(cells, dtype=np.int32)
    nodes_np = np.asarray(nodes, dtype=float)

    cell_ids = np.full(points_np.shape[0], -1, dtype=np.int32)
    weights = np.zeros((points_np.shape[0], 3), dtype=float)

    for point_id, point in enumerate(points_np):
        distances = np.linalg.norm(nodes_np - point, axis=1)
        nearest_node = int(np.argmin(distances))
        if distances[nearest_node] <= tol:
            adjacent_cells = np.flatnonzero(np.any(cells_np == nearest_node, axis=1))
            if adjacent_cells.size > 0:
                cell_id = int(adjacent_cells[0])
                local_node = int(np.flatnonzero(cells_np[cell_id] == nearest_node)[0])
                cell_ids[point_id] = cell_id
                weights[point_id, local_node] = 1.0
                continue

        for cell_id, cell in enumerate(cells_np):
            tri = nodes_np[cell]
            transform = np.column_stack((tri[1] - tri[0], tri[2] - tri[0]))
            local = np.linalg.solve(transform, point - tri[0])
            bary = np.array([1.0 - local.sum(), local[0], local[1]])
            if np.all(bary >= -tol) and np.all(bary <= 1.0 + tol):
                cell_ids[point_id] = cell_id
                weights[point_id] = bary
                break

    if np.any(cell_ids < 0):
        raise ValueError("some points could not be located in the mesh")

    return (
        torch_np.asarray(cell_ids, dtype=INT_DTYPE),
        torch_np.asarray(weights, dtype=FLOAT_DTYPE),
    )


def _q1_shape_values_np(xi: float, eta: float) -> np.ndarray:
    return 0.25 * np.asarray(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ],
        dtype=float,
    )


def _q1_shape_gradients_np(xi: float, eta: float) -> np.ndarray:
    dxi = 0.25 * np.asarray(
        [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
        dtype=float,
    )
    deta = 0.25 * np.asarray(
        [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)],
        dtype=float,
    )
    return np.stack((dxi, deta), axis=1)


def _locate_point_on_cell_edges(cell_nodes: np.ndarray, point: np.ndarray, tol: float) -> np.ndarray | None:
    weights = np.zeros((cell_nodes.shape[0],), dtype=float)
    for start, stop in _cell_edge_pairs(int(cell_nodes.shape[0])):
        edge = cell_nodes[stop] - cell_nodes[start]
        edge_length_sq = float(np.dot(edge, edge))
        if edge_length_sq <= tol * tol:
            continue
        projection = float(np.dot(point - cell_nodes[start], edge) / edge_length_sq)
        if projection < -tol or projection > 1.0 + tol:
            continue
        projection = min(max(projection, 0.0), 1.0)
        closest = cell_nodes[start] + projection * edge
        if float(np.linalg.norm(point - closest)) > tol:
            continue
        weights[start] = 1.0 - projection
        weights[stop] = projection
        return weights
    return None


def _locate_point_in_quad(quad_nodes: np.ndarray, point: np.ndarray, tol: float) -> np.ndarray | None:
    """Return bilinear Q1 weights for a point in a convex quadrilateral."""

    xi = 0.0
    eta = 0.0
    for _ in range(32):
        shape = _q1_shape_values_np(xi, eta)
        mapped = shape @ quad_nodes
        residual = mapped - point
        inside_reference = (
            xi >= -1.0 - tol
            and xi <= 1.0 + tol
            and eta >= -1.0 - tol
            and eta <= 1.0 + tol
        )
        if float(np.linalg.norm(residual)) <= tol and inside_reference:
            break

        gradients = _q1_shape_gradients_np(xi, eta)
        jacobian = gradients.T @ quad_nodes
        try:
            step = np.linalg.solve(jacobian, residual)
        except np.linalg.LinAlgError:
            return None
        xi -= float(step[0])
        eta -= float(step[1])
        if float(np.linalg.norm(step)) <= 1e-12:
            break

    if xi < -1.0 - tol or xi > 1.0 + tol or eta < -1.0 - tol or eta > 1.0 + tol:
        return None
    xi = min(max(xi, -1.0), 1.0)
    eta = min(max(eta, -1.0), 1.0)
    shape = _q1_shape_values_np(xi, eta)
    if np.any(shape < -tol) or np.any(shape > 1.0 + tol):
        return None
    return shape


def locate_points_in_quadrilaterals(
    nodes: Array,
    cells: Array,
    points: Array,
    tol: float = 1e-4,
) -> tuple[Array, Array]:
    """Locate points in a quadrilateral mesh and return containing cell ids and Q1 weights."""

    points_np = np.asarray(points, dtype=float)
    cells_np = np.asarray(cells, dtype=np.int32)
    nodes_np = np.asarray(nodes, dtype=float)

    cell_ids = np.full(points_np.shape[0], -1, dtype=np.int32)
    weights = np.zeros((points_np.shape[0], 4), dtype=float)

    for point_id, point in enumerate(points_np):
        distances = np.linalg.norm(nodes_np - point, axis=1)
        nearest_node = int(np.argmin(distances))
        if distances[nearest_node] <= tol:
            adjacent_cells = np.flatnonzero(np.any(cells_np == nearest_node, axis=1))
            if adjacent_cells.size > 0:
                cell_id = int(adjacent_cells[0])
                local_node = int(np.flatnonzero(cells_np[cell_id] == nearest_node)[0])
                cell_ids[point_id] = cell_id
                weights[point_id, local_node] = 1.0
                continue

        for cell_id, cell in enumerate(cells_np):
            quad = nodes_np[cell]
            quad_weights = _locate_point_on_cell_edges(quad, point, tol)
            if quad_weights is None:
                quad_weights = _locate_point_in_quad(quad, point, tol)
            if quad_weights is None:
                continue
            cell_ids[point_id] = cell_id
            weights[point_id] = quad_weights
            break

    if np.any(cell_ids < 0):
        raise ValueError("some points could not be located in the mesh")

    return (
        torch_np.asarray(cell_ids, dtype=INT_DTYPE),
        torch_np.asarray(weights, dtype=FLOAT_DTYPE),
    )


def refine_triangle_mesh(nodes: Array, cells: Array) -> tuple[Array, Array, Array, dict[tuple[int, int], int]]:
    """Uniformly refine each triangle into four H2-style child triangles."""

    nodes_np = np.asarray(nodes, dtype=float)
    cells_np = np.asarray(cells, dtype=np.int32)

    refined_nodes = nodes_np.tolist()
    edge_midpoints: dict[tuple[int, int], int] = {}
    refined_cells: list[list[int]] = []
    parent_cells: list[int] = []

    def midpoint_node(node_a: int, node_b: int) -> int:
        edge = (min(node_a, node_b), max(node_a, node_b))
        if edge in edge_midpoints:
            return edge_midpoints[edge]

        midpoint = 0.5 * (nodes_np[edge[0]] + nodes_np[edge[1]])
        edge_midpoints[edge] = len(refined_nodes)
        refined_nodes.append(midpoint.tolist())
        return edge_midpoints[edge]

    for parent_id, (n0, n1, n2) in enumerate(cells_np):
        n01 = midpoint_node(int(n0), int(n1))
        n12 = midpoint_node(int(n1), int(n2))
        n20 = midpoint_node(int(n2), int(n0))

        refined_cells.extend(
            [
                [int(n0), n01, n20],
                [int(n1), n12, n01],
                [int(n2), n20, n12],
                [n01, n12, n20],
            ]
        )
        parent_cells.extend([parent_id] * 4)

    return (
        torch_np.asarray(refined_nodes, dtype=FLOAT_DTYPE),
        torch_np.asarray(refined_cells, dtype=INT_DTYPE),
        torch_np.asarray(parent_cells, dtype=INT_DTYPE),
        edge_midpoints,
    )


def _orient_triangle_ccw(nodes: list[list[float]] | np.ndarray, triangle: list[int]) -> list[int]:
    """Return the triangle with positive signed area."""

    tri_nodes = np.asarray([nodes[node_id] for node_id in triangle], dtype=float)
    edge_1 = tri_nodes[1] - tri_nodes[0]
    edge_2 = tri_nodes[2] - tri_nodes[0]
    cross = edge_1[0] * edge_2[1] - edge_1[1] * edge_2[0]
    if cross > 0.0:
        return triangle
    return [triangle[0], triangle[2], triangle[1]]


def insert_surface_points_into_triangle_mesh(
    nodes: Array,
    cells: Array,
    boundary_edges: Array,
    boundary_edge_cells: Array,
    surface_node_ids: Array,
    points: Array,
    tol: float = 1e-4,
) -> tuple[Array, Array, Array, Array, Array]:
    """Insert points that lie on the top surface by splitting boundary-adjacent cells."""

    nodes_np = np.asarray(nodes, dtype=float)
    cells_np = np.asarray(cells, dtype=np.int32)
    boundary_edges_np = np.asarray(boundary_edges, dtype=np.int32)
    boundary_edge_cells_np = np.asarray(boundary_edge_cells, dtype=np.int32)
    surface_node_ids_np = np.asarray(surface_node_ids, dtype=np.int32)
    points_np = np.asarray(points, dtype=float)

    if points_np.shape[0] == 0 or surface_node_ids_np.shape[0] < 2:
        return (
            torch_np.asarray(nodes_np, dtype=FLOAT_DTYPE),
            torch_np.asarray(cells_np, dtype=INT_DTYPE),
            torch_np.arange(cells_np.shape[0], dtype=INT_DTYPE),
            torch_np.empty((0,), dtype=INT_DTYPE),
            torch_np.asarray(surface_node_ids_np, dtype=INT_DTYPE),
        )

    boundary_cell_by_edge = {
        tuple(sorted((int(edge[0]), int(edge[1])))): int(cell_id)
        for edge, cell_id in zip(boundary_edges_np.tolist(), boundary_edge_cells_np.tolist(), strict=False)
    }
    surface_segments = [(int(start), int(stop)) for start, stop in zip(surface_node_ids_np[:-1], surface_node_ids_np[1:], strict=False)]

    point_node_ids = np.full(points_np.shape[0], -1, dtype=np.int32)
    segment_points: dict[int, list[tuple[float, int, np.ndarray]]] = {}

    for point_id, point in enumerate(points_np):
        distances = np.linalg.norm(nodes_np - point, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= tol:
            point_node_ids[point_id] = nearest
            continue

        best_segment = -1
        best_projection = 0.0
        best_distance = float("inf")
        for segment_id, (start_node, stop_node) in enumerate(surface_segments):
            start = nodes_np[start_node]
            stop = nodes_np[stop_node]
            edge = stop - start
            edge_length_sq = float(np.dot(edge, edge))
            if edge_length_sq <= tol:
                continue
            projection = float(np.dot(point - start, edge) / edge_length_sq)
            if projection < -tol or projection > 1.0 + tol:
                continue
            closest = start + projection * edge
            distance = float(np.linalg.norm(point - closest))
            if distance <= tol and distance < best_distance:
                best_segment = segment_id
                best_projection = min(max(projection, 0.0), 1.0)
                best_distance = distance

        if best_segment < 0:
            raise ValueError("surface point could not be matched to a top boundary segment")

        start_node, stop_node = surface_segments[best_segment]
        if best_projection <= tol:
            point_node_ids[point_id] = start_node
            continue
        if best_projection >= 1.0 - tol:
            point_node_ids[point_id] = stop_node
            continue
        segment_points.setdefault(best_segment, []).append((best_projection, point_id, point))

    if not segment_points:
        return (
            torch_np.asarray(nodes_np, dtype=FLOAT_DTYPE),
            torch_np.asarray(cells_np, dtype=INT_DTYPE),
            torch_np.arange(cells_np.shape[0], dtype=INT_DTYPE),
            torch_np.asarray(point_node_ids, dtype=INT_DTYPE),
            torch_np.asarray(surface_node_ids_np, dtype=INT_DTYPE),
        )

    refined_nodes = nodes_np.tolist()
    split_cells: dict[int, tuple[int, int, list[int]]] = {}

    for segment_id, segment_data in segment_points.items():
        start_node, stop_node = surface_segments[segment_id]
        boundary_edge = tuple(sorted((start_node, stop_node)))
        cell_id = boundary_cell_by_edge[boundary_edge]
        if cell_id in split_cells:
            raise ValueError("surface cell has multiple top boundary segments; unsupported topology")

        ordered_points = sorted(segment_data, key=lambda item: item[0])
        inserted_node_ids: list[int] = []
        previous_projection = None
        previous_node_id = None
        for projection, point_id, point in ordered_points:
            if previous_projection is not None and abs(projection - previous_projection) <= tol:
                assert previous_node_id is not None
                point_node_ids[point_id] = previous_node_id
                continue
            node_id = len(refined_nodes)
            refined_nodes.append(point.tolist())
            point_node_ids[point_id] = node_id
            inserted_node_ids.append(node_id)
            previous_projection = projection
            previous_node_id = node_id

        split_cells[cell_id] = (start_node, stop_node, inserted_node_ids)

    refined_cells: list[list[int]] = []
    parent_cells: list[int] = []
    for cell_id, cell in enumerate(cells_np.tolist()):
        split = split_cells.get(cell_id)
        if split is None:
            refined_cells.append([int(cell[0]), int(cell[1]), int(cell[2])])
            parent_cells.append(cell_id)
            continue

        start_node, stop_node, inserted_node_ids = split
        interior_candidates = [int(node_id) for node_id in cell if int(node_id) not in (start_node, stop_node)]
        if len(interior_candidates) != 1:
            raise ValueError("surface boundary cell must contain exactly one interior vertex")
        interior_node = interior_candidates[0]

        segment_nodes = [start_node, *inserted_node_ids, stop_node]
        for segment_start, segment_stop in zip(segment_nodes[:-1], segment_nodes[1:], strict=False):
            refined_cells.append(_orient_triangle_ccw(refined_nodes, [segment_start, segment_stop, interior_node]))
            parent_cells.append(cell_id)

    if np.any(point_node_ids < 0):
        raise ValueError("some inserted surface points did not receive node ids")

    refined_surface_node_ids = [int(surface_node_ids_np[0])]
    for segment_id, (_, stop_node) in enumerate(surface_segments):
        segment_data = segment_points.get(segment_id, [])
        if segment_data:
            ordered_point_ids = sorted(segment_data, key=lambda item: item[0])
            refined_surface_node_ids.extend([int(point_node_ids[point_id]) for _, point_id, _ in ordered_point_ids])
        refined_surface_node_ids.append(int(stop_node))

    return (
        torch_np.asarray(refined_nodes, dtype=FLOAT_DTYPE),
        torch_np.asarray(refined_cells, dtype=INT_DTYPE),
        torch_np.asarray(parent_cells, dtype=INT_DTYPE),
        torch_np.asarray(point_node_ids, dtype=INT_DTYPE),
        torch_np.asarray(refined_surface_node_ids, dtype=INT_DTYPE),
    )


def expand_columnar_triangle_mesh(
    nodes: Array,
    cells: Array,
    tol: float = 1e-6,
) -> tuple[Array, Array, Array] | None:
    """Expand structured terrain-following single-triangle cells into full quad strips."""

    nodes_np = np.asarray(nodes, dtype=float)
    cells_np = np.asarray(cells, dtype=np.int32)
    if nodes_np.shape[0] == 0 or cells_np.shape[0] == 0:
        return None

    order = np.argsort(nodes_np[:, 0], kind="mergesort")
    sorted_ids = order.tolist()
    columns: list[list[int]] = []
    for node_id in sorted_ids:
        x_value = nodes_np[node_id, 0]
        if not columns:
            columns.append([int(node_id)])
            continue
        reference_x = nodes_np[columns[-1][0], 0]
        if abs(x_value - reference_x) <= tol:
            columns[-1].append(int(node_id))
        else:
            columns.append([int(node_id)])

    if len(columns) < 2:
        return None

    column_sizes = [len(column) for column in columns]
    if len(columns) < 10 or min(column_sizes) < 2 or max(column_sizes) - min(column_sizes) > 1:
        return None

    column_node_ids: list[list[int]] = []
    node_to_column = np.full((nodes_np.shape[0],), -1, dtype=np.int32)
    node_to_row = np.full((nodes_np.shape[0],), -1, dtype=np.int32)
    expanded_nodes = nodes_np.tolist()
    for column_id, column in enumerate(columns):
        ordered_column = sorted(column, key=lambda node_id: (-nodes_np[node_id, 1], nodes_np[node_id, 0]))
        column_node_ids.append(ordered_column)
        for row_id, node_id in enumerate(ordered_column):
            node_to_column[node_id] = column_id
            node_to_row[node_id] = row_id

    expanded_cells: list[list[int]] = []
    parent_cells: list[int] = []
    for parent_id, cell in enumerate(cells_np):
        column_ids = node_to_column[cell]
        row_ids = node_to_row[cell]
        if np.any(column_ids < 0) or np.any(row_ids < 0):
            return None

        unique_columns, counts = np.unique(column_ids, return_counts=True)
        if unique_columns.shape[0] != 2 or unique_columns[1] != unique_columns[0] + 1:
            return None
        if tuple(sorted(counts.tolist())) != (1, 2):
            return None

        pair_column = int(unique_columns[np.argmax(counts)])
        lone_column = int(unique_columns[np.argmin(counts)])
        pair_mask = column_ids == pair_column
        pair_nodes = cell[pair_mask]
        pair_rows = row_ids[pair_mask]
        row_order = np.argsort(pair_rows)
        pair_nodes = pair_nodes[row_order]
        pair_rows = pair_rows[row_order]
        if int(pair_rows[1]) != int(pair_rows[0]) + 1:
            return None

        lone_node = int(cell[~pair_mask][0])
        lone_row = int(row_ids[~pair_mask][0])
        if lone_row == int(pair_rows[0]):
            missing_row = int(pair_rows[1])
        elif lone_row == int(pair_rows[1]):
            missing_row = int(pair_rows[0])
        else:
            return None

        if missing_row >= len(column_node_ids[lone_column]):
            if missing_row != len(column_node_ids[lone_column]) or len(column_node_ids[lone_column]) + 1 != max(column_sizes):
                return None
            pair_surface = nodes_np[column_node_ids[pair_column][0], 1]
            pair_depth = nodes_np[int(pair_nodes[1]), 1] - pair_surface
            lone_surface = nodes_np[column_node_ids[lone_column][0], 1]
            new_node_id = len(expanded_nodes)
            expanded_nodes.append([float(nodes_np[column_node_ids[lone_column][0], 0]), float(lone_surface + pair_depth)])
            column_node_ids[lone_column].append(new_node_id)
        missing_node = int(column_node_ids[lone_column][missing_row])
        if missing_node in cell.tolist():
            return None

        expanded_cells.append([int(cell[0]), int(cell[1]), int(cell[2])])
        parent_cells.append(parent_id)
        expanded_cells.append(_orient_triangle_ccw(expanded_nodes, [missing_node, int(pair_nodes[0]), int(pair_nodes[1])]))
        parent_cells.append(parent_id)

    return (
        torch_np.asarray(expanded_nodes, dtype=FLOAT_DTYPE),
        torch_np.asarray(expanded_cells, dtype=INT_DTYPE),
        torch_np.asarray(parent_cells, dtype=INT_DTYPE),
    )


def build_quadratic_triangle_mesh(
    nodes: Array,
    cells: Array,
    boundary_edges: Array,
) -> tuple[Array, Array, Array]:
    """Build shared-edge P2 topology on top of a linear triangle mesh."""

    nodes_np = np.asarray(nodes, dtype=float)
    cells_np = np.asarray(cells, dtype=np.int32)
    boundary_edges_np = np.asarray(boundary_edges, dtype=np.int32)

    quadratic_nodes = nodes_np.tolist()
    edge_midpoints: dict[tuple[int, int], int] = {}

    def midpoint_node(node_a: int, node_b: int) -> int:
        edge = (min(node_a, node_b), max(node_a, node_b))
        if edge in edge_midpoints:
            return edge_midpoints[edge]

        midpoint = 0.5 * (nodes_np[edge[0]] + nodes_np[edge[1]])
        edge_midpoints[edge] = len(quadratic_nodes)
        quadratic_nodes.append(midpoint.tolist())
        return edge_midpoints[edge]

    quadratic_cells: list[list[int]] = []
    for n0, n1, n2 in cells_np:
        m01 = midpoint_node(int(n0), int(n1))
        m12 = midpoint_node(int(n1), int(n2))
        m20 = midpoint_node(int(n2), int(n0))
        quadratic_cells.append([int(n0), int(n1), int(n2), m01, m12, m20])

    quadratic_boundary_edges: list[list[int]] = []
    for node_a, node_b in boundary_edges_np:
        midpoint = midpoint_node(int(node_a), int(node_b))
        quadratic_boundary_edges.append([int(node_a), int(node_b), midpoint])

    return (
        torch_np.asarray(quadratic_nodes, dtype=FLOAT_DTYPE),
        torch_np.asarray(quadratic_cells, dtype=INT_DTYPE),
        torch_np.asarray(quadratic_boundary_edges, dtype=INT_DTYPE),
    )


@dataclass(frozen=True)
class Mesh:
    """2D triangle or quadrilateral mesh with precomputed topology data."""

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
    cell_areas: Array

    @classmethod
    def from_arrays(cls, nodes: Array, cells: Array, *, surface_node_ids: Array | None = None) -> "Mesh":
        """Build a mesh from node coordinates and triangle or quadrilateral connectivity."""

        node_array = torch_np.asarray(nodes, dtype=FLOAT_DTYPE)
        cell_array = torch_np.asarray(cells, dtype=INT_DTYPE)
        surface_node_array = None if surface_node_ids is None else np.asarray(surface_node_ids, dtype=np.int32)

        if node_array.ndim != 2 or node_array.shape[1] != 2:
            raise ValueError("nodes must have shape (num_nodes, 2)")
        if cell_array.ndim != 2 or cell_array.shape[1] not in (3, 4):
            raise ValueError("cells must have shape (num_cells, 3) or (num_cells, 4)")
        node_array_np = np.asarray(node_array, dtype=NP_FLOAT_DTYPE)
        cell_array_np = np.asarray(cell_array, dtype=np.int32)
        if np.any(cell_array_np < 0):
            raise ValueError("cells contain negative node indices")
        if cell_array_np.size and np.any(cell_array_np >= int(node_array.shape[0])):
            raise ValueError("cells reference nodes outside the mesh")

        used_node_ids = np.unique(cell_array_np.reshape(-1))
        if used_node_ids.size != int(node_array.shape[0]):
            remapped_ids = np.full((int(node_array.shape[0]),), -1, dtype=np.int32)
            remapped_ids[used_node_ids] = np.arange(used_node_ids.size, dtype=np.int32)
            node_array_np = node_array_np[used_node_ids]
            node_array = torch_np.asarray(node_array_np, dtype=FLOAT_DTYPE)
            cell_array_np = remapped_ids[cell_array_np]
            cell_array = torch_np.asarray(cell_array_np, dtype=INT_DTYPE)
            if surface_node_array is not None:
                surface_node_array = remapped_ids[surface_node_array]

        cell_areas = cell_areas_2d(node_array, cell_array)
        if np.any(np.asarray(cell_areas, dtype=float) <= 0.0):
            raise ValueError("cells must define non-degenerate 2D polygons")

        boundary_edges, boundary_edge_cells = _boundary_topology(cell_array)
        boundary_edge_centers, boundary_edge_lengths, boundary_edge_normals = _boundary_geometry(
            node_array,
            cell_array,
            boundary_edges,
            boundary_edge_cells,
        )
        if surface_node_array is None:
            surface_node_ids_out, surface_edge_mask, surface_nodes, surface_reference_level, flat_surface = _surface_topology(
                node_array,
                boundary_edges,
            )
        else:
            (
                surface_node_ids_out,
                surface_edge_mask,
                surface_nodes,
                surface_reference_level,
                flat_surface,
            ) = _surface_topology_from_node_path(
                node_array,
                boundary_edges,
                torch_np.asarray(surface_node_array, dtype=INT_DTYPE),
            )
        return cls(
            nodes=node_array,
            cells=cell_array,
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
            cell_areas=cell_areas,
        )

    @classmethod
    def from_meshio(cls, mesh: meshio.Mesh) -> "Mesh":
        """Create a mesh from a meshio mesh object."""

        if "quad" in mesh.cells_dict:
            cells = mesh.cells_dict["quad"]
        elif "quadrilateral" in mesh.cells_dict:
            cells = mesh.cells_dict["quadrilateral"]
        else:
            try:
                cells = mesh.cells_dict["triangle"]
            except KeyError as exc:
                raise ValueError("meshio mesh does not contain triangle or quadrilateral cells") from exc

        nodes = mesh.points[:, :2]
        return cls.from_arrays(nodes=nodes, cells=cells)

    @classmethod
    def from_file(cls, path: str | Path) -> "Mesh":
        """Load a 2D triangle or quadrilateral mesh via meshio."""

        return cls.from_meshio(meshio.read(path))

    @property
    def node_count(self) -> int:
        """Number of mesh vertices."""

        return int(self.nodes.shape[0])

    @property
    def cell_count(self) -> int:
        """Number of mesh cells."""

        return int(self.cells.shape[0])

    @property
    def cell_node_count(self) -> int:
        """Number of vertices per cell."""

        return int(self.cells.shape[1])

    @property
    def is_triangle_mesh(self) -> bool:
        """Whether all cells are triangles."""

        return self.cell_node_count == 3

    @property
    def is_quadrilateral_mesh(self) -> bool:
        """Whether all cells are quadrilaterals."""

        return self.cell_node_count == 4

    @property
    def is_flat_surface(self) -> bool:
        """Whether the inferred top boundary is flat within tolerance."""

        return bool(self.flat_surface)

    def locate_points(self, points: Array, tol: float = 1e-4) -> tuple[Array, Array]:
        """Return containing cell ids and interpolation weights for points."""

        if self.is_triangle_mesh:
            return locate_points_in_triangles(self.nodes, self.cells, points, tol=tol)
        return locate_points_in_quadrilaterals(self.nodes, self.cells, points, tol=tol)

    def refine_uniform(self) -> tuple["Mesh", Array]:
        """Uniformly refine all triangles and return the refined mesh and parent ids."""

        if not self.is_triangle_mesh:
            raise ValueError("uniform refinement is currently implemented for triangle meshes only")
        refined_nodes, refined_cells, parent_cells, edge_midpoints = refine_triangle_mesh(self.nodes, self.cells)
        refined_surface_node_ids = [int(self.surface_node_ids[0])]
        surface_node_ids_np = np.asarray(self.surface_node_ids, dtype=np.int32)
        for start_node, stop_node in zip(surface_node_ids_np[:-1], surface_node_ids_np[1:], strict=False):
            midpoint = edge_midpoints[(min(int(start_node), int(stop_node)), max(int(start_node), int(stop_node)))]
            refined_surface_node_ids.extend((midpoint, int(stop_node)))
        return (
            Mesh.from_arrays(
                refined_nodes,
                refined_cells,
                surface_node_ids=torch_np.asarray(refined_surface_node_ids, dtype=INT_DTYPE),
            ),
            parent_cells,
        )

    def insert_surface_points(self, points: Array, tol: float = 1e-4) -> tuple["Mesh", Array, Array]:
        """Insert top-surface points as explicit mesh vertices and return parent-cell ids."""

        if not self.is_triangle_mesh:
            raise ValueError("surface point insertion is currently implemented for triangle meshes only")
        refined_nodes, refined_cells, parent_cells, point_node_ids, refined_surface_node_ids = (
            insert_surface_points_into_triangle_mesh(
                self.nodes,
                self.cells,
                self.boundary_edges,
                self.boundary_edge_cells,
                self.surface_node_ids,
                points,
                tol=tol,
            )
        )
        return (
            Mesh.from_arrays(refined_nodes, refined_cells, surface_node_ids=refined_surface_node_ids),
            parent_cells,
            point_node_ids,
        )

    def expand_columnar_cells(self, tol: float = 1e-6) -> tuple["Mesh", Array] | None:
        """Expand structured terrain-following single-triangle strips into full triangularized quads."""

        if not self.is_triangle_mesh:
            return None
        expanded = expand_columnar_triangle_mesh(self.nodes, self.cells, tol=tol)
        if expanded is None:
            return None
        expanded_nodes, expanded_cells, parent_cells = expanded
        return Mesh.from_arrays(expanded_nodes, expanded_cells, surface_node_ids=self.surface_node_ids), parent_cells

    def build_quadratic_topology(self) -> tuple[Array, Array, Array]:
        """Return shared-edge P2 nodes, cell connectivity, and boundary-edge connectivity."""

        if not self.is_triangle_mesh:
            raise ValueError("quadratic topology is currently implemented for triangle meshes only")
        return build_quadratic_triangle_mesh(self.nodes, self.cells, self.boundary_edges)
