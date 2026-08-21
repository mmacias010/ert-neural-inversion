"""Spatial and temporal regularization interfaces for Deepert inversions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import scipy.sparse as sp

from deepert.mesh import Mesh


class SpatialRegularization(Protocol):
    """Interface for spatial regularization terms that have a linear operator."""

    name: str

    def matrix(self, forward, n_cells: int, *, z_weight: float = 1.0) -> sp.csr_matrix:
        """Build the spatial regularization matrix for the current parameter mesh."""

    def linearized_system(
        self,
        forward,
        current_model: np.ndarray,
        n_cells: int,
        *,
        reference_roughness: np.ndarray,
        scale: float = 1.0,
        z_weight: float = 1.0,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        """Build ``(A, b)`` for the current spatial regularization update."""


class TemporalRegularization(Protocol):
    """Interface for temporal regularization terms that have a linear operator."""

    name: str

    def matrix(self, n_cells: int, n_times: int, *, scale: float = 1.0) -> sp.csr_matrix:
        """Build the temporal regularization matrix."""

    def linearized_system(
        self,
        current_model: np.ndarray,
        n_cells: int,
        n_times: int,
        *,
        scale: float = 1.0,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        """Build ``(A, b)`` for the current temporal regularization update."""


def regularization_mesh(forward) -> Mesh:
    """Return the mesh used for spatial regularization."""

    mesh = getattr(forward, "regularization_mesh", None)
    if mesh is not None:
        if isinstance(mesh, Mesh):
            return mesh
        raise TypeError("forward.regularization_mesh must be a deepert Mesh")
    if hasattr(forward, "_resolved_mesh"):
        resolved = forward._resolved_mesh()
        if isinstance(resolved, Mesh):
            return resolved
    mesh = getattr(forward, "mesh", None)
    if isinstance(mesh, Mesh):
        return mesh
    raise TypeError("forward must expose a Mesh for first-order regularization")


def cell_edges(cell: np.ndarray) -> list[tuple[int, int]]:
    """Return unordered edge node-id pairs for a polygon cell."""

    return [
        (int(cell[index]), int(cell[(index + 1) % cell.size]))
        for index in range(cell.size)
    ]


def first_order_constraint_matrix(mesh: Mesh, *, z_weight: float = 1.0) -> sp.csr_matrix:
    """Build first-order neighbor constraints for cell models."""

    nodes = np.asarray(mesh.nodes, dtype=float)
    cells = np.asarray(mesh.cells, dtype=np.int32)
    edge_cells: dict[tuple[int, int], list[int]] = {}
    for cell_index, cell in enumerate(cells):
        for edge in cell_edges(cell):
            key = tuple(sorted(edge))
            edge_cells.setdefault(key, []).append(cell_index)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    row = 0
    for edge, owners in edge_cells.items():
        if len(owners) != 2:
            continue
        p0, p1 = nodes[list(edge)]
        tangent = p1 - p0
        length = float(np.linalg.norm(tangent))
        if length <= 0.0:
            continue
        normal_z = abs(float(tangent[0])) / length
        weight = 1.0 + normal_z * (float(z_weight) - 1.0)
        left, right = owners
        rows.extend((row, row))
        cols.extend((left, right))
        data.extend((weight, -weight))
        row += 1

    return sp.coo_matrix((data, (rows, cols)), shape=(row, int(mesh.cell_count))).tocsr()


def structure_guided_constraint_matrix(
    mesh: Mesh,
    structural_ids: np.ndarray,
    *,
    z_weight: float = 1.0,
    cross_structure_weight: float = 0.05,
) -> sp.csr_matrix:
    """Build first-order constraints with weakened weights across structural boundaries."""

    labels = np.asarray(structural_ids, dtype=np.int32).reshape(-1)
    if labels.shape != (int(mesh.cell_count),):
        raise ValueError(
            "structural_ids must have one value per regularization cell "
            f"({labels.shape} != ({int(mesh.cell_count)},))"
        )
    if cross_structure_weight < 0.0:
        raise ValueError("cross_structure_weight must be non-negative")

    nodes = np.asarray(mesh.nodes, dtype=float)
    cells = np.asarray(mesh.cells, dtype=np.int32)
    edge_cells: dict[tuple[int, int], list[int]] = {}
    for cell_index, cell in enumerate(cells):
        for edge in cell_edges(cell):
            key = tuple(sorted(edge))
            edge_cells.setdefault(key, []).append(cell_index)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    row = 0
    for edge, owners in edge_cells.items():
        if len(owners) != 2:
            continue
        p0, p1 = nodes[list(edge)]
        tangent = p1 - p0
        length = float(np.linalg.norm(tangent))
        if length <= 0.0:
            continue
        normal_z = abs(float(tangent[0])) / length
        direction_weight = 1.0 + normal_z * (float(z_weight) - 1.0)
        left, right = owners
        structural_weight = 1.0 if labels[left] == labels[right] else float(cross_structure_weight)
        weight = direction_weight * structural_weight
        if weight == 0.0:
            continue
        rows.extend((row, row))
        cols.extend((left, right))
        data.extend((weight, -weight))
        row += 1

    return sp.coo_matrix((data, (rows, cols)), shape=(row, int(mesh.cell_count))).tocsr()


@dataclass(frozen=True)
class IdentitySpatialRegularization:
    """Damping regularization in model space."""

    name: str = "identity"

    def matrix(self, forward, n_cells: int, *, z_weight: float = 1.0) -> sp.csr_matrix:
        return sp.eye(int(n_cells), format="csr")

    def linearized_system(
        self,
        forward,
        current_model: np.ndarray,
        n_cells: int,
        *,
        reference_roughness: np.ndarray,
        scale: float = 1.0,
        z_weight: float = 1.0,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        matrix = float(scale) * self.matrix(forward, n_cells, z_weight=z_weight)
        roughness = matrix @ np.asarray(current_model, dtype=float).reshape(-1)
        reference = float(scale) * np.asarray(reference_roughness, dtype=float).reshape(-1)
        return matrix, reference - roughness


@dataclass(frozen=True)
class FirstOrderSpatialRegularization:
    """First-order neighbor smoothness regularization."""

    name: str = "first_order"

    def matrix(self, forward, n_cells: int, *, z_weight: float = 1.0) -> sp.csr_matrix:
        matrix = first_order_constraint_matrix(
            regularization_mesh(forward),
            z_weight=z_weight,
        )
        if matrix.shape[1] != int(n_cells):
            raise ValueError(
                "regularization mesh cell count does not match inversion model size "
                f"({matrix.shape[1]} != {int(n_cells)})"
            )
        return matrix

    def linearized_system(
        self,
        forward,
        current_model: np.ndarray,
        n_cells: int,
        *,
        reference_roughness: np.ndarray,
        scale: float = 1.0,
        z_weight: float = 1.0,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        matrix = float(scale) * self.matrix(forward, n_cells, z_weight=z_weight)
        roughness = matrix @ np.asarray(current_model, dtype=float).reshape(-1)
        reference = float(scale) * np.asarray(reference_roughness, dtype=float).reshape(-1)
        return matrix, reference - roughness


@dataclass(frozen=True)
class StructuralPriorSpatialRegularization:
    """Structure-guided first-order smoothness using known subsurface units."""

    name: str = "structural_prior"

    def matrix(self, forward, n_cells: int, *, z_weight: float = 1.0) -> sp.csr_matrix:
        structural_ids = getattr(forward, "structural_prior_cell_ids", None)
        if structural_ids is None:
            structural_ids = getattr(forward, "structure_cell_ids", None)
        if structural_ids is None:
            raise ValueError(
                "structural_prior regularization requires forward.structural_prior_cell_ids "
                "with one structural-unit id per inversion cell"
            )
        cross_weight = float(getattr(forward, "structural_cross_weight", 0.05))
        matrix = structure_guided_constraint_matrix(
            regularization_mesh(forward),
            np.asarray(structural_ids, dtype=np.int32),
            z_weight=z_weight,
            cross_structure_weight=cross_weight,
        )
        if matrix.shape[1] != int(n_cells):
            raise ValueError(
                "regularization mesh cell count does not match inversion model size "
                f"({matrix.shape[1]} != {int(n_cells)})"
            )
        return matrix

    def linearized_system(
        self,
        forward,
        current_model: np.ndarray,
        n_cells: int,
        *,
        reference_roughness: np.ndarray,
        scale: float = 1.0,
        z_weight: float = 1.0,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        matrix = float(scale) * self.matrix(forward, n_cells, z_weight=z_weight)
        roughness = matrix @ np.asarray(current_model, dtype=float).reshape(-1)
        reference = float(scale) * np.asarray(reference_roughness, dtype=float).reshape(-1)
        return matrix, reference - roughness


@dataclass(frozen=True)
class ModelDifferenceSpatialRegularization(FirstOrderSpatialRegularization):
    """Spatial smoothness of time-lapse model differences."""

    name: str = "model_difference_smoothness"


@dataclass(frozen=True)
class FirstOrderSpatialTVRegularization:
    """IRLS-smoothed first-order spatial TV/L1 regularization."""

    epsilon: float = 1.0e-3
    name: str = "first_order_tv"

    def matrix(self, forward, n_cells: int, *, z_weight: float = 1.0) -> sp.csr_matrix:
        return FirstOrderSpatialRegularization().matrix(forward, n_cells, z_weight=z_weight)

    def _sqrt_irls_weight(self, residual: np.ndarray) -> np.ndarray:
        eps = max(float(self.epsilon), np.finfo(float).eps)
        return np.power(residual**2 + eps**2, -0.25)

    def linearized_system(
        self,
        forward,
        current_model: np.ndarray,
        n_cells: int,
        *,
        reference_roughness: np.ndarray,
        scale: float = 1.0,
        z_weight: float = 1.0,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        base = self.matrix(forward, n_cells, z_weight=z_weight)
        current = np.asarray(current_model, dtype=float).reshape(-1)
        reference = np.asarray(reference_roughness, dtype=float).reshape(-1)
        residual = np.asarray(base @ current - reference, dtype=float)
        sqrt_weight = self._sqrt_irls_weight(residual)
        matrix = sp.diags(float(scale) * sqrt_weight, format="csr") @ base
        rhs = float(scale) * sqrt_weight * (reference - base @ current)
        return matrix.tocsr(), np.asarray(rhs, dtype=float)


@dataclass(frozen=True)
class FirstOrderSpatialHuberRegularization:
    """First-order spatial Huber regularization with IRLS linearization."""

    delta: float = 1.0
    epsilon: float = 1.0e-12
    name: str = "first_order_huber"

    def matrix(self, forward, n_cells: int, *, z_weight: float = 1.0) -> sp.csr_matrix:
        return FirstOrderSpatialRegularization().matrix(forward, n_cells, z_weight=z_weight)

    def _sqrt_irls_weight(self, residual: np.ndarray) -> np.ndarray:
        delta = max(float(self.delta), np.finfo(float).eps)
        eps = max(float(self.epsilon), np.finfo(float).eps)
        abs_residual = np.abs(residual)
        weight_sq = np.ones_like(abs_residual, dtype=float)
        mask = abs_residual > delta
        weight_sq[mask] = delta / np.maximum(abs_residual[mask], eps)
        return np.sqrt(weight_sq)

    def linearized_system(
        self,
        forward,
        current_model: np.ndarray,
        n_cells: int,
        *,
        reference_roughness: np.ndarray,
        scale: float = 1.0,
        z_weight: float = 1.0,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        base = self.matrix(forward, n_cells, z_weight=z_weight)
        current = np.asarray(current_model, dtype=float).reshape(-1)
        reference = np.asarray(reference_roughness, dtype=float).reshape(-1)
        residual = np.asarray(base @ current - reference, dtype=float)
        sqrt_weight = self._sqrt_irls_weight(residual)
        matrix = sp.diags(float(scale) * sqrt_weight, format="csr") @ base
        rhs = float(scale) * sqrt_weight * (reference - base @ current)
        return matrix.tocsr(), np.asarray(rhs, dtype=float)


@dataclass(frozen=True)
class FirstOrderTemporalRegularization:
    """First-order L2 temporal smoothness between neighboring time steps."""

    name: str = "first_order_l2"

    def matrix(self, n_cells: int, n_times: int, *, scale: float = 1.0) -> sp.csr_matrix:
        n_cells = int(n_cells)
        n_times = int(n_times)
        row_count = n_cells * (n_times - 1)
        col_count = n_cells * n_times
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for time_index in range(1, n_times):
            row_offset = (time_index - 1) * n_cells
            previous_offset = (time_index - 1) * n_cells
            current_offset = time_index * n_cells
            for cell_index in range(n_cells):
                row = row_offset + cell_index
                rows.extend((row, row))
                cols.extend((current_offset + cell_index, previous_offset + cell_index))
                data.extend((float(scale), -float(scale)))
        return sp.coo_matrix((data, (rows, cols)), shape=(row_count, col_count)).tocsr()

    def linearized_system(
        self,
        current_model: np.ndarray,
        n_cells: int,
        n_times: int,
        *,
        scale: float = 1.0,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        matrix = self.matrix(n_cells, n_times, scale=scale)
        roughness = matrix @ np.asarray(current_model, dtype=float).reshape(-1)
        return matrix, -np.asarray(roughness, dtype=float)


@dataclass(frozen=True)
class SecondOrderTemporalRegularization:
    """Second-order L2 temporal smoothness for time-lapse model curvature."""

    name: str = "second_order_l2"

    def matrix(self, n_cells: int, n_times: int, *, scale: float = 1.0) -> sp.csr_matrix:
        n_cells = int(n_cells)
        n_times = int(n_times)
        row_count = n_cells * max(n_times - 2, 0)
        col_count = n_cells * n_times
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for time_index in range(1, n_times - 1):
            row_offset = (time_index - 1) * n_cells
            previous_offset = (time_index - 1) * n_cells
            current_offset = time_index * n_cells
            next_offset = (time_index + 1) * n_cells
            for cell_index in range(n_cells):
                row = row_offset + cell_index
                rows.extend((row, row, row))
                cols.extend((previous_offset + cell_index, current_offset + cell_index, next_offset + cell_index))
                data.extend((float(scale), -2.0 * float(scale), float(scale)))
        return sp.coo_matrix((data, (rows, cols)), shape=(row_count, col_count)).tocsr()

    def linearized_system(
        self,
        current_model: np.ndarray,
        n_cells: int,
        n_times: int,
        *,
        scale: float = 1.0,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        matrix = self.matrix(n_cells, n_times, scale=scale)
        roughness = matrix @ np.asarray(current_model, dtype=float).reshape(-1)
        return matrix, -np.asarray(roughness, dtype=float)


@dataclass(frozen=True)
class FirstOrderTemporalTVRegularization:
    """IRLS-smoothed first-order temporal TV/L1 regularization."""

    epsilon: float = 1.0e-3
    name: str = "first_order_tv"

    def matrix(self, n_cells: int, n_times: int, *, scale: float = 1.0) -> sp.csr_matrix:
        return FirstOrderTemporalRegularization().matrix(n_cells, n_times, scale=scale)

    def _sqrt_irls_weight(self, residual: np.ndarray) -> np.ndarray:
        eps = max(float(self.epsilon), np.finfo(float).eps)
        return np.power(residual**2 + eps**2, -0.25)

    def linearized_system(
        self,
        current_model: np.ndarray,
        n_cells: int,
        n_times: int,
        *,
        scale: float = 1.0,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        base = FirstOrderTemporalRegularization().matrix(n_cells, n_times, scale=1.0)
        roughness = np.asarray(base @ np.asarray(current_model, dtype=float).reshape(-1), dtype=float)
        sqrt_weight = self._sqrt_irls_weight(roughness)
        matrix = sp.diags(float(scale) * sqrt_weight, format="csr") @ base
        return matrix.tocsr(), -float(scale) * sqrt_weight * roughness


@dataclass(frozen=True)
class FirstOrderTemporalHuberRegularization:
    """First-order temporal Huber regularization with IRLS linearization."""

    delta: float = 1.0
    epsilon: float = 1.0e-12
    name: str = "first_order_huber"

    def matrix(self, n_cells: int, n_times: int, *, scale: float = 1.0) -> sp.csr_matrix:
        return FirstOrderTemporalRegularization().matrix(n_cells, n_times, scale=scale)

    def _sqrt_irls_weight(self, residual: np.ndarray) -> np.ndarray:
        delta = max(float(self.delta), np.finfo(float).eps)
        eps = max(float(self.epsilon), np.finfo(float).eps)
        abs_residual = np.abs(residual)
        weight_sq = np.ones_like(abs_residual, dtype=float)
        mask = abs_residual > delta
        weight_sq[mask] = delta / np.maximum(abs_residual[mask], eps)
        return np.sqrt(weight_sq)

    def linearized_system(
        self,
        current_model: np.ndarray,
        n_cells: int,
        n_times: int,
        *,
        scale: float = 1.0,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        base = FirstOrderTemporalRegularization().matrix(n_cells, n_times, scale=1.0)
        roughness = np.asarray(base @ np.asarray(current_model, dtype=float).reshape(-1), dtype=float)
        sqrt_weight = self._sqrt_irls_weight(roughness)
        matrix = sp.diags(float(scale) * sqrt_weight, format="csr") @ base
        return matrix.tocsr(), -float(scale) * sqrt_weight * roughness


@dataclass(frozen=True)
class ActiveTimeConstraintRegularization:
    """Adaptive first-order temporal smoothness following active time constraints."""

    threshold: float = 0.05
    minimum_weight: float = 0.05
    name: str = "active_time_constraint"

    def matrix(self, n_cells: int, n_times: int, *, scale: float = 1.0) -> sp.csr_matrix:
        return FirstOrderTemporalRegularization().matrix(n_cells, n_times, scale=scale)

    def linearized_system(
        self,
        current_model: np.ndarray,
        n_cells: int,
        n_times: int,
        *,
        scale: float = 1.0,
        threshold: float | None = None,
        minimum_weight: float | None = None,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        base = FirstOrderTemporalRegularization().matrix(n_cells, n_times, scale=1.0)
        roughness = np.asarray(base @ np.asarray(current_model, dtype=float).reshape(-1), dtype=float)
        threshold_value = self.threshold if threshold is None else float(threshold)
        min_weight = self.minimum_weight if minimum_weight is None else float(minimum_weight)
        if threshold_value <= 0.0:
            raise ValueError("active time threshold must be positive")
        if min_weight < 0.0 or min_weight > 1.0:
            raise ValueError("active time minimum weight must be in [0, 1]")
        abs_change = np.abs(roughness)
        adaptive = min_weight + (1.0 - min_weight) / (1.0 + (abs_change / threshold_value) ** 2)
        matrix = sp.diags(float(scale) * adaptive, format="csr") @ base
        return matrix.tocsr(), -float(scale) * adaptive * roughness


@dataclass(frozen=True)
class BaselineReferenceTemporalRegularization:
    """Cross-model constraint to a baseline/reference time model."""

    name: str = "baseline_reference"

    def matrix(self, n_cells: int, n_times: int, *, scale: float = 1.0) -> sp.csr_matrix:
        n_cells = int(n_cells)
        n_times = int(n_times)
        row_count = n_cells * max(n_times - 1, 0)
        col_count = n_cells * n_times
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for time_index in range(1, n_times):
            row_offset = (time_index - 1) * n_cells
            baseline_offset = 0
            current_offset = time_index * n_cells
            for cell_index in range(n_cells):
                row = row_offset + cell_index
                rows.extend((row, row))
                cols.extend((current_offset + cell_index, baseline_offset + cell_index))
                data.extend((float(scale), -float(scale)))
        return sp.coo_matrix((data, (rows, cols)), shape=(row_count, col_count)).tocsr()

    def linearized_system(
        self,
        current_model: np.ndarray,
        n_cells: int,
        n_times: int,
        *,
        scale: float = 1.0,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        matrix = self.matrix(n_cells, n_times, scale=scale)
        roughness = matrix @ np.asarray(current_model, dtype=float).reshape(-1)
        return matrix, -np.asarray(roughness, dtype=float)


_SPATIAL_REGULARIZATIONS: dict[str, SpatialRegularization] = {
    "damping": IdentitySpatialRegularization(),
    "identity": IdentitySpatialRegularization(),
    "first_order": FirstOrderSpatialRegularization(),
    "first_order_smoothness": FirstOrderSpatialRegularization(),
    "smoothness_constrained": FirstOrderSpatialRegularization(),
    "structural_prior": StructuralPriorSpatialRegularization(),
    "structure_guided": StructuralPriorSpatialRegularization(),
    "structure_guided_first_order": StructuralPriorSpatialRegularization(),
    "structurally_constrained": StructuralPriorSpatialRegularization(),
    "model_difference_smoothness": ModelDifferenceSpatialRegularization(),
    "change_smoothness": ModelDifferenceSpatialRegularization(),
    "change_model_smoothness": ModelDifferenceSpatialRegularization(),
    "spatial_total_variation": FirstOrderSpatialTVRegularization(),
    "first_order_tv": FirstOrderSpatialTVRegularization(),
    "spatial_tv": FirstOrderSpatialTVRegularization(),
    "tv": FirstOrderSpatialTVRegularization(),
    "first_order_l1": FirstOrderSpatialTVRegularization(),
    "spatial_l1": FirstOrderSpatialTVRegularization(),
    "first_order_huber": FirstOrderSpatialHuberRegularization(),
    "spatial_huber": FirstOrderSpatialHuberRegularization(),
    "huber": FirstOrderSpatialHuberRegularization(),
}

_TEMPORAL_REGULARIZATIONS: dict[str, TemporalRegularization] = {
    "first_order": FirstOrderTemporalRegularization(),
    "first_order_l2": FirstOrderTemporalRegularization(),
    "l2": FirstOrderTemporalRegularization(),
    "temporal_smoothness": FirstOrderTemporalRegularization(),
    "second_order": SecondOrderTemporalRegularization(),
    "second_order_l2": SecondOrderTemporalRegularization(),
    "second_order_temporal_smoothness": SecondOrderTemporalRegularization(),
    "curvature_l2": SecondOrderTemporalRegularization(),
    "temporal_total_variation": FirstOrderTemporalTVRegularization(),
    "first_order_tv": FirstOrderTemporalTVRegularization(),
    "temporal_tv": FirstOrderTemporalTVRegularization(),
    "tv": FirstOrderTemporalTVRegularization(),
    "first_order_l1": FirstOrderTemporalTVRegularization(),
    "temporal_l1": FirstOrderTemporalTVRegularization(),
    "first_order_huber": FirstOrderTemporalHuberRegularization(),
    "temporal_huber": FirstOrderTemporalHuberRegularization(),
    "huber": FirstOrderTemporalHuberRegularization(),
    "active_time_constraint": ActiveTimeConstraintRegularization(),
    "active_time_constrained": ActiveTimeConstraintRegularization(),
    "atc": ActiveTimeConstraintRegularization(),
    "4d_atc": ActiveTimeConstraintRegularization(),
    "baseline_reference": BaselineReferenceTemporalRegularization(),
    "reference_model_constraint": BaselineReferenceTemporalRegularization(),
    "cross_model_constraint": BaselineReferenceTemporalRegularization(),
}

_PUBLIC_SPATIAL_REGULARIZATIONS = (
    "damping",
    "first_order_smoothness",
    "structural_prior",
    "model_difference_smoothness",
    "spatial_total_variation",
    "spatial_huber",
)

_PUBLIC_TEMPORAL_REGULARIZATIONS = (
    "temporal_smoothness",
    "second_order_temporal_smoothness",
    "temporal_total_variation",
    "temporal_huber",
    "active_time_constraint",
    "baseline_reference",
)


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def available_spatial_regularizations() -> tuple[str, ...]:
    """Return canonical spatial regularization names for user-facing configuration."""

    return tuple(_PUBLIC_SPATIAL_REGULARIZATIONS)


def available_temporal_regularizations() -> tuple[str, ...]:
    """Return canonical temporal regularization names for user-facing configuration."""

    return tuple(_PUBLIC_TEMPORAL_REGULARIZATIONS)


def build_spatial_regularization(name: str | SpatialRegularization) -> SpatialRegularization:
    """Resolve a spatial regularization object from a registered name."""

    if hasattr(name, "matrix"):
        return name  # type: ignore[return-value]
    key = _normalize_name(str(name))
    try:
        return _SPATIAL_REGULARIZATIONS[key]
    except KeyError as exc:
        choices = ", ".join(available_spatial_regularizations())
        raise ValueError(f"unknown spatial_regularization={name!r}; available choices: {choices}") from exc


def build_temporal_regularization(name: str | TemporalRegularization) -> TemporalRegularization:
    """Resolve a temporal regularization object from a registered name."""

    if hasattr(name, "matrix"):
        return name  # type: ignore[return-value]
    key = _normalize_name(str(name))
    try:
        return _TEMPORAL_REGULARIZATIONS[key]
    except KeyError as exc:
        choices = ", ".join(available_temporal_regularizations())
        raise ValueError(f"unknown temporal_regularization_type={name!r}; available choices: {choices}") from exc


def build_spatial_regularization_matrix(
    name: str | SpatialRegularization,
    forward,
    n_cells: int,
    *,
    z_weight: float = 1.0,
) -> sp.csr_matrix:
    """Build a spatial regularization matrix from a registered interface."""

    return build_spatial_regularization(name).matrix(forward, n_cells, z_weight=z_weight)
