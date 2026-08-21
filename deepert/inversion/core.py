"""Log-space ERT inversion routines built on the native differentiable forward path."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
import time
from typing import Any

from deepert.utils.torch_runtime import torch_np
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.sparse.linalg import cg, lsqr

from deepert.forward import ERTForward2p5D, ERTForwardModeling
from deepert.inversion.misfit import (
    DataMisfit,
    build_data_misfit,
)
from deepert.inversion.optimizers import build_linearized_optimizer, build_optimization_algorithm
from deepert.inversion.petrophysics import (
    available_petrophysical_transforms,
    build_petrophysical_transform,
)
from deepert.inversion.regularization import (
    build_spatial_regularization,
    build_temporal_regularization,
    regularization_mesh,
)
from deepert.mesh import Mesh
from deepert.utils.dtypes import FLOAT_DTYPE


ArrayLike = Any
ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class InversionConfig:
    """Controls for nonlinear log-resistivity inversion."""

    max_iterations: int = 8
    data_std: float | ArrayLike = 0.05
    data_misfit: str = "weighted_log_l2"
    regularization: float = 1.0e-2
    regularization_mode: str = "model"
    temporal_regularization: float = 0.0
    temporal_regularization_mode: str = "separate"
    temporal_regularization_type: str = "temporal_smoothness"
    spatial_regularization: str = "damping"
    regularization_domain: str = "state"
    physical_regularization_quantity: str = "parameter"
    z_weight: float = 1.0
    model_transform: str = "log"
    model_bounds: tuple[float, float] | None = None
    petrophysical_transform: str = "log_resistivity"
    petrophysical_parameters: dict[str, ArrayLike] | None = field(default=None, repr=False, compare=False)
    saturation_floor: float = 1.0e-4
    step_length: float = 1.0
    max_log_step: float | None = 1.0
    line_search: bool = False
    target_chi2: float | None = None
    step_tolerance: float = 1.0e-4
    active_time_threshold: float = 0.05
    active_time_minimum_weight: float = 0.05
    optimization_algorithm: str = "gauss_newton_cgls"
    linearized_solver: str = "lsqr"
    lm_damping: float = 1.0e-2
    optimizer_max_step: float = 1.0
    lbfgs_history: int = 10
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1.0e-8
    lsqr_atol: float = 1.0e-6
    lsqr_btol: float = 1.0e-6
    lsqr_iter_limit: int | None = None
    cgls_max_iterations: int = 2000
    cgls_tolerance: float = 1.0e-8
    include_robin_boundary_derivative: bool = False
    normal_sensitivity: bool = True
    # If True, keep the first time-step state fixed to its initial value during
    # time-lapse inversion. This is useful for baseline-anchored inversions.
    freeze_first_timestep: bool = False
    # Optional soft constraint from external physical observations (e.g., sensor
    # water content). The operator maps cell parameters to observation space:
    #     H @ parameter_model[:, t] ~= targets[:, t]
    # and contributes lambda_s * ||H m - y||^2.
    sensor_constraint: float = 0.0
    sensor_constraint_operator: ArrayLike | None = field(default=None, repr=False, compare=False)
    sensor_constraint_targets: ArrayLike | None = field(default=None, repr=False, compare=False)
    sensor_constraint_weights: ArrayLike | None = field(default=None, repr=False, compare=False)
    progress_callback: ProgressCallback | None = field(default=None, repr=False, compare=False)


def _emit_progress(config: InversionConfig, event: str, **payload: Any) -> None:
    callback = config.progress_callback
    if callback is not None:
        callback({"event": event, **payload})


@dataclass(frozen=True)
class ERTInversionResult:
    """Single-time inversion result."""

    final_model: np.ndarray
    final_log_model: np.ndarray
    predicted_data: np.ndarray
    predicted_log_data: np.ndarray
    coverage: np.ndarray
    iteration_chi2: list[float]
    final_parameter_model: np.ndarray | None = None
    final_parameter_name: str = "resistivity"


@dataclass(frozen=True)
class TimeLapseERTInversionResult:
    """Time-lapse inversion result with models stored as ``(n_cells, n_times)``."""

    final_models: np.ndarray
    final_log_models: np.ndarray
    predicted_data: np.ndarray
    predicted_log_data: np.ndarray
    coverage: np.ndarray
    all_coverage: list[np.ndarray]
    all_chi2: np.ndarray
    iteration_chi2: list[float]
    window_reports: list[dict[str, float | int | None]] = field(default_factory=list)
    final_parameter_models: np.ndarray | None = None
    final_parameter_name: str = "resistivity"


class ParameterizedERTForward2p5D:
    """ERT forward wrapper with a full solve mesh and a smaller parameter mesh."""

    def __init__(
        self,
        forward: ERTForward2p5D,
        parameter_cell_ids: ArrayLike,
        *,
        regularization_mesh: Mesh | None = None,
        forward_cell_parameter_ids: ArrayLike | None = None,
        background_mode: str = "pygimli_prolongation",
    ) -> None:
        self.forward_operator = forward
        self.survey = forward.survey
        self.mesh = forward.mesh
        self.regularization_mesh = regularization_mesh
        forward_cell_count = int(forward.mesh.cell_count)
        self.parameter_cell_ids = np.asarray(parameter_cell_ids, dtype=np.int32).ravel()
        if self.parameter_cell_ids.ndim != 1:
            raise ValueError("parameter_cell_ids must be a 1D array")
        if self.parameter_cell_ids.size == 0:
            raise ValueError("parameter_cell_ids must not be empty")
        if np.any(self.parameter_cell_ids < 0) or np.any(self.parameter_cell_ids >= forward_cell_count):
            raise ValueError("parameter_cell_ids reference cells outside the forward mesh")
        if np.unique(self.parameter_cell_ids).size != self.parameter_cell_ids.size:
            raise ValueError("parameter_cell_ids must be unique")
        if background_mode not in ("pygimli_prolongation", "nearest", "fixed_mean"):
            raise ValueError("background_mode must be 'pygimli_prolongation', 'nearest', or 'fixed_mean'")
        self.background_mode = background_mode
        self.forward_cell_parameter_ids = self._resolve_forward_cell_parameter_ids(
            forward_cell_parameter_ids,
            forward_cell_count=forward_cell_count,
        )
        self._n_parameters = int(np.max(self.forward_cell_parameter_ids)) + 1
        if self.parameter_cell_ids.size != self._n_parameters:
            raise ValueError(
                "parameter_cell_ids must contain one representative forward cell per inversion parameter "
                f"({self.parameter_cell_ids.size} != {self._n_parameters})"
            )
        representative_ids = self.forward_cell_parameter_ids[self.parameter_cell_ids]
        expected_ids = np.arange(self._n_parameters, dtype=np.int32)
        if not np.array_equal(representative_ids, expected_ids):
            raise ValueError("parameter_cell_ids must be ordered representatives of forward_cell_parameter_ids")
        if regularization_mesh is not None and int(regularization_mesh.cell_count) != self._n_parameters:
            raise ValueError(
                "regularization mesh cell count must match inversion parameter count "
                f"({regularization_mesh.cell_count} != {self._n_parameters})"
            )
        all_cell_ids = np.arange(forward_cell_count, dtype=np.int32)
        self._active_forward_mask = self.forward_cell_parameter_ids >= 0
        self._active_forward_cell_ids = all_cell_ids[self._active_forward_mask]
        self._active_parameter_ids = self.forward_cell_parameter_ids[self._active_forward_cell_ids]
        self.background_cell_ids = all_cell_ids[~self._active_forward_mask]
        self._background_parameter_ids = (
            self._build_background_parameter_ids()
            if self.background_mode == "nearest"
            else np.empty((0,), dtype=np.int32)
        )
        self._resistivity_prolongation_matrix = (
            self._build_resistivity_prolongation_matrix()
            if self.background_mode == "pygimli_prolongation"
            else None
        )
        self._jacobian_projection = self._build_jacobian_projection(forward_cell_count)

    def _resolve_forward_cell_parameter_ids(
        self,
        forward_cell_parameter_ids: ArrayLike | None,
        *,
        forward_cell_count: int,
    ) -> np.ndarray:
        if forward_cell_parameter_ids is None:
            cell_parameter_ids = np.full((forward_cell_count,), -1, dtype=np.int32)
            cell_parameter_ids[self.parameter_cell_ids] = np.arange(self.parameter_cell_ids.size, dtype=np.int32)
            return cell_parameter_ids

        cell_parameter_ids = np.asarray(forward_cell_parameter_ids, dtype=np.int32).ravel()
        if cell_parameter_ids.shape != (forward_cell_count,):
            raise ValueError(
                "forward_cell_parameter_ids must have one entry per forward mesh cell "
                f"({cell_parameter_ids.shape} != ({forward_cell_count},))"
            )
        if np.any(cell_parameter_ids < -1):
            raise ValueError("forward_cell_parameter_ids may only contain -1 or non-negative parameter ids")
        active_ids = cell_parameter_ids[cell_parameter_ids >= 0]
        if active_ids.size == 0:
            raise ValueError("forward_cell_parameter_ids must contain at least one active parameter cell")
        unique_ids = np.unique(active_ids)
        expected_ids = np.arange(int(unique_ids[-1]) + 1, dtype=np.int32)
        if not np.array_equal(unique_ids, expected_ids):
            raise ValueError("forward_cell_parameter_ids must use contiguous ids starting at 0")
        return cell_parameter_ids

    @classmethod
    def from_mesh_survey(
        cls,
        mesh: Mesh,
        survey,
        parameter_cell_ids: ArrayLike,
        *,
        regularization_mesh: Mesh | None = None,
        background_mode: str = "pygimli_prolongation",
        **forward_kwargs: Any,
    ) -> "ParameterizedERTForward2p5D":
        forward_cell_parameter_ids = forward_kwargs.pop("forward_cell_parameter_ids", None)
        forward = ERTForward2p5D.from_mesh_survey(mesh, survey, **forward_kwargs)
        return cls(
            forward,
            parameter_cell_ids,
            regularization_mesh=regularization_mesh,
            forward_cell_parameter_ids=forward_cell_parameter_ids,
            background_mode=background_mode,
        )

    @property
    def cell_count(self) -> int:
        return self._n_parameters

    def close(self) -> None:
        self.forward_operator.close()

    def _cell_centers(self) -> np.ndarray:
        nodes = np.asarray(self.forward_operator.mesh.nodes, dtype=float)
        cells = np.asarray(self.forward_operator.mesh.cells, dtype=np.int32)
        return np.mean(nodes[cells], axis=1)

    def _parameter_centers(self, centers: np.ndarray) -> np.ndarray:
        sums = np.zeros((self.cell_count, centers.shape[1]), dtype=float)
        counts = np.zeros((self.cell_count,), dtype=float)
        np.add.at(sums, self._active_parameter_ids, centers[self._active_forward_cell_ids])
        np.add.at(counts, self._active_parameter_ids, 1.0)
        if np.any(counts <= 0.0):
            raise ValueError("each inversion parameter must own at least one forward mesh cell")
        return sums / counts[:, None]

    def _build_background_parameter_ids(self) -> np.ndarray:
        if self.background_cell_ids.size == 0:
            return np.empty((0,), dtype=np.int32)
        centers = self._cell_centers()
        parameter_centers = self._parameter_centers(centers)
        _, nearest = cKDTree(parameter_centers).query(centers[self.background_cell_ids])
        return np.asarray(nearest, dtype=np.int32)

    def _build_jacobian_projection(self, forward_cell_count: int) -> sp.csr_matrix:
        rows = [self._active_forward_cell_ids]
        cols = [self._active_parameter_ids]
        data = [np.ones(self._active_forward_cell_ids.size, dtype=float)]
        if self.background_mode == "pygimli_prolongation":
            return sp.coo_matrix(
                (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                shape=(forward_cell_count, self.cell_count),
            ).tocsr()
        if self.background_cell_ids.size and self.background_mode == "nearest":
            rows.append(self.background_cell_ids)
            cols.append(self._background_parameter_ids)
            data.append(np.ones(self.background_cell_ids.size, dtype=float))
        elif self.background_cell_ids.size and self.background_mode == "fixed_mean":
            rows.append(np.repeat(self.background_cell_ids, self.cell_count))
            cols.append(np.tile(np.arange(self.cell_count, dtype=np.int32), self.background_cell_ids.size))
            data.append(np.full(self.background_cell_ids.size * self.cell_count, 1.0 / self.cell_count, dtype=float))
        return sp.coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(forward_cell_count, self.cell_count),
        ).tocsr()

    def _build_resistivity_prolongation_matrix(self) -> sp.csr_matrix:
        """Replicate PyGIMLi's marker model prolongation for background cells."""

        forward_cell_count = int(self.forward_operator.mesh.cell_count)
        matrix = np.zeros((forward_cell_count, self.cell_count), dtype=float)
        matrix[self._active_forward_cell_ids, self._active_parameter_ids] = 1.0
        if self.background_cell_ids.size == 0:
            return sp.csr_matrix(matrix)

        nodes = np.asarray(self.forward_operator.mesh.nodes, dtype=float)
        cells = np.asarray(self.forward_operator.mesh.cells, dtype=np.int32)
        edge_cells: dict[tuple[int, int], list[int]] = {}
        for cell_id, cell in enumerate(cells):
            for edge in _cell_edges(cell):
                edge_cells.setdefault(tuple(sorted(edge)), []).append(cell_id)

        neighbors: list[list[tuple[int, float]]] = [[] for _ in range(forward_cell_count)]
        for edge, owners in edge_cells.items():
            if len(owners) != 2:
                continue
            p0, p1 = nodes[list(edge)]
            tangent = p1 - p0
            length = float(np.linalg.norm(tangent))
            if length <= 0.0:
                continue
            weight = abs(float(tangent[1])) / length + 1.0e-6
            left, right = int(owners[0]), int(owners[1])
            neighbors[left].append((right, weight))
            neighbors[right].append((left, weight))

        known = self._active_forward_mask.copy()
        unknown = set(int(cell_id) for cell_id in self.background_cell_ids)
        while unknown:
            assignments: list[tuple[int, np.ndarray]] = []
            for cell_id in sorted(unknown):
                weighted = np.zeros((self.cell_count,), dtype=float)
                total_weight = 0.0
                for neighbor_id, weight in neighbors[cell_id]:
                    if known[neighbor_id]:
                        weighted += matrix[neighbor_id] * weight
                        total_weight += weight
                if total_weight > 1.0e-8:
                    assignments.append((cell_id, weighted / total_weight))
            if not assignments:
                raise ValueError("could not prolongate background forward cells from active parameter cells")
            for cell_id, row in assignments:
                matrix[cell_id] = row
                known[cell_id] = True
                unknown.remove(cell_id)

        return sp.csr_matrix(matrix)

    def _full_log_model(self, log_resistivity: ArrayLike) -> np.ndarray:
        full_log, _ = self._full_log_model_and_projection(log_resistivity)
        return full_log

    def _full_log_model_and_projection(self, log_resistivity: ArrayLike) -> tuple[np.ndarray, sp.csr_matrix]:
        parameter_log = np.asarray(log_resistivity, dtype=float).ravel()
        if parameter_log.shape != (self.cell_count,):
            raise ValueError(f"log_resistivity must have shape ({self.cell_count},)")
        if self.background_mode == "pygimli_prolongation":
            if self._resistivity_prolongation_matrix is None:
                raise ValueError("resistivity prolongation matrix has not been initialized")
            parameter_resistivity = np.exp(parameter_log)
            full_resistivity = np.asarray(self._resistivity_prolongation_matrix @ parameter_resistivity, dtype=float).ravel()
            if np.any(full_resistivity <= 0.0) or not np.all(np.isfinite(full_resistivity)):
                raise ValueError("prolongated forward resistivity contains non-positive or non-finite values")
            if self._jacobian_projection is None:
                raise ValueError("Jacobian projection has not been initialized")
            return np.log(full_resistivity), self._jacobian_projection

        full_log = np.empty((self.forward_operator.mesh.cell_count,), dtype=float)
        full_log[self._active_forward_cell_ids] = parameter_log[self._active_parameter_ids]
        if self.background_cell_ids.size:
            if self.background_mode == "nearest":
                full_log[self.background_cell_ids] = parameter_log[self._background_parameter_ids]
            else:
                full_log[self.background_cell_ids] = float(np.mean(parameter_log))
        if self._jacobian_projection is None:
            raise ValueError("Jacobian projection has not been initialized")
        return full_log, self._jacobian_projection

    def forward_and_jacobian(
        self,
        resistivity_model: ArrayLike,
        log_transform: bool = True,
        *,
        include_robin_boundary_derivative: bool | None = None,
        normal_sensitivity: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        log_model = _as_log_model(
            resistivity_model,
            expected_size=self.cell_count,
            log_model=log_transform,
            name="resistivity_model",
        )
        full_log_model, projection = self._full_log_model_and_projection(log_model)
        if (
            self.background_mode == "pygimli_prolongation"
            and (normal_sensitivity is None or bool(normal_sensitivity))
            and not (bool(include_robin_boundary_derivative) if include_robin_boundary_derivative is not None else False)
        ):
            conductivity = torch_np.asarray(np.exp(-full_log_model), dtype=FLOAT_DTYPE)
            response, resistance_jacobian = self.forward_operator.solve_with_jacobian(
                conductivity,
                include_robin_boundary_derivative=False,
                normal_sensitivity=True,
                jacobian_cell_parameter_ids=self.forward_cell_parameter_ids,
                jacobian_parameter_count=self.cell_count,
            )
            apparent_jacobian_sigma = torch_np.abs(self.forward_operator._geometric_factors())[:, None] * resistance_jacobian
            parameter_conductivity = torch_np.asarray(np.exp(-log_model), dtype=apparent_jacobian_sigma.dtype)
            jacobian = apparent_jacobian_sigma * (-parameter_conductivity[None, :])
            jacobian = jacobian / response.apparent_resistivity[:, None]
            return (
                np.log(np.asarray(response.apparent_resistivity, dtype=float)),
                np.asarray(jacobian, dtype=float),
            )
        predicted, full_jacobian = _forward_and_jacobian_log(
            self.forward_operator,
            full_log_model,
            include_robin_boundary_derivative=bool(include_robin_boundary_derivative)
            if include_robin_boundary_derivative is not None
            else False,
            normal_sensitivity=bool(normal_sensitivity) if normal_sensitivity is not None else True,
        )
        jacobian = np.asarray((projection.T @ np.asarray(full_jacobian, dtype=float).T).T, dtype=float)
        return predicted, jacobian

    def response(self, resistivity_model: ArrayLike) -> np.ndarray:
        return self.forward(resistivity_model, log_transform=False)

    def forward(self, resistivity_model: ArrayLike, log_transform: bool = True) -> np.ndarray:
        log_model = _as_log_model(
            resistivity_model,
            expected_size=self.cell_count,
            log_model=log_transform,
            name="resistivity_model",
        )
        response = _forward_log_response(self.forward_operator, self._full_log_model(log_model))
        if log_transform:
            return response
        return np.exp(response)


def _check_config(config: InversionConfig) -> None:
    if config.max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    build_data_misfit(config.data_misfit)
    if config.regularization < 0.0:
        raise ValueError("regularization must be non-negative")
    if config.regularization_mode not in ("model", "update"):
        raise ValueError("regularization_mode must be 'model' or 'update'")
    regularization_domain = _normalize_regularization_domain(config.regularization_domain)
    if regularization_domain not in {"state", "physical"}:
        raise ValueError("regularization_domain must be 'state' or 'physical'")
    physical_quantity = _normalize_physical_regularization_quantity(config.physical_regularization_quantity)
    if physical_quantity not in {"parameter", "water_content"}:
        raise ValueError("physical_regularization_quantity must be 'parameter' or 'theta'/'water_content'")
    if config.temporal_regularization < 0.0:
        raise ValueError("temporal_regularization must be non-negative")
    if config.temporal_regularization_mode not in ("separate", "joint_frame"):
        raise ValueError("temporal_regularization_mode must be 'separate' or 'joint_frame'")
    if not isinstance(config.freeze_first_timestep, (bool, np.bool_)):
        raise ValueError("freeze_first_timestep must be a boolean flag")
    if config.sensor_constraint < 0.0:
        raise ValueError("sensor_constraint must be non-negative")
    if config.sensor_constraint > 0.0:
        if config.sensor_constraint_operator is None:
            raise ValueError("sensor_constraint_operator is required when sensor_constraint > 0")
        if config.sensor_constraint_targets is None:
            raise ValueError("sensor_constraint_targets is required when sensor_constraint > 0")
    if config.sensor_constraint_weights is not None and config.sensor_constraint_targets is None:
        raise ValueError("sensor_constraint_weights requires sensor_constraint_targets")
    if config.sensor_constraint > 0.0 and regularization_domain != "physical":
        raise ValueError("sensor_constraint currently requires regularization_domain='physical'")
    build_temporal_regularization(config.temporal_regularization_type)
    build_spatial_regularization(config.spatial_regularization)
    if config.z_weight <= 0.0:
        raise ValueError("z_weight must be positive")
    if config.model_transform not in ("log", "log_lu"):
        raise ValueError("model_transform must be 'log' or 'log_lu'")
    if config.model_transform == "log_lu" and config.model_bounds is None:
        raise ValueError("model_bounds are required for model_transform='log_lu'")
    petrophysical_key = str(config.petrophysical_transform).strip().lower().replace("-", "_")
    petrophysical_choices = set(available_petrophysical_transforms()) | {
        "resistivity",
        "rho",
        "conductivity",
        "sigma",
        "water_saturation",
        "relative_archie",
        "water_content",
        "theta",
    }
    if petrophysical_key not in petrophysical_choices:
        choices = ", ".join(available_petrophysical_transforms())
        raise ValueError(f"unknown petrophysical_transform={config.petrophysical_transform!r}; available choices: {choices}")
    if (
        regularization_domain == "physical"
        and physical_quantity == "water_content"
        and petrophysical_key
        not in {
            "saturation",
            "water_saturation",
            "relative_archie_water_content",
            "relative_archie",
            "water_content",
            "theta",
        }
    ):
        raise ValueError(
            "physical_regularization_quantity='theta'/'water_content' requires "
            "petrophysical_transform='saturation', 'water_content', or 'relative_archie_water_content'"
        )
    if not (0.0 < config.saturation_floor < 1.0):
        raise ValueError("saturation_floor must be in (0, 1)")
    if config.step_length <= 0.0:
        raise ValueError("step_length must be positive")
    if config.max_log_step is not None and config.max_log_step <= 0.0:
        raise ValueError("max_log_step must be positive when set")
    if config.target_chi2 is not None and config.target_chi2 <= 0.0:
        raise ValueError("target_chi2 must be positive when set")
    if config.active_time_threshold <= 0.0:
        raise ValueError("active_time_threshold must be positive")
    if not (0.0 <= config.active_time_minimum_weight <= 1.0):
        raise ValueError("active_time_minimum_weight must be in [0, 1]")
    build_optimization_algorithm(config.optimization_algorithm)
    build_linearized_optimizer(config.linearized_solver)
    if config.lm_damping < 0.0:
        raise ValueError("lm_damping must be non-negative")
    if config.optimizer_max_step <= 0.0:
        raise ValueError("optimizer_max_step must be positive")
    if config.lbfgs_history < 1:
        raise ValueError("lbfgs_history must be >= 1")
    if not (0.0 <= config.adam_beta1 < 1.0):
        raise ValueError("adam_beta1 must be in [0, 1)")
    if not (0.0 <= config.adam_beta2 < 1.0):
        raise ValueError("adam_beta2 must be in [0, 1)")
    if config.adam_epsilon <= 0.0:
        raise ValueError("adam_epsilon must be positive")
    if config.cgls_max_iterations < 1:
        raise ValueError("cgls_max_iterations must be >= 1")
    if config.cgls_tolerance <= 0.0:
        raise ValueError("cgls_tolerance must be positive")
    if config.progress_callback is not None and not callable(config.progress_callback):
        raise ValueError("progress_callback must be callable when set")
    if config.model_bounds is not None:
        lo, hi = config.model_bounds
        if not (0.0 < lo < hi):
            raise ValueError("model_bounds must be positive and ordered as (min, max)")


def _model_size(forward: ERTForward2p5D | ERTForwardModeling) -> int:
    if isinstance(forward, ERTForward2p5D):
        return forward.mesh.cell_count
    if isinstance(forward, ERTForwardModeling):
        return forward.cell_count
    if hasattr(forward, "cell_count"):
        return int(forward.cell_count)
    if hasattr(forward, "mesh") and hasattr(forward.mesh, "cell_count"):
        return int(forward.mesh.cell_count)
    raise TypeError("forward must expose a deepert-compatible cell count")


def _measurement_count(forward: ERTForward2p5D | ERTForwardModeling) -> int:
    if isinstance(forward, ERTForward2p5D):
        return forward.survey.measurement_count
    if isinstance(forward, ERTForwardModeling):
        return forward.forward_operator.survey.measurement_count
    if hasattr(forward, "survey") and hasattr(forward.survey, "measurement_count"):
        return int(forward.survey.measurement_count)
    if hasattr(forward, "forward_operator") and hasattr(forward.forward_operator, "survey"):
        return int(forward.forward_operator.survey.measurement_count)
    raise TypeError("forward must expose a deepert-compatible measurement count")


def _as_log_model(
    model: ArrayLike,
    *,
    expected_size: int,
    log_model: bool,
    name: str,
) -> np.ndarray:
    values = np.asarray(model, dtype=float).ravel()
    if values.shape != (expected_size,):
        raise ValueError(f"{name} must have shape ({expected_size},)")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    if log_model:
        return values.copy()
    if np.any(values <= 0.0):
        raise ValueError(f"{name} must contain positive resistivity values")
    return np.log(values)


def _as_log_model_matrix(
    model: ArrayLike,
    *,
    expected_size: int,
    n_times: int,
    log_model: bool,
    name: str,
) -> np.ndarray:
    values = np.asarray(model, dtype=float)
    if values.ndim == 1:
        return np.column_stack(
            [
                _as_log_model(values, expected_size=expected_size, log_model=log_model, name=name)
                for _ in range(n_times)
            ]
        )
    if values.shape == (expected_size, n_times):
        matrix = values.copy()
    elif values.shape == (n_times, expected_size):
        matrix = values.T.copy()
    else:
        raise ValueError(f"{name} must have shape ({expected_size},), ({expected_size}, {n_times}), or ({n_times}, {expected_size})")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    if log_model:
        return matrix
    if np.any(matrix <= 0.0):
        raise ValueError(f"{name} must contain positive resistivity values")
    return np.log(matrix)


def _as_observed_log_vector(data: ArrayLike, *, log_data: bool, expected_size: int | None = None) -> np.ndarray:
    values = np.asarray(data, dtype=float).ravel()
    if expected_size is not None and values.shape != (expected_size,):
        raise ValueError(f"observed_data must have shape ({expected_size},)")
    if not np.all(np.isfinite(values)):
        raise ValueError("observed_data contains non-finite values")
    if log_data:
        return values.copy()
    if np.any(values <= 0.0):
        raise ValueError("observed_data must contain positive apparent resistivity values")
    return np.log(values)


def _as_observed_log_matrix(
    data: ArrayLike,
    *,
    log_data: bool,
    measurement_count: int,
) -> np.ndarray:
    values = np.asarray(data, dtype=float)
    if values.ndim != 2:
        raise ValueError("observed_data must be a 2D array for time-lapse inversion")
    if values.shape[1] == measurement_count:
        matrix = values.copy()
    elif values.shape[0] == measurement_count:
        matrix = values.T.copy()
    else:
        raise ValueError(
            "observed_data must have shape (n_times, n_measurements) "
            "or (n_measurements, n_times)"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("observed_data contains non-finite values")
    if log_data:
        return matrix
    if np.any(matrix <= 0.0):
        raise ValueError("observed_data must contain positive apparent resistivity values")
    return np.log(matrix)


def _weights(data_std: float | ArrayLike, shape: tuple[int, ...]) -> np.ndarray:
    std = np.asarray(data_std, dtype=float)
    if std.ndim == 0:
        std = np.full(shape, float(std), dtype=float)
    else:
        std = np.broadcast_to(std, shape).astype(float, copy=True)
    if not np.all(np.isfinite(std)):
        raise ValueError("data_std contains non-finite values")
    if np.any(std <= 0.0):
        raise ValueError("data_std must be positive")
    return 1.0 / std


def _model_cell_count_from_array(values: np.ndarray) -> int:
    array = np.asarray(values)
    if array.ndim == 2:
        return int(array.shape[0])
    return int(array.size)


def _petrophysical_transform_for_array(config: InversionConfig, values: np.ndarray):
    return build_petrophysical_transform(
        config.petrophysical_transform,
        n_cells=_model_cell_count_from_array(np.asarray(values)),
        model_transform=config.model_transform,
        model_bounds=config.model_bounds,
        saturation_floor=float(config.saturation_floor),
        parameters=config.petrophysical_parameters,
    )


def _normalize_regularization_domain(domain: str) -> str:
    return str(domain).strip().lower().replace("-", "_")


def _normalize_physical_regularization_quantity(quantity: str) -> str:
    key = str(quantity).strip().lower().replace("-", "_")
    if key in {"theta", "water_content", "moisture_content"}:
        return "water_content"
    if key in {"parameter", "native", "physical_parameter"}:
        return "parameter"
    return key


def _log_model_to_state(log_model: np.ndarray, config: InversionConfig) -> np.ndarray:
    transform = _petrophysical_transform_for_array(config, np.asarray(log_model))
    return transform.state_from_log_resistivity(np.asarray(log_model, dtype=float))


def _state_to_log_model(state: np.ndarray, config: InversionConfig) -> np.ndarray:
    transform = _petrophysical_transform_for_array(config, np.asarray(state))
    return transform.log_resistivity_from_state(np.asarray(state, dtype=float))


def _clip_model_state(state: np.ndarray, config: InversionConfig) -> np.ndarray:
    transform = _petrophysical_transform_for_array(config, np.asarray(state))
    return transform.clip_state(np.asarray(state, dtype=float))


def _d_log_model_d_state(state: np.ndarray, config: InversionConfig) -> np.ndarray:
    transform = _petrophysical_transform_for_array(config, np.asarray(state))
    return transform.d_log_resistivity_d_state(np.asarray(state, dtype=float))


def _d_parameter_model_d_state(state: np.ndarray, config: InversionConfig) -> np.ndarray:
    transform = _petrophysical_transform_for_array(config, np.asarray(state))
    return transform.d_parameter_d_state(np.asarray(state, dtype=float))


def _parameter_model_from_state(state: np.ndarray, config: InversionConfig) -> np.ndarray:
    transform = _petrophysical_transform_for_array(config, np.asarray(state))
    return transform.parameter_from_state(np.asarray(state, dtype=float))


def _parameter_name_from_state(state: np.ndarray, config: InversionConfig) -> str:
    transform = _petrophysical_transform_for_array(config, np.asarray(state))
    return transform.parameter_name


def _cell_parameter_array(values: ArrayLike, *, n_cells: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        result = np.full((int(n_cells),), float(array), dtype=float)
    else:
        result = np.asarray(array, dtype=float).reshape(-1)
    if result.shape != (int(n_cells),):
        raise ValueError(f"{name} must be scalar or have shape ({int(n_cells)},)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _regularization_domain_value_and_derivative(
    state: np.ndarray,
    config: InversionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    state_array = np.asarray(state, dtype=float)
    if _normalize_regularization_domain(config.regularization_domain) == "state":
        return state_array, np.ones_like(state_array, dtype=float)

    parameter = _parameter_model_from_state(state_array, config)
    derivative = _d_parameter_model_d_state(state_array, config)
    quantity = _normalize_physical_regularization_quantity(config.physical_regularization_quantity)
    if quantity == "parameter":
        return np.asarray(parameter, dtype=float), np.asarray(derivative, dtype=float)

    parameter_name = _parameter_name_from_state(state_array, config)
    if parameter_name == "water_content":
        return np.asarray(parameter, dtype=float), np.asarray(derivative, dtype=float)
    if parameter_name != "saturation":
        raise ValueError(
            "physical_regularization_quantity='theta'/'water_content' requires "
            "petrophysical_transform='saturation' or a transform whose parameter is water_content"
        )
    parameters = config.petrophysical_parameters or {}
    if "phi" not in parameters or parameters["phi"] is None:
        raise ValueError(
            "physical_regularization_quantity='theta'/'water_content' requires petrophysical_parameters['phi']"
        )
    phi = _cell_parameter_array(parameters["phi"], n_cells=int(state_array.shape[0]), name="phi")
    if state_array.ndim == 2:
        phi = phi[:, None]
    return np.asarray(phi * parameter, dtype=float), np.asarray(phi * derivative, dtype=float)


def _chain_rule_projection(derivative: np.ndarray, *, order: str = "C") -> sp.csr_matrix:
    diagonal = np.asarray(derivative, dtype=float).reshape(-1, order=order)
    if not np.all(np.isfinite(diagonal)):
        raise ValueError("regularization projection derivative contains non-finite values")
    return sp.diags(diagonal, format="csr")


def _as_csr_matrix(values: ArrayLike, *, name: str) -> sp.csr_matrix:
    if sp.issparse(values):
        matrix = values.tocsr()
    else:
        array = np.asarray(values, dtype=float)
        if array.ndim != 2:
            raise ValueError(f"{name} must be a 2D matrix")
        matrix = sp.csr_matrix(array)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix")
    if matrix.nnz > 0 and not np.all(np.isfinite(matrix.data)):
        raise ValueError(f"{name} contains non-finite entries")
    return matrix


def _broadcast_sensor_constraint_weights(
    weights: ArrayLike | None,
    shape: tuple[int, int],
) -> np.ndarray:
    if weights is None:
        return np.ones(shape, dtype=float)
    values = np.asarray(weights, dtype=float)
    if values.ndim == 0:
        result = np.full(shape, float(values), dtype=float)
    elif values.ndim == 1:
        if values.shape[0] == shape[0]:
            result = np.broadcast_to(values[:, None], shape).astype(float, copy=True)
        elif values.shape[0] == shape[1]:
            result = np.broadcast_to(values[None, :], shape).astype(float, copy=True)
        elif values.shape[0] == shape[0] * shape[1]:
            result = values.reshape(shape, order="F").astype(float, copy=True)
        else:
            raise ValueError(
                "sensor_constraint_weights 1D shape must match n_constraints, n_times, "
                "or n_constraints*n_times"
            )
    elif values.ndim == 2:
        result = np.broadcast_to(values, shape).astype(float, copy=True)
    else:
        raise ValueError("sensor_constraint_weights must be scalar, 1D, or 2D")
    if not np.all(np.isfinite(result)):
        raise ValueError("sensor_constraint_weights contains non-finite values")
    if np.any(result < 0.0):
        raise ValueError("sensor_constraint_weights must be non-negative")
    return result


def _prepare_sensor_constraint(
    config: InversionConfig,
    *,
    n_cells: int,
    n_times: int,
) -> tuple[sp.csr_matrix, np.ndarray] | None:
    if config.sensor_constraint <= 0.0:
        return None
    operator = _as_csr_matrix(config.sensor_constraint_operator, name="sensor_constraint_operator")
    if operator.shape[1] != int(n_cells):
        raise ValueError(
            "sensor_constraint_operator second dimension must match n_cells "
            f"({operator.shape[1]} != {int(n_cells)})"
        )
    targets = np.asarray(config.sensor_constraint_targets, dtype=float)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    if targets.ndim != 2:
        raise ValueError("sensor_constraint_targets must be a 2D matrix [n_constraints, n_times]")
    if targets.shape[0] != operator.shape[0]:
        raise ValueError(
            "sensor_constraint_targets first dimension must match operator rows "
            f"({targets.shape[0]} != {operator.shape[0]})"
        )
    if targets.shape[1] == 1 and int(n_times) > 1:
        targets = np.repeat(targets, int(n_times), axis=1)
    if targets.shape[1] != int(n_times):
        raise ValueError(
            "sensor_constraint_targets second dimension must match n_times "
            f"({targets.shape[1]} != {int(n_times)})"
        )
    weights = _broadcast_sensor_constraint_weights(
        config.sensor_constraint_weights,
        (int(operator.shape[0]), int(n_times)),
    )
    valid = np.isfinite(targets) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        raise ValueError("sensor_constraint has no valid (finite, positive-weight) entries")

    full_operator = sp.kron(sp.eye(int(n_times), format="csr"), operator, format="csr")
    target_vec = targets.reshape(-1, order="F")
    weight_vec = weights.reshape(-1, order="F")
    valid_rows = np.flatnonzero(valid.reshape(-1, order="F"))
    matrix = full_operator[valid_rows].tocsr()
    row_scale = np.sqrt(weight_vec[valid_rows])
    if row_scale.size and not np.allclose(row_scale, 1.0):
        matrix = sp.diags(row_scale, format="csr") @ matrix
    target_scaled = row_scale * target_vec[valid_rows]
    if not np.all(np.isfinite(target_scaled)):
        raise ValueError("sensor_constraint targets contain non-finite values after scaling")
    return matrix.tocsr(), np.asarray(target_scaled, dtype=float)


def _limit_delta(delta: np.ndarray, max_log_step: float | None) -> np.ndarray:
    if max_log_step is None:
        return delta
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    if max_abs <= max_log_step:
        return delta
    return delta * (max_log_step / max_abs)


def _is_log_data_difference_misfit(data_misfit: DataMisfit) -> bool:
    return getattr(data_misfit, "name", "") == "log_data_difference_l2"


def _difference_weights(weights: np.ndarray) -> np.ndarray:
    weight_array = np.asarray(weights, dtype=float)
    sigma = 1.0 / weight_array
    baseline_sigma = sigma[0]
    return 1.0 / np.sqrt(sigma[1:] ** 2 + baseline_sigma[None, :] ** 2)


def _log_data_difference_residual(
    predicted_log: np.ndarray,
    observed_log: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    predicted = np.asarray(predicted_log, dtype=float)
    observed = np.asarray(observed_log, dtype=float)
    if predicted.shape != observed.shape:
        raise ValueError("predicted_log and observed_log must have the same shape")
    if predicted.ndim != 2 or predicted.shape[0] < 2:
        raise ValueError("log data-difference misfit requires at least two time steps")
    base = ((predicted[0] - observed[0]) * np.asarray(weights, dtype=float)[0]).reshape(1, -1)
    diff = (predicted[1:] - predicted[0][None, :]) - (observed[1:] - observed[0][None, :])
    return np.vstack((base, diff * _difference_weights(weights)))


def _log_data_difference_phi(
    predicted_log: np.ndarray,
    observed_log: np.ndarray,
    weights: np.ndarray,
) -> float:
    residual = _log_data_difference_residual(predicted_log, observed_log, weights).reshape(-1)
    return float(np.dot(residual, residual))


def _log_data_difference_chi2(
    predicted_log: np.ndarray,
    observed_log: np.ndarray,
    weights: np.ndarray,
) -> float:
    residual = _log_data_difference_residual(predicted_log, observed_log, weights)
    return float(np.mean(residual**2))


def _mesh_cell_areas_np(mesh: Mesh) -> np.ndarray:
    nodes = np.asarray(mesh.nodes, dtype=float)
    cells = np.asarray(mesh.cells, dtype=np.int32)
    cell_nodes = nodes[cells]
    x_values = cell_nodes[:, :, 0]
    y_values = cell_nodes[:, :, 1]
    cross_sum = np.sum(
        x_values * np.roll(y_values, -1, axis=1) - np.roll(x_values, -1, axis=1) * y_values,
        axis=1,
    )
    areas = 0.5 * np.abs(cross_sum)
    if np.any(areas <= 0.0) or not np.all(np.isfinite(areas)):
        raise ValueError("regularization mesh contains non-positive or non-finite cell areas")
    return areas


def _pygimli_style_coverage_from_jacobian(
    forward: ERTForward2p5D | ERTForwardModeling,
    jacobian: np.ndarray,
) -> np.ndarray:
    """Return PyGIMLi-style log10 coverage used for default plot masking.

    PyGIMLi's ERT coverage path applies the data/model log transform to the
    sensitivity matrix, sums absolute transformed sensitivities over data, and
    normalizes by parameter-cell size before taking log10.
    """

    matrix = np.asarray(jacobian, dtype=float)
    mesh = regularization_mesh(forward)
    areas = _mesh_cell_areas_np(mesh)
    if matrix.shape[1] != areas.shape[0]:
        raise ValueError(
            "coverage Jacobian column count does not match regularization mesh cell count "
            f"({matrix.shape[1]} != {areas.shape[0]})"
        )
    sensitivity_sum = np.sum(np.abs(matrix), axis=0)
    normalized = np.maximum(sensitivity_sum / areas, np.finfo(float).tiny)
    return np.log10(normalized)


def _model_phi(
    state: np.ndarray,
    regularization_matrix: sp.spmatrix,
    reference_roughness: np.ndarray,
) -> float:
    roughness = regularization_matrix @ state - reference_roughness
    return float(np.dot(roughness, roughness))


def _line_search_tau(
    *,
    state: np.ndarray,
    step: np.ndarray,
    predicted_log: np.ndarray,
    candidate_predicted_log: np.ndarray,
    observed_log: np.ndarray,
    weights: np.ndarray,
    regularization_matrix: sp.spmatrix,
    reference_roughness: np.ndarray,
    regularization: float,
    data_misfit: DataMisfit | None = None,
    data_phi: Callable[[np.ndarray], float] | None = None,
) -> float:
    data_direction = candidate_predicted_log - predicted_log
    data_misfit = data_misfit or build_data_misfit("weighted_log_l2")
    objective_data_phi = data_phi or (lambda values: data_misfit.phi(values, observed_log, weights))
    best_tau = 0.0
    best_phi = objective_data_phi(predicted_log) + regularization * _model_phi(
        state,
        regularization_matrix,
        reference_roughness,
    )
    candidate_phi = objective_data_phi(candidate_predicted_log) + regularization * _model_phi(
        state + step,
        regularization_matrix,
        reference_roughness,
    )
    if candidate_phi < best_phi:
        return 1.0
    for index in range(1, 101):
        tau = 0.01 * index
        state_tau = state + tau * step
        predicted_tau = predicted_log + tau * data_direction
        phi = objective_data_phi(predicted_tau) + regularization * _model_phi(
            state_tau,
            regularization_matrix,
            reference_roughness,
        )
        if phi < best_phi:
            best_phi = phi
            best_tau = tau
    if 0.0 < best_tau < 0.03:
        return 0.03
    return best_tau


def _cell_edges(cell: np.ndarray) -> list[tuple[int, int]]:
    return [
        (int(cell[index]), int(cell[(index + 1) % cell.size]))
        for index in range(cell.size)
    ]


def _forward_and_jacobian_log(
    forward: ERTForward2p5D | ERTForwardModeling,
    log_resistivity: np.ndarray,
    *,
    include_robin_boundary_derivative: bool = False,
    normal_sensitivity: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(forward, ERTForwardModeling):
        return forward.forward_and_jacobian(
            log_resistivity,
            log_transform=True,
            include_robin_boundary_derivative=include_robin_boundary_derivative,
            normal_sensitivity=normal_sensitivity,
        )

    if not isinstance(forward, ERTForward2p5D):
        if hasattr(forward, "forward_and_jacobian"):
            return forward.forward_and_jacobian(
                log_resistivity,
                log_transform=True,
                include_robin_boundary_derivative=include_robin_boundary_derivative,
                normal_sensitivity=normal_sensitivity,
            )
        raise TypeError("forward must be ERTForward2p5D, ERTForwardModeling, or expose forward_and_jacobian")

    resistivity = np.exp(log_resistivity)
    conductivity = torch_np.asarray(1.0 / resistivity, dtype=FLOAT_DTYPE)
    response, resistance_jacobian = forward.solve_with_jacobian(
        conductivity,
        include_robin_boundary_derivative=include_robin_boundary_derivative,
        normal_sensitivity=normal_sensitivity,
    )
    apparent_jacobian_sigma = torch_np.abs(forward._geometric_factors())[:, None] * resistance_jacobian
    jacobian = apparent_jacobian_sigma * (-conductivity[None, :])
    jacobian = jacobian / response.apparent_resistivity[:, None]
    return (
        np.log(np.asarray(response.apparent_resistivity, dtype=float)),
        np.asarray(jacobian, dtype=float),
    )


ForwardJacobianCache = OrderedDict[tuple[tuple[int, ...], str, bytes, bool, bool], tuple[np.ndarray, np.ndarray]]


def _forward_and_jacobian_log_cached(
    forward: ERTForward2p5D | ERTForwardModeling,
    log_resistivity: np.ndarray,
    *,
    include_robin_boundary_derivative: bool = False,
    normal_sensitivity: bool = True,
    cache: ForwardJacobianCache | None = None,
    max_entries: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    if cache is None or max_entries < 1:
        return _forward_and_jacobian_log(
            forward,
            log_resistivity,
            include_robin_boundary_derivative=include_robin_boundary_derivative,
            normal_sensitivity=normal_sensitivity,
        )

    key_array = np.ascontiguousarray(log_resistivity, dtype=np.float64)
    key = (
        tuple(int(size) for size in key_array.shape),
        key_array.dtype.str,
        key_array.tobytes(),
        bool(include_robin_boundary_derivative),
        bool(normal_sensitivity),
    )
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
        return cached

    result = _forward_and_jacobian_log(
        forward,
        log_resistivity,
        include_robin_boundary_derivative=include_robin_boundary_derivative,
        normal_sensitivity=normal_sensitivity,
    )
    cache[key] = result
    cache.move_to_end(key)
    while len(cache) > max_entries:
        cache.popitem(last=False)
    return result


def _forward_log_response(
    forward: ERTForward2p5D | ERTForwardModeling,
    log_resistivity: np.ndarray,
) -> np.ndarray:
    if isinstance(forward, ERTForwardModeling):
        return np.asarray(forward.forward(log_resistivity, log_transform=True), dtype=float)

    if not isinstance(forward, ERTForward2p5D):
        if hasattr(forward, "forward"):
            return np.asarray(forward.forward(log_resistivity, log_transform=True), dtype=float)
        raise TypeError("forward must be ERTForward2p5D, ERTForwardModeling, or expose forward")

    resistivity = np.exp(log_resistivity)
    conductivity = torch_np.asarray(1.0 / resistivity, dtype=FLOAT_DTYPE)
    response = forward.apparent_resistivity_values(conductivity)
    return np.log(np.asarray(response, dtype=float))


def _linearized_objective_gradient(matrix: sp.spmatrix, rhs: np.ndarray) -> np.ndarray:
    """Gradient of ``||A dm - b||^2`` at ``dm = 0`` up to a constant factor."""

    return -np.asarray(matrix.T @ np.asarray(rhs, dtype=float).reshape(-1), dtype=float).reshape(-1)


def _step_limit(config: InversionConfig) -> float | None:
    """Use the explicit Gauss-Newton cap when set, otherwise cap first-order methods."""

    if config.max_log_step is not None:
        return config.max_log_step
    return config.optimizer_max_step


def _as_descent_direction(direction: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    """Fall back to steepest descent if a quasi-Newton/CG update loses descent."""

    direction = np.asarray(direction, dtype=float).reshape(-1)
    gradient = np.asarray(gradient, dtype=float).reshape(-1)
    if direction.shape != gradient.shape:
        raise ValueError("optimizer direction and gradient shape mismatch")
    if not np.all(np.isfinite(direction)) or float(np.dot(direction, gradient)) >= 0.0:
        return -gradient
    return direction


def _lbfgs_direction(gradient: np.ndarray, history: list[tuple[np.ndarray, np.ndarray, float]]) -> np.ndarray:
    """Return the L-BFGS inverse-Hessian direction from stored ``(s, y, rho)`` pairs."""

    if not history:
        return -gradient

    q = np.asarray(gradient, dtype=float).copy()
    alphas: list[float] = []
    for s_vec, y_vec, rho in reversed(history):
        alpha = float(rho * np.dot(s_vec, q))
        alphas.append(alpha)
        q -= alpha * y_vec

    s_last, y_last, _ = history[-1]
    yy = float(np.dot(y_last, y_last))
    gamma = float(np.dot(s_last, y_last) / yy) if yy > 0.0 else 1.0
    r = gamma * q

    for (s_vec, y_vec, rho), alpha in zip(history, reversed(alphas)):
        beta = float(rho * np.dot(y_vec, r))
        r += s_vec * (alpha - beta)
    return -r


def _first_order_optimizer_direction(
    *,
    current_state: np.ndarray,
    gradient: np.ndarray,
    optimizer_state: dict[str, Any],
    config: InversionConfig,
) -> np.ndarray:
    """Build a first-order/quasi-Newton model increment from the current gradient."""

    algorithm = build_optimization_algorithm(config.optimization_algorithm).name
    current = np.asarray(current_state, dtype=float).reshape(-1)
    grad = np.asarray(gradient, dtype=float).reshape(-1)
    if current.shape != grad.shape:
        raise ValueError("current_state and gradient shape mismatch")
    if not np.all(np.isfinite(grad)):
        raise ValueError("optimizer gradient contains non-finite values")
    if not np.any(grad):
        return np.zeros_like(grad)

    if algorithm == "nonlinear_cg":
        previous_gradient = optimizer_state.get("nonlinear_cg_gradient")
        previous_direction = optimizer_state.get("nonlinear_cg_direction")
        if previous_gradient is None or previous_direction is None:
            direction = -grad
        else:
            prev_grad = np.asarray(previous_gradient, dtype=float).reshape(-1)
            prev_dir = np.asarray(previous_direction, dtype=float).reshape(-1)
            denominator = max(float(np.dot(prev_grad, prev_grad)), np.finfo(float).eps)
            beta = max(0.0, float(np.dot(grad, grad - prev_grad) / denominator))
            direction = -grad + beta * prev_dir
        direction = _as_descent_direction(direction, grad)
        optimizer_state["nonlinear_cg_gradient"] = grad.copy()
        optimizer_state["nonlinear_cg_direction"] = direction.copy()
        return _limit_delta(direction, _step_limit(config))

    if algorithm in ("lbfgs", "lbfgs_b"):
        history = optimizer_state.setdefault("lbfgs_history", [])
        previous_state = optimizer_state.get("lbfgs_state")
        previous_gradient = optimizer_state.get("lbfgs_gradient")
        if previous_state is not None and previous_gradient is not None:
            s_vec = current - np.asarray(previous_state, dtype=float).reshape(-1)
            y_vec = grad - np.asarray(previous_gradient, dtype=float).reshape(-1)
            ys = float(np.dot(y_vec, s_vec))
            if ys > 1.0e-12 and np.all(np.isfinite(s_vec)) and np.all(np.isfinite(y_vec)):
                history.append((s_vec.copy(), y_vec.copy(), 1.0 / ys))
                del history[:-int(config.lbfgs_history)]
        direction = _as_descent_direction(_lbfgs_direction(grad, history), grad)
        optimizer_state["lbfgs_state"] = current.copy()
        optimizer_state["lbfgs_gradient"] = grad.copy()
        return _limit_delta(direction, _step_limit(config))

    if algorithm == "adam":
        beta1 = float(config.adam_beta1)
        beta2 = float(config.adam_beta2)
        step_number = int(optimizer_state.get("adam_step", 0)) + 1
        first_moment = np.asarray(optimizer_state.get("adam_m", np.zeros_like(grad)), dtype=float)
        second_moment = np.asarray(optimizer_state.get("adam_v", np.zeros_like(grad)), dtype=float)
        first_moment = beta1 * first_moment + (1.0 - beta1) * grad
        second_moment = beta2 * second_moment + (1.0 - beta2) * (grad * grad)
        m_hat = first_moment / (1.0 - beta1**step_number)
        v_hat = second_moment / (1.0 - beta2**step_number)
        direction = -m_hat / (np.sqrt(v_hat) + float(config.adam_epsilon))
        optimizer_state["adam_step"] = step_number
        optimizer_state["adam_m"] = first_moment
        optimizer_state["adam_v"] = second_moment
        return _limit_delta(direction, _step_limit(config))

    raise ValueError(f"optimizer={config.optimization_algorithm!r} is not a first-order optimizer")


def _add_levenberg_marquardt_damping(
    matrix: sp.spmatrix,
    rhs: np.ndarray,
    *,
    n_parameters: int,
    config: InversionConfig,
) -> tuple[sp.csr_matrix, np.ndarray]:
    if config.lm_damping <= 0.0:
        return matrix.tocsr(), np.asarray(rhs, dtype=float).reshape(-1)
    damping = float(np.sqrt(config.lm_damping))
    damping_matrix = damping * sp.eye(int(n_parameters), format="csr")
    return (
        sp.vstack((matrix, damping_matrix), format="csr"),
        np.concatenate((np.asarray(rhs, dtype=float).reshape(-1), np.zeros(int(n_parameters), dtype=float))),
    )


def _optimizer_increment(
    matrix: sp.spmatrix,
    rhs: np.ndarray,
    *,
    current_state: np.ndarray,
    optimizer_state: dict[str, Any],
    config: InversionConfig,
) -> np.ndarray:
    """Compute a model increment using the configured outer optimization algorithm."""

    algorithm = build_optimization_algorithm(config.optimization_algorithm)
    if algorithm.uses_linearized_solver:
        solve_matrix = matrix.tocsr()
        solve_rhs = np.asarray(rhs, dtype=float).reshape(-1)
        if algorithm.name == "levenberg_marquardt":
            solve_matrix, solve_rhs = _add_levenberg_marquardt_damping(
                solve_matrix,
                solve_rhs,
                n_parameters=int(np.asarray(current_state).size),
                config=config,
            )
        return _limit_delta(_solve_increment(solve_matrix, solve_rhs, config), config.max_log_step)

    gradient = _linearized_objective_gradient(matrix, rhs)
    return _first_order_optimizer_direction(
        current_state=np.asarray(current_state, dtype=float).reshape(-1),
        gradient=gradient,
        optimizer_state=optimizer_state,
        config=config,
    )


def _solve_increment(
    matrix: sp.spmatrix,
    rhs: np.ndarray,
    config: InversionConfig,
) -> np.ndarray:
    if config.linearized_solver == "gpu_cgls":
        return _cupy_cgls(
            matrix,
            rhs,
            max_iterations=config.cgls_max_iterations,
            tolerance=config.cgls_tolerance,
        )

    if config.linearized_solver == "pyhydro_cgls":
        normal_matrix = (matrix.T @ matrix).tocsr()
        normal_rhs = np.asarray(matrix.T @ rhs, dtype=float).reshape(-1, 1)
        solution = _pyhydro_cgls(
            normal_matrix,
            normal_rhs,
            max_iterations=config.cgls_max_iterations,
            tolerance=config.cgls_tolerance,
        ).ravel()
        if not np.all(np.isfinite(solution)):
            raise ValueError("linearized inversion update contains non-finite values")
        return solution

    if config.linearized_solver == "normal_cg":
        normal_matrix = (matrix.T @ matrix).tocsr()
        normal_rhs = np.asarray(matrix.T @ rhs, dtype=float).ravel()
        solution, info = cg(
            normal_matrix,
            normal_rhs,
            rtol=config.cgls_tolerance,
            atol=0.0,
            maxiter=config.cgls_max_iterations,
        )
        if info < 0:
            raise ValueError(f"normal_cg failed with illegal input/info={info}")
        if not np.all(np.isfinite(solution)):
            raise ValueError("linearized inversion update contains non-finite values")
        return np.asarray(solution, dtype=float)

    solution = lsqr(
        matrix,
        rhs,
        atol=config.lsqr_atol,
        btol=config.lsqr_btol,
        iter_lim=config.lsqr_iter_limit,
    )[0]
    if not np.all(np.isfinite(solution)):
        raise ValueError("linearized inversion update contains non-finite values")
    return solution


def _cupy_cgls(
    matrix: sp.spmatrix,
    rhs: np.ndarray,
    *,
    max_iterations: int,
    tolerance: float,
) -> np.ndarray:
    """Solve ``min ||A x - b||`` with CGLS using CuPy sparse matvecs."""

    try:
        import cupy as cp
        import cupyx.scipy.sparse as cupy_sparse
    except ImportError as exc:
        raise ImportError("linearized_solver='gpu_cgls' requires CuPy") from exc

    system_cpu = matrix.tocsr()
    dtype = np.float64 if system_cpu.dtype == np.float64 or np.asarray(rhs).dtype == np.float64 else np.float32
    system = cupy_sparse.csr_matrix(
        (
            cp.asarray(system_cpu.data, dtype=dtype),
            cp.asarray(system_cpu.indices, dtype=cp.int32),
            cp.asarray(system_cpu.indptr, dtype=cp.int32),
        ),
        shape=system_cpu.shape,
    )
    b = cp.asarray(np.asarray(rhs, dtype=dtype).ravel())
    x = cp.zeros((system.shape[1],), dtype=dtype)
    r = b.copy()
    s = system.T @ r
    p = s.copy()
    gamma = cp.dot(s, s)
    rr0 = cp.dot(r, r)
    gamma_value = float(gamma)
    gamma0_value = gamma_value
    rr0_value = float(rr0)
    if gamma_value <= 0.0 or rr0_value <= 0.0:
        return cp.asnumpy(x)

    for _ in range(int(max_iterations)):
        q = system @ p
        denominator = cp.dot(q, q)
        denominator_value = float(denominator)
        if denominator_value <= 0.0:
            break
        alpha = gamma / denominator
        x = x + alpha * p
        r = r - alpha * q
        s = system.T @ r
        gamma_new = cp.dot(s, s)
        gamma_new_value = float(gamma_new)
        if gamma_new_value <= 0.0:
            break
        if gamma_new_value / gamma0_value < float(tolerance):
            break
        p = s + (gamma_new / gamma) * p
        gamma = gamma_new

    solution = cp.asnumpy(x)
    if not np.all(np.isfinite(solution)):
        raise ValueError("linearized inversion update contains non-finite values")
    return np.asarray(solution, dtype=float)


def _pyhydro_cgls(
    matrix: sp.spmatrix,
    rhs: np.ndarray,
    *,
    max_iterations: int,
    tolerance: float,
) -> np.ndarray:
    """Replicate PyHydroGeophysX's CGLS routine for the linearized update."""

    system = matrix.tocsr() if sp.issparse(matrix) else np.asarray(matrix, dtype=float)
    b = np.asarray(rhs, dtype=float)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    x = np.zeros((system.shape[1], 1), dtype=float)
    r = b.copy()
    s = system.T.dot(r)
    if np.ndim(s) == 1:
        s = np.asarray(s, dtype=float).reshape(-1, 1)
    else:
        s = np.asarray(s, dtype=float)
    p = s.copy()
    gamma = float((s.T @ s).item())
    rr = float((r.T @ r).item())
    rr0 = rr
    if rr0 <= 0.0 or gamma <= 0.0:
        return x

    for _ in range(int(max_iterations)):
        q = system.dot(p)
        if np.ndim(q) == 1:
            q = np.asarray(q, dtype=float).reshape(-1, 1)
        else:
            q = np.asarray(q, dtype=float)
        denominator = float((q.T @ q).item())
        if denominator <= 0.0:
            break
        alpha = gamma / denominator
        x += alpha * p
        r -= alpha * q
        s = system.T.dot(r)
        if np.ndim(s) == 1:
            s = np.asarray(s, dtype=float).reshape(-1, 1)
        else:
            s = np.asarray(s, dtype=float)
        gamma_new = float((s.T @ s).item())
        if gamma <= 0.0:
            break
        p = s + float(gamma_new / gamma) * p
        gamma = gamma_new
        rr = float((r.T @ r).item())
        if rr / rr0 < float(tolerance):
            break
    return x


def invert_single_log_resistivity(
    forward: ERTForward2p5D | ERTForwardModeling,
    observed_data: ArrayLike,
    initial_model: ArrayLike,
    *,
    reference_model: ArrayLike | None = None,
    config: InversionConfig | None = None,
    observed_log_data: bool = False,
    initial_log_model: bool = False,
    reference_log_model: bool = False,
) -> ERTInversionResult:
    """Invert one ERT dataset for cell log-resistivity.

    ``observed_data`` is interpreted as apparent resistivity unless
    ``observed_log_data=True``. ``initial_model`` and ``reference_model`` are
    interpreted as resistivity unless their corresponding ``*_log_model`` flag
    is set.
    """

    config = config or InversionConfig()
    _check_config(config)

    n_cells = _model_size(forward)
    n_data = _measurement_count(forward)
    observed_log = _as_observed_log_vector(observed_data, log_data=observed_log_data, expected_size=n_data)
    weight = _weights(config.data_std, observed_log.shape)
    data_misfit = build_data_misfit(config.data_misfit)
    if _is_log_data_difference_misfit(data_misfit):
        raise ValueError("log data-difference misfit is only defined for time-lapse inversions")
    initial_log = _as_log_model(
        initial_model,
        expected_size=n_cells,
        log_model=initial_log_model,
        name="initial_model",
    )
    model = _log_model_to_state(initial_log, config)
    if reference_model is None:
        reference = model.copy() if config.spatial_regularization == "identity" else None
    else:
        reference_log = _as_log_model(
            reference_model,
            expected_size=n_cells,
            log_model=reference_log_model,
            name="reference_model",
        )
        reference = _log_model_to_state(reference_log, config)

    iteration_chi2: list[float] = []
    predicted_log = np.empty_like(observed_log)
    jacobian = np.empty((n_data, n_cells), dtype=float)
    linearization_valid = False
    optimizer_state: dict[str, Any] = {}
    spatial_regularization = build_spatial_regularization(config.spatial_regularization)
    regularization_matrix = spatial_regularization.matrix(forward, n_cells, z_weight=config.z_weight)

    _emit_progress(
        config,
        "single_start",
        n_cells=int(n_cells),
        n_data=int(n_data),
        max_iterations=int(config.max_iterations),
    )
    stop_reason = "max_iterations"
    for iteration_index in range(config.max_iterations):
        iteration = iteration_index + 1
        _emit_progress(
            config,
            "single_iteration_start",
            iteration=int(iteration),
            max_iterations=int(config.max_iterations),
        )
        if not linearization_valid:
            log_model = _state_to_log_model(model, config)
            predicted_log, jacobian_log = _forward_and_jacobian_log(
                forward,
                log_model,
                include_robin_boundary_derivative=config.include_robin_boundary_derivative,
                normal_sensitivity=config.normal_sensitivity,
            )
            jacobian = jacobian_log * _d_log_model_d_state(model, config)[None, :]
        data_matrix_values, data_rhs = data_misfit.linearized_system(
            predicted_log,
            observed_log,
            weight,
            jacobian,
        )
        data_matrix = sp.csr_matrix(data_matrix_values)
        rhs_blocks = [data_rhs]
        matrix_blocks: list[sp.spmatrix] = [data_matrix]
        reference_roughness = np.zeros(regularization_matrix.shape[0], dtype=float)
        objective_matrix = regularization_matrix
        objective_reference = reference_roughness
        objective_regularization = config.regularization

        if config.regularization > 0.0:
            scale = float(np.sqrt(config.regularization))
            regularization_domain = _normalize_regularization_domain(config.regularization_domain)
            if regularization_domain == "state":
                domain_model = model
                projection = None
                reference_domain = reference
            else:
                domain_model, d_domain_d_state = _regularization_domain_value_and_derivative(model, config)
                projection = _chain_rule_projection(d_domain_d_state)
                reference_domain = (
                    None
                    if reference is None
                    else _regularization_domain_value_and_derivative(reference, config)[0]
                )

            current_roughness = regularization_matrix @ domain_model
            if config.regularization_mode == "update":
                reference_roughness = current_roughness
            elif reference_domain is None:
                if spatial_regularization.name == "identity":
                    reference_roughness = current_roughness
                else:
                    reference_roughness = np.zeros_like(current_roughness)
            else:
                reference_roughness = regularization_matrix @ reference_domain

            if spatial_regularization.name in ("identity", "first_order"):
                spatial_matrix_domain = scale * regularization_matrix
                spatial_rhs = scale * (reference_roughness - current_roughness)
                if projection is None:
                    spatial_matrix = spatial_matrix_domain
                else:
                    spatial_matrix = (spatial_matrix_domain @ projection).tocsr()
            else:
                spatial_matrix_domain, spatial_rhs = spatial_regularization.linearized_system(
                    forward,
                    np.asarray(domain_model, dtype=float),
                    n_cells,
                    reference_roughness=reference_roughness,
                    scale=scale,
                    z_weight=config.z_weight,
                )
                if projection is None:
                    spatial_matrix = spatial_matrix_domain
                else:
                    spatial_matrix = (spatial_matrix_domain @ projection).tocsr()
            objective_reference = spatial_matrix @ model + spatial_rhs
            matrix_blocks.append(spatial_matrix)
            rhs_blocks.append(spatial_rhs)
            objective_matrix = spatial_matrix
            objective_regularization = 1.0

        matrix = sp.vstack(matrix_blocks, format="csr")
        rhs = np.concatenate(rhs_blocks)
        delta = _optimizer_increment(
            matrix,
            rhs,
            current_state=model,
            optimizer_state=optimizer_state,
            config=config,
        )
        step = config.step_length * delta
        candidate_model = _clip_model_state(model + step, config)
        candidate_step = candidate_model - model

        log_model = _state_to_log_model(candidate_model, config)
        candidate_predicted_log, candidate_jacobian_log = _forward_and_jacobian_log(
            forward,
            log_model,
            include_robin_boundary_derivative=config.include_robin_boundary_derivative,
            normal_sensitivity=config.normal_sensitivity,
        )
        candidate_jacobian = candidate_jacobian_log * _d_log_model_d_state(candidate_model, config)[None, :]

        actual_step = candidate_step
        if config.line_search:
            tau = _line_search_tau(
                state=model,
                step=candidate_step,
                predicted_log=predicted_log,
                candidate_predicted_log=candidate_predicted_log,
                observed_log=observed_log,
                weights=weight,
                regularization_matrix=objective_matrix,
                reference_roughness=np.asarray(objective_reference, dtype=float),
                regularization=objective_regularization,
                data_misfit=data_misfit,
            )
            actual_step = tau * candidate_step
            if tau < 0.95:
                model = _clip_model_state(model + actual_step, config)
                log_model = _state_to_log_model(model, config)
                predicted_log, jacobian_log = _forward_and_jacobian_log(
                    forward,
                    log_model,
                    include_robin_boundary_derivative=config.include_robin_boundary_derivative,
                    normal_sensitivity=config.normal_sensitivity,
                )
                jacobian = jacobian_log * _d_log_model_d_state(model, config)[None, :]
            else:
                model = candidate_model
                predicted_log = candidate_predicted_log
                jacobian = candidate_jacobian
        else:
            model = candidate_model
            predicted_log = candidate_predicted_log
            jacobian = candidate_jacobian

        linearization_valid = True
        chi2 = data_misfit.chi2(predicted_log, observed_log, weight)
        iteration_chi2.append(chi2)
        step_metric = float(np.linalg.norm(actual_step) / max(float(np.sqrt(n_cells)), 1.0))
        _emit_progress(
            config,
            "single_iteration_done",
            iteration=int(iteration),
            max_iterations=int(config.max_iterations),
            chi2=float(chi2),
            step_norm=step_metric,
            target_chi2=None if config.target_chi2 is None else float(config.target_chi2),
        )
        if config.target_chi2 is not None and chi2 < config.target_chi2:
            stop_reason = "target_chi2"
            break
        if step_metric < config.step_tolerance:
            stop_reason = "step_tolerance"
            break

    coverage = _pygimli_style_coverage_from_jacobian(forward, jacobian)
    predicted_data = np.exp(predicted_log)
    final_log_model = _state_to_log_model(model, config)
    final_parameter_model = _parameter_model_from_state(model, config)
    final_parameter_name = _parameter_name_from_state(model, config)
    _emit_progress(
        config,
        "single_done",
        iterations=int(len(iteration_chi2)),
        max_iterations=int(config.max_iterations),
        final_chi2=float(iteration_chi2[-1]) if iteration_chi2 else None,
        stop_reason=stop_reason,
    )
    return ERTInversionResult(
        final_model=np.exp(final_log_model),
        final_log_model=final_log_model,
        predicted_data=predicted_data,
        predicted_log_data=predicted_log,
        coverage=coverage,
        iteration_chi2=iteration_chi2,
        final_parameter_model=final_parameter_model,
        final_parameter_name=final_parameter_name,
    )


def _timelapse_data_system(
    data_misfit: DataMisfit,
    predicted_log: np.ndarray,
    observed_log: np.ndarray,
    weight: np.ndarray,
    jacobians: list[np.ndarray],
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Build the time-lapse data term, including coupled data-difference rows."""

    if not _is_log_data_difference_misfit(data_misfit):
        data_blocks: list[sp.csr_matrix] = []
        rhs_blocks: list[np.ndarray] = []
        for time_index, jac_t in enumerate(jacobians):
            data_matrix_t, data_rhs_t = data_misfit.linearized_system(
                predicted_log[time_index],
                observed_log[time_index],
                weight[time_index],
                jac_t,
            )
            data_blocks.append(sp.csr_matrix(data_matrix_t))
            rhs_blocks.append(data_rhs_t)
        return sp.block_diag(data_blocks, format="csr"), np.concatenate(rhs_blocks)

    n_times = int(predicted_log.shape[0])
    if n_times < 2:
        raise ValueError("log data-difference misfit requires at least two time steps")
    n_cells = int(jacobians[0].shape[1])
    zero = sp.csr_matrix((predicted_log.shape[1], n_cells))
    matrix_rows: list[sp.csr_matrix] = []
    rhs_rows: list[np.ndarray] = []

    w0 = weight[0]
    row_blocks = [sp.csr_matrix(jacobians[0] * w0[:, None])]
    row_blocks.extend([zero] * (n_times - 1))
    matrix_rows.append(sp.hstack(row_blocks, format="csr"))
    rhs_rows.append(-((predicted_log[0] - observed_log[0]) * w0))

    diff_weights = _difference_weights(weight)
    for time_index in range(1, n_times):
        w_t = diff_weights[time_index - 1]
        residual = (predicted_log[time_index] - predicted_log[0]) - (
            observed_log[time_index] - observed_log[0]
        )
        row_blocks = []
        for block_index in range(n_times):
            if block_index == 0:
                row_blocks.append(sp.csr_matrix(-jacobians[0] * w_t[:, None]))
            elif block_index == time_index:
                row_blocks.append(sp.csr_matrix(jacobians[time_index] * w_t[:, None]))
            else:
                row_blocks.append(zero)
        matrix_rows.append(sp.hstack(row_blocks, format="csr"))
        rhs_rows.append(-(residual * w_t))

    rhs = np.concatenate(rhs_rows)
    return sp.vstack(matrix_rows, format="csr"), rhs


def _timelapse_data_chi2(
    data_misfit: DataMisfit,
    predicted_log: np.ndarray,
    observed_log: np.ndarray,
    weight: np.ndarray,
) -> float:
    if _is_log_data_difference_misfit(data_misfit):
        return _log_data_difference_chi2(predicted_log, observed_log, weight)
    return data_misfit.chi2(predicted_log, observed_log, weight)


def _timelapse_data_phi(
    data_misfit: DataMisfit,
    predicted_log: np.ndarray,
    observed_log: np.ndarray,
    weight: np.ndarray,
) -> float:
    if _is_log_data_difference_misfit(data_misfit):
        return _log_data_difference_phi(predicted_log, observed_log, weight)
    return data_misfit.phi(predicted_log, observed_log, weight)


def _model_difference_spatial_system(
    spatial_regularization: sp.spmatrix,
    current_vec: np.ndarray,
    reference_vec: np.ndarray | None,
    *,
    n_cells: int,
    n_times: int,
    scale: float,
    regularization_mode: str,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Build spatial constraints on the baseline model and time-lapse changes."""

    base = spatial_regularization.tocsr()
    zero = sp.csr_matrix(base.shape)
    rows: list[sp.csr_matrix] = []
    rows.append(sp.hstack([base, *([zero] * (n_times - 1))], format="csr"))
    for time_index in range(1, n_times):
        blocks = []
        for block_index in range(n_times):
            if block_index == 0:
                blocks.append(-base)
            elif block_index == time_index:
                blocks.append(base)
            else:
                blocks.append(zero)
        rows.append(sp.hstack(blocks, format="csr"))
    unscaled = sp.vstack(rows, format="csr")
    current_roughness = unscaled @ current_vec
    if regularization_mode == "update":
        reference_roughness = current_roughness
    elif reference_vec is None:
        reference_roughness = np.zeros_like(current_roughness)
    else:
        reference_roughness = unscaled @ reference_vec
    matrix = float(scale) * unscaled
    return matrix, float(scale) * (reference_roughness - current_roughness), float(scale) * reference_roughness


def invert_timelapse_log_resistivity(
    forward: ERTForward2p5D | ERTForwardModeling,
    observed_data: ArrayLike,
    initial_model: ArrayLike,
    *,
    reference_model: ArrayLike | None = None,
    config: InversionConfig | None = None,
    observed_log_data: bool = False,
    initial_log_model: bool = False,
    reference_log_model: bool = False,
    _forward_jacobian_cache: ForwardJacobianCache | None = None,
    _forward_jacobian_cache_max_entries: int = 128,
) -> TimeLapseERTInversionResult:
    """Jointly invert time-lapse ERT data with optional temporal smoothing.

    Observations are accepted as ``(n_times, n_measurements)`` or
    ``(n_measurements, n_times)``. Returned models are shaped
    ``(n_cells, n_times)`` to match the notebook artifact convention.
    """

    config = config or InversionConfig(temporal_regularization=1.0)
    _check_config(config)

    n_cells = _model_size(forward)
    n_measurements = _measurement_count(forward)
    observed_log = _as_observed_log_matrix(
        observed_data,
        log_data=observed_log_data,
        measurement_count=n_measurements,
    )
    n_times = int(observed_log.shape[0])
    if n_times < 2:
        raise ValueError("time-lapse inversion needs at least two timesteps")

    weight = _weights(config.data_std, observed_log.shape)
    data_misfit = build_data_misfit(config.data_misfit)
    initial_logs = _as_log_model_matrix(
        initial_model,
        expected_size=n_cells,
        n_times=n_times,
        log_model=initial_log_model,
        name="initial_model",
    )
    models = _log_model_to_state(initial_logs, config)
    baseline_state: np.ndarray | None = None
    if bool(config.freeze_first_timestep):
        baseline_state = np.asarray(models[:, 0], dtype=float).copy()
        models[:, 0] = baseline_state
    if reference_model is None:
        reference = (
            models.copy()
            if config.spatial_regularization == "identity"
            and config.temporal_regularization_mode == "separate"
            else None
        )
    else:
        reference_logs = _as_log_model_matrix(
            reference_model,
            expected_size=n_cells,
            n_times=n_times,
            log_model=reference_log_model,
            name="reference_model",
        )
        reference = _log_model_to_state(reference_logs, config)

    total_size = n_cells * n_times
    iteration_chi2: list[float] = []
    predicted_log = np.empty_like(observed_log)
    jacobians: list[np.ndarray] = []
    linearization_valid = False
    optimizer_state: dict[str, Any] = {}
    spatial_regularization_obj = build_spatial_regularization(config.spatial_regularization)
    spatial_regularization = spatial_regularization_obj.matrix(forward, n_cells, z_weight=config.z_weight)
    spatial_regularization_all = sp.block_diag(
        [spatial_regularization] * n_times,
        format="csr",
    )
    temporal_regularization = build_temporal_regularization(config.temporal_regularization_type)
    sensor_constraint = _prepare_sensor_constraint(
        config,
        n_cells=n_cells,
        n_times=n_times,
    )

    _emit_progress(
        config,
        "timelapse_start",
        n_cells=int(n_cells),
        n_measurements=int(n_measurements),
        n_times=int(n_times),
        max_iterations=int(config.max_iterations),
    )
    stop_reason = "max_iterations"
    for iteration_index in range(config.max_iterations):
        iteration = iteration_index + 1
        _emit_progress(
            config,
            "timelapse_iteration_start",
            iteration=int(iteration),
            max_iterations=int(config.max_iterations),
            n_times=int(n_times),
        )
        if not linearization_valid:
            predicted_rows: list[np.ndarray] = []
            jacobians = []
            for time_index in range(n_times):
                _emit_progress(
                    config,
                    "timelapse_time_start",
                    iteration=int(iteration),
                    max_iterations=int(config.max_iterations),
                    stage="linearization",
                    time_index=int(time_index),
                    time_number=int(time_index + 1),
                    n_times=int(n_times),
                )
                state_t = models[:, time_index]
                time_config = _config_for_time_index(config, time_index)
                pred_t, jac_log_t = _forward_and_jacobian_log_cached(
                    forward,
                    _state_to_log_model(state_t, time_config),
                    include_robin_boundary_derivative=config.include_robin_boundary_derivative,
                    normal_sensitivity=config.normal_sensitivity,
                    cache=_forward_jacobian_cache,
                    max_entries=_forward_jacobian_cache_max_entries,
                )
                predicted_rows.append(pred_t)
                jacobians.append(jac_log_t * _d_log_model_d_state(state_t, time_config)[None, :])
                _emit_progress(
                    config,
                    "timelapse_time_done",
                    iteration=int(iteration),
                    max_iterations=int(config.max_iterations),
                    stage="linearization",
                    time_index=int(time_index),
                    time_number=int(time_index + 1),
                    n_times=int(n_times),
                )
            predicted_log = np.vstack(predicted_rows)

        data_matrix, data_rhs = _timelapse_data_system(
            data_misfit,
            predicted_log,
            observed_log,
            weight,
            jacobians,
        )

        matrix_blocks: list[sp.spmatrix] = [data_matrix]
        rhs_all: list[np.ndarray] = [data_rhs]
        objective_blocks: list[sp.spmatrix] = []
        objective_references: list[np.ndarray] = []

        current_vec = models.reshape(total_size, order="F")
        reference_vec = None if reference is None else reference.reshape(total_size, order="F")
        regularization_domain = _normalize_regularization_domain(config.regularization_domain)
        needs_projection = (
            config.regularization > 0.0
            or (config.temporal_regularization_mode == "separate" and config.temporal_regularization > 0.0)
            or config.sensor_constraint > 0.0
        )
        use_physical_regularization = regularization_domain == "physical" and needs_projection
        if use_physical_regularization:
            domain_models, domain_derivative = _regularization_domain_value_and_derivative(models, config)
            domain_vec = np.asarray(domain_models, dtype=float).reshape(total_size, order="F")
            projection_all = _chain_rule_projection(domain_derivative, order="F")
            if reference is None:
                reference_domain = None
                reference_domain_vec = None
            else:
                reference_domain, _ = _regularization_domain_value_and_derivative(reference, config)
                reference_domain_vec = np.asarray(reference_domain, dtype=float).reshape(total_size, order="F")
        else:
            domain_models = np.asarray(models, dtype=float)
            domain_derivative = np.ones_like(domain_models, dtype=float)
            domain_vec = current_vec
            projection_all = None
            reference_domain = reference
            reference_domain_vec = reference_vec

        if (
            config.temporal_regularization_mode == "joint_frame"
            and config.temporal_regularization > 0.0
            and temporal_regularization.name not in ("first_order_l2", "second_order_l2")
        ):
            raise ValueError(
                "robust temporal regularization is currently supported for "
                "temporal_regularization_mode='separate' only"
            )
        if (
            config.temporal_regularization_mode == "joint_frame"
            and config.regularization > 0.0
            and spatial_regularization_obj.name not in ("identity", "first_order")
        ):
            raise ValueError(
                "robust spatial regularization is currently supported for "
                "temporal_regularization_mode='separate' only"
            )
        if config.temporal_regularization_mode == "joint_frame":
            if config.regularization > 0.0:
                frame_blocks: list[sp.spmatrix] = [spatial_regularization_all]
                if config.temporal_regularization > 0.0:
                    frame_blocks.append(
                        temporal_regularization.matrix(
                            n_cells,
                            n_times,
                            scale=float(config.temporal_regularization),
                        )
                    )
                frame_constraint_domain = sp.vstack(frame_blocks, format="csr")
                current_roughness = frame_constraint_domain @ domain_vec
                if config.regularization_mode == "update":
                    reference_roughness = current_roughness
                elif reference_domain_vec is None:
                    reference_roughness = np.zeros_like(current_roughness)
                else:
                    reference_roughness = frame_constraint_domain @ reference_domain_vec
                scale = float(np.sqrt(config.regularization))
                if projection_all is None:
                    frame_constraint = frame_constraint_domain
                else:
                    frame_constraint = (frame_constraint_domain @ projection_all).tocsr()
                frame_matrix = scale * frame_constraint
                frame_rhs = scale * (reference_roughness - current_roughness)
                matrix_blocks.append(frame_matrix)
                rhs_all.append(frame_rhs)
                objective_blocks.append(frame_matrix)
                objective_references.append(frame_matrix @ current_vec + frame_rhs)
        elif config.regularization > 0.0:
            scale = float(np.sqrt(config.regularization))
            if spatial_regularization_obj.name == "model_difference_smoothness":
                spatial_matrix_domain_all, spatial_rhs_all, _ = (
                    _model_difference_spatial_system(
                        spatial_regularization,
                        domain_vec,
                        reference_domain_vec,
                        n_cells=n_cells,
                        n_times=n_times,
                        scale=scale,
                        regularization_mode=config.regularization_mode,
                    )
                )
                if projection_all is None:
                    spatial_matrix_all = spatial_matrix_domain_all
                else:
                    spatial_matrix_all = (spatial_matrix_domain_all @ projection_all).tocsr()
                spatial_objective_reference_all = spatial_matrix_all @ current_vec + spatial_rhs_all
            elif spatial_regularization_obj.name in ("identity", "first_order"):
                spatial_matrix_domain_all = scale * spatial_regularization_all
                current_roughness = spatial_regularization_all @ domain_vec
                if config.regularization_mode == "update":
                    reference_roughness = current_roughness
                elif reference_domain is None:
                    if spatial_regularization_obj.name == "identity":
                        reference_roughness = current_roughness
                    else:
                        reference_roughness = np.zeros_like(current_roughness)
                else:
                    reference_roughness = spatial_regularization_all @ reference_domain_vec
                spatial_rhs_all = scale * (reference_roughness - current_roughness)
                if projection_all is None:
                    spatial_matrix_all = spatial_matrix_domain_all
                else:
                    spatial_matrix_all = (spatial_matrix_domain_all @ projection_all).tocsr()
                spatial_objective_reference_all = spatial_matrix_all @ current_vec + spatial_rhs_all
            else:
                spatial_matrices: list[sp.spmatrix] = []
                spatial_rhs_rows: list[np.ndarray] = []
                spatial_objective_references: list[np.ndarray] = []
                for time_index in range(n_times):
                    state_t = models[:, time_index]
                    domain_t = np.asarray(domain_models[:, time_index], dtype=float)
                    current_roughness_t = spatial_regularization @ domain_t
                    if config.regularization_mode == "update":
                        reference_roughness_t = current_roughness_t
                    elif reference_domain is None:
                        reference_roughness_t = np.zeros_like(current_roughness_t)
                    else:
                        reference_roughness_t = spatial_regularization @ np.asarray(
                            reference_domain[:, time_index],
                            dtype=float,
                        )
                    spatial_matrix_domain_t, spatial_rhs_t = spatial_regularization_obj.linearized_system(
                        forward,
                        domain_t,
                        n_cells,
                        reference_roughness=reference_roughness_t,
                        scale=scale,
                        z_weight=config.z_weight,
                    )
                    if projection_all is None:
                        spatial_matrix_t = spatial_matrix_domain_t
                    else:
                        chain_t = _chain_rule_projection(domain_derivative[:, time_index], order="C")
                        spatial_matrix_t = (spatial_matrix_domain_t @ chain_t).tocsr()
                    spatial_matrices.append(spatial_matrix_t)
                    spatial_rhs_rows.append(spatial_rhs_t)
                    spatial_objective_references.append(spatial_matrix_t @ state_t + spatial_rhs_t)
                spatial_matrix_all = sp.block_diag(spatial_matrices, format="csr")
                spatial_rhs_all = np.concatenate(spatial_rhs_rows)
                spatial_objective_reference_all = np.concatenate(spatial_objective_references)
            matrix_blocks.append(spatial_matrix_all)
            rhs_all.append(spatial_rhs_all)
            objective_blocks.append(spatial_matrix_all)
            objective_references.append(spatial_objective_reference_all)

        if config.temporal_regularization_mode == "separate" and config.temporal_regularization > 0.0:
            scale = float(np.sqrt(config.temporal_regularization))
            if temporal_regularization.name == "active_time_constraint":
                temporal_matrix_domain, temporal_rhs = temporal_regularization.linearized_system(
                    domain_vec,
                    n_cells,
                    n_times,
                    scale=scale,
                    threshold=config.active_time_threshold,
                    minimum_weight=config.active_time_minimum_weight,
                )
            else:
                temporal_matrix_domain, temporal_rhs = temporal_regularization.linearized_system(
                    domain_vec,
                    n_cells,
                    n_times,
                    scale=scale,
                )
            if projection_all is None:
                temporal_matrix = temporal_matrix_domain
            else:
                temporal_matrix = (temporal_matrix_domain @ projection_all).tocsr()
            matrix_blocks.append(temporal_matrix)
            rhs_all.append(temporal_rhs)
            objective_blocks.append(temporal_matrix)
            objective_references.append(temporal_matrix @ current_vec + temporal_rhs)

        if sensor_constraint is not None and config.sensor_constraint > 0.0:
            sensor_matrix_domain, sensor_target_scaled = sensor_constraint
            sensor_scale = float(np.sqrt(config.sensor_constraint))
            sensor_predicted_scaled = sensor_matrix_domain @ domain_vec
            sensor_rhs = sensor_scale * (sensor_target_scaled - sensor_predicted_scaled)
            if projection_all is None:
                sensor_matrix = sensor_scale * sensor_matrix_domain
            else:
                sensor_matrix = (sensor_scale * (sensor_matrix_domain @ projection_all)).tocsr()
            matrix_blocks.append(sensor_matrix)
            rhs_all.append(sensor_rhs)
            objective_blocks.append(sensor_matrix)
            objective_references.append(sensor_matrix @ current_vec + sensor_rhs)

        matrix = sp.vstack(matrix_blocks, format="csr")
        rhs = np.concatenate(rhs_all)
        delta_vec = _optimizer_increment(
            matrix,
            rhs,
            current_state=current_vec,
            optimizer_state=optimizer_state,
            config=config,
        )
        delta = delta_vec.reshape((n_cells, n_times), order="F")
        step = config.step_length * delta
        candidate_models = _clip_model_state(models + step, config)
        if baseline_state is not None:
            candidate_models[:, 0] = baseline_state
        candidate_step_vec = candidate_models.reshape(total_size, order="F") - current_vec

        candidate_rows = []
        candidate_jacobians = []
        for time_index in range(n_times):
            _emit_progress(
                config,
                "timelapse_time_start",
                iteration=int(iteration),
                max_iterations=int(config.max_iterations),
                stage="candidate",
                time_index=int(time_index),
                time_number=int(time_index + 1),
                n_times=int(n_times),
            )
            state_t = candidate_models[:, time_index]
            time_config = _config_for_time_index(config, time_index)
            pred_t, jac_log_t = _forward_and_jacobian_log_cached(
                forward,
                _state_to_log_model(state_t, time_config),
                include_robin_boundary_derivative=config.include_robin_boundary_derivative,
                normal_sensitivity=config.normal_sensitivity,
                cache=_forward_jacobian_cache,
                max_entries=_forward_jacobian_cache_max_entries,
            )
            candidate_rows.append(pred_t)
            candidate_jacobians.append(jac_log_t * _d_log_model_d_state(state_t, time_config)[None, :])
            _emit_progress(
                config,
                "timelapse_time_done",
                iteration=int(iteration),
                max_iterations=int(config.max_iterations),
                stage="candidate",
                time_index=int(time_index),
                time_number=int(time_index + 1),
                n_times=int(n_times),
            )
        candidate_predicted_log = np.vstack(candidate_rows)

        actual_step_vec = candidate_step_vec
        if config.line_search:
            if objective_blocks:
                objective_matrix = sp.vstack(objective_blocks, format="csr")
                objective_reference = np.concatenate(objective_references)
            else:
                objective_matrix = sp.csr_matrix((0, total_size))
                objective_reference = np.zeros((0,), dtype=float)
            tau = _line_search_tau(
                state=current_vec,
                step=candidate_step_vec,
                predicted_log=predicted_log.ravel(),
                candidate_predicted_log=candidate_predicted_log.ravel(),
                observed_log=observed_log.ravel(),
                weights=weight.ravel(),
                regularization_matrix=objective_matrix,
                reference_roughness=objective_reference,
                regularization=1.0,
                data_misfit=data_misfit,
                data_phi=lambda values: _timelapse_data_phi(
                    data_misfit,
                    np.asarray(values, dtype=float).reshape(observed_log.shape),
                    observed_log,
                    weight,
                ),
            )
            actual_step_vec = tau * candidate_step_vec
            if tau < 0.95:
                models = _clip_model_state((current_vec + actual_step_vec).reshape((n_cells, n_times), order="F"), config)
                if baseline_state is not None:
                    models[:, 0] = baseline_state
                predicted_rows = []
                jacobians = []
                for time_index in range(n_times):
                    _emit_progress(
                        config,
                        "timelapse_time_start",
                        iteration=int(iteration),
                        max_iterations=int(config.max_iterations),
                        stage="line_search",
                        time_index=int(time_index),
                        time_number=int(time_index + 1),
                        n_times=int(n_times),
                    )
                    state_t = models[:, time_index]
                    time_config = _config_for_time_index(config, time_index)
                    pred_t, jac_log_t = _forward_and_jacobian_log_cached(
                        forward,
                        _state_to_log_model(state_t, time_config),
                        include_robin_boundary_derivative=config.include_robin_boundary_derivative,
                        normal_sensitivity=config.normal_sensitivity,
                        cache=_forward_jacobian_cache,
                        max_entries=_forward_jacobian_cache_max_entries,
                    )
                    predicted_rows.append(pred_t)
                    jacobians.append(jac_log_t * _d_log_model_d_state(state_t, time_config)[None, :])
                    _emit_progress(
                        config,
                        "timelapse_time_done",
                        iteration=int(iteration),
                        max_iterations=int(config.max_iterations),
                        stage="line_search",
                        time_index=int(time_index),
                        time_number=int(time_index + 1),
                        n_times=int(n_times),
                    )
                predicted_log = np.vstack(predicted_rows)
            else:
                models = candidate_models
                predicted_log = candidate_predicted_log
                jacobians = candidate_jacobians
        else:
            models = candidate_models
            predicted_log = candidate_predicted_log
            jacobians = candidate_jacobians

        if baseline_state is not None:
            models[:, 0] = baseline_state

        linearization_valid = True
        chi2 = _timelapse_data_chi2(data_misfit, predicted_log, observed_log, weight)
        iteration_chi2.append(chi2)
        step_metric = float(np.linalg.norm(actual_step_vec) / max(float(np.sqrt(total_size)), 1.0))
        _emit_progress(
            config,
            "timelapse_iteration_done",
            iteration=int(iteration),
            max_iterations=int(config.max_iterations),
            n_times=int(n_times),
            chi2=float(chi2),
            step_norm=step_metric,
            target_chi2=None if config.target_chi2 is None else float(config.target_chi2),
        )
        if config.target_chi2 is not None and chi2 < config.target_chi2:
            stop_reason = "target_chi2"
            break
        if step_metric < config.step_tolerance:
            stop_reason = "step_tolerance"
            break

    all_coverage = [_pygimli_style_coverage_from_jacobian(forward, jac_t) for jac_t in jacobians]
    coverage = np.nanmedian(np.column_stack(all_coverage), axis=1)
    if baseline_state is not None:
        models[:, 0] = baseline_state
    final_log_models = _state_to_log_model(models, config)
    final_parameter_models = _parameter_model_from_state(models, config)
    final_parameter_name = _parameter_name_from_state(models, config)
    _emit_progress(
        config,
        "timelapse_done",
        iterations=int(len(iteration_chi2)),
        max_iterations=int(config.max_iterations),
        final_chi2=float(iteration_chi2[-1]) if iteration_chi2 else None,
        stop_reason=stop_reason,
    )
    return TimeLapseERTInversionResult(
        final_models=np.exp(final_log_models),
        final_log_models=final_log_models,
        predicted_data=np.exp(predicted_log),
        predicted_log_data=predicted_log,
        coverage=coverage,
        all_coverage=all_coverage,
        all_chi2=np.asarray(iteration_chi2, dtype=float),
        iteration_chi2=iteration_chi2,
        final_parameter_models=final_parameter_models,
        final_parameter_name=final_parameter_name,
    )


def _window_start_indices(n_times: int, window_size: int, window_step: int) -> list[int]:
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    if window_size > n_times:
        raise ValueError(f"window_size={window_size} exceeds n_times={n_times}")
    step = max(1, int(window_step))
    starts = list(range(0, n_times - window_size + 1, step))
    tail_start = n_times - window_size
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return sorted(set(starts))


def _config_for_time_window(
    config: InversionConfig,
    *,
    observed_shape: tuple[int, int],
    start: int,
    end: int,
) -> InversionConfig:
    n_times = int(observed_shape[0])
    data_std = np.asarray(config.data_std, dtype=float)
    updates: dict[str, Any] = {}
    if data_std.ndim == 0:
        pass
    else:
        updates["data_std"] = np.broadcast_to(data_std, observed_shape)[start:end].copy()

    if config.petrophysical_parameters:
        sliced_parameters: dict[str, ArrayLike] = {}
        changed = False
        for key, value in config.petrophysical_parameters.items():
            array = np.asarray(value) if value is not None else None
            if array is not None and array.ndim >= 2 and array.shape[1] == n_times:
                sliced_parameters[key] = np.asarray(array)[:, start:end].copy()
                changed = True
            else:
                sliced_parameters[key] = value
        if changed:
            updates["petrophysical_parameters"] = sliced_parameters

    if config.sensor_constraint_targets is not None:
        targets = np.asarray(config.sensor_constraint_targets)
        if targets.ndim >= 2 and targets.shape[1] == n_times:
            updates["sensor_constraint_targets"] = np.asarray(targets, dtype=float)[:, start:end].copy()
        elif targets.ndim == 1 and targets.shape[0] == n_times:
            updates["sensor_constraint_targets"] = np.asarray(targets, dtype=float)[start:end].copy()

    if config.sensor_constraint_weights is not None:
        weights = np.asarray(config.sensor_constraint_weights)
        if weights.ndim >= 2 and weights.shape[1] == n_times:
            updates["sensor_constraint_weights"] = np.asarray(weights, dtype=float)[:, start:end].copy()
        elif weights.ndim == 1 and weights.shape[0] == n_times:
            updates["sensor_constraint_weights"] = np.asarray(weights, dtype=float)[start:end].copy()

    # Baseline freeze should apply only to the global first timestep.
    # In sliding windows, only the first window contains that timestep.
    if bool(config.freeze_first_timestep):
        updates["freeze_first_timestep"] = bool(start == 0)

    if not updates:
        return config
    return replace(config, **updates)


def _config_for_time_index(config: InversionConfig, time_index: int) -> InversionConfig:
    if not config.petrophysical_parameters:
        return config
    sliced_parameters: dict[str, ArrayLike] = {}
    changed = False
    for key, value in config.petrophysical_parameters.items():
        array = np.asarray(value) if value is not None else None
        if array is not None and array.ndim >= 2:
            if not (0 <= int(time_index) < array.shape[1]):
                raise IndexError(f"time_index={time_index} outside petrophysical parameter {key!r} shape {array.shape}")
            sliced_parameters[key] = np.asarray(array)[:, int(time_index)].copy()
            changed = True
        else:
            sliced_parameters[key] = value
    if not changed:
        return config
    return replace(config, petrophysical_parameters=sliced_parameters)


def invert_windowed_timelapse_log_resistivity(
    forward: ERTForward2p5D | ERTForwardModeling,
    observed_data: ArrayLike,
    initial_model: ArrayLike,
    *,
    window_size: int = 3,
    window_step: int = 1,
    reference_model: ArrayLike | None = None,
    config: InversionConfig | None = None,
    observed_log_data: bool = False,
    initial_log_model: bool = False,
    reference_log_model: bool = False,
) -> TimeLapseERTInversionResult:
    """Run notebook-style sliding-window time-lapse inversion.

    Every overlapping window is inverted independently with
    :func:`invert_timelapse_log_resistivity`. The global model for each
    timestep is the geometric mean of all window contributions, matching the
    project notebook aggregation.
    """

    config = config or InversionConfig(temporal_regularization=1.0)
    _check_config(config)

    n_cells = _model_size(forward)
    n_measurements = _measurement_count(forward)
    observed_log = _as_observed_log_matrix(
        observed_data,
        log_data=observed_log_data,
        measurement_count=n_measurements,
    )
    n_times = int(observed_log.shape[0])
    starts = _window_start_indices(n_times, int(window_size), int(window_step))
    initial_logs = _as_log_model_matrix(
        initial_model,
        expected_size=n_cells,
        n_times=n_times,
        log_model=initial_log_model,
        name="initial_model",
    )
    reference_logs = None
    if reference_model is not None:
        reference_logs = _as_log_model_matrix(
            reference_model,
            expected_size=n_cells,
            n_times=n_times,
            log_model=reference_log_model,
            name="reference_model",
        )

    contributions: list[list[np.ndarray]] = [[] for _ in range(n_times)]
    coverage_bank: list[np.ndarray] = []
    window_final_chi2: list[float] = []
    window_reports: list[dict[str, float | int | None]] = []
    forward_jacobian_cache: ForwardJacobianCache = OrderedDict()
    forward_jacobian_cache_entries = max(
        16,
        min(128, int(window_size) * max(1, int(config.max_iterations) + 2) * 4),
    )

    _emit_progress(
        config,
        "windowed_start",
        n_cells=int(n_cells),
        n_measurements=int(n_measurements),
        n_times=int(n_times),
        n_windows=int(len(starts)),
        window_size=int(window_size),
        window_step=int(window_step),
        max_iterations=int(config.max_iterations),
    )
    for window_index, start in enumerate(starts, start=1):
        end = start + int(window_size)
        _emit_progress(
            config,
            "window_start",
            window_index=int(window_index),
            n_windows=int(len(starts)),
            start_idx=int(start),
            end_idx=int(end - 1),
            window_size=int(window_size),
            max_iterations=int(config.max_iterations),
        )
        window_config = _config_for_time_window(
            config,
            observed_shape=observed_log.shape,
            start=start,
            end=end,
        )
        window_start_time = time.perf_counter()
        window_result = invert_timelapse_log_resistivity(
            forward,
            observed_log[start:end],
            initial_logs[:, start:end],
            reference_model=None if reference_logs is None else reference_logs[:, start:end],
            config=window_config,
            observed_log_data=True,
            initial_log_model=True,
            reference_log_model=True,
            _forward_jacobian_cache=forward_jacobian_cache,
            _forward_jacobian_cache_max_entries=forward_jacobian_cache_entries,
        )
        window_elapsed_sec = time.perf_counter() - window_start_time
        for local_index in range(window_result.final_log_models.shape[1]):
            global_index = start + local_index
            contributions[global_index].append(window_result.final_log_models[:, local_index])
        if window_result.coverage is not None:
            coverage_bank.append(np.asarray(window_result.coverage, dtype=float).ravel())
        final_chi2 = float(window_result.iteration_chi2[-1]) if window_result.iteration_chi2 else None
        if final_chi2 is not None:
            window_final_chi2.append(final_chi2)
        window_reports.append(
            {
                "start_idx": int(start),
                "end_idx": int(end - 1),
                "final_chi2_data": final_chi2,
                "iterations": int(len(window_result.iteration_chi2)),
                "elapsed_sec": float(window_elapsed_sec),
            }
        )
        _emit_progress(
            config,
            "window_done",
            window_index=int(window_index),
            n_windows=int(len(starts)),
            start_idx=int(start),
            end_idx=int(end - 1),
            final_chi2=final_chi2,
        )

    final_log_columns: list[np.ndarray] = []
    for time_index, timestep_contributions in enumerate(contributions):
        if not timestep_contributions:
            raise ValueError(f"no window contribution for timestep index={time_index}")
        stack = np.column_stack(timestep_contributions)
        final_log_columns.append(np.mean(stack, axis=1))
    final_log_models = np.column_stack(final_log_columns)
    final_parameter_state = _log_model_to_state(final_log_models, config)
    final_parameter_models = _parameter_model_from_state(final_parameter_state, config)
    final_parameter_name = _parameter_name_from_state(final_parameter_state, config)

    _emit_progress(
        config,
        "windowed_prediction_start",
        n_times=int(n_times),
    )
    predicted_rows: list[np.ndarray] = []
    for time_index in range(n_times):
        _emit_progress(
            config,
            "windowed_prediction_step",
            time_index=int(time_index),
            time_number=int(time_index + 1),
            n_times=int(n_times),
        )
        predicted_rows.append(_forward_log_response(forward, final_log_models[:, time_index]))
    predicted_log = np.vstack(predicted_rows)
    if coverage_bank:
        coverage = np.nanmedian(np.column_stack(coverage_bank), axis=1)
        all_coverage = coverage_bank
    else:
        coverage = np.zeros((n_cells,), dtype=float)
        all_coverage = []

    _emit_progress(
        config,
        "windowed_done",
        n_windows=int(len(starts)),
        final_chi2=float(window_final_chi2[-1]) if window_final_chi2 else None,
    )
    return TimeLapseERTInversionResult(
        final_models=np.exp(final_log_models),
        final_log_models=final_log_models,
        predicted_data=np.exp(predicted_log),
        predicted_log_data=predicted_log,
        coverage=coverage,
        all_coverage=all_coverage,
        all_chi2=np.asarray(window_final_chi2, dtype=float),
        iteration_chi2=window_final_chi2,
        window_reports=window_reports,
        final_parameter_models=final_parameter_models,
        final_parameter_name=final_parameter_name,
    )


@dataclass
class ERTInversion:
    """Small class wrapper matching the notebook-style ``setup/run`` flow."""

    forward: ERTForward2p5D | ERTForwardModeling
    observed_data: ArrayLike
    config: InversionConfig = field(default_factory=InversionConfig)
    observed_log_data: bool = False

    def setup(self) -> "ERTInversion":
        """Validate basic dimensions and return ``self`` for notebook ergonomics."""

        _check_config(self.config)
        _as_observed_log_vector(
            self.observed_data,
            log_data=self.observed_log_data,
            expected_size=_measurement_count(self.forward),
        )
        return self

    def run(
        self,
        initial_model: ArrayLike,
        *,
        reference_model: ArrayLike | None = None,
        initial_log_model: bool = False,
        reference_log_model: bool = False,
    ) -> ERTInversionResult:
        return invert_single_log_resistivity(
            self.forward,
            self.observed_data,
            initial_model,
            reference_model=reference_model,
            config=self.config,
            observed_log_data=self.observed_log_data,
            initial_log_model=initial_log_model,
            reference_log_model=reference_log_model,
        )


@dataclass
class TimeLapseERTInversion:
    """Notebook-style wrapper for joint time-lapse log-resistivity inversion."""

    forward: ERTForward2p5D | ERTForwardModeling
    observed_data: ArrayLike
    config: InversionConfig = field(default_factory=lambda: InversionConfig(temporal_regularization=1.0))
    observed_log_data: bool = False

    def setup(self) -> "TimeLapseERTInversion":
        """Validate basic dimensions and return ``self`` for notebook ergonomics."""

        _check_config(self.config)
        _as_observed_log_matrix(
            self.observed_data,
            log_data=self.observed_log_data,
            measurement_count=_measurement_count(self.forward),
        )
        return self

    def run(
        self,
        initial_model: ArrayLike,
        *,
        reference_model: ArrayLike | None = None,
        initial_log_model: bool = False,
        reference_log_model: bool = False,
    ) -> TimeLapseERTInversionResult:
        return invert_timelapse_log_resistivity(
            self.forward,
            self.observed_data,
            initial_model,
            reference_model=reference_model,
            config=self.config,
            observed_log_data=self.observed_log_data,
            initial_log_model=initial_log_model,
            reference_log_model=reference_log_model,
        )


@dataclass
class WindowedTimeLapseERTInversion:
    """Notebook-style sliding-window wrapper for large time-lapse inversions."""

    forward: ERTForward2p5D | ERTForwardModeling
    observed_data: ArrayLike
    config: InversionConfig = field(default_factory=lambda: InversionConfig(temporal_regularization=1.0))
    window_size: int = 3
    window_step: int = 1
    observed_log_data: bool = False

    def setup(self) -> "WindowedTimeLapseERTInversion":
        """Validate dimensions and window controls."""

        _check_config(self.config)
        observed_log = _as_observed_log_matrix(
            self.observed_data,
            log_data=self.observed_log_data,
            measurement_count=_measurement_count(self.forward),
        )
        _window_start_indices(
            int(observed_log.shape[0]),
            int(self.window_size),
            int(self.window_step),
        )
        return self

    def run(
        self,
        initial_model: ArrayLike,
        *,
        reference_model: ArrayLike | None = None,
        initial_log_model: bool = False,
        reference_log_model: bool = False,
    ) -> TimeLapseERTInversionResult:
        return invert_windowed_timelapse_log_resistivity(
            self.forward,
            self.observed_data,
            initial_model,
            window_size=self.window_size,
            window_step=self.window_step,
            reference_model=reference_model,
            config=self.config,
            observed_log_data=self.observed_log_data,
            initial_log_model=initial_log_model,
            reference_log_model=reference_log_model,
        )
