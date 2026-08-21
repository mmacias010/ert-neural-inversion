"""ERT forward modelling facade without external modelling dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepert.utils.torch_runtime import torch_np
import numpy as np

from deepert.forward.ert2p5d import ERTForward2p5D
from deepert.mesh import Mesh
from deepert.survey import Survey
from deepert.utils.dtypes import FLOAT_DTYPE


def _point_xy(point: Any) -> list[float]:
    if hasattr(point, "pos"):
        point = point.pos()
    return [float(point[0]), float(point[1])]


def _cell_node_ids(cell: Any) -> list[int]:
    if hasattr(cell, "nodeCount") and hasattr(cell, "node"):
        node_count = int(cell.nodeCount())
        if node_count < 3:
            raise ValueError("mesh cells must have at least three nodes")
        if node_count > 4:
            raise ValueError("only triangular and quadrilateral cells are supported")
        return [int(cell.node(local_id).id()) for local_id in range(node_count)]
    node_ids = [int(node_id) for node_id in cell]
    if len(node_ids) < 3:
        raise ValueError("mesh cells must have at least three nodes")
    if len(node_ids) > 4:
        raise ValueError("only triangular and quadrilateral cells are supported")
    return node_ids


def mesh_to_deepert(mesh: Any) -> Mesh:
    """Convert common mesh-like objects into :class:`deepert.mesh.Mesh`.

    Supported inputs are already-built ``Mesh`` instances, ``(nodes, cells)``
    tuples, objects with ``nodes``/``cells`` arrays, meshio meshes, and
    objects exposing ``nodes()``/``cells()``. The conversion uses duck typing
    and never imports external modelling packages.
    """

    if isinstance(mesh, Mesh):
        return mesh
    if isinstance(mesh, tuple) and len(mesh) == 2:
        return Mesh.from_arrays(nodes=mesh[0], cells=mesh[1])
    if hasattr(mesh, "cells_dict"):
        return Mesh.from_meshio(mesh)

    nodes_attr = getattr(mesh, "nodes", None)
    cells_attr = getattr(mesh, "cells", None)
    if nodes_attr is None or cells_attr is None:
        raise TypeError("mesh must be a Mesh, (nodes, cells), meshio mesh, or mesh-like object")

    raw_nodes = nodes_attr() if callable(nodes_attr) else nodes_attr
    raw_cells = cells_attr() if callable(cells_attr) else cells_attr
    nodes = np.asarray([_point_xy(node) for node in raw_nodes], dtype=float)
    cells = np.asarray([_cell_node_ids(cell) for cell in raw_cells], dtype=np.int32)
    return Mesh.from_arrays(torch_np.asarray(nodes), torch_np.asarray(cells))


def survey_to_deepert(data: Any) -> Survey:
    """Convert common survey/data-like objects into :class:`deepert.survey.Survey`.

    Supported inputs are ``Survey`` instances, ``(electrodes, abmn)`` tuples,
    objects with ``electrode_positions``/``measurements`` arrays, and
    DataContainer-like objects exposing ``sensors()`` or ``sensorPositions()``
    plus ``a/b/m/n`` fields.
    """

    if isinstance(data, Survey):
        return data
    if isinstance(data, tuple) and len(data) == 2:
        return Survey.from_arrays(electrode_positions=data[0], measurements=data[1])
    if hasattr(data, "electrode_positions") and hasattr(data, "measurements"):
        return Survey.from_arrays(data.electrode_positions, data.measurements)

    sensors_attr = getattr(data, "sensors", None)
    if sensors_attr is None:
        sensors_attr = getattr(data, "sensorPositions", None)
    if sensors_attr is None:
        raise TypeError("data must be a Survey, (electrodes, abmn), or data-like object")

    sensors = sensors_attr() if callable(sensors_attr) else sensors_attr
    electrodes = np.asarray([_point_xy(sensor) for sensor in sensors], dtype=float)
    measurements = np.column_stack(
        (
            np.asarray(data["a"], dtype=np.int32),
            np.asarray(data["b"], dtype=np.int32),
            np.asarray(data["m"], dtype=np.int32),
            np.asarray(data["n"], dtype=np.int32),
        )
    )
    return Survey.from_arrays(torch_np.asarray(electrodes), torch_np.asarray(measurements))


def _prepare_resistivity_model(resistivity_model: Any, *, log_transform: bool, expected_size: int) -> np.ndarray:
    values = np.asarray(resistivity_model, dtype=float).ravel()
    if log_transform:
        if not np.all(np.isfinite(values)):
            raise ValueError("resistivity_model contains non-finite log-resistivity values")
        values = np.exp(values)

    if values.shape != (expected_size,):
        raise ValueError(f"resistivity_model must have shape ({expected_size},)")
    if not np.all(np.isfinite(values)):
        raise ValueError("resistivity_model contains non-finite resistivity values")
    if np.any(values <= 0.0):
        raise ValueError(
            "resistivity_model must contain positive resistivity values "
            f"(min={float(np.min(values)):.6e})"
        )
    return values


@dataclass
class ERTForwardModeling:
    """Small forward wrapper with the common ``setData/setMesh/forward`` shape."""

    mesh: Any | None = None
    data: Any | None = None
    quadrature_order: int = 2
    numerical_h2_refined: bool = True
    numerical_p2_refined: bool = True
    topographic_geometric_factor_mode: str = "analytic"
    linear_solver_backend: str = "auto"
    terrain_cache_dir: str | Path | None = None
    include_robin_boundary_derivative: bool = False
    normal_sensitivity: bool = True

    def __post_init__(self) -> None:
        self._forward: ERTForward2p5D | None = None
        self._mesh: Mesh | None = None
        self._survey: Survey | None = None

    def set_data(self, data: Any) -> None:
        """Set the ERT survey/data object."""

        self.data = data
        self._survey = None
        self._forward = None

    def setData(self, data: Any) -> None:  # noqa: N802 - compatibility with common ERT APIs
        """Alias for :meth:`set_data`."""

        self.set_data(data)

    def set_mesh(self, mesh: Any) -> None:
        """Set the forward mesh."""

        self.mesh = mesh
        self._mesh = None
        self._forward = None

    def setMesh(self, mesh: Any) -> None:  # noqa: N802 - compatibility with common ERT APIs
        """Alias for :meth:`set_mesh`."""

        self.set_mesh(mesh)

    def _resolved_mesh(self) -> Mesh:
        if self.mesh is None:
            raise ValueError("mesh has not been set")
        if self._mesh is None:
            self._mesh = mesh_to_deepert(self.mesh)
        return self._mesh

    def _resolved_survey(self) -> Survey:
        if self.data is None:
            raise ValueError("data has not been set")
        if self._survey is None:
            self._survey = survey_to_deepert(self.data)
        return self._survey

    @property
    def cell_count(self) -> int:
        return self._resolved_mesh().cell_count

    @property
    def forward_operator(self) -> ERTForward2p5D:
        if self._forward is None:
            self._forward = ERTForward2p5D.from_mesh_survey(
                self._resolved_mesh(),
                self._resolved_survey(),
                quadrature_order=self.quadrature_order,
                numerical_h2_refined=self.numerical_h2_refined,
                numerical_p2_refined=self.numerical_p2_refined,
                topographic_geometric_factor_mode=self.topographic_geometric_factor_mode,
                linear_solver_backend=self.linear_solver_backend,
                terrain_cache_dir=self.terrain_cache_dir,
            )
        return self._forward

    def prepare(
        self,
        resistivity_model: Any | None = None,
        log_transform: bool = True,
        *,
        include_solver_state: bool = True,
    ) -> None:
        """Warm geometry, cache, and optional solver state."""

        if resistivity_model is None:
            self.forward_operator.prepare(None, include_solver_state=include_solver_state)
            return

        resistivity = _prepare_resistivity_model(
            resistivity_model,
            log_transform=log_transform,
            expected_size=self.cell_count,
        )
        conductivity = torch_np.asarray(1.0 / resistivity, dtype=FLOAT_DTYPE)
        self.forward_operator.prepare(conductivity, include_solver_state=include_solver_state)

    def forward(self, resistivity_model: Any, log_transform: bool = True) -> np.ndarray:
        """Compute apparent resistivity for the current mesh and survey."""

        resistivity = _prepare_resistivity_model(
            resistivity_model,
            log_transform=log_transform,
            expected_size=self.cell_count,
        )
        conductivity = torch_np.asarray(1.0 / resistivity, dtype=FLOAT_DTYPE)
        values = np.asarray(self.forward_operator.apparent_resistivity_values(conductivity), dtype=float)
        if log_transform:
            return np.log(values)
        return values

    def response(self, resistivity_model: Any) -> np.ndarray:
        """Return non-log apparent resistivity values."""

        return self.forward(resistivity_model, log_transform=False)

    def forward_and_jacobian(
        self,
        resistivity_model: Any,
        log_transform: bool = True,
        *,
        include_robin_boundary_derivative: bool | None = None,
        normal_sensitivity: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute apparent-resistivity response and a cellwise Jacobian.

        For ``log_transform=True``, the input model and returned response are
        logarithmic and the Jacobian is ``d log(rhoa) / d log(rho)``. For
        ``log_transform=False``, the Jacobian is ``d rhoa / d rho``.
        The facade defaults to the normal-quadrupole sensitivity convention, which
        omits the mixed Robin boundary derivative.
        """

        resistivity = _prepare_resistivity_model(
            resistivity_model,
            log_transform=log_transform,
            expected_size=self.cell_count,
        )
        conductivity = torch_np.asarray(1.0 / resistivity, dtype=FLOAT_DTYPE)
        forward = self.forward_operator
        if include_robin_boundary_derivative is None:
            include_robin_boundary_derivative = self.include_robin_boundary_derivative
        if normal_sensitivity is None:
            normal_sensitivity = self.normal_sensitivity
        response, resistance_jacobian = forward.solve_with_jacobian(
            conductivity,
            include_robin_boundary_derivative=include_robin_boundary_derivative,
            normal_sensitivity=normal_sensitivity,
        )
        apparent_jacobian_sigma = torch_np.abs(forward._geometric_factors())[:, None] * resistance_jacobian

        if log_transform:
            jacobian = apparent_jacobian_sigma * (-conductivity[None, :])
            jacobian = jacobian / response.apparent_resistivity[:, None]
            return np.log(np.asarray(response.apparent_resistivity, dtype=float)), np.asarray(jacobian, dtype=float)

        jacobian = apparent_jacobian_sigma * (-(conductivity**2)[None, :])
        return np.asarray(response.apparent_resistivity, dtype=float), np.asarray(jacobian, dtype=float)


@dataclass
class MappedERTForwardModeling:
    """Forward facade for inversions whose parameters cover only active cells.

    The forward solve is evaluated on ``mesh`` while the public model vector is
    restricted to ``active_cell_ids``. Cells outside that active set are kept at
    ``inactive_resistivity``.
    """

    mesh: Any
    data: Any
    active_cell_ids: Any
    inactive_resistivity: Any
    regularization_mesh: Any | None = None
    quadrature_order: int = 2
    numerical_h2_refined: bool = True
    numerical_p2_refined: bool = True
    topographic_geometric_factor_mode: str = "analytic"
    linear_solver_backend: str = "auto"
    terrain_cache_dir: str | Path | None = None
    include_robin_boundary_derivative: bool = False
    normal_sensitivity: bool = True

    def __post_init__(self) -> None:
        self._forward_modeling = ERTForwardModeling(
            mesh=self.mesh,
            data=self.data,
            quadrature_order=self.quadrature_order,
            numerical_h2_refined=self.numerical_h2_refined,
            numerical_p2_refined=self.numerical_p2_refined,
            topographic_geometric_factor_mode=self.topographic_geometric_factor_mode,
            linear_solver_backend=self.linear_solver_backend,
            terrain_cache_dir=self.terrain_cache_dir,
            include_robin_boundary_derivative=self.include_robin_boundary_derivative,
            normal_sensitivity=self.normal_sensitivity,
        )
        active = np.asarray(self.active_cell_ids, dtype=np.int32).ravel()
        if active.size == 0:
            raise ValueError("active_cell_ids must be non-empty")
        if np.any(active < 0):
            raise ValueError("active_cell_ids contains negative indices")
        if np.unique(active).size != active.size:
            raise ValueError("active_cell_ids must be unique")
        full_cell_count = self._forward_modeling.cell_count
        if np.any(active >= full_cell_count):
            raise ValueError("active_cell_ids references cells outside the forward mesh")
        self._active_cell_ids = active

        inactive = np.asarray(self.inactive_resistivity, dtype=float)
        if inactive.ndim == 0:
            inactive = np.full(full_cell_count, float(inactive), dtype=float)
        else:
            inactive = inactive.ravel().astype(float, copy=True)
            if inactive.shape != (full_cell_count,):
                raise ValueError(f"inactive_resistivity must be scalar or shape ({full_cell_count},)")
        if not np.all(np.isfinite(inactive)) or np.any(inactive <= 0.0):
            raise ValueError("inactive_resistivity must contain positive finite values")
        self._inactive_resistivity = inactive

        if self.regularization_mesh is not None:
            self.regularization_mesh = mesh_to_deepert(self.regularization_mesh)

    @property
    def cell_count(self) -> int:
        return int(self._active_cell_ids.size)

    @property
    def forward_operator(self) -> ERTForward2p5D:
        return self._forward_modeling.forward_operator

    @property
    def active_cell_ids_array(self) -> np.ndarray:
        return self._active_cell_ids.copy()

    def _expand_resistivity(self, resistivity_model: Any, *, log_transform: bool) -> np.ndarray:
        active_resistivity = _prepare_resistivity_model(
            resistivity_model,
            log_transform=log_transform,
            expected_size=self.cell_count,
        )
        full = self._inactive_resistivity.copy()
        full[self._active_cell_ids] = active_resistivity
        return full

    def prepare(
        self,
        resistivity_model: Any | None = None,
        log_transform: bool = True,
        *,
        include_solver_state: bool = True,
    ) -> None:
        """Warm geometry, cache, and optional solver state."""

        if resistivity_model is None:
            self._forward_modeling.prepare(None, include_solver_state=include_solver_state)
            return
        full_resistivity = self._expand_resistivity(resistivity_model, log_transform=log_transform)
        self._forward_modeling.prepare(
            full_resistivity,
            log_transform=False,
            include_solver_state=include_solver_state,
        )

    def forward(self, resistivity_model: Any, log_transform: bool = True) -> np.ndarray:
        full_resistivity = self._expand_resistivity(resistivity_model, log_transform=log_transform)
        values = self._forward_modeling.forward(full_resistivity, log_transform=False)
        if log_transform:
            return np.log(values)
        return values

    def response(self, resistivity_model: Any) -> np.ndarray:
        return self.forward(resistivity_model, log_transform=False)

    def forward_and_jacobian(
        self,
        resistivity_model: Any,
        log_transform: bool = True,
        *,
        include_robin_boundary_derivative: bool | None = None,
        normal_sensitivity: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        full_resistivity = self._expand_resistivity(resistivity_model, log_transform=log_transform)
        full_model = np.log(full_resistivity) if log_transform else full_resistivity
        response, full_jacobian = self._forward_modeling.forward_and_jacobian(
            full_model,
            log_transform=log_transform,
            include_robin_boundary_derivative=include_robin_boundary_derivative,
            normal_sensitivity=normal_sensitivity,
        )
        return response, np.asarray(full_jacobian[:, self._active_cell_ids], dtype=float)
