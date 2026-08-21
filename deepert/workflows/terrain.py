"""Terrain-following ERT forward helpers used by the ParFlow notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from deepert.utils.torch_runtime import torch_np
import numpy as np

from deepert.forward import ERTForward2p5D
from deepert.mesh import Mesh
from deepert.survey import Survey
from deepert.utils.dtypes import FLOAT_DTYPE


@dataclass(frozen=True)
class ParflowGrid:
    """ParFlow grid settings needed to rebuild a 2D terrain slice."""

    dx: float
    dy: float
    dz_base: float
    nx: int
    ny: int
    nz: int
    dz_scales: np.ndarray


@dataclass(frozen=True)
class TerrainForwardCase:
    """Native forward inputs and metadata for one terrain-following timestep."""

    mesh: Mesh
    survey: Survey
    resistivity: np.ndarray
    elec_x: np.ndarray
    elec_z: np.ndarray
    x_nodes: np.ndarray
    z_top: np.ndarray
    layer_thickness: np.ndarray
    y_index: int

    def with_resistivity(self, resistivity: np.ndarray) -> "TerrainForwardCase":
        """Return the same terrain geometry with a different cell resistivity vector."""

        resistivity_array = np.asarray(resistivity, dtype=float)
        if resistivity_array.shape != self.resistivity.shape:
            raise ValueError(f"resistivity must have shape {self.resistivity.shape}")
        if not np.all(np.isfinite(resistivity_array)):
            raise ValueError("resistivity contains non-finite values")
        if np.any(resistivity_array <= 0.0):
            raise ValueError("resistivity must contain positive values")

        return TerrainForwardCase(
            mesh=self.mesh,
            survey=self.survey,
            resistivity=resistivity_array,
            elec_x=self.elec_x,
            elec_z=self.elec_z,
            x_nodes=self.x_nodes,
            z_top=self.z_top,
            layer_thickness=self.layer_thickness,
            y_index=self.y_index,
        )


@dataclass(frozen=True)
class SourcePositionInversionCase:
    """Triangle inversion mesh generated from source/electrode positions."""

    mesh: Mesh
    forward_mesh: Mesh
    survey: Survey
    parameter_cell_ids: np.ndarray
    cell_markers: np.ndarray
    elec_x: np.ndarray
    elec_z: np.ndarray
    x_nodes: np.ndarray
    z_top: np.ndarray
    layer_thickness: np.ndarray
    y_index: int


@dataclass(frozen=True)
class TerrainForwardData:
    """ERT data parsed from a notebook-style terrain forward ``.dat`` file."""

    rhoa: np.ndarray
    measurements: np.ndarray
    elec_x: np.ndarray
    elec_z: np.ndarray
    err: np.ndarray | None = None


@dataclass(frozen=True)
class TerrainForwardRecord:
    """Manifest row for one terrain-forward timestep."""

    step: int
    input_file: str
    dat_file: str
    npz_file: str
    status: str
    rhoa_min: float | None = None
    rhoa_max: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class TerrainForwardRunner:
    """Reusable terrain-forward operator for a fixed mesh, survey, and topography."""

    case_template: TerrainForwardCase
    forward: ERTForward2p5D
    reuse_solver_state: bool = True

    @classmethod
    def from_case(
        cls,
        case: TerrainForwardCase,
        *,
        linear_solver_backend: str = "auto",
        reuse_solver_state: bool = True,
        terrain_cache_dir: str | Path | None = None,
        prepare_forward: bool = False,
    ) -> "TerrainForwardRunner":
        """Build a reusable forward operator from one terrain case.

        By default cuDSS symbolic/plan/buffer state is retained across models.
        Each changed conductivity still updates matrix values and refactorizes
        before solve.
        """

        forward_kwargs = {"linear_solver_backend": linear_solver_backend}
        if terrain_cache_dir is not None:
            forward_kwargs["terrain_cache_dir"] = terrain_cache_dir
        forward = ERTForward2p5D.from_mesh_survey(case.mesh, case.survey, **forward_kwargs)
        runner = cls(case_template=case, forward=forward, reuse_solver_state=bool(reuse_solver_state))
        if prepare_forward:
            runner.prepare_resistivity(case.resistivity)
        return runner

    def case_with_resistivity(self, resistivity: np.ndarray) -> TerrainForwardCase:
        """Return a case for this fixed geometry and a new resistivity vector."""

        return self.case_template.with_resistivity(resistivity)

    def solve_resistivity(self, resistivity: np.ndarray) -> np.ndarray:
        """Compute apparent resistivity for a cell resistivity vector."""

        conductivity = torch_np.asarray(1.0 / np.asarray(resistivity, dtype=float), dtype=FLOAT_DTYPE)

        if not self.reuse_solver_state:
            self.forward.close()
            return self._solve_conductivity_checked(conductivity)

        rhoa = self._solve_conductivity_checked(conductivity, retry_on_invalid=False)
        if rhoa is not None:
            return rhoa

        self.forward.close()
        return self._solve_conductivity_checked(conductivity)

    def _solve_conductivity_checked(self, conductivity, *, retry_on_invalid: bool = True) -> np.ndarray | None:
        """Solve one model and optionally report invalid fast-path output to callers."""

        rhoa = np.asarray(self.forward.apparent_resistivity_values(conductivity=conductivity), dtype=float)
        if not np.isfinite(rhoa).all() or np.any(rhoa <= 0.0):
            if retry_on_invalid:
                raise ValueError("forward returned non-finite or non-positive apparent resistivity values")
            return None
        return rhoa

    def prepare_resistivity(self, resistivity: np.ndarray) -> None:
        """Pre-populate caches for a representative terrain resistivity vector."""

        conductivity = torch_np.asarray(1.0 / np.asarray(resistivity, dtype=float), dtype=FLOAT_DTYPE)
        self.forward.prepare(conductivity, include_solver_state=self.reuse_solver_state)

    def solve_case(self, case: TerrainForwardCase) -> np.ndarray:
        """Compute apparent resistivity for a terrain case sharing this geometry."""

        return self.solve_resistivity(case.resistivity)

    def close(self) -> None:
        """Release cached GPU solver resources held by the reusable operator."""

        self.forward.close()


def parse_resistivity_slice_name(path: str | Path) -> tuple[int, int]:
    """Parse ``(y_index, timestep)`` from terrain resistivity ``.npy`` names.

    ``resistivity2d_y{y}_t{step}.npy`` carries an explicit y-index. The
    notebook-style ``resistivity_t{step}.npy`` name is accepted with y-index
    ``-1`` so callers can provide the desired slice separately.
    """

    name = Path(path).name
    match = re.search(r"resistivity2d_y(\d+)_t(\d+)\.npy$", name)
    if match is not None:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"resistivity_t(\d+)\.npy$", name)
    if match is not None:
        return -1, int(match.group(1))
    raise ValueError(f"cannot parse y-index and timestep from filename: {name}")


def discover_resistivity_slices(
    input_dir: str | Path,
    *,
    y_index: int | None = None,
    file_stride: int = 1,
    max_steps: int | None = None,
) -> list[tuple[int, Path]]:
    """Return sorted ``(step, path)`` pairs for terrain resistivity slices."""

    if file_stride < 1:
        raise ValueError("file_stride must be >= 1")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be >= 1 when set")

    root = Path(input_dir)
    pairs: list[tuple[int, Path]] = []
    for pattern in ("resistivity2d_y*_t*.npy", "resistivity_t*.npy"):
        for path in root.glob(pattern):
            try:
                found_y_index, step = parse_resistivity_slice_name(path)
            except ValueError:
                continue
            if y_index is not None and found_y_index >= 0 and found_y_index != y_index:
                continue
            pairs.append((step, path))

    pairs = sorted(set(pairs), key=lambda item: item[0])
    pairs = pairs[::file_stride]
    if max_steps is not None:
        pairs = pairs[:max_steps]
    return pairs


def load_terrain_resistivity_slice(path: str | Path, grid: ParflowGrid, *, y_index: int) -> np.ndarray:
    """Load a 2D terrain resistivity slice from a 2D or notebook-style 3D file."""

    values = np.asarray(np.load(path), dtype=float)
    if values.ndim == 2:
        return values
    if values.ndim != 3:
        raise ValueError(f"{path}: resistivity array must be 2D or 3D, got shape={values.shape}")
    if not (0 <= y_index < grid.ny):
        raise ValueError(f"y_index={y_index} out of range for NY={grid.ny}")

    if values.shape == (grid.nz, grid.ny, grid.nx):
        return values[:, y_index, :]
    if values.shape == (grid.ny, grid.nz, grid.nx):
        return values[y_index, :, :]
    if values.shape == (grid.nx, grid.ny, grid.nz):
        return values[:, y_index, :].T
    raise ValueError(
        f"{path}: cannot infer 3D resistivity axis order from shape={values.shape}; "
        f"expected ({grid.nz}, {grid.ny}, {grid.nx}), ({grid.ny}, {grid.nz}, {grid.nx}), "
        f"or ({grid.nx}, {grid.ny}, {grid.nz})"
    )


def parse_pftcl(path: str | Path) -> ParflowGrid:
    """Parse ParFlow grid dimensions and ``dzScale`` values from a pftcl file."""

    pftcl_path = Path(path)
    values: dict[str, float | int] = {}
    dz_scales: dict[int, float] = {}

    for line in pftcl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        for key in ("DX", "DY", "DZ"):
            match = re.search(rf'ComputationalGrid\.{key}\s+"([0-9eE+\-.]+)"', line)
            if match is not None:
                values[key.lower()] = float(match.group(1))
        for key in ("NX", "NY", "NZ"):
            match = re.search(rf'ComputationalGrid\.{key}\s+"(\d+)"', line)
            if match is not None:
                values[key.lower()] = int(match.group(1))
        match = re.search(r'Cell\.(\d+)\.dzScale\.Value\s+"([0-9eE+\-.]+)"', line)
        if match is not None:
            dz_scales[int(match.group(1))] = float(match.group(2))

    missing = [key for key in ("dx", "dy", "dz", "nx", "ny", "nz") if key not in values]
    if missing:
        raise ValueError(f"failed to parse ComputationalGrid settings from {pftcl_path}: missing {missing}")

    nz = int(values["nz"])
    if len(dz_scales) != nz:
        raise ValueError(f"dzScale count {len(dz_scales)} does not match NZ={nz}")

    return ParflowGrid(
        dx=float(values["dx"]),
        dy=float(values["dy"]),
        dz_base=float(values["dz"]),
        nx=int(values["nx"]),
        ny=int(values["ny"]),
        nz=nz,
        dz_scales=np.asarray([dz_scales[idx] for idx in range(nz)], dtype=float),
    )


def read_slope_x(path: str | Path, y_index: int) -> np.ndarray:
    """Read one ``slope_x`` y-slice from a ParFlow PFB file."""

    try:
        from parflow.tools.io import read_pfb
    except ImportError as exc:
        raise ImportError(
            "Reading ParFlow PFB files requires the examples extra: "
            "`uv sync --extra examples`."
        ) from exc

    slope_x_3d = np.asarray(read_pfb(str(path)), dtype=float)
    if slope_x_3d.ndim != 3:
        raise ValueError(f"expected slope_x PFB to load as a 3D array, got shape={slope_x_3d.shape}")
    if not 0 <= y_index < slope_x_3d.shape[1]:
        raise ValueError(f"y_index={y_index} out of range for slope_x shape={slope_x_3d.shape}")
    return slope_x_3d[0, y_index, :]


def build_wenner_alpha_measurements(electrode_count: int) -> np.ndarray:
    """Return reference ``schemeName='wa'`` ABMN ordering for a linear line."""

    if electrode_count < 4:
        raise ValueError("Wenner-alpha surveys need at least four electrodes")

    measurements: list[list[int]] = []
    for spacing in range(1, electrode_count // 3 + 1):
        for start in range(electrode_count - 3 * spacing):
            measurements.append([start, start + 3 * spacing, start + spacing, start + 2 * spacing])
    return np.asarray(measurements, dtype=np.int32)


def load_terrain_forward_dat(path: str | Path) -> TerrainForwardData:
    """Read the notebook ERT ``.dat`` format without external ERT loaders."""

    dat_path = Path(path)
    lines = [line.strip() for line in dat_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 4:
        raise ValueError(f"{dat_path} is too short to be an ERT .dat file")

    cursor = 0
    try:
        sensor_count = int(lines[cursor])
    except ValueError as exc:
        raise ValueError(f"{dat_path}: first line must be sensor count") from exc
    cursor += 1
    while cursor < len(lines) and lines[cursor].startswith("#"):
        cursor += 1

    if cursor + sensor_count > len(lines):
        raise ValueError(f"{dat_path}: sensor table is truncated")
    sensors = np.asarray(
        [[float(part) for part in lines[cursor + index].split()[:3]] for index in range(sensor_count)],
        dtype=float,
    )
    sensors = _gimli_round(sensors, 1.0e-12)
    cursor += sensor_count

    while cursor < len(lines) and lines[cursor].startswith("#"):
        cursor += 1
    if cursor >= len(lines):
        raise ValueError(f"{dat_path}: missing measurement count")
    try:
        data_count = int(lines[cursor])
    except ValueError as exc:
        raise ValueError(f"{dat_path}: measurement count must be an integer") from exc
    cursor += 1

    header: list[str] | None = None
    while cursor < len(lines) and lines[cursor].startswith("#"):
        header = lines[cursor].lstrip("#").split()
        cursor += 1
    if header is None:
        raise ValueError(f"{dat_path}: missing measurement header")
    if cursor + data_count > len(lines):
        raise ValueError(f"{dat_path}: measurement table is truncated")

    columns = {name: index for index, name in enumerate(header)}
    required = ("a", "b", "m", "n", "rhoa")
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"{dat_path}: missing required measurement columns {missing}")

    values = np.asarray(
        [[float(part) for part in lines[cursor + index].split()] for index in range(data_count)],
        dtype=float,
    )
    if values.shape[1] < len(header):
        raise ValueError(f"{dat_path}: measurement rows have fewer columns than the header")

    measurements = np.column_stack(
        (
            values[:, columns["a"]],
            values[:, columns["b"]],
            values[:, columns["m"]],
            values[:, columns["n"]],
        )
    ).astype(np.int32)
    measurements -= 1
    err = np.asarray(values[:, columns["err"]], dtype=float) if "err" in columns else None
    return TerrainForwardData(
        rhoa=np.asarray(values[:, columns["rhoa"]], dtype=float),
        measurements=measurements,
        elec_x=np.asarray(sensors[:, 0], dtype=float),
        elec_z=np.asarray(sensors[:, 1], dtype=float),
        err=err,
    )


def _gimli_round(values: np.ndarray, tolerance: float) -> np.ndarray:
    """Match GIMLi's tolerance rounding, which rounds after division by tol."""

    return np.rint(np.asarray(values, dtype=float) / tolerance) * tolerance


def _geometric_factors_np(electrodes: np.ndarray, measurements: np.ndarray) -> np.ndarray:
    """Return analytic half-space geometric factors using NumPy double inputs."""

    electrode_array = np.asarray(electrodes, dtype=float)
    measurement_array = np.asarray(measurements, dtype=np.int32)
    selected = electrode_array[measurement_array]
    a = selected[:, 0]
    b = selected[:, 1]
    m = selected[:, 2]
    n = selected[:, 3]

    response = (
        1.0 / np.linalg.norm(a - m, axis=-1)
        - 1.0 / np.linalg.norm(a - n, axis=-1)
        - 1.0 / np.linalg.norm(b - m, axis=-1)
        + 1.0 / np.linalg.norm(b - n, axis=-1)
    )
    return 2.0 * np.pi / response


def _triangle_node_adjacency(cells: np.ndarray, node_count: int) -> list[np.ndarray]:
    adjacency = [set() for _ in range(node_count)]
    for cell in cells:
        a, b, c = (int(cell[0]), int(cell[1]), int(cell[2]))
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))
    return [np.asarray(sorted(neighbors), dtype=np.int32) for neighbors in adjacency]


def _triangle_smoothing_fixed_mask(
    cells: np.ndarray,
    markers: np.ndarray,
    node_count: int,
    plc_node_count: int,
) -> np.ndarray:
    fixed = np.zeros(node_count, dtype=bool)
    fixed[: min(max(int(plc_node_count), 0), node_count)] = True

    edge_counts: dict[tuple[int, int], int] = {}
    edge_markers: dict[tuple[int, int], set[int]] = {}
    for cell, marker in zip(cells, markers, strict=True):
        a, b, c = (int(cell[0]), int(cell[1]), int(cell[2]))
        cell_edges = (
            (min(a, b), max(a, b)),
            (min(b, c), max(b, c)),
            (min(c, a), max(c, a)),
        )
        marker_id = int(marker)
        for edge in cell_edges:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
            edge_markers.setdefault(edge, set()).add(marker_id)

    for edge, count in edge_counts.items():
        if count == 1 or len(edge_markers[edge]) > 1:
            fixed[list(edge)] = True
    return fixed


def _smooth_triangle_nodes(
    nodes: np.ndarray,
    cells: np.ndarray,
    markers: np.ndarray,
    *,
    plc_node_count: int,
    iterations: int,
) -> np.ndarray:
    """Replicate pyGIMLi's Triangle mesh smoothing for region-marked PLC meshes."""

    if iterations < 0:
        raise ValueError("smoothing iterations must be non-negative")
    nodes_array = np.asarray(nodes, dtype=float)
    if iterations == 0:
        return nodes_array.copy()

    cells_array = np.asarray(cells, dtype=np.int32)
    markers_array = np.asarray(markers, dtype=np.int32).reshape(-1)
    if cells_array.ndim != 2 or cells_array.shape[1] != 3:
        raise ValueError("triangle cells must have shape (n_cells, 3)")
    if markers_array.shape[0] != cells_array.shape[0]:
        raise ValueError("triangle markers must have one value per cell")

    smoothed = nodes_array.copy()
    adjacency = _triangle_node_adjacency(cells_array, smoothed.shape[0])
    fixed = _triangle_smoothing_fixed_mask(cells_array, markers_array, smoothed.shape[0], plc_node_count)
    for _ in range(iterations):
        for node_id, neighbors in enumerate(adjacency):
            if fixed[node_id] or neighbors.size == 0:
                continue
            smoothed[node_id] = (smoothed[node_id] + smoothed[neighbors].sum(axis=0)) / (neighbors.size + 1)
    return smoothed


@dataclass(frozen=True)
class _SourcePositionTriangleArrays:
    parameter_nodes: np.ndarray
    parameter_cells: np.ndarray
    full_nodes: np.ndarray
    full_cells: np.ndarray
    cell_markers: np.ndarray
    parameter_cell_ids: np.ndarray


def _source_position_triangle_arrays(
    elec_x: np.ndarray,
    elec_z: np.ndarray,
    *,
    quality: float,
    smoothing_iterations: int = 10,
) -> _SourcePositionTriangleArrays:
    try:
        import triangle as triangle_lib
    except ImportError as exc:
        raise ImportError(
            "Building the source-position inversion mesh requires "
            "the optional `triangle` package. Install the example dependencies "
            "with `uv sync --extra examples`."
        ) from exc

    if quality <= 0.0:
        raise ValueError("quality must be positive")

    sensors = np.column_stack((elec_x, elec_z))
    electrode_spacing = float(np.linalg.norm(sensors[1] - sensors[0]))
    x_min = float(np.min(elec_x))
    x_max = float(np.max(elec_x))
    para_bound = electrode_spacing * 2.0
    para_depth = 0.4 * (x_max - x_min)
    x_start = x_min - para_bound
    x_end = x_max + para_bound
    bottom_y = min(float(elec_z[0] - para_depth), float(elec_z[-1] - para_depth))
    boundary_scale = 4.0
    outer_bound = abs(x_max - x_min) * boundary_scale

    vertices: list[list[float]] = []
    segments: list[list[int]] = []
    regions: list[list[float]] = []

    def add_node(x_coord: float, y_coord: float) -> int:
        vertices.append([float(x_coord), float(y_coord)])
        return len(vertices) - 1

    def add_segment(start: int, stop: int) -> None:
        segments.append([start, stop])

    n1 = add_node(x_start, float(elec_z[0]))
    n2 = add_node(x_start, bottom_y)
    n3 = add_node(x_end, bottom_y)
    n4 = add_node(x_end, float(elec_z[-1]))

    if outer_bound > para_bound:
        n11 = add_node(vertices[n1][0] - outer_bound, vertices[n1][1])
        n12 = add_node(vertices[n11][0], vertices[n11][1] - (outer_bound + para_depth))
        n14 = add_node(vertices[n4][0] + outer_bound, vertices[n4][1])
        n13 = add_node(vertices[n14][0], vertices[n14][1] - (outer_bound + para_depth))
        add_segment(n1, n11)
        add_segment(n11, n12)
        add_segment(n12, n13)
        add_segment(n13, n14)
        add_segment(n14, n4)
        regions.append([vertices[n12][0] + 1.0e-3, vertices[n12][1] + 1.0e-3, 1.0, 0.0])

    add_segment(n1, n2)
    add_segment(n2, n3)
    add_segment(n3, n4)
    regions.append([vertices[n2][0] + 1.0e-3, vertices[n2][1] + 1.0e-3, 2.0, 0.0])

    surface = [n1]
    for index, (x_coord, z_coord) in enumerate(sensors):
        surface.append(add_node(float(x_coord), float(z_coord)))
        if index < sensors.shape[0] - 1:
            next_x, next_z = sensors[index + 1]
            surface.append(add_node(float((x_coord + next_x) * 0.5), float((z_coord + next_z) * 0.5)))
    surface.append(n4)

    seen: set[int] = set()
    surface = [node_id for node_id in surface if not (node_id in seen or seen.add(node_id))]
    surface.sort(key=lambda node_id: vertices[node_id][0])
    for index in range(len(surface) - 1, 0, -1):
        add_segment(surface[index], surface[index - 1])

    triangle_input = {
        "vertices": np.asarray(vertices, dtype=float),
        "segments": np.asarray(segments, dtype=np.int32),
        "regions": np.asarray(regions, dtype=float),
    }
    triangle_mesh = triangle_lib.triangulate(triangle_input, f"pzeAq{quality:g}aQ")
    nodes_all = np.asarray(triangle_mesh["vertices"], dtype=float)
    cells_all = np.asarray(triangle_mesh["triangles"], dtype=np.int32)
    attrs = np.rint(np.asarray(triangle_mesh["triangle_attributes"], dtype=float).reshape(-1)).astype(np.int32)
    nodes_all = _smooth_triangle_nodes(
        nodes_all,
        cells_all,
        attrs,
        plc_node_count=len(vertices),
        iterations=int(smoothing_iterations),
    )
    parameter_cell_ids = np.flatnonzero(attrs == 2).astype(np.int32)
    parameter_cells = cells_all[parameter_cell_ids]
    if parameter_cells.size == 0:
        raise ValueError("triangle did not produce any parameter-domain cells")

    used_node_ids_list: list[int] = []
    seen_node_ids: set[int] = set()
    for node_id in parameter_cells.reshape(-1):
        node_id_int = int(node_id)
        if node_id_int not in seen_node_ids:
            seen_node_ids.add(node_id_int)
            used_node_ids_list.append(node_id_int)
    used_node_ids = np.asarray(used_node_ids_list, dtype=np.int32)
    remap = np.full(nodes_all.shape[0], -1, dtype=np.int32)
    remap[used_node_ids] = np.arange(used_node_ids.size, dtype=np.int32)
    return _SourcePositionTriangleArrays(
        parameter_nodes=nodes_all[used_node_ids],
        parameter_cells=remap[parameter_cells],
        full_nodes=nodes_all,
        full_cells=cells_all,
        cell_markers=attrs,
        parameter_cell_ids=parameter_cell_ids,
    )


def _load_mesh_npz_arrays(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        missing = {"nodes", "cells"}.difference(data.files)
        if missing:
            raise KeyError(f"{path} missing required mesh arrays: {sorted(missing)}")
        surface_node_ids = (
            np.asarray(data["surface_node_ids"], dtype=np.int32).ravel()
            if "surface_node_ids" in data.files
            else None
        )
        nodes = np.asarray(data["nodes"], dtype=float)
        cells = np.asarray(data["cells"], dtype=np.int32)
        full_nodes = np.asarray(data["forward_nodes"], dtype=float) if "forward_nodes" in data.files else nodes
        full_cells = np.asarray(data["forward_cells"], dtype=np.int32) if "forward_cells" in data.files else cells
        cell_markers = (
            np.asarray(data["cell_markers"], dtype=np.int32).ravel()
            if "cell_markers" in data.files
            else np.full(full_cells.shape[0], 2, dtype=np.int32)
        )
        parameter_cell_ids = (
            np.asarray(data["parameter_cell_ids"], dtype=np.int32).ravel()
            if "parameter_cell_ids" in data.files
            else np.arange(cells.shape[0], dtype=np.int32)
        )
        return nodes, cells, surface_node_ids, full_nodes, full_cells, cell_markers, parameter_cell_ids


def build_source_position_triangle_inversion_case(
    elec_x: np.ndarray,
    elec_z: np.ndarray,
    measurements: np.ndarray,
    x_nodes: np.ndarray,
    z_top: np.ndarray,
    layer_thickness: np.ndarray,
    *,
    y_index: int,
    depth_levels: int = 11,
    quality: float = 34.0,
    smoothing_iterations: int = 10,
    data_file: str | Path | None = None,
    mesh_file: str | Path | None = None,
) -> SourcePositionInversionCase:
    """Build a source-position driven triangular inversion mesh.

    The notebooks create ``paraDomain`` from ERT source positions instead of
    reusing the structured ParFlow forward grid. This helper mirrors that PLC
    construction in deepert and uses Triangle directly.
    If ``mesh_file`` is provided, saved paraDomain arrays are used as a
    bootstrap cache and no mesh generator is imported.
    """

    elec_x_array = np.asarray(elec_x, dtype=float).ravel()
    elec_z_array = np.asarray(elec_z, dtype=float).ravel()
    if elec_x_array.shape != elec_z_array.shape:
        raise ValueError("elec_x and elec_z must have the same shape")
    if elec_x_array.size < 4:
        raise ValueError("at least four source/electrode positions are required")
    if not np.all(np.isfinite(elec_x_array)) or not np.all(np.isfinite(elec_z_array)):
        raise ValueError("electrode positions contain non-finite values")
    if np.any(np.diff(elec_x_array) <= 0.0):
        raise ValueError("elec_x must be strictly increasing")

    measurement_array = np.asarray(measurements, dtype=np.int32)
    if measurement_array.ndim != 2 or measurement_array.shape[1] != 4:
        raise ValueError("measurements must have shape (n_measurements, 4)")
    if np.any(measurement_array < 0) or np.any(measurement_array >= elec_x_array.size):
        raise ValueError("measurements reference electrodes outside the source positions")

    x_node_array = np.asarray(x_nodes, dtype=float).ravel()
    z_top_array = np.asarray(z_top, dtype=float).ravel()
    thickness_array = np.asarray(layer_thickness, dtype=float).ravel()
    if x_node_array.shape != z_top_array.shape:
        raise ValueError("x_nodes and z_top must have the same shape")
    if x_node_array.size < 2:
        raise ValueError("x_nodes must contain at least two nodes")
    if np.any(np.diff(x_node_array) <= 0.0):
        raise ValueError("x_nodes must be strictly increasing")
    if not np.all(np.isfinite(thickness_array)) or np.any(thickness_array <= 0.0):
        raise ValueError("layer_thickness must contain positive finite values")

    if data_file is not None:
        parsed_data = load_terrain_forward_dat(data_file)
        elec_x_array = parsed_data.elec_x
        elec_z_array = parsed_data.elec_z
        measurement_array = parsed_data.measurements

    surface_node_ids = None
    if mesh_file is not None:
        (
            nodes,
            cells,
            surface_node_ids,
            full_nodes,
            full_cells,
            cell_markers,
            parameter_cell_ids,
        ) = _load_mesh_npz_arrays(mesh_file)
    else:
        if depth_levels < 2:
            raise ValueError("depth_levels must be >= 2")
        triangle_arrays = _source_position_triangle_arrays(
            elec_x_array,
            elec_z_array,
            quality=float(quality),
            smoothing_iterations=int(smoothing_iterations),
        )
        nodes = triangle_arrays.parameter_nodes
        cells = triangle_arrays.parameter_cells
        full_nodes = triangle_arrays.full_nodes
        full_cells = triangle_arrays.full_cells
        cell_markers = triangle_arrays.cell_markers
        parameter_cell_ids = triangle_arrays.parameter_cell_ids

    mesh = Mesh.from_arrays(
        torch_np.asarray(nodes, dtype=FLOAT_DTYPE),
        torch_np.asarray(cells, dtype=torch_np.int32),
        surface_node_ids=None if surface_node_ids is None else torch_np.asarray(surface_node_ids, dtype=torch_np.int32),
    )
    forward_mesh = Mesh.from_arrays(
        torch_np.asarray(full_nodes, dtype=FLOAT_DTYPE),
        torch_np.asarray(full_cells, dtype=torch_np.int32),
    )
    survey = Survey.from_arrays(
        torch_np.asarray(np.column_stack((elec_x_array, elec_z_array)), dtype=FLOAT_DTYPE),
        torch_np.asarray(measurement_array),
    )
    return SourcePositionInversionCase(
        mesh=mesh,
        forward_mesh=forward_mesh,
        survey=survey,
        parameter_cell_ids=np.asarray(parameter_cell_ids, dtype=np.int32),
        cell_markers=np.asarray(cell_markers, dtype=np.int32),
        elec_x=elec_x_array,
        elec_z=elec_z_array,
        x_nodes=x_node_array,
        z_top=z_top_array,
        layer_thickness=thickness_array,
        y_index=int(y_index),
    )


def _terrain_resistivity_vector(rho_2d: np.ndarray, grid: ParflowGrid) -> np.ndarray:
    """Validate and flatten a ParFlow bottom-to-top resistivity slice."""

    rho_array = np.asarray(rho_2d, dtype=float)
    if rho_array.shape != (grid.nz, grid.nx):
        raise ValueError(f"resistivity shape {rho_array.shape} does not match parsed grid {(grid.nz, grid.nx)}")
    if not np.all(np.isfinite(rho_array)):
        raise ValueError("resistivity contains non-finite values")
    if np.any(rho_array <= 0.0):
        raise ValueError("resistivity must contain positive values")

    return np.asarray(rho_array[::-1, :].reshape(-1), dtype=float)


def build_terrain_forward_case(
    rho_2d: np.ndarray,
    grid: ParflowGrid,
    slope_x: np.ndarray,
    *,
    y_index: int,
    n_electrodes: int = 48,
    topo_offset: float = 0.0,
) -> TerrainForwardCase:
    """Build native deepert mesh/survey/model arrays for one ParFlow slice.

    ``rho_2d`` is expected in ParFlow z-order, bottom-to-top. The returned
    resistivity vector is top-to-bottom and cell-aligned with the generated
    terrain-following mesh.
    """

    resistivity = _terrain_resistivity_vector(rho_2d, grid)
    slope_array = np.asarray(slope_x, dtype=float).ravel()
    if not 0 <= y_index < grid.ny:
        raise ValueError(f"y_index={y_index} out of range [0, {grid.ny - 1}]")
    if slope_array.shape != (grid.nx,):
        raise ValueError(f"slope_x shape {slope_array.shape} does not match NX={grid.nx}")

    layer_thickness = (grid.dz_base * grid.dz_scales)[::-1]
    y_offsets = np.concatenate(([0.0], -np.cumsum(layer_thickness)))

    x_nodes = np.arange(grid.nx + 1, dtype=float) * grid.dx
    z_top = np.zeros(grid.nx + 1, dtype=float)
    z_top[1:] = np.cumsum(slope_array * grid.dx)
    z_top = z_top + topo_offset

    nodes = np.asarray(
        [[x_coord, z_coord + offset] for offset in y_offsets for x_coord, z_coord in zip(x_nodes, z_top, strict=False)],
        dtype=float,
    )

    def node_id(layer_index: int, column_index: int) -> int:
        return layer_index * (grid.nx + 1) + column_index

    cells: list[list[int]] = []
    for layer_index in range(grid.nz):
        for column_index in range(grid.nx):
            cells.append(
                [
                    node_id(layer_index, column_index),
                    node_id(layer_index, column_index + 1),
                    node_id(layer_index + 1, column_index + 1),
                    node_id(layer_index + 1, column_index),
                ]
            )

    electrode_count = min(int(n_electrodes), grid.nx + 1)
    elec_x = np.linspace(float(x_nodes.min()), float(x_nodes.max()), electrode_count)
    elec_z = np.interp(elec_x, x_nodes, z_top)
    measurements = build_wenner_alpha_measurements(electrode_count)

    mesh = Mesh.from_arrays(
        torch_np.asarray(nodes, dtype=FLOAT_DTYPE),
        torch_np.asarray(cells),
        surface_node_ids=torch_np.arange(grid.nx + 1, dtype=torch_np.int32),
    )
    survey = Survey.from_arrays(
        torch_np.asarray(np.column_stack((elec_x, elec_z)), dtype=FLOAT_DTYPE),
        torch_np.asarray(measurements),
    )
    return TerrainForwardCase(
        mesh=mesh,
        survey=survey,
        resistivity=resistivity,
        elec_x=elec_x,
        elec_z=elec_z,
        x_nodes=x_nodes,
        z_top=z_top,
        layer_thickness=layer_thickness,
        y_index=int(y_index),
    )


def run_terrain_forward(
    case: TerrainForwardCase,
    *,
    linear_solver_backend: str = "auto",
    reuse_solver_state: bool = True,
    terrain_cache_dir: str | Path | None = None,
    prepare_forward: bool = False,
) -> np.ndarray:
    """Compute apparent resistivity for a terrain case."""

    runner = TerrainForwardRunner.from_case(
        case,
        linear_solver_backend=linear_solver_backend,
        reuse_solver_state=reuse_solver_state,
        terrain_cache_dir=terrain_cache_dir,
        prepare_forward=prepare_forward,
    )
    try:
        return runner.solve_case(case)
    finally:
        runner.close()


def save_terrain_forward_dat(
    path: str | Path,
    case: TerrainForwardCase,
    rhoa: np.ndarray,
    *,
    relative_error: float = 0.03,
) -> None:
    """Save a reference-style ERT ``.dat`` file matching the notebooks."""

    output_path = Path(path)
    rhoa_array = np.asarray(rhoa, dtype=float).ravel()
    if rhoa_array.shape != (case.survey.measurement_count,):
        raise ValueError(f"rhoa must have shape ({case.survey.measurement_count},)")

    electrodes = np.column_stack((np.asarray(case.elec_x, dtype=float), np.asarray(case.elec_z, dtype=float)))
    measurements = np.asarray(case.survey.measurements, dtype=np.int32)
    geometric_factors = _geometric_factors_np(electrodes, measurements)
    err = np.full(case.survey.measurement_count, float(relative_error), dtype=float)

    lines = [
        f"{electrodes.shape[0]}\n",
        "# x y z\n",
        *(f"{x_coord:.14g}\t{z_coord:.14g}\t0\n" for x_coord, z_coord in electrodes),
        f"{measurements.shape[0]}\n",
        "# a b m n err i ip iperr k r rhoa u valid \n",
    ]
    lines.extend(
        (
            f"{int(abmn[0]) + 1}\t{int(abmn[1]) + 1}\t{int(abmn[2]) + 1}\t{int(abmn[3]) + 1}\t"
            f"{error_value:.14e}\t0.00000000000000e+00\t0.00000000000000e+00\t"
            f"0.00000000000000e+00\t{k_value:.14e}\t0.00000000000000e+00\t"
            f"{rhoa_value:.14e}\t0.00000000000000e+00\t1\n"
        )
        for abmn, error_value, k_value, rhoa_value in zip(
            measurements,
            err,
            geometric_factors,
            rhoa_array,
            strict=True,
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding="utf-8")


def save_terrain_forward_npz(
    path: str | Path,
    case: TerrainForwardCase,
    rhoa: np.ndarray,
    *,
    relative_error: float = 0.03,
) -> None:
    """Save compact forward artifacts and terrain metadata for plotting."""

    output_path = Path(path)
    rhoa_array = np.asarray(rhoa, dtype=float).ravel()
    if rhoa_array.shape != (case.survey.measurement_count,):
        raise ValueError(f"rhoa must have shape ({case.survey.measurement_count},)")

    measurements = np.asarray(case.survey.measurements, dtype=np.int32)
    err = np.full(case.survey.measurement_count, float(relative_error), dtype=float)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        rhoa=rhoa_array,
        err=err,
        a=measurements[:, 0],
        b=measurements[:, 1],
        m=measurements[:, 2],
        n=measurements[:, 3],
        elec_x=case.elec_x,
        elec_z=case.elec_z,
        x_nodes=case.x_nodes,
        z_top=case.z_top,
        layer_thickness=case.layer_thickness,
        y_index=np.asarray([case.y_index], dtype=np.int32),
    )


def run_terrain_forward_file(
    input_file: str | Path,
    grid: ParflowGrid,
    slope_x: np.ndarray,
    output_dir: str | Path,
    *,
    y_index: int | None = None,
    n_electrodes: int = 48,
    topo_offset: float = 0.0,
    relative_error: float = 0.03,
    overwrite: bool = True,
    linear_solver_backend: str = "auto",
    reuse_solver_state: bool = True,
    terrain_cache_dir: str | Path | None = None,
    prepare_forward: bool = False,
) -> TerrainForwardRecord:
    """Run and save one terrain-forward timestep from a resistivity ``.npy`` file."""

    path = Path(input_file)
    parsed_y_index, step = parse_resistivity_slice_name(path)
    if y_index is None:
        if parsed_y_index < 0:
            raise ValueError(f"{path}: y_index is required for notebook-style resistivity_t*.npy files")
        y_index = parsed_y_index
    elif parsed_y_index >= 0 and parsed_y_index != y_index:
        raise ValueError(f"input file y-index {parsed_y_index} does not match requested y_index={y_index}")

    output_root = Path(output_dir)
    dat_file = output_root / f"synthetic_ert_terrain_vardz_t{step:05d}.dat"
    npz_file = output_root / f"synthetic_ert_terrain_vardz_t{step:05d}.npz"
    if not overwrite and dat_file.exists() and npz_file.exists():
        return TerrainForwardRecord(
            step=step,
            input_file=str(path),
            dat_file=str(dat_file),
            npz_file=str(npz_file),
            status="skipped_existing",
        )

    rho_2d = load_terrain_resistivity_slice(path, grid, y_index=int(y_index))
    case = build_terrain_forward_case(
        rho_2d,
        grid,
        slope_x,
        y_index=y_index,
        n_electrodes=n_electrodes,
        topo_offset=topo_offset,
    )
    rhoa = run_terrain_forward(
        case,
        linear_solver_backend=linear_solver_backend,
        reuse_solver_state=reuse_solver_state,
        terrain_cache_dir=terrain_cache_dir,
        prepare_forward=prepare_forward,
    )
    save_terrain_forward_dat(dat_file, case, rhoa, relative_error=relative_error)
    save_terrain_forward_npz(npz_file, case, rhoa, relative_error=relative_error)
    return TerrainForwardRecord(
        step=step,
        input_file=str(path),
        dat_file=str(dat_file),
        npz_file=str(npz_file),
        status="ok",
        rhoa_min=float(np.min(rhoa)),
        rhoa_max=float(np.max(rhoa)),
    )


def run_terrain_forward_series(
    input_files: list[str | Path] | list[tuple[int, str | Path]],
    grid: ParflowGrid,
    slope_x: np.ndarray,
    output_dir: str | Path,
    *,
    y_index: int | None = None,
    n_electrodes: int = 48,
    topo_offset: float = 0.0,
    relative_error: float = 0.03,
    overwrite: bool = True,
    linear_solver_backend: str = "auto",
    reuse_solver_state: bool = True,
    terrain_cache_dir: str | Path | None = None,
    prepare_forward: bool = False,
) -> tuple[list[TerrainForwardRecord], list[TerrainForwardRecord]]:
    """Run a sequential terrain-forward series and return ``(manifest, failures)``.

    The terrain mesh, survey, sparse pattern, auxiliary discretization, Torch
    kernels, and auxiliary-field caches are reused across successful timesteps
    with the same y-index. cuDSS symbolic/plan/buffer state is also reused by
    default; if that fast path returns invalid apparent resistivities, the
    runner rebuilds solver state once and retries the timestep.
    """

    normalized_files: list[Path] = []
    for item in input_files:
        if isinstance(item, tuple):
            _, path = item
            normalized_files.append(Path(path))
        else:
            normalized_files.append(Path(item))

    manifest: list[TerrainForwardRecord] = []
    failures: list[TerrainForwardRecord] = []
    runners: dict[int, TerrainForwardRunner] = {}
    output_root = Path(output_dir)
    try:
        for path in normalized_files:
            try:
                parsed_y_index, step = parse_resistivity_slice_name(path)
                resolved_y_index = parsed_y_index
                if y_index is not None:
                    if parsed_y_index >= 0 and parsed_y_index != y_index:
                        raise ValueError(f"input file y-index {parsed_y_index} does not match requested y_index={y_index}")
                    resolved_y_index = int(y_index)
                elif parsed_y_index < 0:
                    raise ValueError(f"{path}: y_index is required for notebook-style resistivity_t*.npy files")

                dat_file = output_root / f"synthetic_ert_terrain_vardz_t{step:05d}.dat"
                npz_file = output_root / f"synthetic_ert_terrain_vardz_t{step:05d}.npz"
                if not overwrite and dat_file.exists() and npz_file.exists():
                    manifest.append(
                        TerrainForwardRecord(
                            step=step,
                            input_file=str(path),
                            dat_file=str(dat_file),
                            npz_file=str(npz_file),
                            status="skipped_existing",
                        )
                    )
                    continue

                rho_2d = load_terrain_resistivity_slice(path, grid, y_index=resolved_y_index)
                runner = runners.get(resolved_y_index)
                if runner is None:
                    case = build_terrain_forward_case(
                        rho_2d,
                        grid,
                        slope_x,
                        y_index=resolved_y_index,
                        n_electrodes=n_electrodes,
                        topo_offset=topo_offset,
                    )
                    runner = TerrainForwardRunner.from_case(
                        case,
                        linear_solver_backend=linear_solver_backend,
                        reuse_solver_state=reuse_solver_state,
                        terrain_cache_dir=terrain_cache_dir,
                        prepare_forward=prepare_forward,
                    )
                    runners[resolved_y_index] = runner
                    resistivity = case.resistivity
                else:
                    case = runner.case_template
                    resistivity = _terrain_resistivity_vector(rho_2d, grid)

                rhoa = runner.solve_resistivity(resistivity)
                save_terrain_forward_dat(dat_file, case, rhoa, relative_error=relative_error)
                save_terrain_forward_npz(npz_file, case, rhoa, relative_error=relative_error)
                manifest.append(
                    TerrainForwardRecord(
                        step=step,
                        input_file=str(path),
                        dat_file=str(dat_file),
                        npz_file=str(npz_file),
                        status="ok",
                        rhoa_min=float(np.min(rhoa)),
                        rhoa_max=float(np.max(rhoa)),
                    )
                )
            except Exception as exc:
                step = -1
                try:
                    _, step = parse_resistivity_slice_name(path)
                except ValueError:
                    pass
                failures.append(
                    TerrainForwardRecord(
                        step=step,
                        input_file=str(path),
                        dat_file="",
                        npz_file="",
                        status="failed",
                        error=str(exc),
                    )
                )
    finally:
        for runner in runners.values():
            runner.close()

    manifest.sort(key=lambda record: record.step)
    failures.sort(key=lambda record: record.step)
    return manifest, failures
