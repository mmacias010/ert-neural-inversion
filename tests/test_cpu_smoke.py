"""Minimal smoke test — runs a small ERT forward solve on CPU."""

import numpy as np
from deepert.mesh import Mesh
from deepert.survey import Survey
from deepert.forward import ERTForward2p5D

# --- tiny structured triangular mesh (11x6 nodes, 100 triangles) ---
# domain: 100 m wide x 50 m deep, flat surface at z=0
nx, nz = 10, 5          # cells per row/column
dx, dz = 10.0, 10.0    # cell size in metres

xs = np.linspace(0.0, nx * dx, nx + 1)
zs = np.linspace(0.0, -(nz * dz), nz + 1)   # negative = depth below surface
XX, ZZ = np.meshgrid(xs, zs)
nodes = np.column_stack([XX.ravel(), ZZ.ravel()])  # shape (66, 2)

# split each rectangle into two triangles
cells = []
for row in range(nz):
    for col in range(nx):
        tl = row * (nx + 1) + col
        tr = tl + 1
        bl = tl + (nx + 1)
        br = bl + 1
        cells.append([tl, tr, bl])
        cells.append([tr, br, bl])
cells = np.array(cells, dtype=np.int32)   # shape (100, 3)

# surface nodes are the top row (z == 0)
surface_node_ids = np.arange(nx + 1, dtype=np.int32)

mesh = Mesh.from_arrays(nodes, cells, surface_node_ids=surface_node_ids)
print(f"Mesh: {mesh.node_count} nodes, {mesh.cell_count} cells")

# --- 5 equally spaced electrodes along the surface ---
electrode_x = np.linspace(20.0, 80.0, 5)
electrode_positions = np.column_stack([electrode_x, np.zeros(5)])   # z=0

# Wenner-alpha scheme: A B M N with spacing a, 2a, 3a
a = int(electrode_x[1] - electrode_x[0]) // 10  # electrode index step
measurements = []
for i in range(5 - 3):
    measurements.append([i, i + 3, i + 1, i + 2])  # A, B, M, N indices
measurements = np.array(measurements, dtype=np.int32)

survey = Survey.from_arrays(electrode_positions, measurements)
print(f"Survey: {survey.electrode_count} electrodes, {survey.measurement_count} measurements")

# --- build forward operator (auto -> scipy on CPU) ---
forward = ERTForward2p5D.from_mesh_survey(mesh, survey)
print(f"Linear solver backend resolved to: {forward.linear_solver_backend!r}")
assert forward.linear_solver_backend == "scipy", "Expected scipy fallback on CPU"

# --- run forward solve with uniform resistivity of 100 ohm-m ---
resistivity = np.full(mesh.cell_count, 100.0)
rhoa = forward.apparent_resistivity_values(1.0 / resistivity)

print(f"Apparent resistivities (ohm-m): {np.asarray(rhoa).round(2)}")
print("Smoke test PASSED")
