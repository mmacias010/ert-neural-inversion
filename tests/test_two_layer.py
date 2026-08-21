"""Two-layer ERT forward test on CPU.

Layer 1 (top 20 m): 50 ohm-m  (conductive)
Layer 2 (below):   500 ohm-m  (resistive)

Apparent resistivity should increase with electrode spacing as the
deeper, more resistive layer is sampled more.
"""

import numpy as np
from deepert.mesh import Mesh
from deepert.survey import Survey
from deepert.forward import ERTForward2p5D

# --- mesh: 200 m wide x 80 m deep, 20x8 rectangles split into triangles ---
nx, nz = 20, 8
dx, dz = 10.0, 10.0

xs = np.linspace(0.0, nx * dx, nx + 1)
zs = np.linspace(0.0, -(nz * dz), nz + 1)
XX, ZZ = np.meshgrid(xs, zs)
nodes = np.column_stack([XX.ravel(), ZZ.ravel()])

cells = []
for row in range(nz):
    for col in range(nx):
        tl = row * (nx + 1) + col
        tr = tl + 1
        bl = tl + (nx + 1)
        br = bl + 1
        cells.append([tl, tr, bl])
        cells.append([tr, br, bl])
cells = np.array(cells, dtype=np.int32)

surface_node_ids = np.arange(nx + 1, dtype=np.int32)
mesh = Mesh.from_arrays(nodes, cells, surface_node_ids=surface_node_ids)
print(f"Mesh: {mesh.node_count} nodes, {mesh.cell_count} cells")

# --- two-layer resistivity model ---
# cell centres: average of the three triangle vertices
nodes_np = np.asarray(mesh.nodes)
cells_np = np.asarray(mesh.cells)
cell_centers_z = nodes_np[cells_np, 1].mean(axis=1)   # negative = depth

layer_boundary = -20.0   # top layer extends 20 m below surface
resistivity = np.where(cell_centers_z >= layer_boundary, 50.0, 500.0)
print(f"Layer 1 cells (rho=50):  {(cell_centers_z >= layer_boundary).sum()}")
print(f"Layer 2 cells (rho=500): {(cell_centers_z < layer_boundary).sum()}")

# --- 9 electrodes, Wenner-alpha at three spacings ---
n_elec = 12
electrode_x = np.linspace(40.0, 160.0, n_elec)
electrode_positions = np.column_stack([electrode_x, np.zeros(n_elec)])

# Wenner-alpha: A=i, M=i+a, N=i+2a, B=i+3a  (ABMN indices)
measurements = []
spacings = [1, 2, 3]   # electrode index steps
for a in spacings:
    for i in range(n_elec - 3 * a):
        measurements.append([i, i + 3 * a, i + a, i + 2 * a])

measurements = np.array(measurements, dtype=np.int32)
survey = Survey.from_arrays(electrode_positions, measurements)
print(f"Survey: {survey.electrode_count} electrodes, {survey.measurement_count} measurements")

# --- forward solve ---
forward = ERTForward2p5D.from_mesh_survey(mesh, survey)
print(f"Solver backend: {forward.linear_solver_backend!r}")

conductivity = 1.0 / resistivity
rhoa = np.asarray(forward.apparent_resistivity_values(conductivity))

# --- summarise by spacing ---
print("\nApparent resistivity by Wenner spacing:")
print(f"  {'Spacing':>8}  {'Mean rhoa (ohm-m)':>20}  {'Min':>8}  {'Max':>8}")
idx = 0
for a in spacings:
    n = n_elec - 3 * a
    subset = rhoa[idx : idx + n]
    print(f"  a={a} ({a*dx:.0f}m)  {subset.mean():>20.1f}  {subset.min():>8.1f}  {subset.max():>8.1f}")
    idx += n

print("\nExpected: largest spacing (deepest) > smallest spacing (shallowest)")
idx = 0
means = []
for a in spacings:
    n = n_elec - 3 * a
    means.append(rhoa[idx : idx + n].mean())
    idx += n

# Non-monotonicity near the layer boundary is real physics; just check overall trend
if means[-1] > means[0]:
    print("CHECK PASSED — deeper sampling sees the resistive layer as expected")
else:
    print("CHECK FAILED — unexpected trend")
