"""Step 0 regression: prove the deepert.inr extraction reproduces the baseline.

The refactor moved mesh/survey/coords/SIREN/training-loop out of
``test_siren_inversion.py`` and into ``deepert/inr/`` without changing any
math. Everything is deterministic (fixed Torch seed for the network, fixed
RandomState for the noise), so the numbers must match the recorded baseline
exactly -- tolerances below only absorb the rounding in the printed table.

If this fails, the extraction changed behaviour and nothing downstream can be
trusted. Fix it before adding a second architecture.

Run on CPU:
    .venv\\Scripts\\python.exe step0_regression.py
"""

from __future__ import annotations

import numpy as np

from deepert.inr import (
    SIREN,
    build_mesh,
    build_survey,
    count_parameters,
    fit_inr,
    normalized_coords,
    seed_networks,
    synthetic_observations,
    two_layer_resistivity,
)
from deepert.inversion.core import ParameterizedERTForward2p5D

# Captured from test_siren_inversion.py before the refactor.
# iteration -> (chi2, rms_percent, top_layer_rho, bottom_layer_rho)
BASELINE = {
    0:   (2150.533, 114.42, 137.6,  94.6),
    50:  (   6.845,   3.95,  75.8, 158.9),
    100: (   3.557,   2.82,  72.1, 160.5),
    150: (   2.608,   2.42,  69.7, 169.3),
    200: (   2.291,   2.27,  69.0, 173.4),
    250: (   2.015,   2.12,  68.5, 177.3),
    300: (   1.777,   2.00,  68.2, 180.9),
    350: (   1.670,   1.94,  68.1, 182.7),
    400: (   1.568,   1.88,  68.0, 184.4),
    450: (   1.472,   1.82,  68.0, 186.1),
    500: (   1.425,   1.79,  68.0, 186.9),
}

# Tolerances match the precision the baseline was printed at, nothing looser.
TOL = {"chi2": 1.0e-3, "rms": 1.0e-2, "rho": 1.0e-1}


def main() -> int:
    mesh = build_mesh()
    survey = build_survey()
    n_cells = int(mesh.cell_count)
    print(f"Mesh: {mesh.node_count} nodes, {n_cells} cells")
    print(f"Survey: {survey.electrode_count} electrodes, {survey.measurement_count} measurements")

    forward = ParameterizedERTForward2p5D.from_mesh_survey(
        mesh, survey, np.arange(n_cells, dtype=np.int32),
        background_mode="pygimli_prolongation",
    )
    print(f"Solver backend: {forward.forward_operator.linear_solver_backend!r}")

    centers, coords = normalized_coords(mesh)
    top_mask = centers[:, 1] >= -15.0
    bot_mask = ~top_mask
    true_resistivity = two_layer_resistivity(centers, boundary=-15.0)

    # Noise stream is independent of the network-init stream (see cases.py).
    _, obs_log = synthetic_observations(forward, true_resistivity, relative_noise=0.015, seed=0)
    data_std = 0.015

    # Seed immediately before construction: nothing else may touch the Torch RNG.
    seed_networks(0)
    net = SIREN()
    print(f"SIREN trainable parameters: {count_parameters(net)}\n")

    observed: dict[int, tuple[float, float, float, float]] = {}

    def record(iteration, chi2, rms, log_rho):
        if iteration in BASELINE:
            rho = np.exp(log_rho)
            observed[iteration] = (chi2, rms, float(rho[top_mask].mean()), float(rho[bot_mask].mean()))

    result = fit_inr(forward, net, coords, obs_log, data_std, n_iters=500, callback=record)

    header = f"{'iter':>5}  {'chi2':>12} {'d':>9}  {'rms(%)':>8} {'d':>7}  {'top':>7} {'d':>7}  {'bot':>7} {'d':>7}"
    print(header)
    print("-" * len(header))

    failures: list[str] = []
    for iteration, expected in BASELINE.items():
        if iteration not in observed:
            failures.append(f"iteration {iteration} was never recorded")
            continue
        got = observed[iteration]
        deltas = [g - e for g, e in zip(got, expected)]
        print(
            f"{iteration:>5}  {got[0]:>12.4f} {deltas[0]:>+9.4f}  {got[1]:>8.3f} {deltas[1]:>+7.3f}  "
            f"{got[2]:>7.2f} {deltas[2]:>+7.2f}  {got[3]:>7.2f} {deltas[3]:>+7.2f}"
        )
        for name, index, tol in (("chi2", 0, TOL["chi2"]), ("rms", 1, TOL["rms"]),
                                 ("top_rho", 2, TOL["rho"]), ("bot_rho", 3, TOL["rho"])):
            if abs(deltas[index]) > tol:
                failures.append(
                    f"iteration {iteration} {name}: got {got[index]:.6g}, "
                    f"baseline {expected[index]:.6g}, delta {deltas[index]:+.6g} > tol {tol:g}"
                )

    print(f"\nElapsed: {result['elapsed']:.1f}s over {result['n_iters']} iterations "
          f"({result['elapsed'] / result['n_iters']:.3f}s per forward+Jacobian)")

    forward.close()

    if failures:
        print("\nSTEP 0 REGRESSION FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nSTEP 0 REGRESSION PASSED - deepert.inr reproduces the pre-refactor baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
