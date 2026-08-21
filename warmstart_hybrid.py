"""Hybrid arm: classical TV inversion -> neural projection -> physics refinement.

The mentor-level question is whether a neural parameterization can IMPROVE the
classical result rather than replace it. Design:

  1. CLASSICAL   Gauss-Newton with the frozen TV configuration, stopping at
                 chi^2 = 1 as usual. (~4 iterations, ~1 s)
  2. PROJECT     Pre-fit the coordinate network to the classical log-model by
                 plain regression -- no FEM solves, a few seconds. This passes
                 the classical solution through the architecture's inductive
                 bias, perturbing both the field and its data fit.
  3. REFINE      Continue physics-in-the-loop optimization from those weights
                 and report the model at the first iteration with chi^2 <= 1.

If the projection happens to preserve chi^2 <= 1, the snapshot fires at
iteration 0 and the hybrid legitimately reports the projected model itself.

Compared arms, identical targets/seeds/configs to the corrected benchmark:
  classical alone            (traditional.json, frozen TV)
  INR from scratch           (evaluation.json, corrected 1500-iter run)
  hybrid                     (this script; siren + fourier at frozen configs)

    .venv\\Scripts\\python.exe warmstart_hybrid.py [n_jobs]
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from concurrent.futures import ProcessPoolExecutor
import json
import sys
import time

import numpy as np
import torch

from inr_benchmark import (
    CHI2_TARGET,
    DATA_STD,
    HELD_OUT_TARGETS,
    RESULTS,
    build_case,
    mean_std,
    model_errors,
)
from deepert.inr import build_network, fit_inr, seed_networks
from deepert.inversion.core import InversionConfig, invert_single_log_resistivity

torch.set_num_threads(1)

SEEDS = (10, 11, 12, 13, 14)          # evaluation seeds, matching the corrected run
PREFIT_STEPS = 2000
REFINE_ITERS = 400
REFINE_STEP_SIZE = 100                # keeps the four-halvings schedule shape


def pre_fit(net, coords_t: torch.Tensor, target_log: np.ndarray, steps: int = PREFIT_STEPS) -> float:
    """Regress the network onto the classical log-model. No FEM solves."""

    target_t = torch.as_tensor(target_log, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=1.0e-3)
    loss_value = float("nan")
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.mean((net(coords_t) - target_t) ** 2)
        loss.backward()
        opt.step()
        loss_value = float(loss.detach())
    return loss_value


def run_hybrid(arch: str, hparams: dict, lr: float, target: str, seed: int,
               trad_reg: float, trad_op: str) -> dict:
    torch.set_num_threads(1)
    base = {"arch": arch, "target": target, "seed": seed,
            "hparams": json.dumps(hparams, sort_keys=True), "lr": lr}
    case = build_case(target, seed)
    try:
        # -- 1. classical -----------------------------------------------------
        config = InversionConfig(
            max_iterations=30, data_std=DATA_STD, regularization=trad_reg,
            spatial_regularization=trad_op, target_chi2=CHI2_TARGET,
        )
        t0 = time.time()
        classical = invert_single_log_resistivity(
            case["forward"], np.exp(case["obs_log"]),
            np.full(case["true_rho"].size, float(np.exp(case["log_rho_mean"]))),
            config=config,
        )
        classical_rho = np.asarray(classical.final_model, dtype=float)
        _, classical_rmse = model_errors(classical_rho, case["true_rho"], case["mask"])

        # -- 2. project onto the network prior (no physics) -------------------
        seed_networks(seed)
        net = build_network(arch, log_rho_mean=case["log_rho_mean"], **hparams)
        coords_t = torch.as_tensor(np.asarray(case["coords"], dtype=float), dtype=torch.float32)
        prefit_mse = pre_fit(net, coords_t, np.log(classical_rho))
        with torch.no_grad():
            projected_log = net(coords_t).numpy().astype(float)
        _, projected_rmse = model_errors(np.exp(projected_log), case["true_rho"], case["mask"])

        # -- 3. refine with physics, report at chi^2 = 1 ----------------------
        result = fit_inr(
            case["forward"], net, case["coords"], case["obs_log"], DATA_STD,
            n_iters=REFINE_ITERS, lr=lr, step_size=REFINE_STEP_SIZE, gamma=0.5,
            snapshot_at_chi2=CHI2_TARGET,
        )
        snapshot = result["snapshot_log_resistivity"]
        if snapshot is None:
            hybrid_rmse = None
        else:
            _, hybrid_rmse = model_errors(np.exp(snapshot), case["true_rho"], case["mask"])
        return {
            **base, "status": "ok",
            "classical_rmse": classical_rmse,
            "classical_chi2": float(classical.iteration_chi2[-1]),
            "prefit_mse": prefit_mse,
            "projected_rmse": projected_rmse,
            "projected_chi2": float(result["chi2_history"][0]),
            "hybrid_rmse": hybrid_rmse,
            "hybrid_iteration": result["snapshot_iteration"],
            "final_chi2": result["chi2"],
            "elapsed": time.time() - t0,
        }
    except Exception as error:
        return {**base, "status": "diverged", "error": f"{type(error).__name__}: {error}",
                "classical_rmse": None, "classical_chi2": None, "prefit_mse": None,
                "projected_rmse": None, "projected_chi2": None,
                "hybrid_rmse": None, "hybrid_iteration": None, "final_chi2": None,
                "elapsed": None}
    finally:
        case["forward"].close()


def main() -> int:
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 6

    frozen_neural = json.loads((RESULTS / "tuning.json").read_text(encoding="utf-8"))["best"]
    frozen_trad = json.loads((RESULTS / "traditional.json").read_text(encoding="utf-8"))["frozen"]
    scratch_rows = json.loads((RESULTS / "evaluation.json").read_text(encoding="utf-8"))

    archs = [a for a in ("siren", "fourier") if a in frozen_neural]
    print("HYBRID: classical TV -> neural projection -> physics refinement")
    print(f"classical: {frozen_trad['operator']} lambda={frozen_trad['regularization']:g}")
    for arch in archs:
        entry = frozen_neural[arch]
        print(f"{arch:9s} {entry['hparams']}  lr={entry['lr']:g}")

    jobs = [(arch, frozen_neural[arch]["hparams"], frozen_neural[arch]["lr"],
             target, seed, frozen_trad["regularization"], frozen_trad["operator"])
            for target in HELD_OUT_TARGETS for arch in archs for seed in SEEDS]
    print(f"\n{len(jobs)} runs on {n_jobs} workers\n")

    t0 = time.time()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = [pool.submit(run_hybrid, *job) for job in jobs]
        for index, future in enumerate(futures, 1):
            rows.append(future.result())
            print(f"    {index}/{len(jobs)} done ({time.time() - t0:.0f}s)", flush=True)

    for target in HELD_OUT_TARGETS:
        print(f"\n=== {target} " + "=" * 64)
        print(f"{'arch':9s}{'classical':>16s}{'projected':>16s}{'hybrid@chi2=1':>16s}"
              f"{'scratch@chi2=1':>16s}{'proj chi2':>11s}")
        print("-" * 84)
        for arch in archs:
            group = [r for r in rows if r["target"] == target and r["arch"] == arch
                     and r["status"] == "ok"]
            if not group:
                print(f"{arch:9s}  all runs diverged")
                continue
            c = mean_std([r["classical_rmse"] for r in group])
            p = mean_std([r["projected_rmse"] for r in group])
            h = mean_std([r["hybrid_rmse"] for r in group])
            pc = mean_std([r["projected_chi2"] for r in group])
            scratch = mean_std([r["rmse_cov_at_chi2"] for r in scratch_rows
                                if r["target"] == target and r["arch"] == arch
                                and r.get("status", "ok") == "ok"])
            n_hit = sum(1 for r in group if r["hybrid_rmse"] is not None)
            h_label = f"{h[0]:>9.4f} +-{h[1]:<5.4f}" if n_hit == len(group) else f"{n_hit}/{len(group)} fit"
            print(f"{arch:9s}{c[0]:>9.4f} +-{c[1]:<5.4f}{p[0]:>9.4f} +-{p[1]:<5.4f}"
                  f"{h_label:>16s}{scratch[0]:>9.4f} +-{scratch[1]:<5.4f}{pc[0]:>11.3f}")

    print("\nReading: 'hybrid < classical' = neural refinement improved the classical result.")
    print("         'hybrid ~= projected' = physics restoration changed little.")
    print("         'projected < classical' = the architectural prior alone denoised the model.")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "hybrid.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved {RESULTS / 'hybrid.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
