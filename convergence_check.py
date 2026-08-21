"""How much iteration budget does each architecture actually need?

The benchmark gave every architecture 600 iterations. Fourier reached chi^2 = 1
by iteration 121 on 'blocks'; ReLU and SIREN never got there at all. Their
reported errors therefore measure non-convergence rather than representational
capacity, and no claim about those architectures is defensible until the budget
stops being the binding constraint.

This runs each architecture at its frozen configuration for a much longer budget
and reports when -- or whether -- each reaches the noise level. The learning-rate
schedule is scaled with the budget (step_size = n_iters // 4) so the effective
decay matches the original 600/150 setting rather than annealing to zero.

    .venv\\Scripts\\python.exe convergence_check.py [n_jobs] [n_iters]
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

from inr_benchmark import CHI2_TARGET, DATA_STD, RESULTS, build_case, model_errors
from deepert.inr import build_network, fit_inr, seed_networks

torch.set_num_threads(1)

TARGETS = ("blocks", "parflow")
SEEDS = (10, 11, 12)
PROBE_ITERS = 3000


def run_probe(arch: str, hparams: dict, lr: float, target: str, seed: int, n_iters: int) -> dict:
    """One long run. Divergence is recorded, not raised.

    At extended budgets some configurations blow up to non-finite resistivity,
    which the forward solver rejects. That is a result about the architecture's
    stability, not a harness error, so it is captured per-run instead of being
    allowed to kill the whole batch.
    """

    torch.set_num_threads(1)
    base = {"arch": arch, "target": target, "seed": seed, "lr": lr,
            "hparams": json.dumps(hparams, sort_keys=True)}
    case = build_case(target, seed)
    try:
        seed_networks(seed)
        net = build_network(arch, log_rho_mean=case["log_rho_mean"], **hparams)
        result = fit_inr(
            case["forward"], net, case["coords"], case["obs_log"], DATA_STD,
            n_iters=n_iters, lr=lr,
            step_size=max(1, n_iters // 4), gamma=0.5,      # same 4 halvings as the 600/150 default
            snapshot_at_chi2=CHI2_TARGET,
        )
        history = result["chi2_history"]
        reached = np.flatnonzero(history <= CHI2_TARGET)
        snapshot = result["snapshot_log_resistivity"]
        _, rmse_cov = model_errors(result["resistivity"], case["true_rho"], case["mask"])
        snap_cov = None
        if snapshot is not None:
            _, snap_cov = model_errors(np.exp(snapshot), case["true_rho"], case["mask"])
        return {
            **base,
            "status": "ok",
            "iters_to_chi2": int(reached[0]) if reached.size else None,
            "chi2_final": float(history[-1]),
            "chi2_min": float(history.min()),
            "chi2_at_600": float(history[600]) if history.size > 600 else None,
            "rmse_cov_final": rmse_cov,
            "rmse_cov_at_chi2": snap_cov,
            "elapsed": result["elapsed"],
        }
    except Exception as error:                      # divergence to non-finite resistivity
        return {**base, "status": "diverged", "error": f"{type(error).__name__}: {error}",
                "iters_to_chi2": None, "chi2_final": None, "chi2_min": None,
                "chi2_at_600": None, "rmse_cov_final": None, "rmse_cov_at_chi2": None,
                "elapsed": None}
    finally:
        case["forward"].close()


def main() -> int:
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    n_iters = int(sys.argv[2]) if len(sys.argv) > 2 else PROBE_ITERS

    frozen = json.loads((RESULTS / "tuning.json").read_text(encoding="utf-8"))["best"]
    print(f"Convergence probe: {n_iters} iterations (benchmark used 600)")
    for arch, entry in frozen.items():
        print(f"  {arch:9s} {entry['hparams']}  lr={entry['lr']:g}")

    jobs = [(arch, entry["hparams"], entry["lr"], target, seed, n_iters)
            for target in TARGETS for arch, entry in frozen.items() for seed in SEEDS]
    print(f"\n{len(jobs)} runs on {n_jobs} workers\n")

    t0 = time.time()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = [pool.submit(run_probe, *job) for job in jobs]
        for index, future in enumerate(futures, 1):
            rows.append(future.result())
            print(f"    {index}/{len(jobs)} done ({time.time() - t0:.0f}s)", flush=True)

    for target in TARGETS:
        print(f"\n=== {target} " + "=" * 66)
        print(f"{'arch':9s}{'iters to chi2=1':>18s}{'chi2 @600':>12s}{'chi2 final':>12s}"
              f"{'rmse@chi2=1':>14s}{'rmse final':>12s}")
        print("-" * 77)
        for arch in ("relu", "siren", "fourier"):
            group = [r for r in rows if r["target"] == target and r["arch"] == arch]
            if not group:
                continue
            good = [r for r in group if r["status"] == "ok"]
            n_diverged = len(group) - len(good)
            if not good:
                print(f"{arch:9s}{'DIVERGED':>18s}{'-':>12s}{'-':>12s}{'-':>14s}{'-':>12s}"
                      f"   ({n_diverged}/{len(group)} runs blew up)")
                continue
            hit = [r["iters_to_chi2"] for r in good if r["iters_to_chi2"] is not None]
            hit_label = f"{int(np.mean(hit))} ({len(hit)}/{len(good)})" if hit else f"never (0/{len(good)})"
            at600 = np.mean([r["chi2_at_600"] for r in good if r["chi2_at_600"] is not None])
            final = np.mean([r["chi2_final"] for r in good])
            snap = [r["rmse_cov_at_chi2"] for r in good if r["rmse_cov_at_chi2"] is not None]
            snap_label = f"{np.mean(snap):.4f}" if snap else "-"
            rmse_final = np.mean([r["rmse_cov_final"] for r in good])
            suffix = f"   ({n_diverged}/{len(group)} diverged)" if n_diverged else ""
            print(f"{arch:9s}{hit_label:>18s}{at600:>12.3f}{final:>12.3f}"
                  f"{snap_label:>14s}{rmse_final:>12.4f}{suffix}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "convergence.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved {RESULTS / 'convergence.json'}")
    print("\nUse the largest 'iters to chi2=1' across architectures to set the benchmark budget,")
    print("so no architecture is reported as failing when it merely ran out of iterations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
