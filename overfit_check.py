"""Does optimizing past the noise level actually degrade the model?

Stage 1 concluded that it does -- "optimizing past the noise level degraded
recovered models by 17-49%" -- and that claim now appears in the conclusions
draft, the poster draft, and the rationale for scoring at chi^2 = 1. Every
number behind it came from the SHARED-CAPACITY configuration, which the
capacity sweep later showed was badly sized for several architectures.

A single re-tuned Fourier run (talk_figures.py semiconvergence) showed no
degradation at all: error 0.419 at chi^2 = 1 and 0.419 after fitting 16x
tighter. One run is not enough to overturn a documented claim, so this measures
it properly: every architecture, both configurations, several seeds, both
held-out targets.

Protocol
--------
Both configurations run to the SAME fixed budget so the comparison isolates
configuration rather than budget. 1500 iterations is roughly 10-12x what any
architecture needs to reach chi^2 = 1 here, which is ample room for
over-fitting to appear if it is going to.

Degradation is measured from the chi^2 = 1 snapshot to the end of the budget --
the decision a practitioner actually faces. It is NOT measured from the global
minimum of the error curve, which typically sits in the first few iterations
where the model is still essentially the homogeneous starting guess.

The true model is used only to SCORE the trajectory. Nothing here selects a
configuration or a stopping point using it.

Usage:
    .venv\\Scripts\\python.exe overfit_check.py [workers]
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import torch

from deepert.inr import build_network, fit_inr, seed_networks
from inr_benchmark import GAMMA, build_case

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "inr_results"

N_ITERS = 1500
CHI2_TARGET = 1.0
SEEDS = (10, 11, 12)
TARGETS = ("blocks", "parflow")
ARCHS = ("relu", "siren", "fourier", "dip")


def configurations() -> list[dict]:
    """Stage-1 (shared capacity) and re-tuned (per-architecture) settings."""

    stage1 = json.loads((RESULTS / "tuning.json").read_text(encoding="utf-8"))["best"]
    retuned = json.loads((RESULTS / "capacity_retune.json").read_text(encoding="utf-8"))["best"]

    out: list[dict] = []
    for arch in ARCHS:
        if arch in stage1 and stage1[arch].get("lr") is not None:
            out.append({"arch": arch, "config": "stage1", "width": None,
                        "hparams": stage1[arch]["hparams"], "lr": float(stage1[arch]["lr"])})
        if arch in retuned:
            out.append({"arch": arch, "config": "retuned", "width": int(retuned[arch]["width"]),
                        "hparams": retuned[arch]["hparams"], "lr": float(retuned[arch]["lr"])})
    return out


def _run(job: tuple) -> dict:
    """One inversion, scored at chi^2 = 1 and at the end of the budget.

    Module-level so ProcessPoolExecutor can pickle it (a local def cannot).
    """

    from capacity_sweep import size_kwargs

    cfg, target, seed = job
    torch.set_num_threads(1)
    base = {"arch": cfg["arch"], "config": cfg["config"], "width": cfg["width"],
            "hparams": json.dumps(cfg["hparams"], sort_keys=True), "lr": cfg["lr"],
            "target": target, "seed": seed}

    case = build_case(target, seed)
    try:
        true_log = np.log(case["true_rho"])
        mask = case["mask"]
        error: list[float] = []

        def track(_i: int, _c: float, _r: float, log_rho: np.ndarray) -> None:
            residual = log_rho - true_log
            error.append(float(np.sqrt(np.mean(residual[mask] ** 2))))

        seed_networks(seed)
        size = {} if cfg["width"] is None else size_kwargs(cfg["arch"], cfg["width"])
        net = build_network(cfg["arch"], log_rho_mean=case["log_rho_mean"],
                            **size, **cfg["hparams"])
        result = fit_inr(
            case["forward"], net, case["coords"], case["obs_log"], case["data_std"],
            n_iters=N_ITERS, lr=cfg["lr"], step_size=max(1, N_ITERS // 4), gamma=GAMMA,
            snapshot_at_chi2=CHI2_TARGET, callback=track,
        )

        chi2 = np.asarray(result["chi2_history"], dtype=float)
        arr = np.asarray(error, dtype=float)
        stop = result["snapshot_iteration"]
        if stop is None or not np.all(np.isfinite(arr)):
            return {**base, "status": "never_fit", "stop": None, "error_at_stop": None,
                    "error_at_end": None, "degradation": None, "chi2_end": float(chi2[-1])}
        return {**base, "status": "ok", "stop": int(stop),
                "error_at_stop": float(arr[stop]), "error_at_end": float(arr[-1]),
                "degradation": float((arr[-1] / arr[stop] - 1.0) * 100.0),
                "error_max_after_stop": float(arr[stop:].max()),
                "chi2_end": float(chi2[-1]),
                "fit_gain": float(chi2[stop] / chi2[-1])}
    except Exception as exc:                       # one unstable run must not kill the sweep
        return {**base, "status": f"error: {type(exc).__name__}", "stop": None,
                "error_at_stop": None, "error_at_end": None, "degradation": None}
    finally:
        case["forward"].close()


def main(workers: int) -> None:
    jobs = [(cfg, target, seed)
            for cfg in configurations() for target in TARGETS for seed in SEEDS]
    print(f"{len(jobs)} runs at {N_ITERS} iterations, {workers} workers\n")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_run, jobs))

    (RESULTS / "overfit_check.json").write_text(
        json.dumps({"n_iters": N_ITERS, "seeds": list(SEEDS), "rows": rows}, indent=2),
        encoding="utf-8")

    print(f"{'arch':9s} {'config':9s} {'target':9s} {'n':>2s}  "
          f"{'err@chi2=1':>10s} {'err@end':>8s} {'degradation':>12s}")
    print("-" * 68)
    summary: list[dict] = []
    for arch in ARCHS:
        for config in ("stage1", "retuned"):
            for target in TARGETS:
                group = [r for r in rows if r["arch"] == arch and r["config"] == config
                         and r["target"] == target and r["status"] == "ok"]
                if not group:
                    failed = [r for r in rows if r["arch"] == arch and r["config"] == config
                              and r["target"] == target]
                    note = failed[0]["status"] if failed else "missing"
                    print(f"{arch:9s} {config:9s} {target:9s}  -  {note}")
                    continue
                at_stop = statistics.mean(r["error_at_stop"] for r in group)
                at_end = statistics.mean(r["error_at_end"] for r in group)
                deg = [r["degradation"] for r in group]
                mean_deg = statistics.mean(deg)
                spread = statistics.stdev(deg) if len(deg) > 1 else 0.0
                print(f"{arch:9s} {config:9s} {target:9s} {len(group):2d}  "
                      f"{at_stop:10.3f} {at_end:8.3f} {mean_deg:+8.1f}% +-{spread:4.1f}")
                summary.append({"arch": arch, "config": config, "target": target,
                                "n": len(group), "degradation": mean_deg})

    print()
    for config in ("stage1", "retuned"):
        vals = [s["degradation"] for s in summary if s["config"] == config]
        if vals:
            print(f"{config:9s}: degradation {min(vals):+.0f}% to {max(vals):+.0f}%  "
                  f"(mean {statistics.mean(vals):+.0f}%, {len(vals)} groups)")
    print(f"\nSaved {RESULTS / 'overfit_check.json'}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
