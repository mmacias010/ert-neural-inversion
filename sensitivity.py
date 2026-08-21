"""Sensitivity to initialization and robustness to noise (Stage-1 metrics).

Two studies Hang requested, sharing one runner:

INIT   Same data, different network initializations. Noise realization is held
       FIXED while the Torch initialization seed varies. This is the first
       experiment to actually exploit the split seed streams: everywhere else
       the same integer seeded both, so "sensitive to initialization" and
       "sensitive to this particular noise draw" were confounded. Conventional
       inversion is deterministic given the data, so it has no spread here --
       which is itself the comparison.

NOISE  Same initialization, different noise levels and realizations. Noise
       level varies over {0.5%, 1.5%, 5%} with three realizations each, and
       data_std tracks the level so chi^2 = 1 always means "fit to the noise
       floor". Includes both conventional methods, since the smooth-L2 arm
       already diverged on one realization at the default level -- robustness
       differences exist within Group 1, not only between groups.

Both run every method at its FROZEN configuration; nothing is re-tuned.
Reported per method: mean +- sd of coverage-masked model error at chi^2 = 1,
the coefficient of variation (spread relative to level), and how many runs
fit the data at all.

    python sensitivity.py init  [n_jobs]
    python sensitivity.py noise [n_jobs]
    python sensitivity.py both  [n_jobs]
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
    ARCH_GRID,
    CHI2_TARGET,
    GAMMA,
    HELD_OUT_TARGETS,
    N_ITERS,
    RESULTS,
    STEP_SIZE,
    build_case,
    mean_std,
    model_errors,
)
from deepert.inr import build_network, fit_inr, seed_networks

torch.set_num_threads(1)

INIT_SEEDS = (20, 21, 22, 23, 24)      # vary initialization ...
FIXED_NOISE_SEED = 10                  # ... against this one dataset
FIXED_INIT_SEED = 20                   # noise study holds initialization here
NOISE_LEVELS = (0.005, 0.015, 0.05)
NOISE_SEEDS = (30, 31, 32)
CLASSICAL = ("tv", "smooth_l2")


def frozen_configs() -> tuple[dict, dict]:
    neural = json.loads((RESULTS / "tuning.json").read_text(encoding="utf-8"))["best"]
    tv = json.loads((RESULTS / "traditional.json").read_text(encoding="utf-8"))["frozen"]
    l2 = json.loads((RESULTS / "smooth_l2.json").read_text(encoding="utf-8"))["frozen"]
    return neural, {"tv": tv, "smooth_l2": l2}


def run_neural(arch: str, hparams: dict, lr: float, target: str,
               init_seed: int, noise_seed: int, relative_noise: float) -> dict:
    torch.set_num_threads(1)
    base = {"arch": arch, "target": target, "init_seed": init_seed,
            "noise_seed": noise_seed, "noise_level": relative_noise}
    case = build_case(target, noise_seed, relative_noise=relative_noise)
    try:
        seed_networks(init_seed)
        net = build_network(arch, log_rho_mean=case["log_rho_mean"], **hparams)
        result = fit_inr(
            case["forward"], net, case["coords"], case["obs_log"], case["data_std"],
            n_iters=N_ITERS, lr=lr, step_size=STEP_SIZE, gamma=GAMMA,
            snapshot_at_chi2=CHI2_TARGET,
        )
        snapshot = result["snapshot_log_resistivity"]
        rmse = None
        if snapshot is not None:
            _, rmse = model_errors(np.exp(snapshot), case["true_rho"], case["mask"])
        return {**base, "status": "ok", "rmse_at_chi2": rmse,
                "chi2": result["chi2"], "iters_to_chi2": result["snapshot_iteration"],
                "elapsed": result["elapsed"]}
    except Exception as error:
        return {**base, "status": "diverged", "error": f"{type(error).__name__}: {error}",
                "rmse_at_chi2": None, "chi2": None, "iters_to_chi2": None, "elapsed": None}
    finally:
        case["forward"].close()


def run_classical(name: str, config: dict, target: str,
                  noise_seed: int, relative_noise: float) -> dict:
    from traditional_comparison import run_traditional

    base = {"arch": name, "target": target, "init_seed": None,
            "noise_seed": noise_seed, "noise_level": relative_noise}
    try:
        row = run_traditional(target, noise_seed, config["regularization"],
                              config["operator"], relative_noise=relative_noise)
        fitted = row["chi2"] <= CHI2_TARGET
        return {**base, "status": "ok",
                "rmse_at_chi2": row["rmse_cov"] if fitted else None,
                "chi2": row["chi2"], "iters_to_chi2": row["iters_to_chi2"],
                "elapsed": row["elapsed"]}
    except Exception as error:
        return {**base, "status": "diverged", "error": f"{type(error).__name__}: {error}",
                "rmse_at_chi2": None, "chi2": None, "iters_to_chi2": None, "elapsed": None}


def dispatch(job: tuple) -> dict:
    kind, payload = job
    return run_neural(*payload) if kind == "neural" else run_classical(*payload)


def run_all(jobs: list[tuple], n_jobs: int) -> list[dict]:
    t0 = time.time()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = [pool.submit(dispatch, job) for job in jobs]
        for index, future in enumerate(futures, 1):
            rows.append(future.result())
            if index % max(1, len(jobs) // 20) == 0 or index == len(jobs):
                print(f"    {index}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"  {len(jobs)} runs in {time.time() - t0:.0f}s ({n_jobs} workers)")
    return rows


def summarize(rows: list[dict], group_key, group_label: str, keys: list) -> None:
    """One block per group value; CV% is sd/mean, the spread relative to level."""

    print(f"\n{'method':12s}{group_label:>10s}{'rmse@chi2=1':>21s}{'CV%':>8s}{'fit':>8s}{'chi2':>10s}")
    print("-" * 69)
    for key in keys:
        for arch in [*ARCH_GRID, *CLASSICAL]:
            group = [r for r in rows if r["arch"] == arch and group_key(r) == key]
            if not group:
                continue
            fitted = [r["rmse_at_chi2"] for r in group if r["rmse_at_chi2"] is not None]
            mean, sd = mean_std(fitted)
            chi2, _ = mean_std([r["chi2"] for r in group if r["chi2"] is not None])
            value = f"{mean:>14.4f} +-{sd:<5.4f}" if fitted else f"{'never fit':>21s}"
            cv = f"{100 * sd / mean:>8.1f}" if fitted and np.isfinite(mean) and mean > 0 else f"{'-':>8s}"
            chi2_label = f"{chi2:>10.3f}" if np.isfinite(chi2) else f"{'-':>10s}"
            print(f"{arch:12s}{str(key):>10s}{value}{cv}"
                  f"{f'{len(fitted)}/{len(group)}':>8s}{chi2_label}")
        if len(keys) > 1:
            print()


def study_init(n_jobs: int) -> list[dict]:
    neural, _ = frozen_configs()
    print(f"INIT SENSITIVITY - noise seed fixed at {FIXED_NOISE_SEED}, "
          f"init seeds {INIT_SEEDS}\n")
    jobs = [("neural", (arch, neural[arch]["hparams"], neural[arch]["lr"], target,
                        init_seed, FIXED_NOISE_SEED, 0.015))
            for target in HELD_OUT_TARGETS for arch in ARCH_GRID if arch in neural
            for init_seed in INIT_SEEDS]
    print(f"  {len(jobs)} runs")
    rows = run_all(jobs, n_jobs)
    for target in HELD_OUT_TARGETS:
        print(f"\n=== {target} (identical data; spread is initialization alone)")
        summarize([r for r in rows if r["target"] == target],
                  lambda r: "", "", [""])
    return rows


def study_noise(n_jobs: int) -> list[dict]:
    neural, classical = frozen_configs()
    print(f"NOISE ROBUSTNESS - init fixed at {FIXED_INIT_SEED}, levels {NOISE_LEVELS}, "
          f"realizations {NOISE_SEEDS}\n")
    jobs: list[tuple] = []
    for target in HELD_OUT_TARGETS:
        for level in NOISE_LEVELS:
            for noise_seed in NOISE_SEEDS:
                for arch in ARCH_GRID:
                    if arch in neural:
                        jobs.append(("neural", (arch, neural[arch]["hparams"], neural[arch]["lr"],
                                                target, FIXED_INIT_SEED, noise_seed, level)))
                for name in CLASSICAL:
                    jobs.append(("classical", (name, classical[name], target, noise_seed, level)))
    print(f"  {len(jobs)} runs")
    rows = run_all(jobs, n_jobs)
    for target in HELD_OUT_TARGETS:
        print(f"\n=== {target}")
        summarize([r for r in rows if r["target"] == target],
                  lambda r: f"{100 * r['noise_level']:g}%", "noise",
                  [f"{100 * lv:g}%" for lv in NOISE_LEVELS])
    return rows


def main() -> int:
    stage = sys.argv[1] if len(sys.argv) > 1 else "both"
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, (os.cpu_count() or 2) - 1)
    print(f"stage={stage}  n_jobs={n_jobs}  iters={N_ITERS}\n")
    RESULTS.mkdir(exist_ok=True)

    if stage in ("init", "both"):
        rows = study_init(n_jobs)
        (RESULTS / "sensitivity_init.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nSaved {RESULTS / 'sensitivity_init.json'}")
    if stage in ("noise", "both"):
        rows = study_noise(n_jobs)
        (RESULTS / "sensitivity_noise.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nSaved {RESULTS / 'sensitivity_noise.json'}")
    if stage not in ("init", "noise", "both"):
        raise SystemExit(f"unknown stage {stage!r}; expected init|noise|both")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
