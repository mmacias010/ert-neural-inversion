"""Disentangle capacity from hyperparameters, and budget from capacity.

The capacity sweep (capacity_sweep.py) varied network size while holding
hyperparameters frozen at values chosen when every network had 33,537
parameters. That leaves two confounds in its headline result:

  CONFOUND A   SIREN appeared to peak at ~132k parameters. But w0=3 was
               selected at 33k, and the right frequency for a 132k network
               need not be the right frequency for a 33k one. "SIREN peaks at
               132k" and "SIREN peaks wherever its 33k-tuned w0 happens to
               work" are indistinguishable from that experiment.

  CONFOUND B   The 1500-iteration budget came from a convergence probe run at
               33k. SIREN lost fits at 527k (3/5), which may be
               under-convergence rather than over-capacity -- larger networks
               plausibly need more iterations, not fewer.

Three stages, in dependency order:

  probe    Convergence at each capacity: how many iterations does each
           architecture need to reach chi^2 = 1 when it is larger? Sets an
           honest budget for the stages that follow, resolving CONFOUND B.
  tune     Re-sweep each architecture's own knob (w0, sigma) AT its best
           capacity, using the probe's budget. Resolves CONFOUND A.
  eval     Frozen re-tuned configurations on held-out targets, fresh seeds,
           directly comparable to Stage 1 and to the capacity sweep.

Selection rules match the rest of the benchmark: a configuration is eligible
only if every seed reaches chi^2 <= 1, and it is scored on the model at that
snapshot rather than at the end of the budget.

    python capacity_retune.py probe [n_jobs]
    python capacity_retune.py tune  [n_jobs]
    python capacity_retune.py eval  [n_jobs]
    python capacity_retune.py all   [n_jobs]
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
    DEV_TARGET,
    EVAL_SEEDS,
    GAMMA,
    HELD_OUT_TARGETS,
    LEARNING_RATES,
    RESULTS,
    TUNE_SEEDS,
    build_case,
    mean_std,
    model_errors,
)
from capacity_sweep import WIDTHS, budget_for, size_kwargs
from deepert.inr import build_network, count_parameters, fit_inr, seed_networks

torch.set_num_threads(1)

PROBE_ITERS = 6000                 # generous; the probe measures, it does not cap
PROBE_SEEDS = (10, 11, 12)
PROBE_WIDTHS = (128, 256, 512)     # 64 already fits within budget; no need to re-probe
BUDGET_MARGIN = 1.4                # budget = 1.4x the slowest observed fit, rounded up
MIN_BUDGET = 1500                  # never go below the Stage-1 budget


def schedule_for(n_iters: int) -> tuple[int, float]:
    """Four learning-rate halvings, matching the Stage-1 600/150 shape."""

    return max(1, n_iters // 4), GAMMA


def run_one(arch: str, hparams: dict, lr: float, width: int, target: str,
            seed: int, n_iters: int) -> dict:
    torch.set_num_threads(1)
    base = {"arch": arch, "width": width, "budget": budget_for(width), "lr": lr,
            "hparams": json.dumps(hparams, sort_keys=True), "target": target,
            "seed": seed, "n_iters": n_iters}
    case = build_case(target, seed)
    try:
        seed_networks(seed)
        net = build_network(arch, log_rho_mean=case["log_rho_mean"],
                            **size_kwargs(arch, width), **hparams)
        step_size, gamma = schedule_for(n_iters)
        result = fit_inr(
            case["forward"], net, case["coords"], case["obs_log"], case["data_std"],
            n_iters=n_iters, lr=lr, step_size=step_size, gamma=gamma,
            snapshot_at_chi2=CHI2_TARGET,
        )
        snapshot = result["snapshot_log_resistivity"]
        rmse = None
        if snapshot is not None:
            _, rmse = model_errors(np.exp(snapshot), case["true_rho"], case["mask"])
        return {**base, "status": "ok", "n_params": count_parameters(net),
                "rmse_at_chi2": rmse, "iters_to_chi2": result["snapshot_iteration"],
                "chi2": result["chi2"], "elapsed": result["elapsed"]}
    except Exception as error:
        return {**base, "status": "diverged", "error": f"{type(error).__name__}: {error}",
                "n_params": None, "rmse_at_chi2": None, "iters_to_chi2": None,
                "chi2": None, "elapsed": None}
    finally:
        case["forward"].close()


def run_all(jobs: list[tuple], n_jobs: int, label: str) -> list[dict]:
    print(f"  {label}: {len(jobs)} runs on {n_jobs} workers")
    t0 = time.time()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = [pool.submit(run_one, *job) for job in jobs]
        for index, future in enumerate(futures, 1):
            rows.append(future.result())
            if index % max(1, len(jobs) // 15) == 0 or index == len(jobs):
                print(f"    {index}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"  done in {time.time() - t0:.0f}s\n")
    return rows


def frozen() -> dict:
    return json.loads((RESULTS / "tuning.json").read_text(encoding="utf-8"))["best"]


# ------------------------------------------------------------------- probe
def stage_probe(n_jobs: int) -> dict:
    """How many iterations does each architecture need at each capacity?"""

    best = frozen()
    print(f"PROBE - convergence at widths {PROBE_WIDTHS}, up to {PROBE_ITERS} iterations")
    print("Resolves whether large networks under-converge rather than over-fit.\n")

    jobs = [(arch, best[arch]["hparams"], best[arch]["lr"], width, target, seed, PROBE_ITERS)
            for target in HELD_OUT_TARGETS
            for arch in ARCH_GRID if arch in best
            for width in PROBE_WIDTHS
            for seed in PROBE_SEEDS]
    rows = run_all(jobs, n_jobs, "probe")

    print(f"{'arch':9s}{'width':>7s}{'iters to chi2=1':>18s}{'fit':>7s}{'rmse@chi2=1':>14s}")
    print("-" * 55)
    budgets: dict[str, int] = {}
    for arch in ARCH_GRID:
        for width in PROBE_WIDTHS:
            group = [r for r in rows if r["arch"] == arch and r["width"] == width]
            if not group:
                continue
            hit = [r["iters_to_chi2"] for r in group if r["iters_to_chi2"] is not None]
            rmse = [r["rmse_at_chi2"] for r in group if r["rmse_at_chi2"] is not None]
            label = f"{int(np.mean(hit))} (max {max(hit)})" if hit else "never"
            rmse_label = f"{np.mean(rmse):>14.4f}" if rmse else f"{'-':>14s}"
            print(f"{arch:9s}{width:>7d}{label:>18s}{f'{len(hit)}/{len(group)}':>7s}{rmse_label}")
            if hit:
                budgets[f"{arch}|{width}"] = max(hit)
        print()

    slowest = max(budgets.values()) if budgets else MIN_BUDGET
    recommended = max(MIN_BUDGET, int(np.ceil(slowest * BUDGET_MARGIN / 500) * 500))
    print(f"slowest observed fit: {slowest} iterations")
    print(f"RECOMMENDED BUDGET for tune/eval: {recommended} "
          f"({BUDGET_MARGIN}x margin, rounded to 500)")
    if recommended > MIN_BUDGET:
        print("  -> the Stage-1 budget of 1500 was set at 33k and is too small at")
        print("     these capacities; large-network results were partly under-converged.")
    else:
        print("  -> 1500 remains sufficient; large-network results were NOT")
        print("     budget-limited, so the capacity effect is real.")

    RESULTS.mkdir(exist_ok=True)
    payload = {"recommended_budget": recommended, "per_config": budgets, "rows": rows}
    (RESULTS / "capacity_probe.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {RESULTS / 'capacity_probe.json'}")
    return payload


# -------------------------------------------------------------------- tune
def best_capacity_per_arch() -> dict[str, int]:
    """Width with the lowest mean held-out error in the capacity sweep."""

    rows = json.loads((RESULTS / "capacity_sweep.json").read_text(encoding="utf-8"))
    chosen: dict[str, int] = {}
    for arch in ARCH_GRID:
        scored = []
        for width in WIDTHS:
            group = [r["rmse_at_chi2"] for r in rows
                     if r["arch"] == arch and r["width"] == width
                     and r["rmse_at_chi2"] is not None]
            # Require a majority of runs to have fit, so a single lucky seed
            # cannot nominate a capacity the architecture cannot use.
            total = sum(1 for r in rows if r["arch"] == arch and r["width"] == width)
            if group and len(group) >= 0.6 * total:
                scored.append((float(np.mean(group)), width))
        if scored:
            chosen[arch] = min(scored)[1]
    return chosen


def stage_tune(n_jobs: int, budget: int) -> dict:
    """Re-sweep each architecture's own knob at its best capacity."""

    capacities = best_capacity_per_arch()
    print(f"TUNE - re-sweeping hyperparameters at each architecture's best capacity")
    print(f"budget {budget} iterations, dev target '{DEV_TARGET}', seeds {TUNE_SEEDS}")
    for arch, width in capacities.items():
        print(f"  {arch:9s} width {width:>4d} ({budget_for(width):>7,} params)")
    print()

    jobs = []
    for arch, width in capacities.items():
        for hparams in ARCH_GRID[arch]:
            for lr in LEARNING_RATES:
                for seed in TUNE_SEEDS:
                    jobs.append((arch, hparams, lr, width, DEV_TARGET, seed, budget))
    rows = run_all(jobs, n_jobs, "tune")

    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["arch"], row["hparams"], row["lr"]), []).append(row)

    print(f"{'arch':9s}{'hparams':20s}{'lr':>8s}{'rmse@chi2=1':>20s}{'fit':>7s}")
    print("-" * 64)
    best: dict[str, dict] = {}
    for (arch, hparams, lr), group in sorted(grouped.items()):
        ok = [r for r in group if r["status"] == "ok"]
        eligible = len(ok) == len(group) and all(r["rmse_at_chi2"] is not None for r in ok)
        mean, sd = mean_std([r["rmse_at_chi2"] for r in ok])
        fitted = sum(1 for r in ok if r["rmse_at_chi2"] is not None)
        value = f"{mean:>13.4f} +-{sd:<5.4f}" if np.isfinite(mean) else f"{'-':>20s}"
        print(f"{arch:9s}{hparams:20s}{lr:>8.4f}{value}{f'{fitted}/{len(group)}':>7s}")
        if eligible and (arch not in best or mean < best[arch]["score"]):
            best[arch] = {"arch": arch, "hparams": json.loads(hparams), "lr": lr,
                          "width": capacities[arch], "budget_params": budget_for(capacities[arch]),
                          "n_iters": budget, "score": mean}

    print("\nSELECTED (re-tuned at best capacity):")
    old = frozen()
    for arch, entry in best.items():
        was = old.get(arch, {})
        print(f"  {arch:9s} {entry['hparams']} lr={entry['lr']:.4f} "
              f"@ {entry['budget_params']:,} params   dev={entry['score']:.4f}")
        print(f"            was: {was.get('hparams')} lr={was.get('lr')} @ 33,537 params "
              f"  dev={was.get('score', float('nan')):.4f}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "capacity_retune.json").write_text(
        json.dumps({"best": best, "capacities": capacities, "budget": budget,
                    "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nSaved {RESULTS / 'capacity_retune.json'}")
    return best


# -------------------------------------------------------------------- eval
def stage_eval(n_jobs: int) -> int:
    """Held-out evaluation of the re-tuned configurations."""

    payload = json.loads((RESULTS / "capacity_retune.json").read_text(encoding="utf-8"))
    best, budget = payload["best"], payload["budget"]
    print(f"EVAL - re-tuned configurations, targets {HELD_OUT_TARGETS}, seeds {EVAL_SEEDS}\n")

    jobs = [(arch, entry["hparams"], entry["lr"], entry["width"], target, seed, budget)
            for target in HELD_OUT_TARGETS for arch, entry in best.items()
            for seed in EVAL_SEEDS]
    rows = run_all(jobs, n_jobs, "eval")

    stage1 = json.loads((RESULTS / "evaluation.json").read_text(encoding="utf-8"))
    sweep = json.loads((RESULTS / "capacity_sweep.json").read_text(encoding="utf-8"))

    for target in HELD_OUT_TARGETS:
        print(f"=== {target} " + "=" * 62)
        print(f"{'arch':9s}{'Stage 1 (33k)':>18s}{'sweep (best cap)':>19s}"
              f"{'re-tuned':>19s}{'gain vs Stage 1':>18s}")
        print("-" * 83)
        for arch in ARCH_GRID:
            if arch not in best:
                continue
            s1, _ = mean_std([r["rmse_cov_at_chi2"] for r in stage1
                              if r["target"] == target and r["arch"] == arch
                              and r.get("status", "ok") == "ok"])
            sw, _ = mean_std([r["rmse_at_chi2"] for r in sweep
                              if r["target"] == target and r["arch"] == arch
                              and r["width"] == best[arch]["width"]])
            new_mean, new_sd = mean_std([r["rmse_at_chi2"] for r in rows
                                         if r["target"] == target and r["arch"] == arch])
            gain = (s1 - new_mean) / s1 * 100 if np.isfinite(s1) and np.isfinite(new_mean) else float("nan")
            print(f"{arch:9s}{s1:>18.4f}{sw:>19.4f}{new_mean:>12.4f} +-{new_sd:<5.4f}"
                  f"{gain:>17.1f}%")
        print()

    print("Reading: 'sweep' isolates capacity alone (33k hyperparameters at the new size);")
    print("         're-tuned' adds hyperparameters matched to that size. The difference")
    print("         between them is what CONFOUND A was hiding.")

    (RESULTS / "capacity_retune_eval.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved {RESULTS / 'capacity_retune_eval.json'}")
    return 0


def main() -> int:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, (os.cpu_count() or 2) - 1)
    print(f"stage={stage}  n_jobs={n_jobs}\n")

    if stage == "probe":
        stage_probe(n_jobs)
    elif stage == "tune":
        budget = json.loads((RESULTS / "capacity_probe.json").read_text(encoding="utf-8"))["recommended_budget"]
        stage_tune(n_jobs, budget)
    elif stage == "eval":
        return stage_eval(n_jobs)
    elif stage == "all":
        payload = stage_probe(n_jobs)
        stage_tune(n_jobs, payload["recommended_budget"])
        return stage_eval(n_jobs)
    else:
        raise SystemExit(f"unknown stage {stage!r}; expected probe|tune|eval|all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
