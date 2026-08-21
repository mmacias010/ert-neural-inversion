"""Stage 1b: the benchmark with per-architecture settings, in the right order.

Stage 1 held network capacity and iteration budget fixed across all methods --
standard practice, and the source of three false conclusions. The capacity
retune fixed part of that but inherited three problems of its own:

  ORDERING       Capacity was selected from a sweep run at 1500 iterations,
                 the very budget later shown to be too small. So the capacity
                 choice was made under the artifact it was meant to correct.

  CALIBRATION    Hyperparameters were re-tuned on `two_layer`, which is smooth.
                 That drove SIREN's w0 to 1.0 and made it WORSE on sharp
                 targets (0.685 -> 0.931). Fixing capacity and budget while
                 leaving a spectrally mismatched calibration target in place
                 can hurt rather than help.

  COVERAGE       tanh was excluded from the retune because it had not qualified
                 in the capacity sweep -- which it failed only because of the
                 1500-iteration budget. The artifact excluded the method the
                 artifact created.

This runs the chain in dependency order, with no stage inheriting a setting
that a later stage is supposed to determine:

  probe     convergence at every capacity, generous ceiling, NO prior budget
            assumption -> sets the budget
  capacity  sweep capacities at that budget, on the calibration target
            -> sets per-architecture capacity
  tune      sweep each architecture's own knob at its capacity, on a
            calibration target with mixed spectral content
  eval      held-out targets, fresh seeds, corrected settings throughout

All five architectures are included, tanh among them.

    python stage1b.py probe    [n_jobs]
    python stage1b.py capacity [n_jobs]
    python stage1b.py tune     [n_jobs]
    python stage1b.py eval     [n_jobs]
    python stage1b.py all      [n_jobs]
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

CALIBRATION = "mixed"          # NOT two_layer: needs the full frequency band
PROBE_CEILING = 12000          # generous; the probe measures, it must not cap
PROBE_SEEDS = (0, 1, 2)
BUDGET_MARGIN = 1.4
OUT = RESULTS / "stage1b"


MIN_STEP = 500


def schedule(n_iters: int) -> int:
    """Fallback when no convergence measurement exists yet (the probe stage)."""

    return max(1, n_iters // 4)


def step_sizes_from_probe() -> dict[str, int]:
    """Slowest observed iterations-to-fit per (architecture, width).

    Deriving the learning-rate schedule from the BUDGET systematically
    penalises slow architectures. With step_size = n_iters // 4 and a budget of
    4500, tanh -- which reaches the noise level around 2500 iterations -- had
    already lost three quarters of its learning rate by the time it converged,
    while Fourier converged at ~60 iterations and never saw a decay at all.
    The budget was itself set from tanh's convergence time, so the slowest
    method got the most aggressive schedule: self-defeating.

    Since the reported model is the chi^2 = 1 snapshot, decay applied BEFORE
    convergence serves no purpose. Schedule from each configuration's measured
    convergence instead, so the learning rate holds until the method has
    reached the noise floor.
    """

    probe = json.loads((OUT / "probe.json").read_text(encoding="utf-8"))
    slowest: dict[str, int] = {}
    for row in probe["rows"]:
        if row.get("iters_to_chi2") is None:
            continue
        key = f"{row['arch']}|{row['width']}"
        slowest[key] = max(slowest.get(key, 0), int(row["iters_to_chi2"]))
    return slowest


def step_size_for(arch: str, width: int, slowest: dict[str, int], n_iters: int) -> int:
    observed = slowest.get(f"{arch}|{width}")
    if observed is None:                       # never fit in the probe
        return schedule(n_iters)
    return int(min(n_iters, max(MIN_STEP, observed)))


def run_one(arch: str, hparams: dict, lr: float, width: int, target: str,
            seed: int, n_iters: int, step_size: int | None = None) -> dict:
    torch.set_num_threads(1)
    step = int(step_size) if step_size else schedule(n_iters)
    base = {"arch": arch, "width": width, "params": budget_for(width), "lr": lr,
            "hparams": json.dumps(hparams, sort_keys=True), "target": target,
            "seed": seed, "n_iters": n_iters, "step_size": step}
    case = build_case(target, seed)
    try:
        seed_networks(seed)
        net = build_network(arch, log_rho_mean=case["log_rho_mean"],
                            **size_kwargs(arch, width), **hparams)
        result = fit_inr(
            case["forward"], net, case["coords"], case["obs_log"], case["data_std"],
            n_iters=n_iters, lr=lr, step_size=step, gamma=GAMMA,
            snapshot_at_chi2=CHI2_TARGET,
        )
        snapshot = result["snapshot_log_resistivity"]
        rmse = None
        if snapshot is not None:
            _, rmse = model_errors(np.exp(snapshot), case["true_rho"], case["mask"])
        return {**base, "status": "ok", "n_params": count_parameters(net),
                "rmse": rmse, "iters_to_chi2": result["snapshot_iteration"],
                "chi2": result["chi2"], "elapsed": result["elapsed"],
                # Fields retained so geometry can be scored without re-running --
                # the omission that forced two extra jobs already.
                "resistivity": np.exp(snapshot).tolist() if snapshot is not None else None}
    except Exception as error:
        return {**base, "status": "diverged", "error": f"{type(error).__name__}: {error}",
                "n_params": None, "rmse": None, "iters_to_chi2": None,
                "chi2": None, "elapsed": None, "resistivity": None}
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
            if index % max(1, len(jobs) // 12) == 0 or index == len(jobs):
                print(f"    {index}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"  done in {time.time() - t0:.0f}s\n")
    return rows


def save(name: str, payload: dict, keep_fields: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not keep_fields:
        payload = {k: ([{kk: vv for kk, vv in r.items() if kk != "resistivity"} for r in v]
                       if isinstance(v, list) else v) for k, v in payload.items()}
    (OUT / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {OUT / name}.json")


def default_hparams(arch: str) -> dict:
    """Mid-grid setting, used only to measure convergence before tuning."""

    grid = ARCH_GRID[arch]
    return grid[len(grid) // 2]


# --------------------------------------------------------------- 1. probe
def stage_probe(n_jobs: int) -> int:
    print(f"PROBE - convergence at widths {WIDTHS}, ceiling {PROBE_CEILING} iterations")
    print("No prior budget assumption: this stage DETERMINES the budget.\n")

    jobs = [(arch, default_hparams(arch), 3.0e-3, width, CALIBRATION, seed, PROBE_CEILING)
            for arch in ARCH_GRID for width in WIDTHS for seed in PROBE_SEEDS]
    rows = run_all(jobs, n_jobs, "probe")

    print(f"{'arch':9s}{'width':>7s}{'params':>10s}{'iters to chi2=1':>18s}{'fit':>7s}")
    print("-" * 51)
    slowest = 0
    for arch in ARCH_GRID:
        for width in WIDTHS:
            group = [r for r in rows if r["arch"] == arch and r["width"] == width]
            hit = [r["iters_to_chi2"] for r in group if r["iters_to_chi2"] is not None]
            label = f"{int(np.mean(hit))} (max {max(hit)})" if hit else "never"
            print(f"{arch:9s}{width:>7d}{budget_for(width):>10,}{label:>18s}"
                  f"{f'{len(hit)}/{len(group)}':>7s}")
            if hit:
                slowest = max(slowest, max(hit))
        print()

    budget = int(np.ceil(slowest * BUDGET_MARGIN / 500) * 500) if slowest else PROBE_CEILING
    print(f"slowest observed fit: {slowest}")
    print(f"BUDGET for later stages: {budget}  ({BUDGET_MARGIN}x margin)")
    save("probe", {"budget": budget, "slowest": slowest, "rows": rows})
    return 0


# ------------------------------------------------------------ 2. capacity
def stage_capacity(n_jobs: int) -> int:
    budget = json.loads((OUT / "probe.json").read_text(encoding="utf-8"))["budget"]
    slowest = step_sizes_from_probe()
    print(f"CAPACITY - sweep at the CORRECTED budget ({budget} iterations)")
    print(f"On the calibration target '{CALIBRATION}' only; held-out targets stay unseen.")
    print("Learning-rate step size is per-configuration, from measured convergence.\n")

    jobs = [(arch, default_hparams(arch), 3.0e-3, width, CALIBRATION, seed, budget,
             step_size_for(arch, width, slowest, budget))
            for arch in ARCH_GRID for width in WIDTHS for seed in TUNE_SEEDS]
    rows = run_all(jobs, n_jobs, "capacity")

    print(f"{'arch':9s}" + "".join(f"{budget_for(w)//1000:>9d}k" for w in WIDTHS))
    print("-" * (9 + 10 * len(WIDTHS)))
    chosen: dict[str, int] = {}
    for arch in ARCH_GRID:
        line, scored = f"{arch:9s}", []
        for width in WIDTHS:
            group = [r for r in rows if r["arch"] == arch and r["width"] == width]
            ok = [r["rmse"] for r in group if r["rmse"] is not None]
            # Require a majority to fit, so one lucky seed cannot nominate a
            # capacity the architecture cannot actually use.
            if ok and len(ok) >= 0.6 * len(group):
                line += f"{np.mean(ok):>10.3f}"
                scored.append((float(np.mean(ok)), width))
            else:
                line += f"{'x':>10s}"
        print(line)
        if scored:
            chosen[arch] = min(scored)[1]

    print("\nSELECTED capacity:")
    for arch in ARCH_GRID:
        if arch in chosen:
            print(f"  {arch:9s} width {chosen[arch]:>4d}  ({budget_for(chosen[arch]):>8,} params)")
        else:
            print(f"  {arch:9s} NO CAPACITY QUALIFIED")
    save("capacity", {"budget": budget, "chosen": chosen, "rows": rows})
    return 0


# ---------------------------------------------------------------- 3. tune
def stage_tune(n_jobs: int) -> int:
    cap = json.loads((OUT / "capacity.json").read_text(encoding="utf-8"))
    budget, chosen = cap["budget"], cap["chosen"]
    slowest = step_sizes_from_probe()
    print(f"TUNE - each architecture's own knob at its own capacity, budget {budget}")
    print(f"Calibration target '{CALIBRATION}' spans smooth, dipping and sharp structure,")
    print("so selection cannot collapse toward one end of the frequency band.")
    for arch in ARCH_GRID:
        if arch in chosen:
            print(f"  {arch:9s} width {chosen[arch]:>4d}  step_size "
                  f"{step_size_for(arch, chosen[arch], slowest, budget):>5d}")
    print()

    jobs = [(arch, hparams, lr, chosen[arch], CALIBRATION, seed, budget,
             step_size_for(arch, chosen[arch], slowest, budget))
            for arch in ARCH_GRID if arch in chosen
            for hparams in ARCH_GRID[arch] for lr in LEARNING_RATES for seed in TUNE_SEEDS]
    rows = run_all(jobs, n_jobs, "tune")

    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["arch"], row["hparams"], row["lr"]), []).append(row)

    print(f"{'arch':9s}{'hparams':20s}{'lr':>8s}{'rmse@chi2=1':>20s}{'fit':>7s}")
    print("-" * 64)
    best: dict[str, dict] = {}
    for (arch, hparams, lr), group in sorted(grouped.items()):
        ok = [r for r in group if r["rmse"] is not None]
        eligible = len(ok) == len(group)
        m, sd = mean_std([r["rmse"] for r in ok])
        value = f"{m:>13.4f} +-{sd:<5.4f}" if np.isfinite(m) else f"{'-':>20s}"
        print(f"{arch:9s}{hparams:20s}{lr:>8.4f}{value}{f'{len(ok)}/{len(group)}':>7s}")
        if eligible and (arch not in best or m < best[arch]["score"]):
            best[arch] = {"arch": arch, "hparams": json.loads(hparams), "lr": lr,
                          "width": chosen[arch], "n_iters": budget, "score": m}

    print("\nSELECTED configuration:")
    for arch in ARCH_GRID:
        if arch in best:
            e = best[arch]
            print(f"  {arch:9s} {str(e['hparams']):20s} lr={e['lr']:<7.4f} "
                  f"{budget_for(e['width']):>8,} params   calib={e['score']:.4f}")
        else:
            print(f"  {arch:9s} NO CONFIGURATION FIT ON ALL SEEDS")
    save("tune", {"budget": budget, "best": best, "rows": rows})
    return 0


# ---------------------------------------------------------------- 4. eval
def stage_eval(n_jobs: int) -> int:
    best = json.loads((OUT / "tune.json").read_text(encoding="utf-8"))["best"]
    slowest = step_sizes_from_probe()
    print(f"EVAL - held-out targets {HELD_OUT_TARGETS}, seeds {EVAL_SEEDS}\n")

    jobs = [(e["arch"], e["hparams"], e["lr"], e["width"], target, seed, e["n_iters"],
             step_size_for(e["arch"], e["width"], slowest, e["n_iters"]))
            for target in HELD_OUT_TARGETS for e in best.values() for seed in EVAL_SEEDS]
    rows = run_all(jobs, n_jobs, "eval")

    stage1 = json.loads((RESULTS / "evaluation.json").read_text(encoding="utf-8"))
    for target in HELD_OUT_TARGETS:
        print(f"=== {target} " + "=" * 58)
        print(f"{'arch':9s}{'Stage 1 (shared)':>20s}{'Stage 1b':>22s}{'change':>12s}{'fit':>7s}")
        print("-" * 70)
        for arch in ARCH_GRID:
            group = [r for r in rows if r["target"] == target and r["arch"] == arch]
            if not group:
                continue
            s1, _ = mean_std([r["rmse_cov_at_chi2"] for r in stage1
                              if r["target"] == target and r["arch"] == arch
                              and r.get("status", "ok") == "ok"])
            m, sd = mean_std([r["rmse"] for r in group if r["rmse"] is not None])
            n_fit = sum(1 for r in group if r["rmse"] is not None)
            change = f"{(s1 - m) / s1 * 100:+.1f}%" if np.isfinite(s1) and np.isfinite(m) else "-"
            s1_label = f"{s1:>20.4f}" if np.isfinite(s1) else f"{'never fit':>20s}"
            new_label = f"{m:>15.4f} +-{sd:<5.4f}" if np.isfinite(m) else f"{'never fit':>22s}"
            print(f"{arch:9s}{s1_label}{new_label}{change:>12s}{f'{n_fit}/{len(group)}':>7s}")
        print()

    save("eval", {"best": best, "rows": rows}, keep_fields=True)
    print("\nFields retained in eval.json, so geometry metrics need no further runs.")
    return 0


# -------------------------------------------------------- 5. conventional
def _call_traditional(args: tuple) -> dict:
    """Module-level so ProcessPoolExecutor can pickle it (a local def cannot)."""

    from traditional_comparison import run_traditional
    return run_traditional(*args)


def stage_conventional(n_jobs: int) -> int:
    """Re-calibrate the conventional arm on the SAME target as the networks.

    Otherwise the comparison is unfair in the opposite direction: networks
    tuned on 'mixed' against conventional methods tuned on 'two_layer'. TV and
    smooth-L2 have no capacity or budget to set -- lambda and the operator are
    their only knobs -- so this stage is just a lambda sweep, and Gauss-Newton
    runs in about a second.
    """

    from traditional_comparison import OPERATORS, REGULARIZATION_WEIGHTS

    print(f"CONVENTIONAL - lambda sweep on the same calibration target '{CALIBRATION}'\n")
    grid = [(w, op) for op in OPERATORS for w in REGULARIZATION_WEIGHTS]

    jobs = [(CALIBRATION, seed, w, op) for w, op in grid for seed in TUNE_SEEDS]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        rows = list(pool.map(_call_traditional, jobs))
    print(f"  {len(jobs)} calibration runs in {time.time() - t0:.0f}s")

    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["operator"], row["regularization"]), []).append(row)
    print(f"\n{'operator':26s}{'lambda':>9s}{'rmse':>18s}{'fit':>7s}")
    print("-" * 60)
    best: dict[str, dict] = {}
    for (operator, weight), group in sorted(grouped.items()):
        ok = [r["rmse_cov"] for r in group if r["chi2"] <= CHI2_TARGET]
        m, sd = mean_std(ok)
        value = f"{m:>11.4f} +-{sd:<5.4f}" if np.isfinite(m) else f"{'-':>18s}"
        print(f"{operator:26s}{weight:>9.4g}{value}{f'{len(ok)}/{len(group)}':>7s}")
        family = "smooth_l2" if "smooth" in operator or operator == "damping" else "tv"
        if len(ok) == len(group) and np.isfinite(m) and (
                family not in best or m < best[family]["score"]):
            best[family] = {"operator": operator, "regularization": weight, "score": m}

    print("\nSELECTED:")
    for family, entry in best.items():
        print(f"  {family:10s} {entry['operator']:26s} lambda={entry['regularization']:g}"
              f"   calib={entry['score']:.4f}")

    eval_jobs = [(t, s, e["regularization"], e["operator"])
                 for t in HELD_OUT_TARGETS for e in best.values() for s in EVAL_SEEDS]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        eval_rows = list(pool.map(_call_traditional, eval_jobs))

    print(f"\n{'family':12s}" + "".join(f"{t:>16s}" for t in HELD_OUT_TARGETS))
    print("-" * (12 + 16 * len(HELD_OUT_TARGETS)))
    for family, entry in best.items():
        line = f"{family:12s}"
        for target in HELD_OUT_TARGETS:
            ok = [r["rmse_cov"] for r in eval_rows
                  if r["target"] == target and r["operator"] == entry["operator"]
                  and r["regularization"] == entry["regularization"] and r["chi2"] <= CHI2_TARGET]
            m, sd = mean_std(ok)
            line += f"{m:>10.4f} +-{sd:<5.4f}" if np.isfinite(m) else f"{'never fit':>16s}"
        print(line)

    strip = lambda rows: [{k: v for k, v in r.items() if k != "resistivity"} for r in rows]
    save("conventional", {"best": best, "calibration": strip(rows), "eval": strip(eval_rows)})
    return 0


def main() -> int:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, (os.cpu_count() or 2) - 1)
    print(f"stage={stage}  n_jobs={n_jobs}  calibration='{CALIBRATION}'\n")
    stages = {"probe": stage_probe, "capacity": stage_capacity,
              "tune": stage_tune, "eval": stage_eval, "conventional": stage_conventional}
    # "rest" reuses an existing probe: the probe is a measurement of convergence
    # and does not depend on the schedule rule, so it never needs repeating.
    if stage == "rest":
        for name in ("capacity", "tune", "eval", "conventional"):
            print("\n" + "#" * 78 + f"\n# {name.upper()}\n" + "#" * 78)
            stages[name](n_jobs)
        return 0
    if stage == "all":
        for name in ("probe", "capacity", "tune", "eval", "conventional"):
            print("\n" + "#" * 78 + f"\n# {name.upper()}\n" + "#" * 78)
            stages[name](n_jobs)
        return 0
    if stage not in stages:
        raise SystemExit(f"unknown stage {stage!r}; expected {sorted(stages)}|all")
    return stages[stage](n_jobs)


if __name__ == "__main__":
    raise SystemExit(main())
