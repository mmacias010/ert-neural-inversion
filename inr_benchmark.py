"""Benchmark ReLU MLP vs SIREN vs Fourier-feature MLP as ERT inversion priors.

Protocol
--------
Hyperparameters are tuned on ONE development target (two_layer) using its own
seeds, then frozen. Reported results come from held-out targets (blocks,
parflow) with fresh seeds, so no reported number comes from a configuration
chosen while looking at that problem. Using the true model to select on the
development target is legitimate -- it is a validation set -- but it never
touches the held-out evaluation.

Fairness controls
-----------------
* Capacity matched: 33,537 / 33,537 / 33,601 trainable parameters. The Fourier
  trunk is narrowed automatically because the encoded input is 256-wide.
* Identical optimizer, schedule, iteration budget, mesh, survey, noise and
  starting field for every architecture.
* Seed streams split: network init (Torch) is independent of noise realization
  (NumPy RandomState), so a win cannot be a lucky noise draw.
* Results reported at BOTH a fixed iteration budget and matched chi^2, because
  a fixed budget silently rewards whichever architecture converges fastest
  rather than whichever represents the field best.
* Model error reported over the whole mesh AND restricted to cells the survey
  actually senses; the mesh reaches 100 m but a 16-electrode Wenner array sees
  roughly the top 20 m, and unresolvable cells otherwise swamp the metric.

Caveat carried from cases.py: synthetic data are generated with the same mesh
and solver used to invert them (an inverse crime). All architectures inherit
that bias identically, so it is fair for ranking them, but these numbers are
not evidence about absolute inversion accuracy.

Usage:
    .venv\\Scripts\\python.exe inr_benchmark.py smoke
    .venv\\Scripts\\python.exe inr_benchmark.py tune
    .venv\\Scripts\\python.exe inr_benchmark.py eval
    .venv\\Scripts\\python.exe inr_benchmark.py all
"""

from __future__ import annotations

import os

# Pin BLAS/OpenMP before Torch or NumPy import: workers run one thread each so
# parallel processes do not oversubscribe the 14 cores and end up slower than
# a serial run.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from deepert.inr import (
    block_anomaly,
    build_mesh,
    build_network,
    build_survey,
    count_parameters,
    coverage_mask,
    fit_inr,
    normalized_coords,
    parflow_slice,
    seed_networks,
    synthetic_observations,
    two_layer,
)
from deepert.inversion.core import ParameterizedERTForward2p5D

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "inr_results"

# ---- fixed experimental settings, identical for every architecture ----------
# Budget set from the convergence probe (inr_results/convergence.json): the
# slowest architecture reached chi^2 = 1 at ~1187 iterations, so 1500 ensures
# no architecture is reported as failing merely because it ran out of budget.
# The step size keeps the same four learning-rate halvings as the original
# 600/150 schedule. The PRIMARY metric is the model at the chi^2 = 1 snapshot.
# Final-budget numbers are secondary by design, because a fixed budget runs each
# architecture a different distance past its own stopping point -- Fourier
# reaches chi^2 = 1 at ~116 iterations and ReLU at ~1048, so 1500 iterations is
# 12.9x past for one and 1.4x for the other. Degradation scales with that
# overshoot (+1.7% below 3x, +5.7% at 3-10x, +10.3% beyond 10x; overfit_check.py),
# so end-of-budget scoring would penalize whichever architecture converges
# fastest rather than whichever represents the field worst.
N_ITERS = 1500
STEP_SIZE = 375
GAMMA = 0.5
DATA_STD = 0.015
RELATIVE_NOISE = 0.015
CHI2_TARGET = 1.0            # noise level; also the matched-fit snapshot trigger
COVERAGE_QUANTILE = 0.5      # keep the better-sensed half of the mesh

DEV_TARGET = "two_layer"
HELD_OUT_TARGETS = ("blocks", "parflow")
TUNE_SEEDS = (0, 1, 2)
EVAL_SEEDS = (10, 11, 12, 13, 14)
PARFLOW_SLICE = ROOT / "resistivity_models_2d" / "resistivity2d_y2_t04368.npy"

# ---- tuning grid: each architecture over its own natural knobs --------------
# ReLU/tanh/DIP have no frequency knob, so learning rate is their only
# tunable. That asymmetry is a property of the methods, not an unfairness to
# correct. Grouping follows the mentor's taxonomy: all five are Group 2
# (single-dataset neural parameterizations, no pretraining database).
ARCH_GRID: dict[str, list[dict]] = {
    "relu": [{}],
    "tanh": [{}],
    "siren": [{"w0": w} for w in (1.0, 3.0, 5.0, 10.0, 30.0)],
    "fourier": [{"sigma": s} for s in (0.5, 1.0, 2.0, 5.0, 10.0)],
    "dip": [{}],
}
LEARNING_RATES = (1.0e-3, 3.0e-3, 1.0e-2)


def target_field(name: str, centers: np.ndarray) -> np.ndarray:
    if name == "two_layer":
        return two_layer(centers)
    if name == "mixed":
        # Stage-1b calibration target: smooth + dipping + sharp, so tuning
        # cannot collapse toward one end of the frequency band.
        from deepert.inr.targets import mixed_calibration
        return mixed_calibration(centers)
    if name == "blocks":
        return block_anomaly(centers)
    if name == "parflow":
        return parflow_slice(PARFLOW_SLICE, centers, depth_extent=25.0)
    raise ValueError(f"unknown target {name!r}")


def build_case(target: str, noise_seed: int, *, relative_noise: float = RELATIVE_NOISE) -> dict:
    """Mesh, survey, forward operator, true field and noisy data for one run.

    ``relative_noise`` defaults to the benchmark's 1.5% so every existing
    result reproduces unchanged; the robustness study overrides it. The
    returned ``data_std`` tracks the noise level, so chi^2 = 1 always means
    "fit to the noise floor" regardless of how noisy the data are.
    """

    mesh = build_mesh()
    survey = build_survey()
    forward = ParameterizedERTForward2p5D.from_mesh_survey(
        mesh, survey, np.arange(int(mesh.cell_count), dtype=np.int32),
        background_mode="pygimli_prolongation",
    )
    centers, coords = normalized_coords(mesh)
    true_rho = target_field(target, centers)
    obs_rhoa, obs_log = synthetic_observations(
        forward, true_rho, relative_noise=relative_noise, seed=noise_seed
    )
    # Sensed-region mask from the Jacobian at the TRUE model: computed once,
    # identical for every architecture, so it cannot favour any of them.
    _, jacobian = forward.forward_and_jacobian(np.log(true_rho), log_transform=True)
    mask = coverage_mask(jacobian, quantile=COVERAGE_QUANTILE)
    return {
        "forward": forward,
        "coords": coords,
        "centers": centers,
        "true_rho": true_rho,
        "obs_log": obs_log,
        "mask": mask,
        # Warm start from the median measurement -- observed data only, no truth.
        "log_rho_mean": float(np.log(np.median(obs_rhoa))),
        "data_std": float(relative_noise),
    }


def model_errors(rho: np.ndarray, true_rho: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    error = np.log(rho) - np.log(true_rho)
    return float(np.sqrt(np.mean(error**2))), float(np.sqrt(np.mean(error[mask] ** 2)))


def run_one(arch: str, hparams: dict, lr: float, target: str, init_seed: int, noise_seed: int) -> dict:
    """One complete inversion. Self-contained so it can run in a worker process.

    Divergence to non-finite resistivity (seen for ReLU on parflow in the
    convergence probe) is recorded as a failed row rather than raised, so one
    unstable configuration cannot kill an entire sweep.
    """

    torch.set_num_threads(1)
    case = build_case(target, noise_seed)
    try:
        seed_networks(init_seed)                     # nothing else may touch the Torch RNG
        net = build_network(arch, log_rho_mean=case["log_rho_mean"], **hparams)
        result = fit_inr(
            case["forward"], net, case["coords"], case["obs_log"], DATA_STD,
            n_iters=N_ITERS, lr=lr, step_size=STEP_SIZE, gamma=GAMMA,
            snapshot_at_chi2=CHI2_TARGET,
        )
        rmse_full, rmse_cov = model_errors(result["resistivity"], case["true_rho"], case["mask"])

        snapshot = result["snapshot_log_resistivity"]
        if snapshot is None:
            snap_full = snap_cov = None
            snapshot_rho = None
        else:
            snap_full, snap_cov = model_errors(np.exp(snapshot), case["true_rho"], case["mask"])
            snapshot_rho = np.exp(snapshot).tolist()

        history = result["chi2_history"]
        reached = np.flatnonzero(history <= CHI2_TARGET)
        return {
            "status": "ok",
            "arch": arch,
            "hparams": json.dumps(hparams, sort_keys=True),
            "lr": lr,
            "target": target,
            "init_seed": init_seed,
            "noise_seed": noise_seed,
            "n_params": count_parameters(net),
            "chi2": result["chi2"],
            "rms": result["rms"],
            "rmse_full": rmse_full,
            "rmse_cov": rmse_cov,
            "rmse_full_at_chi2": snap_full,
            "rmse_cov_at_chi2": snap_cov,
            "iters_to_chi2": int(reached[0]) if reached.size else None,
            "elapsed": result["elapsed"],
            "resistivity": result["resistivity"].tolist(),
            "resistivity_at_chi2": snapshot_rho,
        }
    except Exception as error:                       # divergence: record, don't raise
        return {
            "status": "diverged", "error": f"{type(error).__name__}: {error}",
            "arch": arch, "hparams": json.dumps(hparams, sort_keys=True), "lr": lr,
            "target": target, "init_seed": init_seed, "noise_seed": noise_seed,
            "n_params": None, "chi2": None, "rms": None,
            "rmse_full": None, "rmse_cov": None,
            "rmse_full_at_chi2": None, "rmse_cov_at_chi2": None,
            "iters_to_chi2": None, "elapsed": None,
            "resistivity": None, "resistivity_at_chi2": None,
        }
    finally:
        case["forward"].close()


def run_many(jobs: list[tuple], n_jobs: int) -> list[dict]:
    """Run jobs across worker processes.

    Uses stdlib ``ProcessPoolExecutor`` rather than joblib so the harness has
    no dependency beyond what deepert already needs -- worth avoiding on a
    CentOS 7 cluster where every install is a wheel-compatibility gamble.
    Each worker re-imports this module, so the single-thread environment
    variables at the top are re-applied before Torch loads there.
    """

    from concurrent.futures import ProcessPoolExecutor

    t0 = time.time()
    rows: list[dict] = []
    if n_jobs <= 1:
        for index, job in enumerate(jobs, 1):
            rows.append(run_one(*job))
            print(f"    {index}/{len(jobs)} done", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = [pool.submit(run_one, *job) for job in jobs]
            for index, future in enumerate(futures, 1):
                rows.append(future.result())
                if index % max(1, len(jobs) // 20) == 0 or index == len(jobs):
                    print(f"    {index}/{len(jobs)} done ({time.time() - t0:.0f}s)", flush=True)
    print(f"  {len(jobs)} runs in {time.time() - t0:.0f}s wall ({n_jobs} workers)")
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray([v for v in values if v is not None], dtype=float)
    if array.size == 0:
        return float("nan"), float("nan")
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


# ---------------------------------------------------------------- smoke test
def smoke(n_jobs: int) -> int:
    """Confirm all three architectures run and produce physically sane fields.

    This is a debugging step, NOT a result: the architectures are untuned here,
    and an untuned Fourier sigma in particular can land anywhere on its curve.
    """

    print("SMOKE TEST - untuned defaults, one seed. Not a result.\n")
    jobs = [(arch, {}, 3.0e-3, DEV_TARGET, 0, 0) for arch in ARCH_GRID]
    rows = run_many(jobs, n_jobs)
    print(f"\n{'arch':10s}{'params':>8s}{'chi2':>10s}{'rms%':>8s}{'rmse_cov':>10s}"
          f"{'rho_min':>10s}{'rho_max':>10s}{'sec':>7s}")
    ok = True
    for row in rows:
        rho = np.asarray(row["resistivity"])
        print(f"{row['arch']:10s}{row['n_params']:>8d}{row['chi2']:>10.3f}{row['rms']:>8.2f}"
              f"{row['rmse_cov']:>10.3f}{rho.min():>10.1f}{rho.max():>10.1f}{row['elapsed']:>7.1f}")
        if not np.all(np.isfinite(rho)) or rho.min() <= 0.0:
            print(f"  !! {row['arch']} produced non-physical resistivity")
            ok = False
    print("\nsmoke test " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


# -------------------------------------------------------------------- tuning
def tune(n_jobs: int, only: set[str] | None = None) -> dict:
    """Sweep each architecture on the development target only, then freeze.

    ``only`` restricts the sweep to the named architectures; their results are
    merged into the existing tuning.json so previously tuned architectures
    keep their frozen configurations without being re-run.
    """

    grid_items = {a: g for a, g in ARCH_GRID.items() if only is None or a in only}
    print(f"TUNING on '{DEV_TARGET}' - seeds {TUNE_SEEDS} - archs {sorted(grid_items)}\n")
    jobs = []
    for arch, grid in grid_items.items():
        for hparams in grid:
            for lr in LEARNING_RATES:
                for seed in TUNE_SEEDS:
                    jobs.append((arch, hparams, lr, DEV_TARGET, seed, seed))
    print(f"  {len(jobs)} runs ({len(jobs) // len(TUNE_SEEDS)} configurations x {len(TUNE_SEEDS)} seeds)")
    rows = run_many(jobs, n_jobs)

    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["arch"], row["hparams"], row["lr"]), []).append(row)

    # Selection rule: a configuration is ELIGIBLE only if every seed reached
    # chi^2 <= 1, and it is scored on the model at that snapshot. Scoring the
    # final-budget model instead would penalize fast-converging architectures,
    # which a fixed budget carries much further past the noise floor than slow
    # ones (overfit_check.py: degradation +1.7% below 3x overshoot, +10.3%
    # beyond 10x). Admitting non-fitting configs is separately how the original
    # oracle selected diverged runs.
    print(f"\n{'arch':9s}{'hparams':22s}{'lr':>8s}{'rmse@chi2=1':>18s}{'chi2':>10s}{'to_chi2':>9s}{'note':>12s}")
    print("-" * 88)
    best: dict[str, dict] = {}
    fallback: dict[str, dict] = {}
    for (arch, hparams, lr), group in sorted(grouped.items()):
        ok = [r for r in group if r.get("status", "ok") == "ok"]
        n_diverged = len(group) - len(ok)
        eligible = bool(ok) and n_diverged == 0 and all(r["rmse_cov_at_chi2"] is not None for r in ok)
        mean_snap, std_snap = mean_std([r["rmse_cov_at_chi2"] for r in ok])
        mean_chi2, _ = mean_std([r["chi2"] for r in ok])
        hit = [r["iters_to_chi2"] for r in ok if r["iters_to_chi2"] is not None]
        hit_label = f"{int(np.mean(hit))}" if hit else "-"
        note = "diverged" if n_diverged else ("" if eligible else "no fit")
        snap_label = f"{mean_snap:>11.4f} +-{std_snap:<5.4f}" if np.isfinite(mean_snap) else f"{'-':>18s}"
        print(f"{arch:9s}{hparams:22s}{lr:>8.4f}{snap_label}{mean_chi2:>10.3f}{hit_label:>9s}{note:>12s}")
        if eligible and (arch not in best or mean_snap < best[arch]["score"]):
            best[arch] = {"arch": arch, "hparams": json.loads(hparams), "lr": lr,
                          "score": mean_snap, "score_std": std_snap}
        # Fallback ordering for architectures with no eligible config: the one
        # that came closest to fitting the data, judged by chi^2 alone.
        if ok and np.isfinite(mean_chi2) and (arch not in fallback or mean_chi2 < fallback[arch]["chi2"]):
            fallback[arch] = {"arch": arch, "hparams": json.loads(hparams), "lr": lr,
                              "score": float("nan"), "score_std": float("nan"), "chi2": mean_chi2}

    print("\nSELECTED (lowest mean rmse at the chi^2=1 snapshot; all seeds must fit):")
    for arch in grid_items:
        if arch in best:
            entry = best[arch]
            print(f"  {arch:9s} {entry['hparams']}  lr={entry['lr']:.4f}   dev rmse@chi2=1={entry['score']:.4f}")
        elif arch in fallback:
            entry = fallback[arch]
            best[arch] = entry
            print(f"  {arch:9s} {entry['hparams']}  lr={entry['lr']:.4f}   "
                  f"** NO CONFIG FIT THE DATA on all seeds; fallback = closest chi^2 ({entry['chi2']:.2f}) **")

    RESULTS.mkdir(exist_ok=True)
    # Merge with any existing tuning results so a filtered run (only=...) keeps
    # the frozen configurations of architectures it did not touch.
    merged_best, merged_rows = {}, []
    tuning_path = RESULTS / "tuning.json"
    if tuning_path.exists():
        previous = json.loads(tuning_path.read_text(encoding="utf-8"))
        merged_best = {a: e for a, e in previous.get("best", {}).items() if a not in grid_items}
        merged_rows = [r for r in previous.get("rows", []) if r.get("arch") not in grid_items]
    merged_best.update(best)
    merged_rows.extend(
        {k: v for k, v in r.items() if k not in ("resistivity", "resistivity_at_chi2")} for r in rows
    )
    tuning_path.write_text(json.dumps({"best": merged_best, "rows": merged_rows}, indent=2),
                           encoding="utf-8")
    return merged_best


# ---------------------------------------------------------------- evaluation
def evaluate(best: dict, n_jobs: int, only: set[str] | None = None) -> int:
    """Run frozen configurations on held-out targets with fresh seeds.

    ``only`` restricts new runs to the named architectures; rows for the rest
    are loaded from the existing evaluation.json so the summary tables always
    show every architecture.
    """

    selected = {a: e for a, e in best.items() if only is None or a in only}
    print(f"\nEVALUATION - frozen hyperparameters, targets {HELD_OUT_TARGETS}, "
          f"seeds {EVAL_SEEDS} - archs {sorted(selected)}\n")
    jobs = []
    for target in (*HELD_OUT_TARGETS, DEV_TARGET):
        for arch, entry in selected.items():
            for seed in EVAL_SEEDS:
                jobs.append((arch, entry["hparams"], entry["lr"], target, seed, seed))
    rows = run_many(jobs, n_jobs)

    evaluation_path = RESULTS / "evaluation.json"
    if only is not None and evaluation_path.exists():
        rows = [r for r in json.loads(evaluation_path.read_text(encoding="utf-8"))
                if r.get("arch") not in selected] + rows

    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["target"], row["arch"]), []).append(row)

    for target in (*HELD_OUT_TARGETS, DEV_TARGET):
        label = "HELD OUT" if target in HELD_OUT_TARGETS else "development (hyperparameters chosen here)"
        print(f"\n=== {target}  [{label}] " + "=" * max(0, 40 - len(target)))
        print(f"{'arch':9s}{'rmse_cov (budget)':>22s}{'rmse_cov (chi2=1)':>22s}{'chi2':>9s}{'to_chi2':>9s}{'sec':>7s}")
        print("-" * 78)
        for arch in ARCH_GRID:
            group = grouped.get((target, arch))
            if not group:
                continue
            m_cov, s_cov = mean_std([r["rmse_cov"] for r in group])
            m_snap, s_snap = mean_std([r["rmse_cov_at_chi2"] for r in group])
            m_chi2, _ = mean_std([r["chi2"] for r in group])
            hit = [r["iters_to_chi2"] for r in group if r["iters_to_chi2"] is not None]
            hit_label = f"{int(np.mean(hit))}/{len(group)}" if hit else f"-/{len(group)}"
            snap_label = f"{m_snap:>11.4f} +-{s_snap:<5.4f}" if np.isfinite(m_snap) else f"{'never reached':>22s}"
            m_time, _ = mean_std([r["elapsed"] for r in group])
            print(f"{arch:9s}{m_cov:>15.4f} +-{s_cov:<5.4f}{snap_label}{m_chi2:>9.3f}{hit_label:>9s}{m_time:>7.1f}")

    RESULTS.mkdir(exist_ok=True)
    evaluation_path.write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "resistivity"} for r in rows], indent=2), encoding="utf-8")
    plot_models(rows, best)
    return 0


def plot_models(rows: list[dict], best: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LogNorm

    mesh = build_mesh()
    nodes = np.asarray(mesh.nodes, dtype=float)
    cells = np.asarray(mesh.cells, dtype=np.int32)
    polygons = nodes[cells]
    centers, _ = normalized_coords(mesh)
    targets = (*HELD_OUT_TARGETS, DEV_TARGET)
    seed = EVAL_SEEDS[0]

    n_cols = 1 + len(best)
    fig, axes = plt.subplots(len(targets), n_cols, figsize=(4.8 * n_cols, 3.6 * len(targets)), squeeze=False)
    for row_index, target in enumerate(targets):
        true_rho = target_field(target, centers)
        panels = [(true_rho, f"{target}: true")]
        for arch in (a for a in ARCH_GRID if a in best):
            match = [r for r in rows if r["target"] == target and r["arch"] == arch
                     and r["init_seed"] == seed and r.get("status", "ok") == "ok"]
            if match:
                # Plot the chi^2=1 snapshot (the primary metric's model); the
                # end-of-budget model has overfit past the noise floor.
                row = match[0]
                if row.get("resistivity_at_chi2") is not None:
                    values, tag = np.asarray(row["resistivity_at_chi2"]), "@chi2=1"
                else:
                    values, tag = np.asarray(row["resistivity"]), "no fit"
                panels.append((values, f"{arch} ({best[arch]['hparams']}, {tag})"))
        allv = np.concatenate([p[0] for p in panels])
        norm = LogNorm(vmin=max(allv.min(), 1e-3), vmax=allv.max())
        for col, (values, title) in enumerate(panels):
            ax = axes[row_index][col]
            coll = PolyCollection(polygons, array=values, cmap="turbo", norm=norm, edgecolors="none")
            ax.add_collection(coll)
            ax.set_title(title, fontsize=9)
            ax.set_ylim(-30.0, 2.0)          # sensed zone; the mesh runs to -100 m
            ax.autoscale_view()
            ax.set_aspect("equal")
        fig.colorbar(coll, ax=axes[row_index], fraction=0.012, pad=0.01).set_label("ohm-m")

    path = RESULTS / "inr_comparison.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure to {path}")


def oracle(n_jobs: int) -> int:
    """Per-target hyperparameter selection: an upper bound, not a usable method.

    The transfer experiment froze hyperparameters chosen on a smooth target and
    applied them to sharp ones. That confounds two failures: an architecture
    that cannot represent the field, and an architecture handed the wrong
    frequency setting for it. This stage removes the second by tuning each
    architecture on each held-out target directly.

    Selecting hyperparameters against the true model is cheating -- you could
    not do it on field data -- so these numbers are an ORACLE UPPER BOUND. Their
    only job is diagnostic: the gap between oracle and transfer is what
    hyperparameter mismatch cost, and if SIREN reaches chi^2=1 on 'blocks' under
    its own best w0, then the transfer result was about tuning, not capability.
    """

    print(f"ORACLE - per-target tuning on {HELD_OUT_TARGETS}, seeds {TUNE_SEEDS}")
    print("Upper bound only: hyperparameters selected against the true model.\n")

    jobs = []
    for target in HELD_OUT_TARGETS:
        for arch, grid in ARCH_GRID.items():
            for hparams in grid:
                for lr in LEARNING_RATES:
                    for seed in TUNE_SEEDS:
                        jobs.append((arch, hparams, lr, target, seed, seed))
    print(f"  search: {len(jobs)} runs")
    rows = run_many(jobs, n_jobs)

    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["target"], row["arch"], row["hparams"], row["lr"]), []).append(row)

    # Same eligibility rule as tune(): every seed must reach chi^2 <= 1, and
    # configurations are scored on the snapshot model. Without this the oracle
    # selects diverged runs whose fields coincidentally resemble the truth.
    best: dict[tuple, dict] = {}
    for (target, arch, hparams, lr), group in grouped.items():
        ok = [r for r in group if r.get("status", "ok") == "ok"]
        if len(ok) != len(group) or not all(r["rmse_cov_at_chi2"] is not None for r in ok):
            continue
        score, _ = mean_std([r["rmse_cov_at_chi2"] for r in ok])
        if not np.isfinite(score):
            continue
        key = (target, arch)
        if key not in best or score < best[key]["score"]:
            best[key] = {"hparams": json.loads(hparams), "lr": lr, "score": score}

    print("\nORACLE selections (eligible = all seeds fit the data):")
    for target in HELD_OUT_TARGETS:
        for arch in ARCH_GRID:
            entry = best.get((target, arch))
            if entry is None:
                print(f"  {target:10s} {arch:9s} NO CONFIGURATION FIT THE DATA")
            else:
                print(f"  {target:10s} {arch:9s} {str(entry['hparams']):20s} lr={entry['lr']:.4f}")

    # Re-run the oracle winners on the EVALUATION seeds so they are directly
    # comparable to the transfer numbers rather than to their own search seeds.
    print(f"\n  confirm: re-running winners on eval seeds {EVAL_SEEDS}")
    confirm_jobs = [
        (arch, entry["hparams"], entry["lr"], target, seed, seed)
        for (target, arch), entry in best.items()
        for seed in EVAL_SEEDS
    ]
    confirm = run_many(confirm_jobs, n_jobs)

    transfer_rows = json.loads((RESULTS / "evaluation.json").read_text(encoding="utf-8"))
    transfer: dict[tuple, list[dict]] = {}
    for row in transfer_rows:
        transfer.setdefault((row["target"], row["arch"]), []).append(row)
    oracle_group: dict[tuple, list[dict]] = {}
    for row in confirm:
        oracle_group.setdefault((row["target"], row["arch"]), []).append(row)

    for target in HELD_OUT_TARGETS:
        print(f"\n=== {target} " + "=" * 62)
        print(f"{'arch':9s}{'transfer rmse':>18s}{'oracle rmse':>18s}{'cost of transfer':>18s}"
              f"{'oracle cfg':>22s}{'chi2':>8s}{'to_chi2':>9s}")
        print("-" * 94)
        for arch in ARCH_GRID:
            t_mean, t_std = mean_std([r["rmse_cov_at_chi2"] for r in transfer.get((target, arch), [])])
            entry = best.get((target, arch))
            if entry is None:
                print(f"{arch:9s}{t_mean:>11.4f} +-{t_std:<5.4f}{'NO CONFIGURATION FIT THE DATA':>60s}")
                continue
            o_group = [r for r in oracle_group.get((target, arch), []) if r.get("status", "ok") == "ok"]
            o_mean, o_std = mean_std([r["rmse_cov_at_chi2"] for r in o_group])
            o_chi2, _ = mean_std([r["chi2"] for r in o_group])
            hit = [r["iters_to_chi2"] for r in o_group if r["iters_to_chi2"] is not None]
            hit_label = f"{int(np.mean(hit))}" if hit else "never"
            cfg = f"{entry['hparams']} lr={entry['lr']:g}"
            print(f"{arch:9s}{t_mean:>11.4f} +-{t_std:<5.4f}{o_mean:>11.4f} +-{o_std:<5.4f}"
                  f"{t_mean - o_mean:>+18.4f}{cfg:>22s}{o_chi2:>8.3f}{hit_label:>9s}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "oracle.json").write_text(json.dumps({
        "best": {f"{t}|{a}": v for (t, a), v in best.items()},
        "search": [{k: v for k, v in r.items() if k != "resistivity"} for r in rows],
        "confirm": [{k: v for k, v in r.items() if k != "resistivity"} for r in confirm],
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {RESULTS / 'oracle.json'}")
    return 0


def main() -> int:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else min(8, (os.cpu_count() or 2) - 2)
    # Optional third argument: comma-separated architecture filter, e.g.
    #   inr_benchmark.py tune 9 tanh,dip
    # runs only those architectures and merges results into the stored JSONs.
    only: set[str] | None = None
    if len(sys.argv) > 3:
        only = {a.strip() for a in sys.argv[3].split(",") if a.strip()}
        unknown = only - set(ARCH_GRID)
        if unknown:
            raise SystemExit(f"unknown architectures {sorted(unknown)}; expected {sorted(ARCH_GRID)}")
    print(f"stage={stage}  n_jobs={n_jobs}  iters={N_ITERS}  only={sorted(only) if only else 'all'}\n")

    if stage == "smoke":
        return smoke(n_jobs)
    if stage == "tune":
        tune(n_jobs, only)
        return 0
    if stage == "eval":
        best = json.loads((RESULTS / "tuning.json").read_text(encoding="utf-8"))["best"]
        return evaluate(best, n_jobs, only)
    if stage == "oracle":
        return oracle(n_jobs)
    if stage == "all":
        code = smoke(n_jobs)
        if code:
            return code
        return evaluate(tune(n_jobs), n_jobs)
    raise SystemExit(f"unknown stage {stage!r}; expected smoke|tune|eval|all")


if __name__ == "__main__":
    raise SystemExit(main())
