"""Add a traditional Gauss-Newton arm to the INR architecture benchmark.

The neural benchmark compares three parameterizations against each other. This
adds the method they are meant to replace: classical cell-based inversion with
an explicit spatial regularizer, solved by Gauss-Newton.

Fairness
--------
Traditional gets exactly the same treatment the neural arms got:

* Same mesh, survey, noise realization, data uncertainty and coverage mask.
* Same starting field -- a homogeneous model at the median observed apparent
  resistivity, which is what the networks warm-start to.
* A 15-configuration grid (5 regularization weights x 3 operators), matching
  the 15 configurations SIREN and Fourier each received.
* Tuned on the development target only, then frozen and applied to held-out
  targets, plus a per-target oracle upper bound.

The operator grid deliberately includes total variation. TV is the standard
classical tool for recovering blocky structure, so omitting it would handicap
the traditional arm on the 'blocks' target in precisely the way an untuned
sigma handicapped Fourier features in the smoke test.

One asymmetry that cannot be removed: Gauss-Newton stops itself at
``target_chi2``, while the networks run a fixed iteration budget. Comparisons
are therefore drawn at matched chi^2 wherever the networks reached it.

This lives in its own file rather than inside inr_benchmark.py so it can be
written and run while a benchmark sweep is still in flight.

Usage:
    .venv\\Scripts\\python.exe traditional_comparison.py
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import time

import numpy as np

from inr_benchmark import (
    CHI2_TARGET,
    DATA_STD,
    DEV_TARGET,
    EVAL_SEEDS,
    HELD_OUT_TARGETS,
    RESULTS,
    TUNE_SEEDS,
    build_case,
    mean_std,
    model_errors,
)
from deepert.inversion.core import InversionConfig, invert_single_log_resistivity

MAX_GN_ITERATIONS = 30
REGULARIZATION_WEIGHTS = (1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0)
OPERATORS = ("first_order_smoothness", "spatial_total_variation", "damping")


def run_traditional(target: str, noise_seed: int, regularization: float, operator: str,
                    relative_noise: float = DATA_STD) -> dict:
    """One classical Gauss-Newton inversion, matched to the neural setup.

    ``relative_noise`` defaults to the benchmark's 1.5%; the robustness study
    overrides it, and ``data_std`` tracks it so the discrepancy-principle stop
    always means "fit to the noise floor".
    """

    case = build_case(target, noise_seed, relative_noise=relative_noise)
    try:
        observed_rhoa = np.exp(case["obs_log"])                       # cases store log data
        initial = np.full(case["true_rho"].size, float(np.exp(case["log_rho_mean"])))
        config = InversionConfig(
            max_iterations=MAX_GN_ITERATIONS,
            data_std=case["data_std"],
            regularization=float(regularization),
            spatial_regularization=operator,
            target_chi2=CHI2_TARGET,
        )
        t0 = time.time()
        result = invert_single_log_resistivity(case["forward"], observed_rhoa, initial, config=config)
        elapsed = time.time() - t0

        rho = np.asarray(result.final_model, dtype=float)
        rmse_full, rmse_cov = model_errors(rho, case["true_rho"], case["mask"])
        chi2_history = [float(v) for v in result.iteration_chi2]
        reached = [i for i, v in enumerate(chi2_history) if v <= CHI2_TARGET]
        return {
            "arch": "traditional",
            "target": target,
            "noise_seed": noise_seed,
            "init_seed": noise_seed,
            "regularization": float(regularization),
            "operator": operator,
            "chi2": chi2_history[-1],
            "rmse_full": rmse_full,
            "rmse_cov": rmse_cov,
            "iters_to_chi2": reached[0] if reached else None,
            "n_iterations": len(chi2_history),
            "elapsed": elapsed,
            "resistivity": rho.tolist(),
        }
    finally:
        case["forward"].close()


def run_parallel(jobs: list[tuple], n_jobs: int) -> list[dict]:
    t0 = time.time()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = [pool.submit(run_traditional, *job) for job in jobs]
        for index, future in enumerate(futures, 1):
            rows.append(future.result())
            if index % max(1, len(jobs) // 10) == 0 or index == len(jobs):
                print(f"    {index}/{len(jobs)} done ({time.time() - t0:.0f}s)", flush=True)
    return rows


def select_best(rows: list[dict], key=lambda r: (r["target"],)) -> dict:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault((*key(row), row["regularization"], row["operator"]), []).append(row)
    best: dict[tuple, dict] = {}
    for composite, group in grouped.items():
        outer, regularization, operator = composite[:-2], composite[-2], composite[-1]
        score, _ = mean_std([r["rmse_cov"] for r in group])
        if outer not in best or score < best[outer]["score"]:
            best[outer] = {"regularization": regularization, "operator": operator, "score": score}
    return best


def main(n_jobs: int = 8) -> int:
    grid = [(w, op) for op in OPERATORS for w in REGULARIZATION_WEIGHTS]
    print(f"TRADITIONAL arm - {len(grid)} configurations "
          f"({len(REGULARIZATION_WEIGHTS)} weights x {len(OPERATORS)} operators)\n")

    # ---- tune on the development target only -------------------------------
    print(f"Tuning on '{DEV_TARGET}', seeds {TUNE_SEEDS}")
    tune_jobs = [(DEV_TARGET, s, w, op) for w, op in grid for s in TUNE_SEEDS]
    tune_rows = run_parallel(tune_jobs, n_jobs)

    print(f"\n{'operator':26s}{'lambda':>9s}{'rmse_cov':>18s}{'chi2':>9s}{'iters':>7s}")
    print("-" * 69)
    grouped: dict[tuple, list[dict]] = {}
    for row in tune_rows:
        grouped.setdefault((row["operator"], row["regularization"]), []).append(row)
    for (operator, weight), group in sorted(grouped.items()):
        m, s = mean_std([r["rmse_cov"] for r in group])
        c, _ = mean_std([r["chi2"] for r in group])
        n, _ = mean_std([r["n_iterations"] for r in group])
        print(f"{operator:26s}{weight:>9.4g}{m:>11.4f} +-{s:<5.4f}{c:>9.3f}{n:>7.0f}")

    frozen = select_best(tune_rows, key=lambda r: ())[()]
    print(f"\nSELECTED (dev): {frozen['operator']}  lambda={frozen['regularization']:g}  "
          f"rmse_cov={frozen['score']:.4f}")

    # ---- freeze and evaluate on held-out targets ---------------------------
    print(f"\nEvaluating frozen config on {(*HELD_OUT_TARGETS, DEV_TARGET)}, seeds {EVAL_SEEDS}")
    eval_jobs = [(t, s, frozen["regularization"], frozen["operator"])
                 for t in (*HELD_OUT_TARGETS, DEV_TARGET) for s in EVAL_SEEDS]
    eval_rows = run_parallel(eval_jobs, n_jobs)

    # ---- per-target oracle upper bound -------------------------------------
    print(f"\nOracle search on {HELD_OUT_TARGETS} (upper bound; selects against truth)")
    oracle_jobs = [(t, s, w, op) for t in HELD_OUT_TARGETS for w, op in grid for s in TUNE_SEEDS]
    oracle_rows = run_parallel(oracle_jobs, n_jobs)
    oracle_best = select_best(oracle_rows)
    confirm_jobs = [(t, s, entry["regularization"], entry["operator"])
                    for (t,), entry in oracle_best.items() for s in EVAL_SEEDS]
    confirm_rows = run_parallel(confirm_jobs, n_jobs)

    # ---- combined table ----------------------------------------------------
    neural_eval = json.loads((RESULTS / "evaluation.json").read_text(encoding="utf-8"))
    try:
        neural_oracle = json.loads((RESULTS / "oracle.json").read_text(encoding="utf-8"))["confirm"]
    except FileNotFoundError:
        neural_oracle = []

    def collect(rows, target, arch, field="rmse_cov"):
        return [r[field] for r in rows if r["target"] == target and r["arch"] == arch]

    print("\n" + "=" * 92)
    print("COMBINED - coverage-masked model RMSE (log), mean +- sd over 5 evaluation seeds")
    print("=" * 92)
    for target in (*HELD_OUT_TARGETS, DEV_TARGET):
        label = "HELD OUT" if target in HELD_OUT_TARGETS else "development (all methods tuned here)"
        print(f"\n--- {target}  [{label}]")
        print(f"{'method':14s}{'transfer':>20s}{'oracle':>20s}{'chi2':>9s}{'reached chi2=1':>16s}")
        print("-" * 79)
        for arch in ("traditional", "relu", "siren", "fourier"):
            if arch == "traditional":
                transfer = [r for r in eval_rows if r["target"] == target]
                oracle = [r for r in confirm_rows if r["target"] == target]
            else:
                transfer = [r for r in neural_eval if r["target"] == target and r["arch"] == arch]
                oracle = [r for r in neural_oracle if r["target"] == target and r["arch"] == arch]
            if not transfer:
                continue
            t_m, t_s = mean_std([r["rmse_cov"] for r in transfer])
            o_m, o_s = mean_std([r["rmse_cov"] for r in oracle]) if oracle else (float("nan"), 0.0)
            c_m, _ = mean_std([r["chi2"] for r in transfer])
            hit = sum(1 for r in transfer if r["iters_to_chi2"] is not None)
            o_label = f"{o_m:>13.4f} +-{o_s:<5.4f}" if np.isfinite(o_m) else f"{'-':>20s}"
            print(f"{arch:14s}{t_m:>13.4f} +-{t_s:<5.4f}{o_label}{c_m:>9.3f}{f'{hit}/{len(transfer)}':>16s}")

    RESULTS.mkdir(exist_ok=True)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "resistivity"} for r in rows]
    (RESULTS / "traditional.json").write_text(json.dumps({
        "frozen": frozen,
        "oracle_best": {t[0]: v for t, v in oracle_best.items()},
        "tune": strip(tune_rows), "eval": strip(eval_rows),
        "oracle_search": strip(oracle_rows), "oracle_confirm": strip(confirm_rows),
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {RESULTS / 'traditional.json'}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 8))
