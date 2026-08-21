"""Re-analysis of saved benchmark runs. No new compute.

Fixes two defects in the original reporting:

1. ORACLE SELECTION. The original oracle minimized model error without
   requiring the data to be fit, so a diverged run whose field happened to
   resemble the truth could win -- which is how SIREN's oracle values came from
   runs at chi^2 = 64.3 that fit the data in 0/5 seeds. Here a configuration is
   only eligible if EVERY seed reached chi^2 <= 1.

2. MATCHED CHI^2 AS THE PRIMARY METRIC. Architectures converge at very
   different rates (Fourier reached chi^2 = 1 by iteration 121 on 'blocks';
   ReLU and SIREN never did within 600). Comparing them at a fixed budget
   therefore conflates "cannot represent the field" with "had not finished".
   Classical Gauss-Newton stops itself at chi^2 = 1, so its reported result is
   already a matched-fit result; the networks are now reported the same way,
   using the snapshots fit_inr recorded.

Where a method never reached chi^2 = 1, that is stated -- never silently
replaced by its fixed-budget number.

    .venv\\Scripts\\python.exe reanalysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "inr_results"
CHI2_TARGET = 1.0
NEURAL = ("relu", "siren", "fourier")
TARGETS = ("blocks", "parflow", "two_layer")
HELD_OUT = ("blocks", "parflow")


def stats(values) -> tuple[float, float]:
    array = np.asarray([v for v in values if v is not None], dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def fmt(mean: float, sd: float, width: int = 13) -> str:
    if not np.isfinite(mean):
        return f"{'never reached':>{width + 7}}"
    return f"{mean:>{width}.4f} +-{sd:<5.4f}"


def load() -> tuple[list, dict, dict]:
    evaluation = json.loads((RESULTS / "evaluation.json").read_text(encoding="utf-8"))
    oracle = json.loads((RESULTS / "oracle.json").read_text(encoding="utf-8"))
    traditional = json.loads((RESULTS / "traditional.json").read_text(encoding="utf-8"))
    return evaluation, oracle, traditional


# ------------------------------------------------------------------ part 1
def matched_chi2_tables(evaluation: list, traditional: dict) -> None:
    print("=" * 96)
    print("PRIMARY - coverage-masked model RMSE at MATCHED DATA FIT (chi^2 = 1)")
    print("Fixed-budget numbers shown alongside. 5 evaluation seeds.")
    print("=" * 96)

    for target in TARGETS:
        label = "HELD OUT" if target in HELD_OUT else "development (tuned here)"
        print(f"\n--- {target}  [{label}]")
        print(f"{'method':14s}{'at chi2=1 (PRIMARY)':>22s}{'at fixed budget':>22s}"
              f"{'seeds reaching chi2=1':>24s}")
        print("-" * 82)

        rows = []
        trad = [r for r in traditional["eval"] if r["target"] == target]
        if trad:
            # Gauss-Newton stops at target_chi2, so its result IS the matched-fit result.
            m, s = stats([r["rmse_cov"] for r in trad])
            hit = sum(1 for r in trad if r["iters_to_chi2"] is not None)
            rows.append(("traditional", (m, s), (m, s), hit, len(trad)))

        for arch in NEURAL:
            group = [r for r in evaluation if r["target"] == target and r["arch"] == arch]
            if not group:
                continue
            snap = stats([r["rmse_cov_at_chi2"] for r in group])
            budget = stats([r["rmse_cov"] for r in group])
            hit = sum(1 for r in group if r["rmse_cov_at_chi2"] is not None)
            rows.append((arch, snap, budget, hit, len(group)))

        ranked = sorted(rows, key=lambda r: (not np.isfinite(r[1][0]), r[1][0]))
        for name, snap, budget, hit, total in ranked:
            print(f"{name:14s}{fmt(*snap)}{fmt(*budget)}{f'{hit}/{total}':>24s}")


# ------------------------------------------------------------------ part 2
def fixed_oracle(oracle: dict, traditional: dict) -> dict:
    """Re-select per-target oracle configs, requiring every seed to fit the data."""

    print("\n" + "=" * 96)
    print("ORACLE RE-SELECTION - eligible only if EVERY seed reached chi^2 <= 1")
    print("=" * 96)

    grouped: dict[tuple, list] = {}
    for row in oracle["search"]:
        grouped.setdefault((row["target"], row["arch"], row["hparams"], row["lr"]), []).append(row)
    for row in traditional["oracle_search"]:
        key = (row["target"], "traditional", row["operator"], row["regularization"])
        grouped.setdefault(key, []).append(row)

    selected: dict[tuple, dict] = {}
    rejected: dict[tuple, int] = {}
    for (target, arch, hparams, lr), group in grouped.items():
        eligible = all(r["chi2"] <= CHI2_TARGET for r in group)
        if not eligible:
            rejected[(target, arch)] = rejected.get((target, arch), 0) + 1
            continue
        field = "rmse_cov_at_chi2" if arch in NEURAL else "rmse_cov"
        score, _ = stats([r.get(field) for r in group])
        if not np.isfinite(score):
            continue
        key = (target, arch)
        if key not in selected or score < selected[key]["score"]:
            selected[key] = {"config": f"{hparams} lr={lr}", "score": score, "n_configs": len(group)}

    for target in HELD_OUT:
        print(f"\n--- {target}")
        print(f"{'method':14s}{'eligible config':>34s}{'search score':>16s}{'configs rejected':>19s}")
        print("-" * 83)
        for arch in ("traditional", *NEURAL):
            entry = selected.get((target, arch))
            n_rejected = rejected.get((target, arch), 0)
            if entry is None:
                print(f"{arch:14s}{'NO CONFIGURATION FIT THE DATA':>34s}{'-':>16s}{n_rejected:>19d}")
            else:
                print(f"{arch:14s}{entry['config']:>34s}{entry['score']:>16.4f}{n_rejected:>19d}")
    return selected


# ------------------------------------------------------------------ part 3
def verdict(evaluation: list, traditional: dict) -> None:
    print("\n" + "=" * 96)
    print("DO THE CONCLUSIONS SURVIVE?")
    print("=" * 96)

    for target in HELD_OUT:
        trad = [r for r in traditional["eval"] if r["target"] == target]
        trad_mean, _ = stats([r["rmse_cov"] for r in trad])
        best_neural, best_score = None, float("inf")
        unfit = []
        for arch in NEURAL:
            group = [r for r in evaluation if r["target"] == target and r["arch"] == arch]
            mean, _ = stats([r["rmse_cov_at_chi2"] for r in group])
            if not np.isfinite(mean):
                unfit.append(arch)
                continue
            if mean < best_score:
                best_neural, best_score = arch, mean

        print(f"\n{target}:")
        print(f"  traditional (Gauss-Newton, stops at chi2=1) : {trad_mean:.4f}")
        if best_neural is None:
            print("  best neural at matched chi2                  : none reached chi2 = 1")
        else:
            gap = best_score - trad_mean
            print(f"  best neural at matched chi2 ({best_neural:<8s})       : {best_score:.4f}")
            print(f"  gap (positive = traditional better)         : {gap:+.4f}")
        if unfit:
            print(f"  never reached chi2 = 1                      : {', '.join(unfit)}")
            print("    -> their fixed-budget errors measure non-convergence, not")
            print("       representational capacity. Budget must be raised before")
            print("       any claim is made about these architectures.")


def main() -> int:
    evaluation, oracle, traditional = load()
    matched_chi2_tables(evaluation, traditional)
    fixed_oracle(oracle, traditional)
    verdict(evaluation, traditional)
    print("\nNo new compute: every number above comes from runs already in inr_results/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
