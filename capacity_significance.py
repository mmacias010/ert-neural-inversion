"""Are the capacity-sweep crossings real, or is the detector too literal?

capacity_sweep.py flags a "ranking change" whenever the ordering of mean
errors differs between capacities. That test ignores uncertainty: a 0.05 swap
between two architectures whose seed spread is 0.10 is noise, not a crossing.

Every run at a given seed uses the SAME noise realization, so comparisons are
naturally PAIRED and can be tested with far more power than comparing
independent means. Two questions:

  within   Does capacity change an architecture's own error? (33k vs each
           other width, paired by seed.)
  between  Does the ordering of two architectures change with capacity, and
           is either ordering statistically supported?

With five seeds the smallest attainable two-sided Wilcoxon p is 0.0625, so
sign counts are reported alongside: 5/5 in one direction is as strong as this
design can be, and is treated as the practical significance threshold.

    python capacity_significance.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS = Path(__file__).resolve().parent / "inr_results"
REFERENCE_WIDTH = 128        # the Stage-1 budget, 33,537 parameters
WIDTHS = (64, 128, 256, 512)
ARCHS = ("relu", "tanh", "siren", "fourier", "dip")


def budget(width: int) -> int:
    return (2 * width + width) + 2 * (width * width + width) + (width + 1)


def paired(rows, target, arch_a, width_a, arch_b, width_b, field="rmse_at_chi2"):
    """Values for seeds where BOTH configurations produced a fit."""

    def by_seed(arch, width):
        return {r["seed"]: r[field] for r in rows
                if r["target"] == target and r["arch"] == arch
                and r["width"] == width and r.get(field) is not None}

    a, b = by_seed(arch_a, width_a), by_seed(arch_b, width_b)
    seeds = sorted(set(a) & set(b))
    return np.array([a[s] for s in seeds]), np.array([b[s] for s in seeds]), seeds


def verdict(a: np.ndarray, b: np.ndarray) -> str:
    """Paired comparison of a vs b. Positive difference means b is better."""

    if len(a) < 3:
        return f"n={len(a)} too few to test"
    diff = a - b
    wins = int(np.sum(diff > 0))
    mean = float(np.mean(diff))
    try:
        p = float(stats.wilcoxon(diff).pvalue)
    except ValueError:
        p = float("nan")
    t_p = float(stats.ttest_rel(a, b).pvalue)
    flag = "SIGNIFICANT" if (wins == len(diff) or wins == 0) and len(diff) >= 5 else \
           "significant" if t_p < 0.05 else "not significant"
    return (f"delta {mean:+.4f}  {wins}/{len(diff)} seeds  "
            f"wilcoxon p={p:.4f}  t p={t_p:.4f}  -> {flag}")


def within_architecture(rows: list[dict]) -> None:
    print("=" * 92)
    print("WITHIN ARCHITECTURE - does capacity change its own error?")
    print("Reference: 33,537 parameters (the Stage-1 budget). Positive delta = the")
    print("other capacity is BETTER than 33k.")
    print("=" * 92)
    for target in ("blocks", "parflow"):
        print(f"\n--- {target}")
        for arch in ARCHS:
            lines = []
            for width in WIDTHS:
                if width == REFERENCE_WIDTH:
                    continue
                a, b, _ = paired(rows, target, arch, REFERENCE_WIDTH, arch, width)
                if len(a):
                    lines.append(f"    33k vs {budget(width) // 1000:>3d}k: {verdict(a, b)}")
            if lines:
                print(f"  {arch}")
                print("\n".join(lines))
            else:
                print(f"  {arch}\n    no capacity produced enough paired fits")


def between_architectures(rows: list[dict]) -> None:
    print("\n" + "=" * 92)
    print("BETWEEN ARCHITECTURES - is any ordering actually supported?")
    print("Only pairs whose mean ordering CHANGES across capacity are shown, since")
    print("those are the crossings the sweep flagged.")
    print("=" * 92)
    for target in ("blocks", "parflow"):
        print(f"\n--- {target}")
        for arch_a, arch_b in itertools.combinations(ARCHS, 2):
            orderings, rendered = set(), []
            for width in WIDTHS:
                a, b, _ = paired(rows, target, arch_a, width, arch_b, width)
                if len(a) < 3:
                    continue
                orderings.add(bool(np.mean(a) < np.mean(b)))
                rendered.append(f"    {budget(width) // 1000:>3d}k: "
                                f"{arch_a} {np.mean(a):.3f} vs {arch_b} {np.mean(b):.3f}  "
                                f"{verdict(a, b)}")
            if len(orderings) > 1:        # the ordering flipped somewhere
                print(f"  {arch_a} vs {arch_b}  [ORDERING FLIPS]")
                print("\n".join(rendered))


def contrast_trend(rows: list[dict]) -> None:
    print("\n" + "=" * 92)
    print("CONTRAST RATIO vs CAPACITY - is 'excess structure' capacity-tunable?")
    print("1.0 = correct amplitude. Positive delta = more capacity moved it CLOSER to 1.")
    print("=" * 92)
    for target in ("blocks", "parflow"):
        print(f"\n--- {target}")
        for arch in ARCHS:
            lines = []
            for width in WIDTHS:
                if width == REFERENCE_WIDTH:
                    continue
                a, b, _ = paired(rows, target, arch, REFERENCE_WIDTH, arch, width, field="contrast")
                if len(a) >= 3:
                    # distance from 1.0, so "better" means closer to correct amplitude
                    lines.append(f"    33k vs {budget(width) // 1000:>3d}k: "
                                 f"{verdict(np.abs(a - 1.0), np.abs(b - 1.0))}")
            if lines:
                print(f"  {arch}")
                print("\n".join(lines))


def main() -> int:
    rows = json.loads((RESULTS / "capacity_sweep.json").read_text(encoding="utf-8"))
    rows = [r for r in rows if r.get("status") == "ok"]
    within_architecture(rows)
    between_architectures(rows)
    contrast_trend(rows)
    print("\nNote: with 5 seeds the smallest possible two-sided Wilcoxon p is 0.0625,")
    print("so a unanimous 5/5 sign count is the strongest evidence this design allows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
