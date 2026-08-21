"""Anomaly geometry for the Stage 1b configurations.

Model error and structural accuracy came apart at the re-tuned configuration:
the leading network matched TV on log-RMSE while still misplacing anomaly
centroids 2.8x farther (4.7 m vs 1.7 m) and recovering a weaker footprint
(IoU 0.66 vs 0.83). Stage 1b changed how those configurations were selected --
calibration target instead of ground truth -- so the question is whether the
divergence survived.

Neural fields come straight from ``inr_results/stage1b/eval.json``; Stage 1b
retained them, so no network is re-inverted here. The conventional arm did NOT
retain fields, so TV and smooth-L2 are re-run locally at the weights Stage 1b
selected on the calibration target. Each classical inversion takes a couple of
seconds.

Metrics come from ``structure_metrics.score`` unchanged, so these numbers are
directly comparable to the Stage-1 table.

Usage:
    .venv\\Scripts\\python.exe stage1b_geometry.py [workers]
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

from inr_benchmark import build_case, target_field
from structure_metrics import score

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "inr_results"
STAGE1B = RESULTS / "stage1b"

TARGETS = ("blocks", "parflow")
ARCH_ORDER = ("relu", "tanh", "siren", "fourier", "dip")
LABELS = {"relu": "ReLU INR", "tanh": "tanh INR", "siren": "SIREN",
          "fourier": "Fourier INR", "dip": "CNN-DIP",
          "smooth_l2": "smooth-L2", "tv": "TV"}

# Stage-1 geometry at the ground-truth-tuned configuration, for comparison.
# Source: docs/stage1_results.md Table 2 and the capacity-retune geometry pass.
STAGE1_REFERENCE = {"fourier": {"iou": 0.662, "centroid": 4.7},
                    "tv": {"iou": 0.826, "centroid": 1.7}}


def _classical(job: tuple) -> dict:
    """One classical inversion, kept module-level so it can be pickled."""

    from traditional_comparison import run_traditional

    family, operator, lam, target, seed = job
    row = run_traditional(target, seed, lam, operator)
    return {"family": family, "target": target, "seed": seed,
            "resistivity": row["resistivity"], "rmse_cov": row["rmse_cov"]}


def main(workers: int) -> int:
    if not (STAGE1B / "eval.json").exists():
        raise SystemExit(f"missing {STAGE1B / 'eval.json'} -- copy inr_results/stage1b down from Argon")

    evaluation = json.loads((STAGE1B / "eval.json").read_text(encoding="utf-8"))
    neural = [r for r in evaluation["rows"] if r.get("status") == "ok" and r.get("resistivity")]

    conventional = json.loads((STAGE1B / "conventional.json").read_text(encoding="utf-8"))["best"]
    seeds = sorted({r["seed"] for r in neural})
    print(f"{len(neural)} neural fields from eval.json | seeds {seeds}")

    jobs = [(fam, cfg["operator"], float(cfg["regularization"]), target, seed)
            for fam, cfg in conventional.items()
            for target in TARGETS for seed in seeds]
    print(f"re-running {len(jobs)} classical inversions for their fields "
          f"(conventional.json did not retain them)\n")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        classical = list(pool.map(_classical, jobs))

    # Truth and coverage mask per target, computed once. The mask comes from the
    # Jacobian at the true model, so it is identical for every method.
    truths, masks = {}, {}
    for target in TARGETS:
        case = build_case(target, seeds[0])
        try:
            truths[target] = case["true_rho"]
            masks[target] = case["mask"]
        finally:
            case["forward"].close()

    rows: list[dict] = []
    for r in neural:
        rows.append({"method": r["arch"], "target": r["target"], "seed": r["seed"],
                     "rmse": r["rmse"],
                     **score(np.asarray(r["resistivity"], dtype=float),
                             truths[r["target"]], masks[r["target"]], r["target"])})
    for r in classical:
        rows.append({"method": r["family"], "target": r["target"], "seed": r["seed"],
                     "rmse": r["rmse_cov"],
                     **score(np.asarray(r["resistivity"], dtype=float),
                             truths[r["target"]], masks[r["target"]], r["target"])})

    order = ["tv", "smooth_l2", *ARCH_ORDER]
    for target in TARGETS:
        geometry = target == "blocks"      # IoU and centroid are defined for the block target
        print(f"\n=== {target} " + "=" * 58)
        header = f"{'method':14s}{'n':>3s}{'model err':>11s}{'SSIM':>8s}{'contrast':>10s}"
        if geometry:
            header += f"{'IoU':>8s}{'centroid m':>12s}{'magnitude':>11s}"
        print(header)
        print("-" * len(header))
        for method in order:
            group = [r for r in rows if r["method"] == method and r["target"] == target]
            if not group:
                print(f"{LABELS.get(method, method):14s}  no runs")
                continue

            def stat(key: str) -> tuple[float, float]:
                vals = [r[key] for r in group if r.get(key) is not None and np.isfinite(r[key])]
                if not vals:
                    return float("nan"), 0.0
                return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

            rmse, _ = stat("rmse")
            ssim_mean, _ = stat("ssim")
            contrast, _ = stat("contrast")
            line = (f"{LABELS.get(method, method):14s}{len(group):3d}{rmse:11.3f}"
                    f"{ssim_mean:8.3f}{contrast:10.2f}")
            if geometry:
                iou, _ = stat("iou")
                cen, _ = stat("centroid_err")
                mag, _ = stat("magnitude")
                line += f"{iou:8.3f}{cen:12.1f}{mag:11.2f}"
            print(line)

    # Did the error/geometry divergence survive calibration-only selection?
    print("\n" + "=" * 70)
    print("Model error vs geometry on blocks -- Stage 1b against the truth-tuned run")
    print("=" * 70)
    for method in ("tv", "fourier"):
        group = [r for r in rows if r["method"] == method and r["target"] == "blocks"]
        if not group:
            continue
        iou = statistics.mean(r["iou"] for r in group if r.get("iou") is not None)
        cens = [r["centroid_err"] for r in group
                if r.get("centroid_err") is not None and np.isfinite(r["centroid_err"])]
        cen = statistics.mean(cens) if cens else float("nan")
        ref = STAGE1_REFERENCE.get(method)
        note = f"   (truth-tuned: IoU {ref['iou']:.3f}, centroid {ref['centroid']:.1f} m)" if ref else ""
        print(f"  {LABELS[method]:12s} IoU {iou:.3f}   centroid {cen:.1f} m{note}")

    (STAGE1B / "geometry.json").write_text(
        json.dumps({"stage1_reference": STAGE1_REFERENCE, "rows": rows}, indent=2),
        encoding="utf-8")
    print(f"\nSaved {STAGE1B / 'geometry.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 7))
