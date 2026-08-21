"""Structural metrics: anomaly geometry, recovered magnitude, and similarity.

Model log-RMSE says how wrong a recovered field is on average, but not whether
the anomaly was put in the right PLACE with the right AMPLITUDE. Those are
separate questions, and a method can do well on one and badly on the other.

Computed here, all on the chi^2 = 1 snapshot models already saved by the
benchmark (no new inversions for the neural arms; the classical arm is re-run
because its fields were stripped before writing traditional.json):

  SSIM            structural similarity of the log-resistivity image, over the
                  survey-sensed depth range. Sensitive to structure and
                  contrast rather than to average level.
  contrast        std(recovered log rho) / std(true log rho) -- how much of the
                  true heterogeneity amplitude survived. 1.0 is ideal; below 1
                  means over-smoothing, above 1 means spurious structure.
  IoU             intersection-over-union of the recovered anomaly footprint
                  against the true one (blocks target only, where anomalies are
                  discrete). Pure geometry: did the anomaly land in the right
                  place with the right extent?
  centroid err    distance in metres between true and recovered anomaly
                  centroids (blocks only).
  magnitude       recovered log-contrast inside the true anomaly, as a fraction
                  of the true log-contrast (blocks only). 1.0 = full amplitude
                  recovery, 0 = anomaly not expressed at all.

The mesh is a structured grid split into triangles (2 per rectangle), so cell
values map back to a (nz, nx) image exactly, by averaging each triangle pair.

    .venv\\Scripts\\python.exe structure_metrics.py
"""

from __future__ import annotations

import json

import numpy as np

from inr_benchmark import HELD_OUT_TARGETS, RESULTS, EVAL_SEEDS, build_case, mean_std, target_field
from deepert.inr import build_mesh, normalized_coords

NX, NZ = 25, 20
DX, DZ = 5.0, 5.0


def cells_to_image(values: np.ndarray) -> np.ndarray:
    """(1000,) triangle-cell values -> (nz, nx) image; row 0 is the surface."""

    v = np.asarray(values, dtype=float).reshape(NZ * NX, 2).mean(axis=1)
    return v.reshape(NZ, NX)


def ssim(a: np.ndarray, b: np.ndarray, *, window: int = 5) -> float:
    """Structural similarity with a uniform window (small images here)."""

    from scipy.ndimage import uniform_filter

    data_range = float(a.max() - a.min())
    if data_range <= 0:
        return float("nan")
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_a = uniform_filter(a, window, mode="nearest")
    mu_b = uniform_filter(b, window, mode="nearest")
    saa = uniform_filter(a * a, window, mode="nearest") - mu_a * mu_a
    sbb = uniform_filter(b * b, window, mode="nearest") - mu_b * mu_b
    sab = uniform_filter(a * b, window, mode="nearest") - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * sab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (saa + sbb + c2)
    return float(np.mean(num / den))


def sensed_rows(mask: np.ndarray) -> slice:
    """Rows where the survey has sensitivity in most cells."""

    covered = cells_to_image(mask.astype(float)) > 0.5
    rows = np.flatnonzero(covered.any(axis=1))
    return slice(int(rows.min()), int(rows.max()) + 1)


def anomaly_metrics(rec_log: np.ndarray, true_log: np.ndarray, rows: slice) -> dict:
    """Geometry and amplitude of discrete anomalies (blocks target)."""

    rec, true = rec_log[rows], true_log[rows]
    background = float(np.median(true))
    out: dict[str, float] = {}
    ious, centroids, mags = [], [], []

    for sign in (-1.0, +1.0):                      # conductive, then resistive
        true_mask = (sign * (true - background)) > 0.05
        if true_mask.sum() < 3:
            continue
        # Threshold the recovery halfway (in log space) to the true anomaly level.
        level = background + 0.5 * sign * abs(float(true[true_mask].mean()) - background)
        rec_mask = (sign * (rec - level)) > 0
        union = (true_mask | rec_mask).sum()
        ious.append(float((true_mask & rec_mask).sum() / union) if union else 0.0)

        zz, xx = np.mgrid[0:true.shape[0], 0:true.shape[1]]
        ct = np.array([(zz[true_mask].mean()) * DZ, (xx[true_mask].mean()) * DX])
        if rec_mask.sum() >= 3:
            cr = np.array([(zz[rec_mask].mean()) * DZ, (xx[rec_mask].mean()) * DX])
            centroids.append(float(np.linalg.norm(cr - ct)))
        else:
            centroids.append(float("nan"))         # anomaly not expressed at all

        true_contrast = float(true[true_mask].mean()) - background
        rec_contrast = float(rec[true_mask].mean()) - background
        mags.append(rec_contrast / true_contrast if true_contrast else float("nan"))

    out["iou"] = float(np.mean(ious)) if ious else float("nan")
    out["centroid_err"] = float(np.nanmean(centroids)) if centroids else float("nan")
    out["magnitude"] = float(np.mean(mags)) if mags else float("nan")
    return out


def score(rec_rho: np.ndarray, true_rho: np.ndarray, mask: np.ndarray, target: str) -> dict:
    rec_log = np.log10(cells_to_image(rec_rho))
    true_log = np.log10(cells_to_image(true_rho))
    rows = sensed_rows(mask)
    result = {
        "ssim": ssim(true_log[rows], rec_log[rows]),
        "contrast": float(rec_log[rows].std() / true_log[rows].std()),
    }
    if target == "blocks":
        result.update(anomaly_metrics(rec_log, true_log, rows))
    return result


def classical_fields() -> dict[tuple, np.ndarray]:
    """Re-run the classical arm on held-out targets, keeping its fields."""

    from traditional_comparison import run_traditional

    frozen = json.loads((RESULTS / "traditional.json").read_text(encoding="utf-8"))["frozen"]
    print(f"Re-running classical arm ({frozen['operator']}, lambda={frozen['regularization']:g}) "
          f"to recover fields ...")
    fields = {}
    for target in HELD_OUT_TARGETS:
        for seed in EVAL_SEEDS:
            row = run_traditional(target, seed, frozen["regularization"], frozen["operator"])
            fields[(target, seed)] = np.asarray(row["resistivity"], dtype=float)
    print(f"  {len(fields)} classical fields recovered\n")
    return fields


def main() -> int:
    mesh = build_mesh()
    centers, _ = normalized_coords(mesh)
    evaluation = json.loads((RESULTS / "evaluation.json").read_text(encoding="utf-8"))
    classical = classical_fields()

    masks, truths = {}, {}
    for target in HELD_OUT_TARGETS:
        case = build_case(target, EVAL_SEEDS[0])
        masks[target] = case["mask"]
        truths[target] = target_field(target, centers)
        case["forward"].close()

    # Smooth-L2 fields were saved by its eval script; keep only runs that fit
    # the data (chi^2 <= 1) -- the diverged noise realization would otherwise
    # contribute geometry scores for a garbage field.
    smooth_l2 = json.loads((RESULTS / "smooth_l2.json").read_text(encoding="utf-8"))
    l2_ok = {(r["target"], r["noise_seed"]) for r in smooth_l2["eval"] if r["chi2"] <= 1.0}

    rows: list[dict] = []
    for target in HELD_OUT_TARGETS:
        for seed in EVAL_SEEDS:
            rows.append({"target": target, "arch": "traditional", "seed": seed,
                         **score(classical[(target, seed)], truths[target], masks[target], target)})
            if (target, seed) in l2_ok:
                field = np.asarray(smooth_l2["fields"][f"{target}|{seed}"], dtype=float)
                rows.append({"target": target, "arch": "smooth_l2", "seed": seed,
                             **score(field, truths[target], masks[target], target)})
        for record in evaluation:
            if record["target"] != target or record.get("status", "ok") != "ok":
                continue
            if record.get("resistivity_at_chi2") is None:
                continue
            rows.append({"target": target, "arch": record["arch"], "seed": record["init_seed"],
                         **score(np.asarray(record["resistivity_at_chi2"]),
                                 truths[target], masks[target], target)})

    for target in HELD_OUT_TARGETS:
        print(f"=== {target} " + "=" * 62)
        keys = ["ssim", "contrast"] + (["iou", "centroid_err", "magnitude"] if target == "blocks" else [])
        header = f"{'method':13s}" + "".join(f"{k:>17s}" for k in keys)
        print(header)
        print("-" * len(header))
        for arch in ("traditional", "smooth_l2", "siren", "fourier", "relu", "tanh", "dip"):
            group = [r for r in rows if r["target"] == target and r["arch"] == arch]
            if not group:
                print(f"{arch:13s}{'no runs fit the data':>17s}")
                continue
            line = f"{arch:13s}"
            for k in keys:
                m, s = mean_std([r[k] for r in group if np.isfinite(r.get(k, np.nan))])
                line += f"{m:>10.3f} +-{s:<5.3f}" if np.isfinite(m) else f"{'not expressed':>17s}"
            print(line)
        print()

    print("Reading: SSIM 1.0 = identical structure. contrast 1.0 = full heterogeneity")
    print("         amplitude recovered (<1 over-smoothed). IoU 1.0 = perfect anomaly")
    print("         footprint. magnitude 1.0 = full log-contrast recovered.")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "structure_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved {RESULTS / 'structure_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
