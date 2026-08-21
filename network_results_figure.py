"""What each method actually recovered -- all seven, side by side.

Rebuilds the recovered-model comparison from fields already saved on disk, so
it needs no new inversions. Fixes two problems with inr_comparison.png:

  COLOUR SCALE   The old figure set each row's scale from the min/max across
                 all panels, so a single outlier flattened the parflow row to
                 uniform teal and hid every real feature. Here the scale comes
                 from the TRUE model's 2nd-98th percentiles, so recovered
                 outliers cannot destroy the row.

  MISSING BASELINE   The old figure showed only neural methods. Conventional
                 smooth-L2 and TV are the comparison the whole study is about,
                 so they are included.

Every panel is the model at chi^2 = 1 (the discrepancy-principle stopping
point), labelled with mean +- sd of coverage-masked log-RMSE over five seeds.

IMPORTANT SCOPE: these are the SHARED-configuration results -- 33,537
parameters and 1500 iterations for every network. That configuration was later
shown to understate the networks (Fourier scores 0.744 here, 0.405 when
properly configured). This figure is an accurate picture of Stage 1, not of
what the networks can do.

    python network_results_figure.py
"""

from __future__ import annotations

import json

import numpy as np

from inr_benchmark import HELD_OUT_TARGETS, RESULTS, build_case, mean_std, target_field
from deepert.inr import build_mesh, normalized_coords

SEED = 10                       # panel shown; statistics use all five
SEEDS = (10, 11, 12, 13, 14)
COLUMNS = [
    ("smooth_l2", "Smooth L2"),
    ("tv", "Total variation"),
    ("relu", "ReLU INR"),
    ("tanh", "tanh INR"),
    ("siren", "SIREN"),
    ("fourier", "Fourier-feature INR"),
    ("dip", "CNN Deep Image Prior"),
]


def gather() -> dict:
    """Recovered fields and errors for every method, from saved results."""

    evaluation = json.loads((RESULTS / "evaluation.json").read_text(encoding="utf-8"))
    smooth = json.loads((RESULTS / "smooth_l2.json").read_text(encoding="utf-8"))
    tv_cfg = json.loads((RESULTS / "traditional.json").read_text(encoding="utf-8"))["frozen"]
    from traditional_comparison import run_traditional

    out: dict[tuple, dict] = {}

    # Neural arms: fields were saved by the benchmark.
    for row in evaluation:
        if row.get("status", "ok") != "ok" or row["target"] not in HELD_OUT_TARGETS:
            continue
        out[(row["target"], row["arch"], row["init_seed"])] = {
            "rho": row.get("resistivity_at_chi2"), "rmse": row.get("rmse_cov_at_chi2")}

    # Smooth L2: fields saved by its evaluation script; skip diverged runs.
    ok = {(r["target"], r["noise_seed"]) for r in smooth["eval"] if r["chi2"] <= 1.0}
    for row in smooth["eval"]:
        key = (row["target"], row["noise_seed"])
        if row["target"] in HELD_OUT_TARGETS and key in ok:
            out[(row["target"], "smooth_l2", row["noise_seed"])] = {
                "rho": smooth["fields"][f"{row['target']}|{row['noise_seed']}"],
                "rmse": row["rmse_cov"]}

    # TV: fields were stripped, but a Gauss-Newton run costs under a second.
    print("Re-running TV to recover its fields (a few seconds) ...")
    for target in HELD_OUT_TARGETS:
        for seed in SEEDS:
            row = run_traditional(target, seed, tv_cfg["regularization"], tv_cfg["operator"])
            out[(target, "tv", seed)] = {"rho": row["resistivity"], "rmse": row["rmse_cov"]}
    return out


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LogNorm

    data = gather()
    mesh = build_mesh()
    polygons = np.asarray(mesh.nodes, dtype=float)[np.asarray(mesh.cells, dtype=np.int32)]
    centers, _ = normalized_coords(mesh)

    n_cols = 1 + len(COLUMNS)
    fig, axes = plt.subplots(len(HELD_OUT_TARGETS), n_cols,
                             figsize=(2.45 * n_cols, 2.9 * len(HELD_OUT_TARGETS)), squeeze=False)

    for r, target in enumerate(HELD_OUT_TARGETS):
        true_rho = target_field(target, centers)
        # Scale from the TRUE model only -- this is the fix for the flattened row.
        norm = LogNorm(vmin=float(np.percentile(true_rho, 2)),
                       vmax=float(np.percentile(true_rho, 98)))

        panels = [(true_rho, "True model", "")]
        for key, label in COLUMNS:
            shown = data.get((target, key, SEED), {}).get("rho")
            errors = [data[(target, key, s)]["rmse"] for s in SEEDS
                      if (target, key, s) in data and data[(target, key, s)]["rmse"] is not None]
            m, sd = mean_std(errors)
            subtitle = f"{m:.3f} $\\pm$ {sd:.3f}" if np.isfinite(m) else "never fit"
            panels.append((np.asarray(shown) if shown is not None else None, label, subtitle))

        coll = None
        for c, (values, label, subtitle) in enumerate(panels):
            ax = axes[r][c]
            if values is not None:
                coll = PolyCollection(polygons, array=values, cmap="turbo",
                                      norm=norm, edgecolors="none")
                ax.add_collection(coll)
            else:
                ax.text(0.5, 0.5, "did not reach\n$\\chi^2 = 1$", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="0.35")
            ax.set_xlim(0.0, 125.0)
            ax.set_ylim(-30.0, 2.0)
            ax.set_aspect("equal")
            ax.set_title(f"{label}\n{subtitle}", fontsize=9)
            ax.set_xticks([0, 60, 120])
            if c == 0:
                ax.set_ylabel(f"{target}\ndepth (m)", fontsize=9)
            else:
                ax.set_yticklabels([])
        if coll is not None:
            fig.colorbar(coll, ax=axes[r], fraction=0.010, pad=0.01).set_label("ohm-m", fontsize=9)

    fig.suptitle("Recovered resistivity at matched data fit ($\\chi^2 = 1$); "
                 "labels are coverage-masked log-RMSE, mean $\\pm$ sd over 5 seeds\n"
                 "Shared configuration: 33,537 parameters, 1500 iterations for every network",
                 fontsize=10, y=1.02)

    path = RESULTS / "network_results.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    print("\nMean coverage-masked log-RMSE at chi^2 = 1 (5 seeds):")
    print(f"{'method':24s}" + "".join(f"{t:>14s}" for t in HELD_OUT_TARGETS))
    print("-" * (24 + 14 * len(HELD_OUT_TARGETS)))
    for key, label in COLUMNS:
        line = f"{label:24s}"
        for target in HELD_OUT_TARGETS:
            errors = [data[(target, key, s)]["rmse"] for s in SEEDS
                      if (target, key, s) in data and data[(target, key, s)]["rmse"] is not None]
            m, _ = mean_std(errors)
            line += f"{m:>14.3f}" if np.isfinite(m) else f"{'never fit':>14s}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
