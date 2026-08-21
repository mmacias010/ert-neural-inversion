"""Poster figure and the geometry metrics missing at the re-tuned configuration.

Two gaps this closes at once.

FIGURE   inr_comparison.png shows only neural methods, all at the superseded
         shared configuration (33,537 parameters, 1500 iterations), and its
         colour scale is stretched by an outlier. The finding it needs to show
         -- the network looks far worse than conventional inversion until it is
         configured properly, at which point the gap closes -- is not visible
         anywhere. Four panels per target make it obvious without text:
         true | conventional TV | Fourier at shared config | Fourier re-tuned.

METRICS  Anomaly geometry (IoU, centroid error), magnitude, and SSIM exist only
         at the shared configuration, because capacity_retune.py saved scalars
         and discarded the recovered fields. So "conventional inversion
         recovered anomaly geometry more accurately" is currently verified only
         under the shared budget -- the one claim in the abstract that could
         overstate. This measures it at the re-tuned configuration.

Both need the same thing: the recovered resistivity fields. This runs the three
configurations with the fields retained, then scores and plots them.

    python poster_figure.py [n_jobs]
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
    CHI2_TARGET,
    GAMMA,
    HELD_OUT_TARGETS,
    N_ITERS,
    RESULTS,
    STEP_SIZE,
    build_case,
    mean_std,
    model_errors,
    target_field,
)
from capacity_sweep import size_kwargs
from capacity_retune import schedule_for
from structure_metrics import score
from deepert.inr import build_mesh, build_network, fit_inr, normalized_coords, seed_networks

torch.set_num_threads(1)

SEEDS = (10, 11, 12, 13, 14)
PANELS = ("tv", "fourier_shared", "fourier_retuned")
LABELS = {
    "tv": "conventional TV",
    "fourier_shared": "Fourier, shared config",
    "fourier_retuned": "Fourier, re-tuned",
}


def configs() -> dict:
    """Frozen configurations, read from disk rather than hardcoded."""

    shared = json.loads((RESULTS / "tuning.json").read_text(encoding="utf-8"))["best"]["fourier"]
    retuned = json.loads((RESULTS / "capacity_retune.json").read_text(encoding="utf-8"))["best"]["fourier"]
    tv = json.loads((RESULTS / "traditional.json").read_text(encoding="utf-8"))["frozen"]
    return {
        "tv": {"regularization": tv["regularization"], "operator": tv["operator"]},
        "fourier_shared": {"hparams": shared["hparams"], "lr": shared["lr"],
                           "width": 128, "n_iters": N_ITERS, "step_size": STEP_SIZE},
        "fourier_retuned": {"hparams": retuned["hparams"], "lr": retuned["lr"],
                            "width": retuned["width"], "n_iters": retuned["n_iters"],
                            "step_size": schedule_for(retuned["n_iters"])[0]},
    }


def run_panel(method: str, config: dict, target: str, seed: int) -> dict:
    """One inversion, RETAINING the recovered field (the point of this script)."""

    torch.set_num_threads(1)
    base = {"method": method, "target": target, "seed": seed}

    if method == "tv":
        from traditional_comparison import run_traditional
        row = run_traditional(target, seed, config["regularization"], config["operator"])
        return {**base, "status": "ok", "rmse": row["rmse_cov"], "chi2": row["chi2"],
                "iters": row["n_iterations"], "elapsed": row["elapsed"],
                "resistivity": row["resistivity"]}

    case = build_case(target, seed)
    try:
        seed_networks(seed)
        net = build_network("fourier", log_rho_mean=case["log_rho_mean"],
                            **size_kwargs("fourier", config["width"]), **config["hparams"])
        result = fit_inr(
            case["forward"], net, case["coords"], case["obs_log"], case["data_std"],
            n_iters=config["n_iters"], lr=config["lr"],
            step_size=config["step_size"], gamma=GAMMA, snapshot_at_chi2=CHI2_TARGET,
        )
        snapshot = result["snapshot_log_resistivity"]
        if snapshot is None:                      # never reached the noise level
            return {**base, "status": "no_fit", "rmse": None, "chi2": result["chi2"],
                    "iters": None, "elapsed": result["elapsed"], "resistivity": None}
        rho = np.exp(snapshot)
        _, rmse = model_errors(rho, case["true_rho"], case["mask"])
        return {**base, "status": "ok", "rmse": rmse, "chi2": result["chi2"],
                "iters": result["snapshot_iteration"], "elapsed": result["elapsed"],
                "resistivity": rho.tolist()}
    except Exception as error:
        return {**base, "status": "diverged", "error": f"{type(error).__name__}: {error}",
                "rmse": None, "chi2": None, "iters": None, "elapsed": None, "resistivity": None}
    finally:
        case["forward"].close()


def plot(rows: list[dict], masks: dict, centers: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LogNorm

    mesh = build_mesh()
    polygons = np.asarray(mesh.nodes, dtype=float)[np.asarray(mesh.cells, dtype=np.int32)]
    seed = SEEDS[0]

    fig, axes = plt.subplots(len(HELD_OUT_TARGETS), 4,
                             figsize=(16.5, 3.1 * len(HELD_OUT_TARGETS)), squeeze=False)
    for r, target in enumerate(HELD_OUT_TARGETS):
        true_rho = target_field(target, centers)
        # Scale from the TRUE model only: recovered outliers must not flatten the row
        # (the old figure's parflow row spanned 10^2-10^7 for exactly that reason).
        norm = LogNorm(vmin=float(np.percentile(true_rho, 1)),
                       vmax=float(np.percentile(true_rho, 99)))

        panels = [(true_rho, f"{target}: true model", None)]
        for method in PANELS:
            match = [x for x in rows if x["target"] == target and x["method"] == method
                     and x["seed"] == seed and x["resistivity"] is not None]
            group = [x["rmse"] for x in rows if x["target"] == target
                     and x["method"] == method and x["rmse"] is not None]
            mean, sd = mean_std(group)
            subtitle = f"error {mean:.3f} +- {sd:.3f}" if np.isfinite(mean) else "never fit"
            panels.append((np.asarray(match[0]["resistivity"]) if match else None,
                           LABELS[method], subtitle))

        for c, (values, title, subtitle) in enumerate(panels):
            ax = axes[r][c]
            if values is not None:
                coll = PolyCollection(polygons, array=values, cmap="turbo",
                                      norm=norm, edgecolors="none")
                ax.add_collection(coll)
            else:
                ax.text(0.5, 0.5, "no fit", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlim(0.0, 125.0)
            ax.set_ylim(-30.0, 2.0)
            ax.set_aspect("equal")
            ax.set_title(title if subtitle is None else f"{title}\n{subtitle}", fontsize=10)
            if c == 0:
                ax.set_ylabel("depth (m)")
            ax.set_xlabel("distance (m)")
        fig.colorbar(coll, ax=axes[r], fraction=0.012, pad=0.01).set_label("ohm-m")

    path = RESULTS / "poster_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {path}")


def main() -> int:
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, (os.cpu_count() or 2) - 1)
    cfg = configs()
    print("Configurations:")
    for method in PANELS:
        print(f"  {LABELS[method]:24s} {cfg[method]}")

    jobs = [(m, cfg[m], t, s) for t in HELD_OUT_TARGETS for m in PANELS for s in SEEDS]
    print(f"\n{len(jobs)} runs on {n_jobs} workers\n")

    t0 = time.time()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = [pool.submit(run_panel, *job) for job in jobs]
        for index, future in enumerate(futures, 1):
            rows.append(future.result())
            print(f"    {index}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"  done in {time.time() - t0:.0f}s\n")

    # ---- geometry at the re-tuned configuration: the measurement that was missing
    mesh = build_mesh()
    centers, _ = normalized_coords(mesh)
    masks, truths = {}, {}
    for target in HELD_OUT_TARGETS:
        case = build_case(target, SEEDS[0])
        masks[target], truths[target] = case["mask"], target_field(target, centers)
        case["forward"].close()

    print("=" * 92)
    print("GEOMETRY AND STRUCTURE -- does the conventional advantage survive re-tuning?")
    print("=" * 92)
    metrics: list[dict] = []
    for target in HELD_OUT_TARGETS:
        keys = ["ssim", "contrast"] + (["iou", "centroid_err", "magnitude"] if target == "blocks" else [])
        print(f"\n--- {target}")
        header = f"{'method':24s}{'error':>16s}" + "".join(f"{k:>17s}" for k in keys)
        print(header)
        print("-" * len(header))
        for method in PANELS:
            group = [x for x in rows if x["target"] == target and x["method"] == method
                     and x["resistivity"] is not None]
            if not group:
                print(f"{LABELS[method]:24s}{'never fit':>16s}")
                continue
            scored = [score(np.asarray(x["resistivity"]), truths[target], masks[target], target)
                      for x in group]
            for x, s in zip(group, scored):
                metrics.append({"method": method, "target": target, "seed": x["seed"],
                                "rmse": x["rmse"], **s})
            m, sd = mean_std([x["rmse"] for x in group])
            line = f"{LABELS[method]:24s}{m:>9.3f} +-{sd:<5.3f}"
            for k in keys:
                km, ksd = mean_std([s[k] for s in scored if np.isfinite(s.get(k, np.nan))])
                line += f"{km:>10.3f} +-{ksd:<5.3f}" if np.isfinite(km) else f"{'-':>17s}"
            print(line)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "poster_metrics.json").write_text(json.dumps(
        {"metrics": metrics,
         "runs": [{k: v for k, v in r.items() if k != "resistivity"} for r in rows]},
        indent=2), encoding="utf-8")
    print(f"\nSaved {RESULTS / 'poster_metrics.json'}")

    plot(rows, masks, centers)
    print("\nRead the IoU and centroid columns: if conventional's geometry advantage")
    print("shrinks between 'shared config' and 're-tuned', the abstract's geometry")
    print("claim must stay scoped to the shared budget. If it holds, it generalizes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
