"""Two figures for the 5-minute lightning talk.

Both are built for a non-expert audience seen from the back of a room: large
type, few words, no jargon in the axis labels. Neither adds a dependency and
neither changes library code.

    nonuniqueness  -- two different subsurface models whose predicted
                      measurements lie on top of each other. This is the
                      premise of the whole project, shown rather than asserted.
                      Reads saved fields from inr_results/evaluation.json and
                      only forward-solves them, so it takes seconds.

    semiconvergence -- data misfit falling while model error rises. This is the
                      talk's takeaway, drawn. Requires one inversion (~5-8 min
                      on a laptop); uses fit_inr's existing ``callback`` hook to
                      record model error per iteration, so train.py is unchanged.

Usage:
    .venv\\Scripts\\python.exe talk_figures.py nonuniqueness
    .venv\\Scripts\\python.exe talk_figures.py semiconvergence [n_iters]
    .venv\\Scripts\\python.exe talk_figures.py all
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json
from pathlib import Path
import sys

import numpy as np
import torch

from deepert.inr import build_mesh, build_network, fit_inr, seed_networks
from inr_benchmark import DATA_STD, GAMMA, build_case

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "inr_results"

TARGET = "blocks"          # sharp structure: the clearest picture at slide distance
SEED = 10                  # any evaluation seed; fields exist for all of 10-14

LABELS = {"relu": "ReLU network", "tanh": "tanh network", "siren": "SIREN",
          "fourier": "Fourier network", "dip": "Deep Image Prior"}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 15, "axes.titlesize": 18, "axes.labelsize": 15,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 15,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def _panel(ax, polygons, values, norm, title):
    from matplotlib.collections import PolyCollection

    coll = PolyCollection(polygons, array=values, cmap="turbo", norm=norm, edgecolors="none")
    ax.add_collection(coll)
    ax.set_xlim(0.0, 125.0)
    ax.set_ylim(-30.0, 2.0)
    ax.set_aspect("equal")
    ax.set_title(title, pad=8)
    ax.set_xlabel("distance along the ground (m)")
    ax.set_ylabel("depth (m)")
    return coll


# ---------------------------------------------------------------------------
# Figure 1 -- nonuniqueness
# ---------------------------------------------------------------------------

def nonuniqueness() -> None:
    """Two different pictures, one set of measurements.

    Both models are read from evaluation.json at the chi^2 = 1 snapshot, so both
    already fit the data to the noise level by construction. The figure's job is
    to make that fact visible: forward-solve each and overlay the predictions.

    The pair is chosen as the two saved models that differ MOST from each other
    in log-resistivity, because the point is strongest when the two pictures are
    obviously not the same picture.
    """

    rows = json.loads((RESULTS / "evaluation.json").read_text(encoding="utf-8"))
    saved = [r for r in rows
             if r.get("status") == "ok" and r.get("target") == TARGET
             and r.get("init_seed") == SEED and r.get("resistivity_at_chi2")]
    if len(saved) < 2:
        raise SystemExit(f"need two saved fields for {TARGET} seed {SEED}; found {len(saved)}")

    # Most visually distinct pair, measured in log space (the space the metrics use).
    best_pair, best_gap = None, -1.0
    for i in range(len(saved)):
        for j in range(i + 1, len(saved)):
            a = np.log(np.asarray(saved[i]["resistivity_at_chi2"], dtype=float))
            b = np.log(np.asarray(saved[j]["resistivity_at_chi2"], dtype=float))
            gap = float(np.sqrt(np.mean((a - b) ** 2)))
            if gap > best_gap:
                best_pair, best_gap = (saved[i], saved[j]), gap
    left, right = best_pair
    print(f"pair: {left['arch']} vs {right['arch']}  (log-RMS difference {best_gap:.3f})")

    case = build_case(TARGET, SEED)
    try:
        forward, true_rho = case["forward"], case["true_rho"]
        obs_log = case["obs_log"]

        panels = []
        for row in (left, right):
            rho = np.asarray(row["resistivity_at_chi2"], dtype=float)
            predicted = np.asarray(forward.response(rho), dtype=float)
            chi2 = float(np.mean(((np.log(predicted) - obs_log) / DATA_STD) ** 2))
            panels.append((row, rho, predicted, chi2))
            print(f"  {row['arch']:8s}  chi2 = {chi2:.2f}   model error = "
                  f"{row['rmse_cov_at_chi2']:.3f}")

        # How far apart the two PREDICTIONS are, against the noise they are fit to.
        spread = float(np.sqrt(np.mean((np.log(panels[0][2]) - np.log(panels[1][2])) ** 2)))
        print(f"  predictions differ by {spread * 100:.2f}%  |  noise level {DATA_STD * 100:.1f}%")

        plt = _style()
        from matplotlib.colors import LogNorm

        mesh = build_mesh()
        polygons = np.asarray(mesh.nodes, dtype=float)[np.asarray(mesh.cells, dtype=np.int32)]
        norm = LogNorm(vmin=float(np.percentile(true_rho, 1)),
                       vmax=float(np.percentile(true_rho, 99)))

        fig = plt.figure(figsize=(15.0, 8.6))
        grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.62, wspace=0.14)

        for col, (row, rho, _predicted, _chi2) in enumerate(panels):
            ax = fig.add_subplot(grid[0, col])
            coll = _panel(ax, polygons, rho, norm, LABELS.get(row["arch"], row["arch"]))
        fig.colorbar(coll, ax=fig.axes, fraction=0.016, pad=0.02,
                     location="right").set_label("resistance of the ground")

        ax = fig.add_subplot(grid[1, :])
        n = np.arange(obs_log.size)
        ax.plot(n, np.exp(obs_log), "o", color="0.25", ms=9, zorder=3,
                label="what we measured")
        styles = [("-", "#0072B2", 3.6), ("--", "#D55E00", 3.0)]
        for (row, _rho, predicted, _chi2), (ls, color, lw) in zip(panels, styles):
            ax.plot(n, predicted, ls, color=color, lw=lw, zorder=4,
                    label=f"predicted by the {LABELS.get(row['arch'], row['arch'])}")
        ax.set_xlabel("measurement number")
        ax.set_ylabel("measured value")
        ax.set_title("Both pictures predict the same measurements", pad=10)
        # Legend below the axes: inside, it sits on top of the curves.
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=3)

        # The quantitative punch line. Stated on the figure so the slide carries
        # it even if the sentence is lost in delivery.
        ax.text(0.5, 1.30,
                f"the two predictions differ by {spread * 100:.2f}%  —  "
                f"smaller than the {DATA_STD * 100:.1f}% error in the measurements",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=17,
                color="#8B2500",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF3E0",
                          edgecolor="#D55E00", linewidth=1.6))

        fig.suptitle("Two different pictures of the ground.  One set of measurements.",
                     fontsize=23, y=0.98)
        path = RESULTS / "talk_nonuniqueness.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved {path}")
    finally:
        case["forward"].close()


# ---------------------------------------------------------------------------
# Figure 2 -- semi-convergence
# ---------------------------------------------------------------------------

def _draw_semiconvergence(chi2: np.ndarray, error: np.ndarray, stop: int,
                          degradation: float) -> None:
    """Render misfit and model error against iteration.

    Separate from the run so a saved history can be redrawn without repeating a
    multi-minute inversion.
    """

    plt = _style()
    fig, ax = plt.subplots(figsize=(13.0, 7.2))
    it = np.arange(chi2.size)

    ax.semilogy(it, chi2, color="#0072B2", lw=3.4, label="how well it fits the measurements")
    ax.set_xlabel("training step")
    ax.set_ylabel("misfit to the measurements\n(lower = fits better)", color="#0072B2")
    ax.tick_params(axis="y", colors="#0072B2")
    ax.axhline(1.0, color="0.55", ls=":", lw=2.0)
    ax.text(chi2.size * 0.985, 1.25, "noise level", ha="right", va="bottom",
            color="0.4", fontsize=13)

    ax2 = ax.twinx()
    ax2.plot(it, error, color="#D55E00", lw=3.4, label="how wrong the picture is")
    ax2.set_ylabel("error in the recovered picture\n(lower = closer to the truth)",
                   color="#D55E00")
    ax2.tick_params(axis="y", colors="#D55E00")
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)

    ax.axvline(stop, color="0.3", lw=2.2)
    ax.text(stop, chi2.max() * 0.5, "  stop here", color="0.2", fontsize=16,
            ha="left", va="top", fontweight="bold")
    ax2.plot([stop], [error[stop]], "o", color="#D55E00", ms=13, zorder=5)

    # Title and annotation state what the run actually did. An earlier version
    # asserted degradation unconditionally; at the re-tuned configuration there
    # is none, and a figure must not claim what its own data contradict.
    fit_gain = chi2[stop] / chi2[-1]
    if degradation >= 10.0:
        title = "Past this point it keeps fitting better — and keeps getting the ground wrong"
        note = f"{degradation:+.0f}% worse"
    else:
        title = (f"Fitting {fit_gain:.0f}x tighter than the noise level "
                 f"changed the recovered picture by {degradation:+.0f}%")
        note = "no degradation"
    ax2.annotate(note, xy=(error.size - 1, error[-1]),
                 xytext=(error.size * 0.62, error[-1]),
                 color="#8B2500", fontsize=17, va="center", ha="right",
                 arrowprops=dict(arrowstyle="->", color="#8B2500", lw=2.0))

    handles = [*ax.get_legend_handles_labels()[0], *ax2.get_legend_handles_labels()[0]]
    labels = [*ax.get_legend_handles_labels()[1], *ax2.get_legend_handles_labels()[1]]
    ax.legend(handles, labels, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.set_title(title, pad=14)

    path = RESULTS / "talk_semiconvergence.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {path}")


def semiconvergence(n_iters: int = 1500) -> None:
    """Data misfit falling while model error rises.

    This is the talk's takeaway as a picture, and it is the one claim currently
    made only in words. Model error is recorded through fit_inr's existing
    ``callback`` hook, so no library code changes -- important because train.py
    is shared with the Argon runs.

    Uses the RE-TUNED Fourier configuration (capacity_retune.json), not the
    Stage-1 one. The Stage-1 configuration was later shown to be badly sized,
    so drawing this curve from it would conflate over-fitting with
    misconfiguration -- and would invite exactly the objection the talk's own
    story raises. The claim here has to survive a well-configured method.

    Nothing selects a stopping point using the true model. The true model is
    used only to PLOT the error the field would never let you see; the stopping
    rule drawn on the figure (chi^2 = 1) reads data misfit alone.

    The quoted degradation is measured from the chi^2 = 1 snapshot to the end of
    the budget, NOT from the global minimum of the error curve. The minimum
    typically sits within the first few iterations, where the model is still
    essentially the homogeneous starting guess -- reporting that as "best" would
    imply stopping before the inversion has learned anything.
    """

    from capacity_sweep import size_kwargs

    best_cfg = json.loads((RESULTS / "capacity_retune.json").read_text(encoding="utf-8"))["best"]
    cfg = best_cfg["fourier"]
    hparams, lr, width = cfg["hparams"], float(cfg["lr"]), int(cfg["width"])
    print(f"fourier (re-tuned)  hparams={hparams}  lr={lr}  width={width}")
    print(f"target={TARGET}  seed={SEED}")
    print(f"running {n_iters} iterations -- expect several minutes\n")

    case = build_case(TARGET, SEED)
    try:
        true_log = np.log(case["true_rho"])
        mask = case["mask"]
        model_error: list[float] = []

        def track(_iteration: int, _chi2: float, _rms: float, log_rho: np.ndarray) -> None:
            error = log_rho - true_log
            model_error.append(float(np.sqrt(np.mean(error[mask] ** 2))))

        seed_networks(SEED)
        net = build_network("fourier", log_rho_mean=case["log_rho_mean"],
                            **size_kwargs("fourier", width), **hparams)
        result = fit_inr(
            case["forward"], net, case["coords"], case["obs_log"], case["data_std"],
            n_iters=int(n_iters), lr=lr, step_size=max(1, int(n_iters) // 4), gamma=GAMMA,
            snapshot_at_chi2=1.0, callback=track,
        )

        chi2 = np.asarray(result["chi2_history"], dtype=float)
        error = np.asarray(model_error, dtype=float)
        stop = result["snapshot_iteration"]
        if stop is None:
            raise SystemExit("run never reached chi^2 = 1; nothing honest to draw")
        degradation = (error[-1] / error[stop] - 1.0) * 100.0
        print(f"chi^2 = 1 reached at iteration {stop}")
        print(f"error there             : {error[stop]:.3f}")
        print(f"error at the end        : {error[-1]:.3f}  ({degradation:+.0f}%)")
        print(f"final chi^2             : {chi2[-1]:.2e}  "
              f"({chi2[stop] / chi2[-1]:.0f}x better fit than at the stop)")

        _draw_semiconvergence(chi2, error, stop, degradation)

        (RESULTS / "talk_semiconvergence.json").write_text(json.dumps(
            {"arch": "fourier", "config": "retuned", "hparams": hparams, "lr": lr,
             "width": width, "target": TARGET, "seed": SEED, "n_iters": int(n_iters),
             "stop_iteration": stop, "error_at_stop": error[stop],
             "error_at_end": error[-1], "degradation_percent": degradation,
             "chi2_history": chi2.tolist(), "model_error_history": error.tolist()},
            indent=2), encoding="utf-8")
    finally:
        case["forward"].close()


def replot() -> None:
    """Redraw the semi-convergence figure from its saved history.

    The inversion takes minutes; the histories are saved, so correcting a label
    should not cost a re-run.
    """

    data = json.loads((RESULTS / "talk_semiconvergence.json").read_text(encoding="utf-8"))
    _draw_semiconvergence(
        np.asarray(data["chi2_history"], dtype=float),
        np.asarray(data["model_error_history"], dtype=float),
        int(data["stop_iteration"]), float(data["degradation_percent"]))


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "all"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    if command == "replot":
        replot()
    else:
        if command in ("nonuniqueness", "all"):
            nonuniqueness()
        if command in ("semiconvergence", "all"):
            semiconvergence(iters)
