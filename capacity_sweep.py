"""Does the architecture ranking depend on network capacity?

Stage 1 matched every neural method to ~33,537 trainable parameters, but that
number was inherited from whatever the original SIREN demo happened to use
(hidden=128, layers=3). Nobody validated it. That leaves a fairness asymmetry
a reviewer can state in one sentence: the conventional methods had their
regularization strength swept over five lambda values, while the neural
methods had theirs -- capacity, which plays the same role in controlling
effective degrees of freedom -- frozen at an arbitrary value.

This sweeps capacity over four budgets (~8k, 33k, 133k, 527k parameters) at
each architecture's FROZEN hyperparameters, and asks three questions:

  flat past 33k     capacity is not the limiter; Stage-1 conclusions are
                    robust and can be defended with evidence.
  still descending  every neural method is starved and Stage 1 understates
                    them; the headline needs softening.
  curves cross      the ranking is capacity-dependent, so "A beats B" is a
                    statement about one arbitrary size rather than about
                    architectures. This is the outcome that would damage the
                    current write-up.

It also probes finding #3 directly. The networks currently generate 20-70%
excess structure (contrast ratios 1.19-1.72 against TV's 0.85). If that is
over-flexibility, smaller networks should suppress it and may score BETTER --
capacity acting as the neural analogue of lambda. The contrast ratio is
therefore recorded alongside model error at every budget.

Hyperparameters are held frozen rather than re-tuned per budget. That is the
cheap first pass: it answers "do the curves cross at all" for 1/10th the runs.
If they do cross, re-tuning per capacity becomes necessary before drawing
conclusions -- a config chosen at 33k need not suit 527k.

    python capacity_sweep.py [n_jobs]
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
    GAMMA,
    HELD_OUT_TARGETS,
    N_ITERS,
    RESULTS,
    STEP_SIZE,
    build_case,
    mean_std,
    model_errors,
)
from deepert.inr import build_network, count_parameters, fit_inr, seed_networks
from structure_metrics import cells_to_image, sensed_rows

torch.set_num_threads(1)

WIDTHS = (64, 128, 256, 512)          # hidden width for the MLP-style trunks
SEEDS = (10, 11, 12, 13, 14)
MLP_ARCHS = ("relu", "tanh", "siren")  # sized directly by hidden width
BUDGET_ARCHS = ("fourier", "dip")      # sized by matcher to the same budget


def budget_for(width: int) -> int:
    """Trainable parameters of a 2 -> width -> width -> width -> 1 MLP."""

    return (2 * width + width) + 2 * (width * width + width) + (width + 1)


def size_kwargs(arch: str, width: int) -> dict:
    """How each architecture is asked for a given capacity."""

    if arch in MLP_ARCHS:
        return {"hidden": width}
    return {"parameter_budget": budget_for(width)}


def run_one(arch: str, hparams: dict, lr: float, width: int, target: str, seed: int) -> dict:
    torch.set_num_threads(1)
    base = {"arch": arch, "width": width, "budget": budget_for(width),
            "target": target, "seed": seed}
    case = build_case(target, seed)
    try:
        seed_networks(seed)
        net = build_network(arch, log_rho_mean=case["log_rho_mean"],
                            **size_kwargs(arch, width), **hparams)
        n_params = count_parameters(net)
        result = fit_inr(
            case["forward"], net, case["coords"], case["obs_log"], case["data_std"],
            n_iters=N_ITERS, lr=lr, step_size=STEP_SIZE, gamma=GAMMA,
            snapshot_at_chi2=CHI2_TARGET,
        )
        snapshot = result["snapshot_log_resistivity"]
        rmse = contrast = None
        if snapshot is not None:
            recovered = np.exp(snapshot)
            _, rmse = model_errors(recovered, case["true_rho"], case["mask"])
            # Contrast ratio over the sensed zone: >1 means excess structure.
            rows = sensed_rows(case["mask"])
            rec_log = np.log10(cells_to_image(recovered))[rows]
            true_log = np.log10(cells_to_image(case["true_rho"]))[rows]
            contrast = float(rec_log.std() / true_log.std())
        return {**base, "status": "ok", "n_params": n_params,
                "rmse_at_chi2": rmse, "contrast": contrast,
                "chi2": result["chi2"], "iters_to_chi2": result["snapshot_iteration"],
                "elapsed": result["elapsed"]}
    except Exception as error:
        return {**base, "status": "diverged", "error": f"{type(error).__name__}: {error}",
                "n_params": None, "rmse_at_chi2": None, "contrast": None,
                "chi2": None, "iters_to_chi2": None, "elapsed": None}
    finally:
        case["forward"].close()


def crossing_report(rows: list[dict], target: str) -> None:
    """Does the ranking change with capacity? That is the decisive question."""

    order_by_width: dict[int, list[str]] = {}
    for width in WIDTHS:
        scored = []
        for arch in ARCH_GRID:
            group = [r for r in rows if r["target"] == target and r["arch"] == arch
                     and r["width"] == width and r["rmse_at_chi2"] is not None]
            if group:
                scored.append((float(np.mean([r["rmse_at_chi2"] for r in group])), arch))
        order_by_width[width] = [a for _, a in sorted(scored)]

    print(f"\n  ranking by capacity ({target}), best first:")
    for width, order in order_by_width.items():
        print(f"    {budget_for(width):>7,} params: {' > '.join(order) if order else '(none fit)'}")

    populated = [o for o in order_by_width.values() if o]
    if len(populated) < 2:
        print("    -> too few capacities produced fits to judge crossing")
        return
    reference = populated[0]
    if all(o == reference for o in populated):
        print("    -> RANKING STABLE across capacity: Stage-1 conclusions are capacity-robust")
    else:
        print("    -> RANKING CHANGES with capacity: Stage-1 comparisons hold only at 33.5k;")
        print("       hyperparameters must be re-tuned per capacity before concluding")


def main() -> int:
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, (os.cpu_count() or 2) - 1)
    frozen = json.loads((RESULTS / "tuning.json").read_text(encoding="utf-8"))["best"]

    print(f"CAPACITY SWEEP - widths {WIDTHS} "
          f"(~{', '.join(f'{budget_for(w) // 1000}k' for w in WIDTHS)} parameters)")
    print(f"frozen hyperparameters, seeds {SEEDS}, targets {HELD_OUT_TARGETS}\n")

    jobs = [(arch, frozen[arch]["hparams"], frozen[arch]["lr"], width, target, seed)
            for target in HELD_OUT_TARGETS
            for arch in ARCH_GRID if arch in frozen
            for width in WIDTHS
            for seed in SEEDS]
    print(f"{len(jobs)} runs on {n_jobs} workers\n")

    t0 = time.time()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = [pool.submit(run_one, *job) for job in jobs]
        for index, future in enumerate(futures, 1):
            rows.append(future.result())
            if index % max(1, len(jobs) // 20) == 0 or index == len(jobs):
                print(f"    {index}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"  {len(jobs)} runs in {time.time() - t0:.0f}s\n")

    for target in HELD_OUT_TARGETS:
        print(f"=== {target} " + "=" * 60)
        header = f"{'arch':9s}" + "".join(f"{budget_for(w) // 1000:>7d}k" for w in WIDTHS)
        print("model error at chi^2 = 1 (mean over seeds; 'x' = no seed fit)")
        print(header)
        print("-" * len(header))
        for arch in ARCH_GRID:
            line = f"{arch:9s}"
            for width in WIDTHS:
                group = [r for r in rows if r["target"] == target and r["arch"] == arch
                         and r["width"] == width and r["rmse_at_chi2"] is not None]
                line += f"{np.mean([r['rmse_at_chi2'] for r in group]):>8.3f}" if group else f"{'x':>8s}"
            print(line)

        print("\ncontrast ratio (1.0 = correct amplitude; >1 = excess structure)")
        print(header)
        print("-" * len(header))
        for arch in ARCH_GRID:
            line = f"{arch:9s}"
            for width in WIDTHS:
                group = [r for r in rows if r["target"] == target and r["arch"] == arch
                         and r["width"] == width and r["contrast"] is not None]
                line += f"{np.mean([r['contrast'] for r in group]):>8.3f}" if group else f"{'x':>8s}"
            print(line)

        print("\nseeds reaching chi^2 = 1, of 5")
        print(header)
        print("-" * len(header))
        for arch in ARCH_GRID:
            line = f"{arch:9s}"
            for width in WIDTHS:
                group = [r for r in rows if r["target"] == target and r["arch"] == arch
                         and r["width"] == width]
                fit = sum(1 for r in group if r["rmse_at_chi2"] is not None)
                line += f"{fit:>8d}" if group else f"{'-':>8s}"
            print(line)

        crossing_report(rows, target)
        print()

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "capacity_sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved {RESULTS / 'capacity_sweep.json'}")
    plot(rows)
    return 0


def plot(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    archs = [a for a in ARCH_GRID if any(r["arch"] == a for r in rows)]
    fig, axes = plt.subplots(1, len(HELD_OUT_TARGETS), figsize=(6.2 * len(HELD_OUT_TARGETS), 4.6),
                             squeeze=False)
    for column, target in enumerate(HELD_OUT_TARGETS):
        ax = axes[0][column]
        for arch in archs:
            xs, ys, es = [], [], []
            for width in WIDTHS:
                group = [r["rmse_at_chi2"] for r in rows
                         if r["target"] == target and r["arch"] == arch
                         and r["width"] == width and r["rmse_at_chi2"] is not None]
                if group:
                    mean, sd = mean_std(group)
                    xs.append(budget_for(width))
                    ys.append(mean)
                    es.append(sd)
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=arch, lw=1.6)
        ax.axvline(33537, color="0.6", ls="--", lw=1)
        ax.annotate("Stage-1\nbudget", xy=(33537, ax.get_ylim()[1]), fontsize=8,
                    color="0.4", ha="center", va="top")
        ax.set_xscale("log")
        ax.set_xlabel("trainable parameters")
        ax.set_ylabel("model error at chi^2 = 1" if column == 0 else "")
        ax.set_title(target)
        ax.grid(alpha=0.25)
        if column == 0:
            ax.legend(fontsize=8)
    path = RESULTS / "capacity_sweep.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    raise SystemExit(main())
