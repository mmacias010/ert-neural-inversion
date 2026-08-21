"""Physics-coupled optimization loop for INR-reparameterized ERT inversion.

The network is not pre-trained. Fitting *is* the inversion: weights start
random and the only gradient they ever receive comes from the data misfit of
one specific survey. Each iteration does

    coords -> f_theta -> log-resistivity per cell     (Torch, autograd live)
           -> detach to NumPy
           -> FEM forward + Jacobian                  (SciPy, no autograd)
           -> dPhi/dm handed back to .backward()      (Torch supplies dm/dtheta)

The chain rule is split by hand because ``forward_and_jacobian`` is NumPy and
Torch cannot differentiate through it.

RNG note: network initialization consumes the Torch RNG in construction order,
and ``nn.Linear`` draws before SIREN's ``uniform_`` overwrites the weight. Seed
with :func:`seed_networks` immediately before constructing the network, and do
not let anything else touch the Torch RNG in between, or results will shift for
reasons unrelated to the architecture under test.
"""

from __future__ import annotations

from collections.abc import Callable
import time

import numpy as np
import torch
import torch.nn as nn


def seed_networks(seed: int) -> None:
    """Seed the Torch RNG that governs network initialization.

    Deliberately leaves NumPy's global RNG alone. Observation noise is drawn
    from its own ``RandomState`` in :func:`deepert.inr.cases.synthetic_observations`
    so that "lucky initialization" and "lucky noise realization" are separate
    axes and can be varied independently across benchmark seeds.
    """

    torch.manual_seed(int(seed))


def fit_inr(
    forward,
    net: nn.Module,
    coords: np.ndarray,
    obs_log: np.ndarray,
    data_std: float | np.ndarray,
    *,
    n_iters: int = 500,
    lr: float = 3.0e-3,
    step_size: int = 150,
    gamma: float = 0.5,
    snapshot_at_chi2: float | None = None,
    callback: Callable[[int, float, float, np.ndarray], None] | None = None,
) -> dict:
    """Optimize network weights against the ERT data misfit.

    Parameters mirror the original demo defaults exactly. ``data_std`` may be a
    scalar or a per-measurement array (real data carries per-reading error
    estimates); it broadcasts elementwise either way.

    Returns a dict with the final model, the predicted data, and the full chi^2
    history. The history is what lets a later harness compare architectures at
    *matched chi^2* rather than only at a matched iteration budget -- different
    architectures converge at different rates, so a fixed budget silently
    rewards whichever is fastest rather than whichever represents the field best.

    No early stopping is applied. Any stopping rule must be driven by chi^2
    alone: selecting an iteration by lowest error against the true model would
    use ground truth that does not exist in the field and would inflate the
    apparent quality of every architecture.

    ``snapshot_at_chi2`` records (without stopping) the model at the first
    iteration where chi^2 falls to or below the given level. That supports
    comparing architectures at *matched data fit* rather than at a matched
    iteration count -- the honest way to separate "represents the field well"
    from "merely converges fast". The trigger reads chi^2 only, so it stays
    usable on field data where no true model exists.
    """

    device = next(net.parameters()).device
    coords_t = torch.as_tensor(np.asarray(coords, dtype=float), dtype=torch.float32, device=device)
    obs_log = np.asarray(obs_log, dtype=float).ravel()

    opt = torch.optim.Adam(net.parameters(), lr=float(lr))
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=int(step_size), gamma=float(gamma))

    chi2_history: list[float] = []
    rms_history: list[float] = []
    log_rho_np = np.empty(0)
    predicted_log = np.empty(0)
    snapshot_log_rho: np.ndarray | None = None
    snapshot_iteration: int | None = None

    t0 = time.time()
    for iteration in range(int(n_iters) + 1):
        log_rho = net(coords_t)                                   # (n_cells,), grad live
        log_rho_np = log_rho.detach().cpu().numpy().astype(float)
        predicted_log, jacobian = forward.forward_and_jacobian(log_rho_np, log_transform=True)

        misfit = predicted_log - obs_log
        chi2 = float(np.mean((misfit / data_std) ** 2))
        rms = float(np.sqrt(np.mean(np.expm1(misfit) ** 2)) * 100.0)
        chi2_history.append(chi2)
        rms_history.append(rms)
        if snapshot_at_chi2 is not None and snapshot_log_rho is None and chi2 <= float(snapshot_at_chi2):
            snapshot_log_rho = log_rho_np.copy()
            snapshot_iteration = iteration
        if callback is not None:
            callback(iteration, chi2, rms, log_rho_np)

        if iteration == int(n_iters):
            break

        grad_m = jacobian.T @ (misfit / data_std**2)              # dPhi/dm at the cells
        opt.zero_grad()
        log_rho.backward(gradient=torch.as_tensor(grad_m, dtype=torch.float32, device=device))
        opt.step()
        scheduler.step()

    elapsed = time.time() - t0

    return {
        "log_resistivity": log_rho_np,
        "resistivity": np.exp(log_rho_np),
        "predicted_log_data": predicted_log,
        "chi2": chi2_history[-1],
        "rms": rms_history[-1],
        "chi2_history": np.asarray(chi2_history, dtype=float),
        "rms_history": np.asarray(rms_history, dtype=float),
        "elapsed": elapsed,
        "n_iters": int(n_iters),
        "snapshot_log_resistivity": snapshot_log_rho,
        "snapshot_iteration": snapshot_iteration,
    }
