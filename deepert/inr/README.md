# INR reparameterization of ERT inversion — benchmark

Neural reparameterization of 2.5D ERT inversion: a controlled comparison of
three implicit neural representations against each other, against classical
cell-based inversion, and in hybrid combination with it.

**This document reflects the CORRECTED protocol** (1500-iteration budget,
matched-chi^2 primary metric, snapshot-based selection). Results from the
earlier 600-iteration runs were budget-confounded and are superseded.

## The idea

Classical ERT inversion optimizes one resistivity value per mesh cell against
an explicit regularizer:

```
min_m  ||G(m) - d_obs||²_W  +  λ R(m)
```

The reparameterization replaces `m` with a coordinate network — resistivity
becomes a continuous function of position, and the inversion optimizes network
weights:

```
min_θ  ||G(f_θ(x, z)) - d_obs||²_W
```

The FEM forward solver `G` is **unchanged**, and there is no explicit `λ R`
term: regularization is implicit in the architecture's inductive bias.

**The network is never pre-trained.** Fitting *is* the inversion — weights
start random (or warm-started, see Hybrid) and the only gradient they receive
comes from one dataset's physics misfit. This is the Deep Image Prior mechanism
with coordinate networks.

## Module map

| File | Contents |
|---|---|
| `networks.py` | `CoordinateNetwork` base, `SIREN`, `ReLUMLP`, `FourierFeatureMLP`, capacity matching, registry |
| `train.py` | `fit_inr` — physics-coupled loop with chi^2=1 snapshotting; `seed_networks` |
| `coords.py` | Cell centers, per-axis normalization to [-1, 1] |
| `targets.py` | True fields (`two_layer`, `block_anomaly`, `parflow_slice`) and `coverage_mask` |
| `cases.py` | Mesh, survey, synthetic observations |

Root-level drivers: `step0_regression.py` (refactor regression),
`inr_benchmark.py` (tune/eval/oracle), `traditional_comparison.py` (classical
arm), `convergence_check.py` (budget probe), `warmstart_hybrid.py` (hybrid),
`reanalysis.py` (matched-chi^2 re-analysis of saved runs).

## How the gradient works

`forward_and_jacobian` is NumPy/SciPy; Torch cannot differentiate through it,
so the chain rule is split by hand:

```python
log_rho = net(coords)                      # Torch, autograd live
m_np    = log_rho.detach().cpu().numpy()   # leave the graph
d_pred, J = forward.forward_and_jacobian(m_np)   # SciPy FEM solve
grad_m  = J.T @ ((d_pred - obs_log) / data_std**2)   # dPhi/dm
log_rho.backward(gradient=grad_m)          # Torch supplies dm/dtheta
```

Any module mapping `(n_cells, 2) -> (n_cells,)` drops in with no solver
changes.

## Architectures

All share the contract `(n_cells, 2) coords in [-1,1] -> (n_cells,) log rho`
and the same output offset (median observed apparent resistivity — measured
data only).

| Architecture | Cure for spectral bias | Trainable params | Knobs |
|---|---|---:|---|
| `ReLUMLP` | none (baseline) | 33,537 | lr |
| `SIREN` | sine activations | 33,537 | `w0`, lr |
| `FourierFeatureMLP` | random Fourier input encoding, fixed **B** | 33,601 | `σ`, lr |

Fourier's trunk is auto-narrowed to width 80 (`match_hidden_width`) because
its 256-wide encoded input would otherwise double its capacity. Note the INR is
33× *over*-parameterized relative to the 1000-cell mesh — this is not
dimensionality reduction in 2D.

## Corrected experimental protocol

Fixed for every method: 25×20 mesh (1000 cells), 16-electrode Wenner survey
(35 measurements), 1.5% noise, `data_std=0.015`, Adam + StepLR, same starting
field.

**Budget: 1500 iterations**, set by a convergence probe
(`inr_results/convergence.json`): the slowest architecture reached chi^2 = 1 at
~1187 iterations, so 1500 ensures no architecture fails merely for lack of
budget. (The earlier 600-iteration budget cut off ReLU and SIREN mid-descent —
their old "never converged" results were artifacts.)

**Primary metric: coverage-masked log-RMSE at the chi^2 = 1 snapshot** — the
first iterate whose data fit reaches the noise level. Classical Gauss-Newton
stops there by construction (discrepancy principle); the networks are reported
the same way. The probe measured 17–49% model degradation from optimizing past
the noise floor, so end-of-budget numbers are secondary by design.

**Selection: tune on `two_layer` (seeds 0–2), freeze, evaluate on held-out
`blocks` and `parflow` (seeds 10–14).** A configuration is eligible only if
every seed reaches chi^2 <= 1, and is scored on the snapshot model. Frozen
configs: relu lr=0.003; siren w0=3, lr=0.01; fourier σ=0.5, lr=0.001;
traditional TV λ=0.1.

**Oracle bound:** per-target re-tuning under the same eligibility rule —
selects against ground truth, so it is a diagnostic upper bound, never a
usable method.

## Results (chi^2 = 1 snapshots, mean ± sd over 5 evaluation seeds)

### `blocks` (held out) — sharp rectangular anomalies

| method | transfer (dev-tuned) | oracle (target-tuned) | cost of transfer |
|---|---|---|---|
| **traditional** (TV) | **0.3620 ± 0.0392** | 0.4065 ± 0.0426 | −0.045 |
| siren | 0.6849 ± 0.0636 | 0.5845 ± 0.0404 (w0=5) | +0.100 |
| fourier | 0.7438 ± 0.1465 | **0.4053 ± 0.0325** (σ=1.0) | **+0.338** |
| relu | 1.0897 ± 0.0880 | no configuration fit the data | — |

### `parflow` (held out) — ParFlow-derived hillslope slice

| method | transfer | oracle | cost |
|---|---|---|---|
| **traditional** | **0.5468 ± 0.0188** | 0.4423 ± 0.0253 | +0.105 |
| fourier | 0.6173 ± 0.0275 | **0.5211 ± 0.0856** (σ=2.0) | +0.096 |
| siren | 0.6566 ± 0.0656 | 0.6566 (w0=3 was already optimal) | +0.000 |
| relu | 0.7220 ± 0.0737 | no configuration fit the data | — |

### `two_layer` (development — all methods tuned here, optimistically biased)

traditional 0.5790 ± 0.1354 · fourier 0.6242 ± 0.0697 · relu 0.6696 ± 0.0208 ·
siren 0.8709 ± 0.0633

### Convergence (iterations to chi^2 = 1, held-out targets)

siren 67–114 · fourier 79–161 · relu 588–1318 (and ineligible at oracle level:
on tuning seeds not all runs fit within 1500; one parflow run diverged to
non-finite resistivity in the 3000-iteration probe)

### Hybrid: classical TV -> neural projection -> physics refinement

| target | arch | classical | projected | hybrid | scratch INR |
|---|---|---|---|---|---|
| blocks | fourier | 0.3620 | 0.3632 | **0.3687 ± 0.0415** | 0.7438 |
| blocks | siren | 0.3620 | 0.3579 | 0.5874 ± 0.0486 | 0.6849 |
| parflow | fourier | 0.5468 | 0.5468 | **0.5468 ± 0.0187** | 0.6173 |
| parflow | siren | 0.5468 | 0.5428 | 0.6660 ± 0.0416 | 0.6566 |

## Findings

1. **Hyperparameter transfer cost exceeds architecture differences.** Tuning
   frequency content (σ, w0) on a smooth development target and applying it to
   sharp structure cost Fourier +0.34 on `blocks` — larger than the gap between
   any two architectures at their per-target best (0.41–0.58). Two different
   selection protocols produced two different victims (SIREN under end-of-budget
   scoring, Fourier under snapshot scoring): the selection, not the
   architecture, dominates. Classical λ-transfer cost only ±0.05–0.10 on the
   same tests. Field data offer no tuning target, so this is the deployment
   risk for neural parameterizations.

2. **Stopping at the data-noise level is essential.** With no explicit
   regularizer, optimizing past chi^2 = 1 fits noise and degraded recovered
   models by 17–49% (semi-convergence). All neural results here are therefore
   reported at the discrepancy-principle stopping point — which classical
   Gauss-Newton applies by construction.

3. **The plain ReLU MLP fails architecturally.** It converges ~10× slower than
   frequency-capable networks, could not fit the data on all seeds within 1500
   iterations at any learning rate, showed divergence instability at extended
   budgets, and where it did fit (`blocks`, eval seeds, ~1318 iters) the
   recovered model was still wrong (1.09): in an underdetermined problem
   (35 data, 1000 cells), fitting the data is not recovering the model —
   the architecture's bias chooses which data-consistent model you get.

4. **Classical TV remains the accuracy and robustness anchor in 2D static
   inversion.** It leads or ties every neural variant at matched data fit on
   both held-out targets, converges in 3–6 Gauss-Newton iterations (~1–2 s vs
   ~1–6 min), and its regularization weight transfers across target types far
   more cheaply than neural frequency settings. At oracle (best-case) tuning
   the families are comparable (blocks 0.362 vs 0.405; parflow 0.442–0.547 vs
   0.521) — the classical advantage in practice is robustness, not ceiling.

5. **Help flows classical -> neural, not neural -> classical (in 2D static).**
   Neural refinement of the classical solution never improved it (Fourier
   preserved it; SIREN's incompatible prior degraded it). But warm-starting the
   INR from the classical solution halved its from-scratch error (blocks:
   0.74 -> 0.37), delivering classical-grade accuracy *in continuous neural
   form* — the natural initialization for time-lapse f_θ(x, z, t), where the
   parameterization's per-timestep compression argument actually favors it.

## Limitations

- **Inverse crime**: data generated with the same mesh/solver used to invert.
  Identical bias for all methods (fair for ranking), but these are not
  absolute-accuracy estimates. Fix: finer forward mesh (planned).
- **Seed streams**: init and noise seeds exist independently in the code but
  reported runs used the same integer for both; sensitivity to initialization
  vs noise realization remains confounded.
- **Narrow base**: two held-out targets, one mesh/survey/noise level, five
  seeds, unpaired statistics.
- **Oracle caveats**: searched on seeds 0–2 then confirmed on 10–14; small
  search-seed count leaves winner's-curse risk (traditional's negative
  "cost of transfer" on blocks is that effect surfacing).
- **Hybrid tested two architectures** at frozen (transfer) configs only; a
  target-tuned hybrid was not run.
- Survey senses ~20 m; deeper structure is unrecoverable by any method.

## Reproducing

CPU only (the cost is a single-threaded SciPy sparse solve; GPUs do not help).
Workers pin one thread each — without this, parallel workers oversubscribe
cores and run slower than serial.

```bash
.venv/Scripts/python.exe step0_regression.py            # refactor regression, ~30 s
.venv/Scripts/python.exe inr_benchmark.py smoke  6      # sanity, ~2 min
.venv/Scripts/python.exe inr_benchmark.py tune   9      # 99 runs, ~55 min
.venv/Scripts/python.exe inr_benchmark.py eval   9      # 45 runs, ~28 min
.venv/Scripts/python.exe inr_benchmark.py oracle 9      # 228 runs, ~2.5 h
.venv/Scripts/python.exe traditional_comparison.py 9    # 160 runs, ~2 min
.venv/Scripts/python.exe convergence_check.py 9 3000    # 18 runs, ~22 min
.venv/Scripts/python.exe warmstart_hybrid.py 9          # 20 runs, ~11 min
.venv/Scripts/python.exe reanalysis.py                  # re-analysis, seconds
```

Outputs in `inr_results/`: `tuning.json`, `evaluation.json`, `oracle.json`,
`traditional.json`, `convergence.json`, `hybrid.json`, `inr_comparison.png`.

## References

- Sitzmann et al. (2020), *Implicit Neural Representations with Periodic
  Activation Functions* — SIREN.
- Tancik et al. (2020), *Fourier Features Let Networks Learn High Frequency
  Functions in Low Dimensional Domains* — fixed **B** ~ N(0, σ²) encoding.
- Ulyanov et al. (2018), *Deep Image Prior* — architecture as implicit
  regularizer, fit per instance.
