# ERT inversion benchmark — method taxonomy and Stage-1 protocol

Organization follows the three-group taxonomy defined by Hang (July 2026).
The grouping criterion is **what the regularization is and whether a training
database is required** — the two properties that determine whether methods can
share an evaluation protocol.

All Stage-1 methods share: the same 2.5D FEM forward solver (never replaced),
the same synthetic models, survey geometry (48-electrode-ready; currently
16-electrode Wenner, 35 measurements), noise level (1.5% multiplicative),
starting field (homogeneous at the median observed apparent resistivity),
optimization budget (1500 iterations for neural arms; Gauss-Newton stops at
the discrepancy level), and evaluation metrics. Primary accuracy metric:
coverage-masked log-RMSE at matched data fit (chi^2 = 1).

## Group 1 — Conventional model parameterizations

One resistivity value per mesh cell, optimized by Gauss-Newton; explicit
regularization; no training database; output resolution fixed by the mesh.

| Method | Regularizer | Status |
|---|---|---|
| Smooth L2 inversion | first-order smoothness, weight lambda | **done** (frozen lambda=0.01) |
| Total-variation inversion | edge-preserving TV, weight lambda | **done** (frozen lambda=0.1) |

## Group 2 — Neural parameterizations optimized per dataset

No pretraining database. Network weights are the inversion unknowns, fit from
scratch against one dataset's physics misfit (Deep Image Prior mechanism).
Regularization is *implicit* — architecture, activation, frequency encoding,
and initialization decide which data-consistent model is recovered. All are
capacity-matched to ~33.5k trainable parameters (within 0.5%).

| Method | Representation | Knobs | Status |
|---|---|---|---|
| ReLU INR | coords -> rho, ReLU MLP | lr | **done** |
| tanh INR | coords -> rho, tanh MLP | lr | running |
| Fourier-feature INR | random fixed B encoding + ReLU MLP | sigma, lr | **done** |
| SIREN | sine activations, w0 scaling | w0, lr | **done** |
| CNN Deep Image Prior | conv decoder from fixed random latent | lr | running |

INR vs DIP distinction (per Hang): an INR maps coordinates to resistivity,
(x,z) -> rho(x,z), so the field is continuous and can be resampled at any
resolution. The CNN DIP generates the whole image from a fixed latent input on
a **fixed output grid** and ignores coordinates; its prior comes from
convolutional locality and upsampling rather than a coordinate encoding.

## Group 3 — Pretrained deep-learning frameworks (future stage)

Require a database of (measurements, model) pairs; after training they map
data to a model directly or estimate a conditional distribution.

- CNN / U-Net direct inversion
- LSTM / ConvLSTM — time-lapse or sequential ERT only, not static surveys
- Transformer / CNN-Transformer
- Conditional invertible neural networks / normalizing flows — primarily for
  representing nonuniqueness and posterior uncertainty, not a single
  deterministic model
- PINN and physics-guided frameworks — note PINN is a *training framework*
  (physics terms in the loss: forward equation, boundary conditions, data
  misfit), not an architecture in the CNN/LSTM sense

Blocking infrastructure for Group 3: a training-pair generator (the 365
ParFlow slices forward-modelled with augmentation), site-held-out splits, and
a decision on variable survey geometry (fixed-grid CNNs cannot transfer
between electrode layouts; set/graph models can).

## Implementation differences

| Method | Model representation | Training database | Output resolution |
|---|---|---|---|
| Conventional (L2, TV) | resistivity per mesh cell | no | fixed by inversion mesh |
| ReLU / tanh / SIREN / Fourier INR | mapping coords -> resistivity | no | continuous; resamplable at any resolution |
| CNN Deep Image Prior | conv decoder generates full model | no | fixed grid (20 x 25 here) |
| CNN / U-Net direct inversion | mapping data -> resistivity image | usually required | usually fixed |
| Invertible NN / flows | conditional posterior over models | usually required | implementation-dependent |

## Stage-1 method summary 

| Method | Input | Output | Parameterization | Pretraining | Loss | Regularization mechanism | Output resolution | Metrics |
|---|---|---|---|---|---|---|---|---|
| Smooth L2 | apparent resistivities | cell resistivities | per-cell values | no | weighted log-L2 misfit + lambda*||L m||^2 | explicit first-order smoothness | mesh | all below |
| TV | apparent resistivities | cell resistivities | per-cell values | no | weighted log-L2 misfit + lambda*TV(m) | explicit edge-preserving TV | mesh | all below |
| ReLU INR | coords (+ data misfit gradient) | continuous log-rho field | MLP weights (33,537) | no | weighted log-L2 misfit only | implicit: spectral bias of ReLU | continuous | all below |
| tanh INR | coords (+ data misfit gradient) | continuous log-rho field | MLP weights (33,537) | no | weighted log-L2 misfit only | implicit: smooth saturating activation | continuous | all below |
| Fourier INR | encoded coords | continuous log-rho field | MLP weights (33,601), B fixed | no | weighted log-L2 misfit only | implicit: frequency content set by sigma | continuous | all below |
| SIREN | coords | continuous log-rho field | MLP weights (33,537) | no | weighted log-L2 misfit only | implicit: sine basis, band set by w0 | continuous | all below |
| CNN DIP | fixed random latent | full model image | conv weights (33,703), latent fixed | no | weighted log-L2 misfit only | implicit: conv locality + upsampling pyramid | fixed 20x25 grid | all below |

Shared metric set: data misfit (chi^2, RMS%), model log-RMSE (full and
coverage-masked, at chi^2=1 and at budget), anomaly geometry (IoU, centroid
error), recovered magnitude (fraction of true log-contrast), structural
similarity (SSIM over the sensed zone), runtime and iterations-to-fit,
sensitivity to initialization (init seed varied at fixed noise), robustness to
noise (noise level and realization varied at frozen configuration).

Guiding caution, verified repeatedly in this benchmark: **a low data misfit
does not indicate that the true model has been recovered.** The clearest
instance so far: the best-fitting configurations (chi^2 ~ 0.03-0.2) carried
the worst model errors, and ReLU reached chi^2 = 1 while misplacing anomalies
by ~17 m.

## Metric status ledger (Stage 1) — COMPLETE

| Metric | Status |
|---|---|
| Data misfit, model error, runtime | done, all 7 methods |
| Anomaly geometry, magnitude, SSIM | done, all 7 methods (`structure_metrics.py`) |
| Sensitivity to initialization | done — 5 init seeds at fixed noise (`sensitivity.py init`) |
| Robustness to noise | done — {0.5%, 1.5%, 5%} x 3 realizations (`sensitivity.py noise`) |

**Full results: [stage1_results.md](stage1_results.md)** — five tables
(accuracy, geometry, initialization sensitivity, noise robustness, cost),
eight findings, and limitations.

Headline: TV leads on every axis; the decisive variable is whether the prior is
edge-preserving or smooth, not whether it is neural — smooth L2 sits between TV
and the networks on geometry and is the least numerically stable method tested.
Networks do not over-smooth, they misplace structure (contrast ratios 1.2-1.7
versus TV's 0.85). Initialization alone moves neural results up to 18% while
conventional inversion is exactly reproducible. tanh failed to fit the data at
realistic noise levels; ReLU fits inconsistently and is reliably wrong when it
does. Warm-starting a network from the conventional solution halves its error —
the useful direction is conventional to neural, not the reverse.
