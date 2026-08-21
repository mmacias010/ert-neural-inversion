# Stage-1 benchmark results — conventional vs single-dataset neural parameterizations

> **Superseded in part by `stage1b_results.md`.** Stage 1 held network capacity,
> iteration budget, and learning-rate schedule fixed across architectures. Six
> conclusions in this document turned out to measure those shared settings
> rather than the architectures. The most serious is Finding 7 (tanh judged
> incapable — retracted). Numbers here are kept as the record of what the shared
> protocol produced; read them alongside Stage 1b.

Complete results for Groups 1 and 2 of the ERT inversion benchmark. Every
method uses the same 2.5D FEM forward solver (never replaced), synthetic
models, survey geometry, noise level, starting field, optimization budget, and
evaluation metrics.

**Protocol.** 25x20 mesh (1000 cells), 16-electrode Wenner survey (35
measurements), 1.5% multiplicative noise unless stated, homogeneous start at
the median observed apparent resistivity. Neural methods: 1500 Adam iterations
with StepLR(375, 0.5) — budget set by a convergence probe so no method fails
merely for lack of iterations. Conventional: Gauss-Newton, stopping at the
discrepancy level. Hyperparameters tuned on one development target
(`two_layer`), frozen, then evaluated on held-out targets (`blocks`,
`parflow`) with fresh seeds. **Primary metric: coverage-masked log-RMSE at
matched data fit (chi^2 = 1)**, since a fixed iteration budget runs
fast-converging architectures far past their stopping point (Fourier reaches
chi^2 = 1 at ~116 iterations against a 1500-iteration budget, ReLU at ~1048)
and models degrade in proportion to that overshoot -- see `overfit_check.py`.

**Frozen configurations.** Smooth L2 lambda=0.01 · TV lambda=0.1 · ReLU
lr=0.003 · tanh lr=0.01 · SIREN w0=3, lr=0.01 · Fourier sigma=0.5, lr=0.001 ·
CNN-DIP lr=0.001. Capacity matched: 33,537 / 33,537 / 33,537 / 33,601 / 33,703
trainable parameters (within 0.5%).

---

## Table 1 — Model accuracy at matched data fit

Coverage-masked log-RMSE, mean ± sd over 5 evaluation seeds. Lower is better.

| Method | Group | `blocks` | `parflow` | `two_layer` (dev) |
|---|---|---|---|---|
| **TV** | 1 | **0.3620 ± 0.0392** | **0.5468 ± 0.0188** | 0.5790 ± 0.1354 |
| Smooth L2 | 1 | 0.5926 ± 0.1103 † | 0.6187 ± 0.0733 † | 0.6014 ± 0.0750 |
| SIREN | 2 | 0.6849 ± 0.0636 | 0.6566 ± 0.0656 | 0.8709 ± 0.0633 |
| Fourier | 2 | 0.7438 ± 0.1465 | **0.6173 ± 0.0275** | 0.6242 ± 0.0697 |
| CNN-DIP | 2 | 1.0031 ± 0.1405 | 0.7442 ± 0.0579 | 0.7389 ± 0.0274 |
| ReLU | 2 | 1.0897 ± 0.0880 | 0.7220 ± 0.0737 | 0.6696 ± 0.0208 |
| tanh | 2 | **never reached chi^2 = 1** | 0.8218 ± 0.0092 | 0.6871 ± 0.0000 |

† Smooth L2 excludes runs that diverged (see Table 4); statistics are over
surviving realizations only.

## Table 2 — Anomaly geometry, magnitude, and structural similarity

`blocks` target, where anomalies are discrete. SSIM and contrast also shown for
`parflow`. IoU 1.0 = perfect anomaly footprint; magnitude 1.0 = full
log-contrast recovered; contrast 1.0 = correct heterogeneity amplitude.

| Method | SSIM | IoU | centroid err (m) | magnitude | contrast |
|---|---|---|---|---|---|
| **TV** | **0.729 ± 0.062** | **0.826 ± 0.090** | **1.7 ± 1.1** | 0.803 ± 0.058 | 0.845 ± 0.035 |
| Smooth L2 | 0.435 ± 0.036 | 0.476 ± 0.119 | 9.9 ± 4.4 | 0.860 ± 0.047 | 1.115 ± 0.122 |
| Fourier | 0.351 ± 0.070 | 0.346 ± 0.087 | 9.7 ± 2.7 | 0.826 ± 0.060 | 1.331 ± 0.208 |
| SIREN | 0.305 ± 0.021 | 0.362 ± 0.058 | 12.2 ± 2.0 | 0.750 ± 0.052 | 1.192 ± 0.108 |
| CNN-DIP | 0.272 ± 0.027 | 0.270 ± 0.024 | 14.6 ± 1.5 | **0.935 ± 0.027** | 1.690 ± 0.139 |
| ReLU | 0.228 ± 0.028 | 0.202 ± 0.015 | 17.3 ± 0.8 | 0.761 ± 0.078 | 1.720 ± 0.133 |
| tanh | — no runs fit the data — | | | | |

`parflow` SSIM: TV 0.428 · smooth L2 0.329 · ReLU 0.313 · Fourier 0.293 ·
tanh 0.260 · CNN-DIP 0.225 · SIREN 0.199.

## Table 3 — Sensitivity to initialization

Identical data (noise seed fixed at 10); only the network initialization seed
varies across 5 runs. CV% = sd/mean, the run-to-run spread relative to level.
**Conventional inversion is deterministic given the data, so its CV is exactly
zero** — that contrast is the result.

| Method | `blocks` rmse | CV% | fit | `parflow` rmse | CV% | fit |
|---|---|---|---|---|---|---|
| **TV / Smooth L2** | — | **0.0** | — | — | **0.0** | — |
| ReLU | 1.2411 ± 0.0346 | 2.8 | 3/5 | 0.7974 ± 0.0211 | 2.6 | 4/5 |
| tanh | never fit | — | 0/5 | 0.8479 ± 0.0158 | 1.9 | 5/5 |
| Fourier | 0.7924 ± 0.0935 | 11.8 | 5/5 | 0.6350 ± 0.0229 | 3.6 | 5/5 |
| SIREN | 0.6981 ± 0.0877 | **12.6** | 5/5 | 0.7028 ± 0.1274 | **18.1** | 5/5 |
| CNN-DIP | 0.9096 ± 0.1318 | 14.5 | 5/5 | 0.8001 ± 0.0479 | 6.0 | 5/5 |

## Table 4 — Robustness to noise

Initialization fixed; noise level and realization vary (3 realizations per
level). `data_std` tracks the noise level, so chi^2 = 1 always means "fit to
the noise floor". `fit` counts realizations reaching chi^2 <= 1.

### `blocks`

| Method | 0.5% | 1.5% | 5% | fit rate |
|---|---|---|---|---|
| **TV** | **0.3178 ± 0.0181** | **0.3735 ± 0.0220** | **0.4869 ± 0.0468** | **9/9** |
| Smooth L2 | 0.5637 (1/3) | 0.4488 (1/3) | 0.5989 (1/3) | 3/9 |
| SIREN | 0.6309 ± 0.0123 | 0.6723 ± 0.0559 | 0.7203 ± 0.0705 | 9/9 |
| Fourier | 0.9178 ± 0.0848 | 0.8074 ± 0.1000 | 0.6770 ± 0.0049 | 8/9 |
| CNN-DIP | 0.9845 (1/3) | 1.0396 ± 0.2764 | 0.6271 ± 0.0406 | 6/9 |
| ReLU | never fit | 1.2392 ± 0.1232 | 1.1298 ± 0.0377 | 4/9 |
| tanh | never fit | never fit | 0.8152 (1/3) | 1/9 |

### `parflow`

| Method | 0.5% | 1.5% | 5% | fit rate |
|---|---|---|---|---|
| **TV** | 0.5346 ± 0.0410 | **0.5404 ± 0.0392** | 0.5885 ± 0.0253 | **9/9** |
| Smooth L2 | **0.5282 ± 0.0278** | 0.6228 ± 0.0918 | 0.6530 (1/3) | 6/9 |
| Fourier | 0.5958 ± 0.0090 | 0.6068 ± 0.0245 | **0.5861 ± 0.1292** | 9/9 |
| SIREN | 0.6449 ± 0.0486 | 0.7419 ± 0.0382 | 0.8173 ± 0.1749 | 9/9 |
| ReLU | 0.6472 (1/3) | 0.6574 ± 0.0189 | 0.7541 ± 0.1814 | 6/9 |
| CNN-DIP | 0.6996 ± 0.0022 | 0.7568 ± 0.0233 | 0.9496 ± 0.0788 | 8/9 |
| tanh | never fit | never fit | 0.6424 ± 0.0132 | 2/9 |

Smooth L2 divergences reached chi^2 of 2.4x10^5, 3.0x10^4, 1.2x10^4, 7.2x10^3
and 1.8x10^4 — orders of magnitude past the noise floor, not marginal misses.

## Table 5 — Convergence and cost

| Method | iterations to chi^2 = 1 | runtime per inversion |
|---|---|---|
| TV / Smooth L2 | 3-6 Gauss-Newton steps | **~1-2 s** |
| SIREN | 67-114 | ~60-340 s |
| Fourier | 79-161 | ~60-435 s |
| CNN-DIP | 161-225 | ~60-2860 s |
| ReLU | 588-1318 | ~60-390 s |
| tanh | 1219-1453 (when it fits at all) | ~60-2340 s |

Conventional inversion is **50-500x cheaper** per inversion. Gauss-Newton uses
the Jacobian for second-order steps; the neural arms feed the same Jacobian
into first-order Adam updates over ~33.5k weights.

---

## Findings

**1. Explicit edge-preserving regularization outperforms every implicit prior
tested.** TV leads on accuracy (Table 1), geometry (Table 2), robustness
(Table 4) and cost (Table 5). The gap is largest in geometry: IoU 0.83 against
0.20-0.35 for the networks, and anomaly centroids placed within 1.7 m versus
9.7-17.3 m.

**2. The decisive variable is the prior's character, not whether it is neural.**
Smooth L2 — a conventional method — lands *between* TV and the networks on
geometry (IoU 0.476, centroid 9.9 m), and it is the least numerically stable
method in the entire study. The story is not "conventional beats neural" but
"edge-preserving beats smooth, explicit or implicit".

**3. Networks do not over-smooth; they misplace structure.** Contrast ratios
are all above 1.0 (ReLU 1.72, CNN-DIP 1.69, Fourier 1.33) against TV's 0.85.
The networks generate 20-70% *excess* heterogeneity and put it in the wrong
places. CNN-DIP is the extreme case: the best magnitude recovery in the study
(0.935, better than TV) at nearly the worst geometry. Recovered amplitude and
recovered position are separate axes, and only measuring both reveals this.

**4. Data misfit does not indicate model recovery.** Two models both at
chi^2 = 1 differ by 1.34 in log-RMS while their predicted data differ by 1.46%
against 1.5% noise -- formally indistinguishable to the survey -- and carry
model errors of 0.731 and 1.221 (`talk_figures.py nonuniqueness`). ReLU reached
chi^2 = 1 while misplacing anomalies by 17 m.

Optimizing past the noise floor degrades models further, but the penalty
tracks **budget overshoot, not architecture** (`overfit_check.py`, 41 runs,
both configurations, both held-out targets): +1.7% below 3x the iterations
needed to reach chi^2 = 1, +5.7% at 3-10x, +10.3% beyond 10x, correlation
r = +0.43. Iterations to chi^2 = 1 differ nearly tenfold across architectures
(ReLU 1048, SIREN 251, DIP 172, Fourier 116), so a shared 1500-iteration budget
runs Fourier 12.9x past its stopping point and ReLU only 1.4x. Apparent
architecture-specific over-fitting is largely that.

An earlier version of this section reported 17-49% degradation. That range
traces to the convergence probe, where Fourier ran far past chi^2 = 1
(+34% to +65%); at the benchmark's own budget the range across all
architectures is -4% to +17%, mean +6%, with per-group spreads often larger
than the means. All neural results are reported at the discrepancy-principle
stopping point, which conventional Gauss-Newton applies by construction and the
neural loop must be told to apply. The correction makes that choice more
necessary, not less: a fixed budget penalizes whichever architecture converges
fastest.

**5. Higher noise improved the over-flexible networks** (Fourier on blocks:
0.918 -> 0.807 -> 0.677 as noise rose 0.5% -> 5%; CNN-DIP: 0.985 -> 0.627).
With noisier data, chi^2 = 1 is reached earlier, so the snapshot is taken
before the model overfits. Semi-convergence observed from the opposite
direction, and further evidence that the stopping rule dominates.

**6. Initialization alone changes neural results by up to 18%** (Table 3), with
SIREN the least reproducible. Conventional inversion is exactly reproducible
from the same data. For field deployment — where no ground truth exists to
detect a bad draw — this is a first-order practical concern.

**7. ~~Two architectures failed as parameterizations.~~ RETRACTED for tanh --
see `stage1b_results.md`.** This section reported that tanh could not fit the
data at realistic noise (1/9 and 2/9 realizations) and asserted the failure was
"architectural rather than a tuning artifact."

That was wrong, and wrong in the same way as the rest of Stage 1. The
learning-rate schedule was derived from the iteration budget
(`step_size = n_iters // 4`), and tanh is slow enough to converge that its rate
had been cut to a quarter before it reached chi^2 = 1. No learning rate in the
grid could fix that, because the grid set the *initial* rate and the schedule
decayed it regardless -- which is exactly why the failure looked architectural.

Scheduling from each configuration's own measured convergence, **tanh fits 5/5
seeds on both held-out targets**: 0.8335 +- 0.1098 on `blocks` and
0.7674 +- 0.0437 on `parflow`. Mid-field, not incapable.

ReLU's inconsistency stands: 4/5 seeds on both targets under Stage 1b, and when
it does fit it recovers the wrong structure with the lowest run-to-run variance
of any network -- reliably wrong.

*(Tables 1-5 below retain the Stage-1 numbers as the record of what the shared
protocol produced. Every "never fit" entry for tanh should be read against this
retraction.)*

**8. Neural parameterization inherits accuracy but does not create it.**
Warm-starting a Fourier network from the TV solution halved its from-scratch
error on blocks (0.744 -> 0.369, matching TV's 0.362), while neural refinement
never improved the conventional model. The useful direction is
conventional -> neural: classical inversion anchors accuracy, and the network
re-expresses it in continuous, resolution-independent form — the natural
initialization for time-lapse f(x, z, t), where the parameterization's
compression argument finally favors the network (1000 cells x 365 timesteps
versus one modest network).

## Limitations

- **Inverse crime**: synthetic data generated with the same mesh and solver
  used to invert. Bias is identical across methods, so ranking is fair, but
  these are not absolute-accuracy estimates.
- **Narrow experimental base**: two held-out targets, one mesh, one survey
  geometry, five seeds; comparisons by overlapping means rather than paired
  tests.
- **Group 2 only** among neural methods — no pretrained frameworks yet
  (Group 3 requires a training-pair generator and site-held-out splits).
- **Survey resolution**: the array senses ~20 m; deeper structure is
  unrecoverable by any method and contributes noise to whole-mesh metrics.
- **Oracle bounds** searched on 3 seeds, leaving winner's-curse risk.
- Structural metrics (Table 2) computed on evaluation runs only, not on the
  sensitivity study runs.

## Reproducing

```bash
# laptop (CPU) or Argon compute node
python step0_regression.py                     # refactor regression gate
python inr_benchmark.py tune   <n_jobs>        # 99 runs
python inr_benchmark.py eval   <n_jobs>        # 45 runs
python inr_benchmark.py oracle <n_jobs>        # 228 runs
python traditional_comparison.py <n_jobs>      # conventional arm
python convergence_check.py <n_jobs> 3000      # budget probe
python warmstart_hybrid.py <n_jobs>            # hybrid arm
python structure_metrics.py                    # geometry/magnitude/SSIM
python sensitivity.py both <n_jobs>            # init + noise studies

# Argon batch (UI queue, 40 cores) — GEOPHYSICS has only 2 nodes and is
# usually saturated; UI has many idle 80-core nodes.
qsub run_sensitivity.job
```

Results land in `inr_results/`: `tuning.json`, `evaluation.json`,
`oracle.json`, `traditional.json`, `smooth_l2.json`, `convergence.json`,
`hybrid.json`, `structure_metrics.json`, `sensitivity_init.json`,
`sensitivity_noise.json`, `inr_comparison.png`.
