# Stage 1b — can a neural inversion be configured without ground truth?

Stage 1 and the capacity re-tune both selected network settings by scoring
against the **true model**. That is legitimate as a measurement of what an
architecture can do, but it is not available in the field, so it says nothing
about whether these methods are deployable.

Stage 1b removes ground truth from every selection step. Capacity, budget,
frequency scale, learning rate, and the conventional regularization weight are
all chosen on a single **calibration target** (`mixed`) and then frozen. The
held-out targets (`blocks`, `parflow`) are inverted once with those frozen
settings and never influence any choice.

It also runs the whole chain in **dependency order** — probe → capacity → tune →
eval → conventional — so no stage inherits a setting a later stage is meant to
determine. Stage 1 violated that in five separate places, each of which produced
a false conclusion about an architecture.

Run: Argon job 6402338, 40 slots, 64 min. Results in
`inr_results/stage1b/{probe,capacity,tune,eval,conventional}.json`.

---

## The calibration target

`two_layer`, the original development target, is perfectly smooth. Tuning on it
selected the lowest frequency setting available for every architecture that has
one — SIREN's `w0` was driven to 1.0 — which then failed badly on sharp
structure. A calibration target has to contain the range of spatial frequencies
the method will be asked to represent.

`mixed` combines a smooth depth gradient, a dipping interface, and two sharp
blocks. Its lateral gradient (0.049) sits between `parflow` (0.044) and `blocks`
(0.072), against `two_layer`'s 0.000. Its correlation with `blocks` is −0.21, so
it is not a disguised copy of the evaluation problem.

---

## Table 1 — Model error at matched data fit (chi^2 = 1), coverage-masked

Mean +- SD over 5 seeds. `fit` counts seeds reaching the noise level.
"Stage 1" is the shared-capacity, shared-budget result for comparison.

### `blocks`

| method | Stage 1 | Stage 1b | change | fit |
|---|---|---|---|---|
| **TV** | 0.362 | **0.3620 +- 0.0392** | — | — |
| **Fourier INR** | 0.7438 | **0.4046 +- 0.0294** | **+45.6%** | 5/5 |
| smooth-L2 | — | 0.5008 +- 0.0390 | — | — |
| CNN-DIP | 1.0031 | 0.7727 +- 0.1763 | +23.0% | 5/5 |
| SIREN | 0.6849 | 0.8122 +- 0.0969 | **−18.6%** | 4/5 |
| tanh INR | *never fit* | **0.8335 +- 0.1098** | — | **5/5** |
| ReLU INR | 1.0897 | 1.1479 +- 0.1117 | −5.3% | 4/5 |

### `parflow`

| method | Stage 1 | Stage 1b | change | fit |
|---|---|---|---|---|
| **Fourier INR** | 0.6173 | **0.5451 +- 0.0127** | **+11.7%** | 5/5 |
| **TV** | 0.547 | **0.5468 +- 0.0188** | — | — |
| smooth-L2 | — | 0.5716 +- 0.0517 | — | — |
| SIREN | 0.6566 | 0.6733 +- 0.0614 | −2.6% | 5/5 |
| CNN-DIP | 0.7442 | 0.6826 +- 0.0152 | +8.3% | 4/5 |
| ReLU INR | 0.7220 | 0.7169 +- 0.0461 | +0.7% | 4/5 |
| tanh INR | 0.8218 | 0.7674 +- 0.0437 | +6.6% | 5/5 |

---

## Table 2 — Anomaly geometry, structure, and amplitude (`blocks`)

Scored from the fields retained in `eval.json` with `structure_metrics.score`,
so these are directly comparable to the Stage-1 table. Conventional fields were
re-run locally at the Stage-1b-selected weights (`stage1b_geometry.py`).

| method | model err | SSIM | contrast | IoU | centroid (m) | magnitude |
|---|---|---|---|---|---|---|
| **TV** | **0.362** | **0.729** | 0.85 | **0.826** | **1.7** | 0.80 |
| **Fourier INR** | 0.405 | 0.560 | 0.88 | 0.662 | 4.7 | 0.70 |
| smooth-L2 | 0.501 | 0.504 | 0.97 | 0.593 | 5.9 | 0.80 |
| CNN-DIP | 0.773 | 0.335 | 1.40 | 0.359 | 11.9 | **0.89** |
| SIREN | 0.812 | 0.296 | 1.43 | 0.300 | 12.8 | 0.84 |
| tanh INR | 0.834 | 0.306 | 1.45 | 0.297 | 13.3 | 0.82 |
| ReLU INR | 1.148 | 0.205 | 1.83 | 0.191 | 17.0 | 0.78 |

On `parflow` (no discrete anomalies, so IoU and centroid are undefined) the same
split appears in SSIM: TV **0.428** against Fourier **0.308**, at model errors of
0.547 and 0.545.

---

## Findings

### 0. Model error and structural accuracy still diverge — and the geometry reproduced exactly.

Fourier matches TV on model error (0.405 vs 0.362, a tie at n = 5) while
recovering a **20% smaller anomaly footprint** (IoU 0.662 vs 0.826) and placing
centroids **2.8x farther** from truth (4.7 m vs 1.7 m). On `parflow`, model
errors are indistinguishable (0.545 vs 0.547) while SSIM differs by 28%.

Both figures are *identical* to the ground-truth-tuned run — IoU 0.662 and
4.7 m there as well. Calibration-only selection reproduced not just the average
error but the spatial structure, which is a stronger reproducibility result than
Table 1 alone shows.

**Aggregate error is not a sufficient report.** A method can tie on log-RMSE and
still put the anomaly in the wrong place, and for a groundwater survey the
position is usually the answer being sought.

### 0b. Proper configuration fixed the over-texturing — for Fourier only.

Stage 1 found all five networks generating 20-70% excess heterogeneity
(contrast ratios 1.19-1.72 against TV's 0.85). At Stage-1b configurations
Fourier sits at **0.88**, essentially TV's value, while ReLU (1.83), tanh
(1.45), SIREN (1.43) and DIP (1.40) still over-texture.

So the excess structure was a configuration artifact for the architecture that
transfers well, and remains a property of the others.

### 0c. Amplitude and position remain independent failure modes.

CNN-DIP recovers the **best anomaly amplitude in the study** (0.89 of the true
log-contrast, above TV's 0.80) at nearly the **worst** position (11.9 m). Fourier
inverts the pattern: second-best geometry, the *weakest* amplitude of any method
(0.70). Reporting either alone would rank these two methods differently.

### 1. tanh fits. The Stage-1 conclusion that it could not was an artifact of the learning-rate schedule.

Stage 1 reported tanh as never reaching the noise level and excluded it. The
schedule was derived from the iteration budget (`step_size = n_iters // 4`),
and tanh converges slowly enough that its learning rate had been cut to a
quarter before it reached chi^2 = 1. Scheduling from each configuration's own
measured convergence instead, tanh fits **5/5 seeds on both held-out targets**
and lands mid-field at 0.8335 and 0.7674.

This is the sixth documented instance of a shared setting producing a false
statement about an architecture, and the only one predicted in advance and then
confirmed by a targeted fix.

### 2. Calibration-only selection reproduced ground-truth tuning exactly — for Fourier.

| | ground-truth tuned | calibration tuned |
|---|---|---|
| `blocks` | 0.405 | **0.4046** |
| `parflow` | 0.545 | **0.5451** |

Nothing in the Stage 1b pipeline ever saw the true model of either held-out
target. For this architecture, a representative calibration target is a complete
substitute for ground truth.

### 3. On `parflow` the best network now edges TV — without ground truth.

Fourier **0.5451 +- 0.0127** against TV **0.5468 +- 0.0188**: a gap of 0.0017,
far inside the seed spread, and Fourier is nominally ahead. Fourier ranks first
of all seven methods on this target.

On `blocks` TV retains a real lead (0.3620 vs 0.4046, gap 0.043; five seeds
resolve differences above ~0.09, so this remains statistically a tie but the
point estimate favours TV consistently).

### 4. The transfer is architecture-dependent, and that is the practical result.

| architecture | change under calibration-only selection |
|---|---|
| Fourier | +45.6% / +11.7% |
| CNN-DIP | +23.0% / +8.3% |
| tanh | recovered from "never fit" |
| ReLU | −5.3% / +0.7% |
| SIREN | **−18.6%** / −2.6% |

Fourier and DIP transfer well; SIREN does not. SIREN is the most
capacity-sensitive architecture in the study, and the capacity chosen on
`mixed` does not suit `blocks`. So "can you configure this without truth?" has
no single answer — it depends on which network you picked.

### 5. Conventional regularization is insensitive to the calibration target. Neural configuration is not.

Re-deriving the regularization weight on `mixed` instead of `two_layer` moved TV
by nothing measurable:

| method | calibrated on `two_layer` | calibrated on `mixed` |
|---|---|---|
| TV, `blocks` | 0.362 | 0.3620 |
| TV, `parflow` | 0.547 | 0.5468 |
| SIREN, `blocks` | 0.685 | 0.812 |

The same protocol change costs SIREN 18.6% and TV nothing. That asymmetry is now
measured rather than argued: TV's single scalar weight is stable across
calibration problems, while a network's capacity and frequency scale are not.

### 6. Calibration ranking does not predict held-out ranking — for conventional methods either.

On the calibration target, smooth-L2 scored **0.5090** against TV's **0.5638**,
so smooth-L2 ranked first. On both held-out targets TV won decisively:

| | smooth-L2 | TV |
|---|---|---|
| `blocks` | 0.5008 | **0.3620** |
| `parflow` | 0.5716 | **0.5468** |

The ranking inverted. The same inversion appears among the networks — Fourier
ranks poorly on calibration and first on held-out `parflow`.

**A calibration target can tune a method. It cannot be used to choose between
methods.** That distinction matters for anyone proposing to select an inversion
scheme on a synthetic test problem.

---

## Selected configurations

Conventional, chosen on `mixed`:

| family | operator | lambda | calibration score |
|---|---|---|---|
| smooth-L2 | first-order smoothness | 1 | 0.5090 |
| TV | spatial total variation | 0.1 | 0.5638 |

TV failed entirely at lambda = 10 (0/3 seeds reached the noise level) and was
unstable at 0.001 (2/3), so its usable range on this problem is narrow.

---

## Limitations

**Five seeds resolve differences above ~0.09.** The `blocks` gap between TV and
Fourier is 0.043 and the `parflow` gap is 0.0017. Both are ties by that
standard; only the point estimates differ.

**Two held-out targets, both synthetic, both inverted with the same mesh and
solver used to generate them** (an inverse crime). All methods inherit that bias
identically, so it is fair for ranking, but these are not statements about
absolute accuracy on field data.

**One calibration target.** Whether `mixed` is representative enough for a
different survey geometry or a different geological setting is untested.

**Fit rates below 5/5** for ReLU (4/5 both targets), SIREN on `blocks` (4/5), and
DIP on `parflow` (4/5). Those means are taken over the seeds that converged, so
they flatter methods that sometimes fail.

---

## Reproducing

```bash
# Argon batch, UI queue, 40 cores, ~65 min
qsub run_stage1b_rest.job          # capacity -> tune -> eval -> conventional
qsub run_stage1b.job               # add the 12000-iteration probe (~2.5-3 h)
```

Fields are retained in `eval.json`, so geometry metrics (IoU, centroid error,
SSIM, contrast ratio) can be scored from the saved run without re-inverting.
