# AGU26 poster draft — RESESS template

Mapped to the five content boxes in the template. Poster text is **not** paper
text: short sentences, numbers up front, figures carrying the argument. Target
word counts are given per box; going over is the most common way posters become
unreadable from two metres away.

**Layout (from the template):**

```
┌──────────────────────────────────────────────────────────────────┐
│  TITLE + AUTHORS                              NSF · NGF · EarthScope │
├──────────────┬───────────────────────────────────────────────────┤
│              │                                                   │
│ 1. WHY ERT   │  2. WHAT WE COMPARED  (hero figure)               │
│    IS HARD   │                                                   │
│    (tall)    │                                                   │
│              │                                                   │
├──────────────┼─────────────────────────────┬─────────────────────┤
│ 3. WHAT WE   │  4. WHAT WE FOUND           │  5. CONCLUSIONS     │
│    MEASURED  │     (wide)                  │     (dark box)      │
└──────────────┴─────────────────────────────┴─────────────────────┘
```

---

## Header

**Benchmarking Deep-Learning Frameworks for Electrical Resistivity Tomography Inversion**

Miranda Macias¹, Zhengyang Fang², Weiyu Guo², Chen Xiong², Hang Chen²
¹University of Texas at El Paso ²University of Iowa, School of Earth, Environment, and Sustainability

---

## Box 1 — "Why ERT inversion is hard" *(~120 words + concept figure)*

Groundwater sustains drinking water, agriculture, and ecosystems. Managing it
requires knowing where that water sits — and electrical resistivity tomography
(ERT) is how we look.

**The problem is nonuniqueness.** Many different subsurface models reproduce the
same surface measurements equally well. To choose one, the inversion must assume
something about the ground. Conventional methods assume smoothness, imposed
explicitly through a regularization term.

**We asked whether a neural network's design can supply that assumption
instead** — keeping the physics-based forward solver unchanged and altering only
how the model is described. Instead of one resistivity value per mesh cell, a
network maps position to resistivity, and the inversion optimizes network
weights.

> **FIGURE A** — the cell-values → network-weights concept diagram (slide 9).
> Small, top or bottom of this box. It does more work than any paragraph here.

---

## Box 2 — "What we compared" *(~110 words + hero figure)*

Seven methods, identical conditions: same forward solver, mesh, survey
(16-electrode Wenner, 35 measurements), noise (1.5%), starting model, and
evaluation criteria. All compared at **matched data fit (χ² = 1)**, not matched
iteration count.

**Group 1 — conventional, explicit regularization**
Smooth-L2 · Total variation (TV)

**Group 2 — neural, optimized per dataset, no training database**
ReLU · tanh · SIREN · Fourier-feature · CNN Deep Image Prior

Two held-out synthetic targets: sharp rectangular blocks, and a ParFlow-derived
hillslope. Hyperparameters were tuned on a separate calibration target and
frozen, so no reported number comes from settings chosen while looking at that
problem.

> **FIGURE B (hero)** — `network_results.png`. Recovered resistivity for all
> seven methods on both targets, each panel labelled with model error.
> This is the largest box on the poster; let the figure fill it.

---

## Box 3 — "What we measured" *(~90 words)*

Reporting data misfit alone is not enough, so every method was scored six ways:

- **Model error** — log-RMSE against truth, over cells the survey senses
- **Anomaly geometry** — intersection-over-union, centroid position error
- **Recovered magnitude** — fraction of true log-contrast
- **Structural similarity** — SSIM
- **Runtime** and iterations to reach the noise level
- **Robustness** — five random seeds, three noise levels

> **A low data misfit does not mean the subsurface has been recovered.**
> Two models both at χ² = 1 differed by 1.34 in log-RMS while their predicted
> data differed by 1.46% — less than the 1.5% measurement noise. Model errors:
> 0.731 and 1.221. The survey cannot choose between them.

---

## Box 4 — "What we found" *(~170 words + figure)*

**Configuration mattered more than architecture.** Under a single shared network
capacity and iteration budget, TV appeared **~50% more accurate**. Tuning
capacity, budget, and frequency scale per architecture erased that gap: the best
network reached **0.405** against TV's **0.362** on sharp blocks (paired t,
n = 5, p = 0.15) and **0.545** vs **0.547** on the hillslope (p = 0.89).

That shared setting produced **six false conclusions**, the starkest being an
architecture (tanh) reported as unable to fit the data at all. Its learning-rate
schedule was derived from the shared budget, so its rate was cut to a quarter
before it converged. Scheduled from its own measured convergence, it fits
**5/5 seeds** on both held-out targets.

**And the parity no longer requires ground truth.** Selecting capacity,
frequency scale, and budget on a separate calibration target — never on the
evaluation problems — reproduced the truth-tuned result exactly (0.4046 and
0.5451), and on the hillslope the network edged TV (0.5451 vs 0.5468).

**But average error and structural accuracy converge differently.** Under
calibration-only selection the leading network matched TV on model error while
still misplacing anomaly centroids **2.8× farther** (4.7 m vs 1.7 m) and
recovering a weaker footprint (IoU 0.66 vs 0.83) — figures identical to the
ground-truth-tuned run. Amplitude and position are independent: CNN-DIP recovers
the best anomaly magnitude in the study (0.89 of true log-contrast, above TV's
0.80) at nearly the worst position (11.9 m).

**But the transfer is architecture-dependent.** Under calibration-only
selection Fourier gained 45.6% on blocks and CNN-DIP 23.0%, while SIREN — the
most capacity-sensitive method tested — lost 18.6%. Re-deriving TV's weight on
the same calibration target moved it by nothing measurable. A single scalar
regularization weight transfers between problems; a network's capacity may not.

**And calibration ranking does not predict held-out ranking.** On the
calibration target smooth-L2 beat TV (0.509 vs 0.564); on both held-out targets
TV won (0.362 vs 0.501, 0.547 vs 0.572). A calibration target can *tune* a
method but cannot be used to *choose between* methods.

> **FIGURE C** — `poster_comparison.png`. Same architecture, badly configured
> vs properly configured, against TV. Annotate each panel with IoU and centroid
> error, not just model error.

---

## Box 5 — Conclusions *(dark box, ~140 words)*

1. **Configuration dominated the comparison more than architecture did.** A
   shared capacity and iteration budget made conventional inversion appear ~50%
   more accurate; per-architecture tuning erased that gap.

2. **Neural configuration can be chosen without ground truth — for some
   architectures.** Selecting capacity, frequency scale, and budget on a
   separate calibration target reproduced ground-truth-tuned accuracy exactly
   for the Fourier network (0.4046 and 0.5451 against 0.405 and 0.545), which
   then matched TV on the hillslope (0.5451 vs 0.5468). The same protocol cost
   SIREN 18.6% and moved TV by nothing measurable. Conventional regularization
   is a single scalar and transfers between problems; a network's capacity does
   not, and whether it transfers depends on the architecture.

3. **The networks misplaced structure rather than blurring it** — contrary to
   the spectral-bias expectation — and average error conceals this. Proper
   configuration fixed the excess structure but not the misplacement.

4. **Fitting the data is not recovering the model.** Two models at identical
   χ² = 1 predicted data within 1.46% of each other — below the noise — yet one
   carried 67% more model error. Degradation past the noise level tracks budget
   overshoot, not architecture (+1.7% below 3×, +10.3% beyond 10×), and
   architectures reach χ² = 1 between 116 and 1048 iterations. A shared budget
   therefore penalizes the fastest-converging method, which is why the
   discrepancy-principle stopping rule Gauss-Newton applies by construction must
   be imposed explicitly on a neural parameterization.

**Next:** pretrained frameworks (CNN/U-Net, recurrent, transformer, invertible,
physics-guided), time-lapse 4D representations, and field surveys at Ashton
Prairie Living Laboratory (Iowa) and East River watershed (Colorado).

---

## Footer

This work was conducted as part of the RESESS internship program, supported by
the NSF NGF operated by EarthScope Consortium. NSF National Geophysical Facility
supported by NSF award 2435260.

---

## Notes on assembly

**Figures do the arguing.** Three figures, in descending size: the hero
comparison (Box 2), the configuration effect (Box 4), the concept diagram
(Box 1). If a fourth is wanted, `capacity_sweep.png` fits Box 3.

**Conclusions dropped from six bullets to four** for the dark box, which is
small. Cut: the prior-character finding (smooth-L2 between TV and the networks)
and reproducibility. Both are worth having ready verbally — they are the two
most likely questions.

**Numbers to have memorized for the poster session**
- TV 0.362 · best network 0.405 (blocks); 0.547 vs 0.545 (parflow)
- IoU 0.83 vs 0.66; centroid 1.7 m vs 4.7 m
- tanh needed 1843–5556 iterations against a budget of 1500
- Five seeds resolve differences above ~0.09; the blocks gap is 0.043

**Stage 1b has reported** (Argon job 6402338; `docs/stage1b_results.md`).
Parity **did** survive calibration-only selection, so Conclusion 2 has been
rewritten from "cannot be tuned without truth" to "can be tuned from a
representative calibration target — for some architectures." Fourier reproduced
its ground-truth-tuned numbers exactly and edged TV on the hillslope; SIREN lost
18.6%; tanh, previously reported as unable to fit at all, converged on 5/5 seeds
once its learning-rate schedule stopped being derived from the shared budget.

**Numbers to swap in before print** — Box 4 and Box 2 still quote the
ground-truth-tuned run:

| | old (truth-tuned) | new (calibration-tuned) |
|---|---|---|
| Fourier, blocks | 0.405 | 0.4046 |
| Fourier, parflow | 0.545 | 0.5451 |
| TV, blocks | 0.362 | 0.3620 |
| TV, parflow | 0.547 | 0.5468 |

The values barely move; what changes is that **none of them now required the
true model**. That is the claim worth making on the poster, and it is stronger
than the one it replaces.

**Also add tanh to the method list.** Box 2 currently lists it among the five
neural methods, which is correct, but Box 4's "four false conclusions" line
should become **six**, and the tanh example is now a retraction rather than an
illustration: it was reported as unable to fit, and it fits.
