# Conclusions — draft

Drawn only from `stage1_results.md`. Nothing here is new: no field data, no
time-lapse, no pretrained methods, no infrastructure. Forward-looking material
is quarantined in the Future Work block below, per the handout's rule that
nothing new belongs in a Conclusions section.

**Ordering decision:** the complementarity result leads. Readers carry away
the first bullet, and this framing says what the two approaches are *for*
rather than which one won. The mechanistic conclusions follow as explanation.

---

## Version 1 — Poster (bulleted)

**Conclusions**

We tested whether replacing per-cell resistivity values with a neural
parameterization improves ERT inversion, holding the physics-based forward
solver, survey, data, noise, optimization budget, and evaluation criteria
identical across seven methods.

1. **Neural parameterizations inherit accuracy from conventional inversion but
   do not create it.** Total-variation inversion produced the lowest model
   error of the seven methods tested, and warm-starting a Fourier-feature
   network from the TV solution halved the network's from-scratch error
   (0.74 → 0.37, matching TV's 0.36) — while neural refinement of the TV model
   never improved it. The productive coupling runs conventional → neural:
   classical inversion supplies the accuracy, and the network re-expresses it
   in continuous, resolution-independent form.

2. **The decisive factor is the character of the prior, not whether it is
   neural.** Smooth-L2 inversion, equally conventional, fell *between* TV and
   the networks on anomaly geometry (IoU 0.48, versus 0.83 for TV and
   0.20–0.35 for the networks), and was the least numerically stable method
   tested. Edge-preserving priors outperformed smooth priors regardless of
   whether the prior was explicit or architectural.

3. **Neural priors misplace structure rather than over-smoothing it.** All
   five networks generated 20–70% *excess* heterogeneity (contrast ratios
   1.19–1.72, against TV's 0.85) while placing anomaly centroids 9.7–17.3 m
   from truth, against TV's 1.7 m. The CNN deep image prior recovered the best
   anomaly amplitude in the study — 0.94 of the true log-contrast, exceeding
   TV's 0.80 — at nearly the worst position. Recovered amplitude and recovered
   position are independent failure modes, and aggregate error conceals both.

4. **Fitting the data is not recovering the model.** With 35 measurements
   constraining 1000 cells, two models that both reach chi^2 = 1 can differ by
   1.34 in log-RMS while their predicted data differ by 1.46% — less than the
   1.5% measurement noise. The data cannot choose between them. ReLU reached
   the noise level while misplacing anomalies by 17 m, and one of those two
   equally-fitting models carried 67% more model error than the other.

   Optimizing past the noise level degrades models further, but **the size of
   that penalty is set by how far past you run, not by the architecture**
   (`overfit_check.py`, 41 runs): +1.7% at under 3x the iterations needed to
   reach chi^2 = 1, +5.7% at 3–10x, +10.3% beyond 10x (r = +0.43). Because
   architectures converge at very different rates — ReLU 1048 iterations,
   Fourier 116 — a shared budget runs some of them an order of magnitude
   further past their stopping point than others. An earlier draft reported
   17–49% here; that range came from the convergence probe, where Fourier ran
   far past chi^2 = 1, and it is not representative of the benchmark budget
   (-4% to +17%, mean +6%). The correction *strengthens* the case for
   discrepancy-principle stopping: a fixed budget systematically penalizes
   whichever architecture converges fastest, so matched-fit scoring is not
   optional.

5. **Neural inversions are not reproducible.** Initialization alone — identical
   data, different random seed — changed model error by up to 18% (SIREN),
   while conventional inversion returns the same answer every time. In the
   field, where no ground truth exists to detect an unlucky initialization,
   this is a first-order practical concern.

6. **Configuration can be chosen without ground truth — but only for some
   architectures.** Selecting capacity, frequency scale, budget, and learning
   rate on a separate calibration target, never on the evaluation problems,
   reproduced the truth-tuned Fourier result exactly (0.4046 and 0.5451 against
   0.405 and 0.545) and brought it level with TV on the hillslope (0.5451 vs
   0.5468). The identical protocol cost SIREN 18.6% and CNN-DIP gained 23.0%.
   Re-deriving TV's regularization weight on that calibration target moved it by
   nothing measurable. A scalar regularization weight transfers between
   problems; a network's capacity may not, and which it does depends on the
   architecture. (`stage1b_results.md`)

7. **A calibration target can tune a method but cannot rank methods.** On the
   calibration problem smooth-L2 outscored TV (0.509 vs 0.564); on both held-out
   targets TV won decisively (0.362 vs 0.501; 0.547 vs 0.572). The same
   inversion appears among the networks. Anyone selecting an inversion scheme by
   its performance on a synthetic test problem is measuring fit to that problem,
   not transferable quality.

*(The earlier "two architectures failed outright" bullet has been withdrawn.
The tanh network's failure was an artifact of a learning-rate schedule derived
from the shared iteration budget: its rate was cut to a quarter before it
converged. Scheduled from its own measured convergence it fits 5/5 seeds on both
held-out targets, at 0.834 and 0.767. ReLU's inconsistency stands — 4/5 seeds on
both targets in Stage 1b.)*

---

## Version 2 — Paper (prose, for conclusions embedded in a Discussion)

We tested whether replacing per-cell resistivity values with a neural
parameterization improves ERT inversion, holding the physics-based forward
solver, survey, data, noise, optimization budget, and evaluation criteria
identical across seven methods. Neural parameterizations inherited accuracy
from conventional inversion but did not create it: total-variation inversion
produced the lowest model error of the seven methods, and warm-starting a
Fourier-feature network from the TV solution halved its from-scratch error
(0.74 → 0.37, matching TV's 0.36), whereas neural refinement of the TV model
never improved it. The decisive factor proved to be the character of the prior
rather than whether it was neural, since smooth-L2 inversion fell between TV
and the networks on anomaly geometry (IoU 0.48, against 0.83 for TV and
0.20–0.35 for the networks) and was the least numerically stable method
tested.

Contrary to the standard expectation that spectral bias produces blurred
models, the networks did not over-smooth: all five generated 20–70% excess
heterogeneity (contrast ratios 1.19–1.72 against TV's 0.85) and misplaced
anomaly centroids by 9.7–17.3 m against TV's 1.7 m, with recovered amplitude
and recovered position behaving as independent failure modes. These results
reinforce that fitting the data is not recovering the model: with 35
measurements constraining 1000 cells, two models both at chi^2 = 1 differed by
1.34 in log-RMS while their predicted data differed by less than the
measurement noise, and one carried 67% more model error than the other.
Optimizing past the noise level degrades models further by an amount set by
budget overshoot rather than architecture (+1.7% below 3x, +10.3% beyond 10x),
so the discrepancy-principle stopping rule that Gauss-Newton applies by
construction must be imposed explicitly on a neural parameterization —
particularly because a shared budget runs fast-converging architectures much
further past their stopping point than slow ones. A final practical limitation is reproducibility:
initialization alone changed model error by up to 18% on identical data, where
conventional inversion returns the same answer every time.

---

## Future Work — NOT for the Conclusions section

Everything forward-looking, kept separate so it cannot leak into the
conclusions:

- Extend the benchmark to pretrained frameworks (Group 3): CNN/U-Net direct
  inversion, recurrent and transformer models, invertible networks, and
  physics-guided training frameworks.
- Time-lapse inversion with a 4D coordinate network f(x, z, t), where the
  parameterization's compression argument favors the network for the first
  time (1000 cells × 365 timesteps versus one modest network).
- Field ERT surveys from the Ashton Prairie Living Laboratory (Iowa) and the
  East River watershed (Colorado).
- Protocol strengthening: second-order optimization for the neural arms,
  a spectrally representative calibration target, an explicit regularization
  term on the network output, capacity validation, and removal of the
  inverse crime.

---

## Flags

1. **Bullet 1 must stand alone**, since readers may see only the conclusions.
   It therefore states that TV produced the lowest error *before* claiming the
   network inherits it — otherwise "inherits accuracy" leaves the reader
   asking from what. Keep that clause if the bullet gets trimmed for space.

2. **Bullet 3's geometry numbers rest on one target.** IoU, centroid error and
   magnitude come from the `blocks` target only, since `parflow` has no
   discrete anomalies to measure placement against. The contrast ratios hold
   on both targets. If pressed, the honest scope is "on the sharp-anomaly
   target."

3. **The warm-start number is also single-target.** 0.74 → 0.37 is the blocks
   result; on parflow the improvement was ~11% (0.62 → 0.55). Bullet 1 is
   written without a target qualifier for readability — add "on the
   sharp-anomaly target" if a reviewer or your mentor asks for precision.
