# Draft email to Hang Chen — Stage 1 / 1b results and Stage 2 scope

Adjust tone to how you normally write to him. Two things to keep: the
corrections belong near the top rather than buried, and the Stage-2 questions
need to be answerable in one reply.

---

**Subject:** Stage 1 benchmark complete — results, three corrections, and Stage 2 scope

Hi Hang,

The Group 1 and Group 2 benchmark is finished. All seven methods, both held-out
targets, five seeds, scored on the six criteria you asked for. Summary table
below; the full write-ups are in `docs/stage1_results.md` and
`docs/stage1b_results.md`.

Three findings changed after I checked the protocol's own assumptions, so I want
to state those first.

## Three corrections to what I reported earlier

**1. tanh does fit. My earlier report that it could not was wrong.**
I had said its failure was architectural because no learning rate in the grid
fit on all seeds. The learning-rate *schedule* was derived from the iteration
budget (`step_size = n_iters // 4`), and tanh converges slowly enough that its
rate was cut to a quarter before it reached chi^2 = 1. The grid set the initial
rate; the schedule decayed it regardless — which is exactly why the failure
looked architectural. Scheduling from each configuration's measured convergence,
tanh fits 5/5 seeds on both targets (0.834 and 0.767).

**2. The ~50% conventional advantage was a configuration artifact.**
Under a shared capacity and iteration budget, TV appeared about 50% more
accurate than any network. Tuning capacity, budget and frequency scale per
architecture erased that: Fourier 0.405 against TV's 0.362 on blocks
(paired t, n = 5, p = 0.15) and 0.545 vs 0.547 on parflow.

**3. "Optimizing past the noise level degrades models 17-49%" was not
representative.** That range came from the convergence probe, where Fourier ran
far past chi^2 = 1. Measured properly across architectures and configurations
(41 runs), degradation tracks *budget overshoot*, not architecture: +1.7% below
3x the iterations needed to reach chi^2 = 1, +5.7% at 3-10x, +10.3% beyond 10x.
Since architectures reach chi^2 = 1 between 94 and 2010 iterations, a shared
budget runs some of them an order of magnitude further past their stopping point
than others. This strengthens rather than weakens the case for scoring at
matched data fit.

In total, six conclusions in the first draft measured a shared setting rather
than an architecture. That is the main methodological result, and I think it is
the part worth writing up.

## Summary table

Model error is coverage-masked log-RMSE at chi^2 = 1, mean over 5 seeds.
Geometry columns are the `blocks` target. Runtime is time to reach the noise
level, not the full budget.

| Method | Group | err blocks | err parflow | IoU | centroid m | magnitude | SSIM | time to chi2=1 | fit |
|---|---|---|---|---|---|---|---|---|---|
| TV | 1 | **0.362** | 0.547 | **0.826** | **1.7** | 0.80 | **0.729** | **0.9 s** | 5/5 |
| smooth-L2 | 1 | 0.501 | 0.572 | 0.593 | 5.9 | 0.80 | 0.504 | 0.9 s | 5/5 |
| Fourier INR | 2 | 0.405 | **0.545** | 0.662 | 4.7 | 0.70 | 0.560 | 13.5 s | 5/5 |
| CNN-DIP | 2 | 0.773 | 0.683 | 0.359 | 11.9 | **0.89** | 0.335 | 15.7 s | 9/10 |
| SIREN | 2 | 0.812 | 0.673 | 0.300 | 12.8 | 0.84 | 0.296 | 44.6 s | 9/10 |
| tanh INR | 2 | 0.834 | 0.767 | 0.297 | 13.3 | 0.82 | 0.306 | 215.7 s | 10/10 |
| ReLU INR | 2 | 1.148 | 0.717 | 0.191 | 17.0 | 0.78 | 0.205 | 112.6 s | 9/10 |

Initialization sensitivity and noise robustness are in `stage1_results.md`
Tables 3 and 4 — those were measured at the shared configuration and I have not
re-run them at the per-architecture settings.

## The result I think matters most

**Configuration can be chosen without ground truth — but only for some
architectures.**

Stage 1b selects capacity, budget, frequency scale, learning rate and the
conventional regularization weight on a separate calibration target containing
smooth, intermediate and sharp structure, then freezes them. The held-out
targets are inverted once and never influence any choice.

For Fourier this reproduced the ground-truth-tuned result *exactly* — 0.4046 and
0.5451 against 0.405 and 0.545 — and the geometry too (IoU 0.662, centroid
4.7 m, identical). On parflow it ranks first of all seven methods.

The same protocol cost SIREN 18.6%. The reason is concrete: calibration selected
width 64 (8,577 parameters) for SIREN and width 512 (528,199) for Fourier. SIREN
is the most capacity-sensitive method in the study and the capacity that suited
the calibration target does not suit blocks.

Re-deriving TV's regularization weight on the same calibration target moved it by
nothing measurable (0.362 -> 0.3620). A scalar weight transfers between problems;
a network's capacity may not.

**One caution worth flagging:** on the calibration target smooth-L2 outscored TV
(0.509 vs 0.564), but TV won on both held-out targets (0.362 vs 0.501; 0.547 vs
0.572). The ranking inverted, and the same happens among the networks. A
calibration target can tune a method but cannot be used to choose between
methods.

## Second result: average error and structure do not converge together

Fourier ties TV on model error while recovering a 20% smaller anomaly footprint
(IoU 0.662 vs 0.826) and placing centroids 2.8x farther from truth (4.7 m vs
1.7 m). On parflow the model errors are indistinguishable while SSIM differs by
28%.

Amplitude and position behave as independent failure modes: CNN-DIP recovers the
best anomaly magnitude in the study (0.89 of true log-contrast, above TV's 0.80)
at nearly the worst position. Fourier is the mirror image — second-best geometry,
weakest amplitude (0.70).

I would not report model error alone for any of these methods.

## Questions on Stage 2

1. **Group 3 scope.** Which pretrained frameworks do you want first — CNN/U-Net,
   recurrent, transformer, invertible, physics-guided? These need a training
   database, which is a different infrastructure problem from Groups 1 and 2, so
   I would rather build for two or three well than all five thinly.

2. **Fixed or variable survey geometry.** This decides the training-set design.
   Fixed geometry lets a network learn a direct data-to-model map; variable
   geometry needs the survey encoded as input and a much larger database. It is
   the biggest single decision for Stage 2 and I do not want to guess.

3. **Training database.** Should I generate synthetic models (ParFlow-derived,
   as in the current parflow target), or is there an existing dataset in the
   group I should use?

4. **Inverse crime.** All current data are generated with the same mesh and
   solver used to invert them. For Group 3 that becomes more serious, since a
   pretrained network can learn solver artifacts. Should Stage 2 generate data on
   a finer mesh and invert on a coarser one?

5. **Field data.** Is the Ashton Prairie survey ready to invert, and would you
   like that in Stage 2 or held for later?

Happy to talk any of this through.

Best,
Miranda

---

## Notes for you, not for the email

**Send the summary table as a real table**, not a code block — paste it into the
email body or attach the two markdown files.

**Question 2 is the one that blocks work.** If he answers nothing else, that is
the one to push on: fixed vs variable geometry changes the database design, the
architecture, and the timeline.

**Question 4 is worth raising even if the answer is "not yet."** It shows you
know the current results are not evidence about absolute accuracy, which is a
limitation he may otherwise think you have missed.

**If he asks why the numbers changed** — the honest answer is that every
correction came from testing an assumption in your own protocol rather than from
new data, and each one is reproducible from a script in the repo
(`capacity_sweep.py`, `overfit_check.py`, `stage1b.py`).
