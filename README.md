# Neural reparameterization for ERT inversion

Benchmarking deep-learning frameworks for electrical resistivity tomography
(ERT) inversion, holding the physics fixed and changing only how the subsurface
model is represented.

Miranda Macias, University of Texas at El Paso
RESESS internship 2026, hosted by the University of Iowa

---

## The question

ERT inversion is underdetermined: 35 measurements constrain roughly 1000 mesh
cells, so many different subsurface models reproduce the same surface data. To
choose one, an inversion must assume something about the ground. Conventional
methods impose smoothness explicitly through a regularization term.

This work asks whether a **neural network's architecture** can supply that
assumption instead. The physics-based forward solver is unchanged; the only
substitution is the model description. Instead of one resistivity value per mesh
cell, a network maps position to resistivity and the inversion optimizes network
weights:

```
conventional :  m (one value per cell)  ->  G(m)  ->  predicted data
neural       :  (x, z) -> f_theta -> m  ->  G(m)  ->  predicted data
```

There is no pre-training and no database. Each inversion fits one dataset from
scratch, in the deep-image-prior sense.

## Seven methods, identical conditions

Same forward solver, mesh, survey (16-electrode Wenner, 35 measurements), noise
(1.5%), starting model, and stopping rule. All compared at **matched data fit**
(chi^2 = 1), not matched iteration count.

**Group 1 — conventional, explicit regularization**
smooth-L2 · total variation

**Group 2 — neural, optimized per dataset**
ReLU · tanh · SIREN · Fourier-feature · CNN deep image prior

## Main results

Full write-ups in [`docs/`](docs/). Headline findings:

**Configuration dominated the comparison more than architecture did.** Under a
shared network capacity and iteration budget, total variation appeared ~50% more
accurate than any network. Six conclusions drawn under that protocol turned out
to measure the shared setting rather than the architecture — including one
network reported as unable to fit the data at all, which fits 5/5 seeds once its
learning-rate schedule is not derived from a budget chosen for a different
method.

**Configuration can be chosen without ground truth — for some architectures.**
Selecting capacity, frequency scale, and budget on a separate calibration target
reproduced ground-truth-tuned accuracy exactly for the Fourier network (0.4046
and 0.5451 against 0.405 and 0.545), which then matched TV on the hillslope
target. The same protocol cost SIREN 18.6% and moved TV by nothing measurable.

**Fitting the data is not recovering the model.** Two models at identical
chi^2 = 1 predicted data within 1.46% of each other — below the 1.5% measurement
noise — while differing by 1.34 in log-RMS and carrying model errors of 0.731 and
1.221. Average error and anomaly geometry also converge differently: the leading
network matched TV on model error while placing anomaly centroids 2.8x farther
from truth.

## Install

```bash
uv sync
```

Requires Python >= 3.11. CPU-only works for smoke tests; the full sweeps were run
on the University of Iowa's Argon cluster.

## Reproducing

```bash
python step0_regression.py            # regression gate -- run this first
python inr_benchmark.py smoke         # quick single-architecture check

python stage1b.py all 8               # probe -> capacity -> tune -> eval -> conventional
python stage1b_geometry.py            # IoU, centroid error, SSIM, magnitude
python overfit_check.py               # does optimizing past the noise level hurt?
python talk_figures.py all            # presentation figures
```

On a scheduler, the `*.job` files are SGE batch scripts (`qsub run_stage1b.job`).

## Layout

| path | contents |
|---|---|
| `deepert/inr/` | architectures, physics-coupled training loop, targets |
| `docs/` | methods, Stage 1 and Stage 1b results, conclusions, poster and talk drafts |
| `inr_results/` | saved runs, including recovered resistivity fields |
| `*.py` | benchmark harnesses and analysis scripts |
| `tests/` | regression and smoke tests |

Architectures live in [`deepert/inr/networks.py`](deepert/inr/networks.py); the
training loop that couples Torch autograd to the NumPy/SciPy forward solver is in
[`deepert/inr/train.py`](deepert/inr/train.py).

## Attribution

The `deepert/` forward-modelling and inversion library — everything outside
`deepert/inr/` — was developed by **Hang Chen's group at the University of Iowa**
and is vendored here so this repository runs standalone. It is their work, not
mine.

My contribution is the neural reparameterization layer (`deepert/inr/`), the
benchmark harnesses, the analysis scripts, and the write-ups in `docs/`.

`resistivity_models_2d/` contains a single ParFlow-derived timestep used as an
evaluation target, included with permission; the raw simulation output stays in
the group's repository.

## Caveats

Synthetic data are generated with the same mesh and solver used to invert them
(an *inverse crime*). Every method inherits that bias identically, so it is fair
for ranking them, but these numbers are not evidence about absolute inversion
accuracy on field data.

Five seeds resolve differences above ~0.09 in model error. Several reported gaps
are smaller than that and are stated as ties.

## Acknowledgements

Hang Chen, Zhengyang Fang, Weiyu Guo, and Chen Xiong at the University of Iowa.

This work was conducted as part of the RESESS (Research Experiences in Solid
Earth Sciences for Students) internship program, supported by the NSF National
Geophysical Facility operated by EarthScope Consortium (NSF award 2435260).
