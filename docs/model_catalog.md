# Deep learning architectures for ERT inversion — candidate catalog

Working catalog for the benchmarking study. Organized by **paradigm**, because
models in different paradigms cannot share a protocol without deciding what is
held constant (training-data budget, forward-solver access, cross-site
generalization). Within a paradigm, comparison is straightforward.

**Status legend** — `DONE` implemented and benchmarked · `DROP-IN` runs today
through `deepert.inr.fit_inr` with no new infrastructure · `NEEDS TRAINING`
requires a supervised training pipeline that does not yet exist · `RESEARCH`
substantial new implementation.

---

## A. Neural reparameterization (implicit priors)

Network *is* the model: `f_θ(x, z) → log ρ`, fit from scratch per dataset with
the FEM solver in the loop. **No training data.** All of these plug into the
existing harness unchanged.

| # | Model | Relevance to ERT | Status |
|---|---|---|---|
| 1 | ReLU MLP | Baseline; exhibits spectral bias | DONE |
| 2 | SIREN (sine activations) | Periodic basis, fine detail | DONE |
| 3 | Random Fourier features + ReLU | Input-stage frequency lift | DONE |
| 4 | Deterministic positional encoding (NeRF ladder) | Ablation: does randomness in **B** matter? | DROP-IN |
| 5 | Hash-grid encoding (Instant-NGP) | Multiresolution hash + tiny MLP; strongest modern INR, fast | DROP-IN |
| 6 | WIRE (Gabor wavelet activations) | Sharp features **and** noise robustness | DROP-IN |
| 7 | Gaussian-activation INR | Simple alternative nonlinearity, smooth spectrum control | DROP-IN |
| 8 | Multiplicative Filter Networks (FourierNet / GaborNet) | Products of filters instead of composition | DROP-IN |
| 9 | BACON (band-limited coordinate network) | Explicit, controllable frequency band | DROP-IN |
| 10 | KAN (Kolmogorov–Arnold Network) | Spline-based; untested as an INR for geophysics | DROP-IN |
| 11 | Learnable feature grid + tiny MLP | Grid-based prior, no coordinate encoding | DROP-IN |
| 12 | Plain learnable grid (no network) | Ablation: is the network doing anything? | DROP-IN |
| 13 | CNN decoder / Deep Image Prior | Convolutional generator from fixed latent — a genuine CNN **inside this paradigm** | DROP-IN |
| 14 | Deep Decoder | Under-parameterized CNN prior; real compression | DROP-IN |
| 15 | Modulated SIREN (FiLM conditioning) | Conditioning hook for time-lapse or multi-site | RESEARCH |

---

## B. Supervised direct inversion

Learn `d_obs → m` from many (data, model) pairs. **Requires a training-pair
generator** — forward-model the 365 ParFlow slices, plus augmentation. Once
built, all of these share it.

| # | Model | Relevance to ERT | Status |
|---|---|---|---|
| 16 | MLP direct inversion | Simplest baseline; without it, CNN gains are unquantified | NEEDS TRAINING |
| 17 | 1D CNN on measurement vector | Treats the survey as a signal | NEEDS TRAINING |
| 18 | 2D CNN on pseudosection | The workhorse of the published ERT-DL literature | NEEDS TRAINING |
| 19 | U-Net | Encoder–decoder with skips; standard for image-to-image inversion | NEEDS TRAINING |
| 20 | Attention U-Net / U-Net++ | Attention-gated variants | NEEDS TRAINING |
| 21 | ResNet / DenseNet backbone | Depth and residual connections | NEEDS TRAINING |
| 22 | FCN / SegNet | Older fully-convolutional baselines | NEEDS TRAINING |
| 23 | Vision Transformer (ViT) | Global receptive field over the pseudosection | NEEDS TRAINING |
| 24 | Swin Transformer | Hierarchical windowed attention | NEEDS TRAINING |
| 25 | **Set Transformer / DeepSets** | ERT data is a *set* of (a,b,m,n,ρₐ) tuples — handles **variable survey geometry**, which fixed-grid CNNs cannot | NEEDS TRAINING |
| 26 | Perceiver / Perceiver IO | Variable-size input, arbitrary output grid | NEEDS TRAINING |
| 27 | **Graph neural network** (GCN / GAT / GraphSAGE) | Operates natively on unstructured triangle meshes; CNNs assume a regular grid | NEEDS TRAINING |
| 28 | MeshGraphNet | GNN designed for FEM meshes specifically | NEEDS TRAINING |
| 29 | PointNet / PointNet++ | Electrode and measurement geometry as a point cloud | NEEDS TRAINING |
| 30 | Hybrid CNN–Transformer | Local features + global context | NEEDS TRAINING |

---

## C. Physics-guided and hybrid

Keep the forward solver in the loop but learn part of the inversion. **This
class directly attacks the finding that classical Gauss-Newton is both more
accurate and ~100× cheaper.**

| # | Model | Relevance to ERT | Status |
|---|---|---|---|
| 31 | Unrolled Gauss-Newton / learned iterative | Unroll N solver steps with learned components; the natural response to the cost gap | RESEARCH |
| 32 | LISTA-style unrolled sparse coding | Learned proximal steps | RESEARCH |
| 33 | Plug-and-play prior (PnP-ADMM) | Swap TV for a learned denoiser inside classical iterations | RESEARCH |
| 34 | RED (Regularization by Denoising) | Denoiser as an explicit regularizer | RESEARCH |
| 35 | Adversarially learned regularizer (ACR) | Learn the regularizer instead of choosing λ and an operator | RESEARCH |
| 36 | PINN (PDE residual in the loss) | Physics as a soft constraint rather than a solver call | RESEARCH |
| 37 | Deep equilibrium model (DEQ) | Infinite-depth unrolling at fixed memory | RESEARCH |
| 38 | Classical warm start + neural refinement | Cheap; tests whether networks can *improve* a good solution | DROP-IN |
| 39 | Learned preconditioner / step size | Accelerate the existing solver rather than replace it | RESEARCH |

---

## D. Generative and latent priors

Optimize a low-dimensional latent instead of a full model. **This is where real
dimensionality reduction happens** — unlike INRs, which are over-parameterized
relative to the mesh.

| # | Model | Relevance to ERT | Status |
|---|---|---|---|
| 40 | VAE latent-space inversion | Invert in a learned compressed code space | NEEDS TRAINING |
| 41 | GAN latent-space inversion | Strong prior, mode-collapse risk | NEEDS TRAINING |
| 42 | Autoencoder code-space inversion | Simplest compression baseline | NEEDS TRAINING |
| 43 | Normalizing flow prior | Tractable likelihood; **gives uncertainty quantification** | RESEARCH |
| 44 | Diffusion / score-based prior (DPS, DDRM) | State of the art for imaging inverse problems; novel for ERT | RESEARCH |
| 45 | Latent diffusion | Cheaper diffusion in code space | RESEARCH |
| 46 | Bayesian NN / MC-dropout | Posterior uncertainty on the recovered model | RESEARCH |

---

## E. Sequence models — time-lapse only

RNN-family models assume sequence structure. Static 2D ERT has none; **time-lapse
does**. If these are in scope, the study should be framed around time-lapse or a
reviewer will ask why they are present. The repo has 365 daily slices and a
classical time-lapse baseline to compare against.

| # | Model | Relevance to ERT | Status |
|---|---|---|---|
| 47 | RNN | Baseline sequence model over survey times | NEEDS TRAINING |
| 48 | LSTM | Long-range temporal dependence (wetting/drying cycles) | NEEDS TRAINING |
| 49 | GRU | Cheaper LSTM variant | NEEDS TRAINING |
| 50 | ConvLSTM | Spatial + temporal jointly | NEEDS TRAINING |
| 51 | Temporal CNN / TCN | Dilated causal convolutions | NEEDS TRAINING |
| 52 | Temporal transformer | Attention over time steps | NEEDS TRAINING |
| 53 | State-space model (S4 / Mamba) | Long sequences at low cost — 365 time steps | RESEARCH |
| 54 | Neural ODE | Continuous-time evolution of the resistivity field | RESEARCH |
| 55 | **4D INR** `f_θ(x, z, t)` | Temporal regularization for free; ~1000×365 unknowns compressed into one network | DROP-IN |

---

## F. Operator learning / forward surrogates

Learn the forward map `G` (or its inverse) as an operator. A fast surrogate
would accelerate **every** method in this catalog, since the FEM solve is the
bottleneck at ~0.065 s per iteration.

| # | Model | Relevance to ERT | Status |
|---|---|---|---|
| 56 | Fourier Neural Operator (FNO) | Resolution-independent operator learning | NEEDS TRAINING |
| 57 | DeepONet | Branch/trunk operator approximation | NEEDS TRAINING |
| 58 | Graph neural operator | Operator learning on unstructured meshes | NEEDS TRAINING |
| 59 | CNN forward surrogate | Replace the FEM solve; needs a differentiability/accuracy audit | NEEDS TRAINING |

---

## Infrastructure gates

Nothing in the catalog is blocked by architecture difficulty — it is blocked by
missing pipelines. In dependency order:

1. **Exists today.** Everything marked DROP-IN (15 models across A, C, E) runs
   through `deepert.inr.fit_inr` with the current harness, protocol and metrics.
2. **Training-pair generator.** Forward-model the 365 ParFlow slices (plus
   augmentation and multi-site geometries) into (d_obs, m) pairs, with
   train/val/test splits held out **by site**, not randomly. Unlocks all of B,
   D and F — roughly 30 models — and is the single highest-leverage thing to
   build.
3. **Solver-internal access.** Unrolling and plug-and-play need hooks inside
   the Gauss-Newton loop. Unlocks C.
4. **Field data.** `examples/8_real_data` expects a `ProcessedData/` directory
   that is absent from the repo. Required for any cross-site claim.

## Protocol note

The controls already established for the INR comparison should carry to every
model added: capacity matching where meaningful, hyperparameters tuned on a
development set and frozen before held-out evaluation, multiple seeds with
reported spread, error measured only where the survey has sensitivity, and a
classical Gauss-Newton baseline in every table. Supervised models additionally
need **site-held-out** splits — random splits over augmented synthetic data
will overstate generalization badly.

Cross-paradigm comparisons must state what is held constant. A supervised CNN
that has seen 10,000 training models is not comparable to an INR that has seen
none unless the training cost is accounted for somewhere in the table.
