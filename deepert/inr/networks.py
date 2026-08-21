"""Coordinate networks used as the model parameterization f_theta(x, z) -> log rho.

Three architectures share one contract: take (n_cells, 2) coordinates in
[-1, 1] and return (n_cells,) log-resistivity. They differ only in how they
fight spectral bias -- the tendency of a plain coordinate MLP to learn smooth,
low-frequency functions long before sharp ones.

    ReLUMLP            no cure. The baseline that exhibits the bias.
    SIREN              sine activations throughout (Sitzmann et al., 2020).
    FourierFeatureMLP  random Fourier encoding on the input, then a ReLU
                       trunk (Tancik et al., 2020).

All three carry the same ``log_rho_mean`` output offset, so all three start
from the same homogeneous field and differ only in inductive bias.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# Reference capacity: SIREN/ReLU at hidden=128, layers=3, 2 input coordinates.
DEFAULT_PARAMETER_BUDGET = 33537


class CoordinateNetwork(nn.Module):
    """Base contract: (n_cells, 2) coords in [-1, 1] -> (n_cells,) log-resistivity.

    The constant ``log_rho_mean`` offset is the warm start. At initialization
    the trunk emits roughly zero-mean values, so the field begins near
    ``exp(log_rho_mean)`` everywhere -- the neural equivalent of the
    homogeneous starting model a traditional cell-based inversion uses. In
    practice callers set it from the median observed apparent resistivity,
    which uses only measured data and leaks nothing from the true model.
    """

    def __init__(self, log_rho_mean: float = float(np.log(100.0))):
        super().__init__()
        self.log_rho_mean = float(log_rho_mean)

    def trunk(self, coords: torch.Tensor) -> torch.Tensor:
        """Return (n_cells, 1) raw output before the offset."""

        raise NotImplementedError

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.trunk(coords).squeeze(-1) + self.log_rho_mean


class SineLayer(nn.Module):
    """Linear layer followed by ``sin(w0 * .)``, with the SIREN weight init.

    ``nn.Linear`` draws its own weight and bias from the Torch RNG before
    ``uniform_`` overwrites the weight, so each layer consumes three draws and
    only two survive. Construction order therefore fixes the RNG stream.
    """

    def __init__(self, in_f: int, out_f: int, w0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        with torch.no_grad():
            bound = 1.0 / in_f if is_first else np.sqrt(6.0 / in_f) / w0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))


class SIREN(CoordinateNetwork):
    """Sine-activation coordinate network.

    ``w0`` scales the pre-activation and sets the frequency band the network
    can express. The original paper uses 30; the demo this was extracted from
    used 5. It is swept rather than assumed.
    """

    def __init__(
        self,
        hidden: int = 128,
        layers: int = 3,
        w0: float = 5.0,
        log_rho_mean: float = float(np.log(100.0)),
    ):
        super().__init__(log_rho_mean)
        net = [SineLayer(2, hidden, w0=w0, is_first=True)]
        for _ in range(layers - 1):
            net.append(SineLayer(hidden, hidden, w0=w0))
        net.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*net)

    def trunk(self, coords):
        return self.net(coords)


class ReLUMLP(CoordinateNetwork):
    """Plain coordinate MLP -- SIREN minus the sine trick.

    Identical topology to SIREN (2 -> hidden -> ... -> 1) so the parameter
    count matches exactly at the same width and depth. Initialization is
    PyTorch's ``nn.Linear`` default, which is what "plain ReLU MLP baseline"
    means in the INR literature; no special scheme is applied, deliberately.

    This architecture has no frequency knob, so its only tunable is the
    learning rate. That asymmetry is a property of the method, not an
    unfairness to be corrected.
    """

    def __init__(
        self,
        hidden: int = 128,
        layers: int = 3,
        in_features: int = 2,
        log_rho_mean: float = float(np.log(100.0)),
    ):
        super().__init__(log_rho_mean)
        net: list[nn.Module] = [nn.Linear(in_features, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            net.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        net.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*net)

    def trunk(self, coords):
        return self.net(coords)


class TanhMLP(CoordinateNetwork):
    """Coordinate MLP with tanh activations.

    The classic pre-ReLU coordinate network: smooth and infinitely
    differentiable, with a strong bias toward low-frequency structure --
    historically the reason early coordinate networks produced blurry fields.
    Included as the second no-frequency-knob baseline alongside ReLU: the two
    isolate what the activation's smoothness alone does to recovered structure.
    Topology matches ReLUMLP exactly, so parameters match at 33,537.
    """

    def __init__(
        self,
        hidden: int = 128,
        layers: int = 3,
        in_features: int = 2,
        log_rho_mean: float = float(np.log(100.0)),
    ):
        super().__init__(log_rho_mean)
        net: list[nn.Module] = [nn.Linear(in_features, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            net.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        net.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*net)

    def trunk(self, coords):
        return self.net(coords)


def match_dip_channels(
    target_parameters: int,
    latent_channels: int,
    *,
    max_channels: int = 256,
) -> int:
    """Channel width whose DIP decoder parameter count is closest to the target.

    Decoder layout (3x3 convs): latent->c, c->c, c->c, c->1, so
    params(c) = (9*latent*c + c) + 2*(9*c*c + c) + (9*c + 1).
    """

    best_c, best_error = 1, None
    for c in range(1, int(max_channels) + 1):
        count = (9 * latent_channels * c + c) + 2 * (9 * c * c + c) + (9 * c + 1)
        error = abs(count - int(target_parameters))
        if best_error is None or error < best_error:
            best_c, best_error = c, error
    return best_c


class DIPDecoder(CoordinateNetwork):
    """CNN Deep Image Prior: a convolutional decoder generates the whole field.

    Following Ulyanov et al., a FIXED random latent tensor (a buffer, never
    trained -- the analogue of Fourier's fixed B) is decoded through 3x3
    convolutions with two bilinear 2x upsamplings into a (nz, nx) image on the
    structured grid; each rectangle's value is shared by its two triangles.
    The implicit prior comes from convolutional locality and the upsampling
    pyramid rather than from a coordinate encoding.

    Unlike the INRs this parameterization has a FIXED output grid and ignores
    the input coordinates entirely -- it accepts them only to honour the
    CoordinateNetwork contract (and validates the cell count against them).
    It therefore cannot be resampled at other resolutions, one of the
    taxonomy distinctions the benchmark is designed to surface.
    """

    def __init__(
        self,
        nz: int = 20,
        nx: int = 25,
        latent_channels: int = 8,
        channels: int | None = None,
        parameter_budget: int = DEFAULT_PARAMETER_BUDGET,
        log_rho_mean: float = float(np.log(100.0)),
    ):
        super().__init__(log_rho_mean)
        self.nz, self.nx = int(nz), int(nx)
        z0 = -(-self.nz // 4)                     # two 2x upsamplings must cover (nz, nx)
        x0 = -(-self.nx // 4)
        if channels is None:
            channels = match_dip_channels(parameter_budget, latent_channels)
        self.channels = int(channels)
        self.register_buffer("latent", torch.randn(1, latent_channels, z0, x0))
        c = self.channels
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, c, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(c, 1, 3, padding=1),
        )

    def trunk(self, coords):
        image = self.decoder(self.latent)[0, 0, : self.nz, : self.nx]
        values = image.reshape(-1).repeat_interleave(2)   # rectangle -> its 2 triangles
        if values.shape[0] != coords.shape[0]:
            raise ValueError(
                f"DIPDecoder grid ({self.nz}x{self.nx} -> {values.shape[0]} cells) "
                f"does not match the mesh ({coords.shape[0]} cells)"
            )
        return values.unsqueeze(-1)


class FourierFeatures(nn.Module):
    """Random Fourier encoding gamma(v) = [cos(2 pi B v), sin(2 pi B v)].

    Following Tancik et al. (2020), ``B`` has entries drawn from N(0, sigma^2)
    and is **fixed, not trained** -- it is registered as a buffer so it moves
    with the module and is excluded from the trainable-parameter count.

    ``sigma`` is the method's critical hyperparameter, not a detail. Too large
    and the network fits high-frequency noise; too small and the encoding
    degenerates until the model is effectively a plain ReLU MLP. There is no
    safe default, which is precisely why it is swept.
    """

    def __init__(self, in_features: int = 2, n_features: int = 128, sigma: float = 1.0):
        super().__init__()
        self.register_buffer("B", torch.randn(n_features, in_features) * float(sigma))
        self.sigma = float(sigma)

    @property
    def out_features(self) -> int:
        return 2 * int(self.B.shape[0])

    def forward(self, x):
        projected = 2.0 * np.pi * (x @ self.B.T)
        return torch.cat([torch.cos(projected), torch.sin(projected)], dim=-1)


def match_hidden_width(
    target_parameters: int,
    in_features: int,
    layers: int,
    *,
    max_width: int = 1024,
) -> int:
    """Return the hidden width whose parameter count is closest to the target.

    Fourier encoding widens the input from 2 to 2*n_features, which inflates
    the first layer enormously (at n_features=128 the input is 256-wide). Left
    alone, the Fourier model would carry roughly twice SIREN's parameters and
    any advantage it showed would be confounded with extra capacity. Shrinking
    the trunk restores a like-for-like comparison.
    """

    best_width, best_error = 1, None
    for width in range(1, int(max_width) + 1):
        count = (in_features * width + width) + (layers - 1) * (width * width + width) + (width + 1)
        error = abs(count - int(target_parameters))
        if best_error is None or error < best_error:
            best_width, best_error = width, error
    return best_width


class FourierFeatureMLP(CoordinateNetwork):
    """Random Fourier feature encoding followed by a ReLU trunk.

    Width defaults to whatever matches ``parameter_budget`` trainable
    parameters, so capacity is comparable to SIREN and ReLUMLP rather than
    inflated by the wide encoded input.
    """

    def __init__(
        self,
        hidden: int | None = None,
        layers: int = 3,
        n_features: int = 128,
        sigma: float = 1.0,
        parameter_budget: int = DEFAULT_PARAMETER_BUDGET,
        log_rho_mean: float = float(np.log(100.0)),
    ):
        super().__init__(log_rho_mean)
        self.encoding = FourierFeatures(2, n_features, sigma)
        in_features = self.encoding.out_features
        if hidden is None:
            hidden = match_hidden_width(parameter_budget, in_features, layers)
        self.hidden = int(hidden)

        net: list[nn.Module] = [nn.Linear(in_features, self.hidden), nn.ReLU()]
        for _ in range(layers - 1):
            net.extend([nn.Linear(self.hidden, self.hidden), nn.ReLU()])
        net.append(nn.Linear(self.hidden, 1))
        self.net = nn.Sequential(*net)

    def trunk(self, coords):
        return self.net(self.encoding(coords))


ARCHITECTURES: dict[str, type[CoordinateNetwork]] = {
    "relu": ReLUMLP,
    "tanh": TanhMLP,
    "siren": SIREN,
    "fourier": FourierFeatureMLP,
    "dip": DIPDecoder,
}


def build_network(name: str, **kwargs) -> CoordinateNetwork:
    """Construct an architecture by registry name.

    Seed the Torch RNG immediately before calling this; every architecture
    consumes the RNG during construction, and Fourier draws ``B`` as well.
    """

    if name not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {name!r}; expected one of {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[name](**kwargs)


def count_parameters(net: nn.Module, *, trainable_only: bool = True) -> int:
    """Count parameters, for matching capacity across architectures.

    Buffers -- notably the Fourier ``B`` matrix -- are excluded, because they
    are fixed random constants rather than degrees of freedom the inversion
    optimizes.
    """

    return sum(p.numel() for p in net.parameters() if p.requires_grad or not trainable_only)
