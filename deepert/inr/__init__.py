"""Implicit neural representation (INR) reparameterization of ERT inversion.

Instead of one free resistivity value per mesh cell, a coordinate network
f_theta(x, z) -> log rho represents the field continuously, and the inversion
optimizes theta. The FEM forward solver is unchanged.

The network is never pre-trained: fitting *is* the inversion, run from scratch
for each dataset, driven only by that dataset's physics misfit.
"""

from deepert.inr.cases import (
    build_mesh,
    build_survey,
    synthetic_observations,
    two_layer_resistivity,
)
from deepert.inr.coords import cell_centers, normalized_coords
from deepert.inr.networks import (
    ARCHITECTURES,
    CoordinateNetwork,
    DEFAULT_PARAMETER_BUDGET,
    DIPDecoder,
    FourierFeatureMLP,
    FourierFeatures,
    ReLUMLP,
    SIREN,
    SineLayer,
    TanhMLP,
    build_network,
    count_parameters,
    match_dip_channels,
    match_hidden_width,
)
from deepert.inr.targets import block_anomaly, coverage_mask, parflow_slice, two_layer
from deepert.inr.train import fit_inr, seed_networks

__all__ = [
    "ARCHITECTURES",
    "DEFAULT_PARAMETER_BUDGET",
    "CoordinateNetwork",
    "DIPDecoder",
    "FourierFeatureMLP",
    "FourierFeatures",
    "ReLUMLP",
    "SIREN",
    "SineLayer",
    "TanhMLP",
    "block_anomaly",
    "build_mesh",
    "build_network",
    "build_survey",
    "cell_centers",
    "count_parameters",
    "coverage_mask",
    "fit_inr",
    "match_hidden_width",
    "normalized_coords",
    "parflow_slice",
    "seed_networks",
    "synthetic_observations",
    "two_layer",
    "two_layer_resistivity",
]
