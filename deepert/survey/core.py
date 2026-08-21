"""Survey primitives for ABMN-style ERT measurements."""

from __future__ import annotations

from dataclasses import dataclass

from deepert.utils.torch_runtime import Array
from deepert.utils.torch_runtime import torch_np

from deepert.utils.dtypes import FLOAT_DTYPE, INT_DTYPE


def _pairwise_distance(lhs: Array, rhs: Array) -> Array:
    return torch_np.linalg.norm(lhs - rhs, axis=-1)


@dataclass(frozen=True)
class Survey:
    """Electrode geometry and ABMN measurement indexing."""

    electrode_positions: Array
    measurements: Array

    @classmethod
    def from_arrays(cls, electrode_positions: Array, measurements: Array) -> "Survey":
        """Build a survey from electrode coordinates and ABMN indices."""

        positions = torch_np.asarray(electrode_positions, dtype=FLOAT_DTYPE)
        quads = torch_np.asarray(measurements, dtype=INT_DTYPE)

        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("electrode_positions must have shape (num_electrodes, 2)")
        if quads.ndim != 2 or quads.shape[1] != 4:
            raise ValueError("measurements must have shape (num_measurements, 4)")
        if bool(torch_np.any(quads < 0)):
            raise ValueError("measurements contain negative electrode indices")
        if quads.size and bool(torch_np.any(quads >= positions.shape[0])):
            raise ValueError("measurements reference electrodes outside the survey")

        sorted_quads = torch_np.sort(quads, axis=1)
        duplicates = torch_np.diff(sorted_quads, axis=1) == 0
        if bool(torch_np.any(duplicates)):
            raise ValueError("each ABMN measurement must use four distinct electrodes")

        return cls(electrode_positions=positions, measurements=quads)

    @property
    def electrode_count(self) -> int:
        """Number of electrodes."""

        return int(self.electrode_positions.shape[0])

    @property
    def measurement_count(self) -> int:
        """Number of ABMN measurements."""

        return int(self.measurements.shape[0])

    def geometric_factors(self) -> Array:
        """Return analytic half-space geometric factors for each ABMN row."""

        electrodes = self.electrode_positions[self.measurements]
        a = electrodes[:, 0]
        b = electrodes[:, 1]
        m = electrodes[:, 2]
        n = electrodes[:, 3]

        response = (
            1.0 / _pairwise_distance(a, m)
            - 1.0 / _pairwise_distance(a, n)
            - 1.0 / _pairwise_distance(b, m)
            + 1.0 / _pairwise_distance(b, n)
        )
        return 2.0 * torch_np.pi / response

    def apparent_resistivity(self, voltages: Array, currents: Array | float = 1.0) -> Array:
        """Convert measured voltages to apparent resistivity."""

        voltage_array = torch_np.asarray(voltages, dtype=FLOAT_DTYPE)
        current_array = torch_np.asarray(currents, dtype=FLOAT_DTYPE)
        return self.geometric_factors() * voltage_array / current_array
