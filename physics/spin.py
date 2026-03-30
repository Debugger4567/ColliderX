from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SpinState:
    """
    Minimal spin carrier for sequential decays.
    """

    helicity: float
    quantization_axis: np.ndarray

    def __post_init__(self):
        if self.helicity not in (1.0, -1.0, 0.0):
            raise ValueError(f"Helicity must be +1, 01 or 0. Got {self.helicity}")

        axis = np.asarray(self.quantization_axis, dtype=float)
        mag = float(np.linalg.norm(axis))
        if mag < 1e-10:
            raise ValueError("Quantization axis cannot be zero vector")

        axis = axis / mag
        axis.setflags(write=False)
        object.__setattr__(self, "quantization_axis", axis)

    @classmethod
    def unpolarized(cls) -> "SpinState":
        return cls(
            helicity=0.0,
            quantization_axis=np.array([0.0, 0.0, 1.0], dtype=float),
        )

    @classmethod
    def from_p4(
        cls, p4: tuple[float, float, float, float], helicity: float
    ) -> "SpinState":
        _, px, py, pz = p4
        p_vec = np.array([px, py, pz], dtype=float)
        mag = float(np.linalg.norm(p_vec))
        axis = np.array([0.0, 0.0, 1.0], dtype=float) if mag < 1e-10 else p_vec / mag
        return cls(helicity=helicity, quantization_axis=axis)

    @property
    def is_polarized(self) -> bool:
        return self.helicity != 0.0

    def __eq__(self, other: object) -> bool:
        """Compare spin states: helicity + quantization axis (robust to numpy arrays)."""
        if not isinstance(other, SpinState):
            return False
        return self.helicity == other.helicity and np.allclose(
            self.quantization_axis, other.quantization_axis
        )

    def __hash__(self) -> int:
        """Hash for frozen dataclass with numpy array field."""
        return hash((self.helicity, tuple(self.quantization_axis.round(10))))

    def cos_theta_wrt(self, direction: np.ndarray) -> float:
        d = np.asarray(direction, dtype=float)
        d_mag = float(np.linalg.norm(d))
        if d_mag < 1e-10:
            return 0.0
        return float(np.dot(self.quantization_axis, d / d_mag))
