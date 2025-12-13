"""
Kinematics helpers for ColliderX.

Units: MeV (natural units c = 1).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
import warnings

# -----------------------------
# FourVector
# -----------------------------
@dataclass
class FourVector:
    E: float
    px: float
    py: float
    pz: float

    @property
    def p(self) -> np.ndarray:
        return np.array([self.px, self.py, self.pz], dtype=float)

    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(self.p))

    @property
    def mass(self) -> float:
        m2 = self.E * self.E - self.magnitude * self.magnitude
        return math.sqrt(max(m2, 0.0))

    def beta(self) -> np.ndarray:
        if self.E == 0.0:
            return np.zeros(3, dtype=float)
        return self.p / self.E

    def boost(self, beta: np.ndarray) -> "FourVector":
        p4 = np.array([self.E, self.px, self.py, self.pz], dtype=float)
        b = np.asarray(beta, dtype=float)
        boosted = lorentz_boost_array(p4, b)
        return FourVector(float(boosted[0]), float(boosted[1]), float(boosted[2]), float(boosted[3]))

    def __add__(self, other: "FourVector") -> "FourVector":
        return FourVector(self.E + other.E, self.px + other.px, self.py + other.py, self.pz + other.pz)

    def __sub__(self, other: "FourVector") -> "FourVector":
        return FourVector(self.E - other.E, self.px - other.px, self.py - other.py, self.pz - other.pz)

    def __repr__(self) -> str:
        return f"FourVector(E={self.E:.6f}, px={self.px:.6f}, py={self.py:.6f}, pz={self.pz:.6f})"


# -----------------------------
# Lorentz boost
# -----------------------------
def lorentz_boost_array(p4: np.ndarray, beta: np.ndarray) -> np.ndarray:
    beta = np.asarray(beta, dtype=float)
    p4 = np.asarray(p4, dtype=float)
    beta2 = float(np.dot(beta, beta))
    if beta2 >= 1.0:
        raise ValueError("beta^2 < 1 required.")
    if beta2 <= 1e-18:
        return p4.copy()
    gamma = 1.0 / math.sqrt(1.0 - beta2)
    bp = float(np.dot(beta, p4[1:]))
    Eprime = gamma * (p4[0] + bp)
    factor = ((gamma - 1.0) * bp / beta2) + gamma * p4[0]
    pprime = p4[1:] + factor * beta
    return np.array([Eprime, pprime[0], pprime[1], pprime[2]], dtype=float)


# -----------------------------
# Isotropic direction
# -----------------------------
def isotropic_direction(rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    u = rng.uniform(-1.0, 1.0)
    phi = rng.uniform(0.0, 2.0 * math.pi)
    sint = math.sqrt(max(0.0, 1.0 - u * u))
    return np.array([sint * math.cos(phi), sint * math.sin(phi), u], dtype=float)


# -----------------------------
# Two-body decay
# -----------------------------
def generate_two_body_decay(parent_mass: float,
                            daughter_masses: Tuple[float, float],
                            rng: Optional[np.random.Generator] = None) -> List[FourVector]:
    warnings.warn("generate_two_body_decay is deprecated. Use phase_space.generate_n_body_decay instead.", DeprecationWarning, stacklevel=2)
    rng = rng or np.random.default_rng()
    m0 = float(parent_mass)
    m1, m2 = map(float, daughter_masses)

    if m0 + 1e-12 < (m1 + m2):
        raise ValueError("Two-body decay kinematically forbidden.")

    if abs(m0 - (m1 + m2)) < 1e-12:
        # threshold: both at rest
        return [FourVector(m1, 0.0, 0.0, 0.0), FourVector(m2, 0.0, 0.0, 0.0)]

    term1 = m0 * m0 - (m1 + m2) ** 2
    term2 = m0 * m0 - (m1 - m2) ** 2
    p_mag = math.sqrt(max(term1 * term2, 0.0)) / (2.0 * m0)

    dirv = isotropic_direction(rng)
    p1 = p_mag * dirv
    p2 = -p1
    E1 = math.sqrt(m1 * m1 + p_mag * p_mag)
    E2 = math.sqrt(m2 * m2 + p_mag * p_mag)

    return [
        FourVector(E1, float(p1[0]), float(p1[1]), float(p1[2])),
        FourVector(E2, float(p2[0]), float(p2[1]), float(p2[2]))
    ]


# -----------------------------
# Three-body decay (phase space)
# -----------------------------
def generate_three_body_decay(parent_mass: float,
                              daughter_masses: Tuple[float, float, float],
                              rng: Optional[np.random.Generator] = None) -> List[FourVector]:
    warnings.warn("generate_three_body_decay is deprecated. Use phase_space.generate_n_body_decay instead.", DeprecationWarning, stacklevel=2)
    rng = rng or np.random.default_rng()
    m0 = float(parent_mass)
    m1, m2, m3 = map(float, daughter_masses)

    if m0 + 1e-12 < (m1 + m2 + m3):
        raise ValueError("Three-body decay kinematically forbidden.")

    s12_min = (m1 + m2) ** 2
    s12_max = (m0 - m3) ** 2
    s12 = rng.uniform(s12_min, s12_max)
    m12 = math.sqrt(max(s12, 0.0))

    # Parent -> (12) + 3
    t1 = m0 * m0 - (m12 + m3) ** 2
    t2 = m0 * m0 - (m12 - m3) ** 2
    p3_mag = math.sqrt(max(t1 * t2, 0.0)) / (2.0 * m0)
    dir3 = isotropic_direction(rng)
    p3 = p3_mag * dir3
    E3 = math.sqrt(m3 * m3 + p3_mag * p3_mag)
    E12 = m0 - E3
    p12 = -p3

    # (12) -> 1 + 2 in (12) RF
    u1 = m12 * m12 - (m1 + m2) ** 2
    u2 = m12 * m12 - (m1 - m2) ** 2
    p12_mag = math.sqrt(max(u1 * u2, 0.0)) / (2.0 * m12) if m12 > 0 else 0.0

    dir12 = isotropic_direction(rng)
    p1_rf = p12_mag * dir12
    p2_rf = -p1_rf
    E1_rf = math.sqrt(m1 * m1 + p12_mag * p12_mag)
    E2_rf = math.sqrt(m2 * m2 + p12_mag * p12_mag)

    beta12 = p12 / E12 if E12 != 0.0 else np.zeros(3, dtype=float)

    fv1 = FourVector(E1_rf, p1_rf[0], p1_rf[1], p1_rf[2]).boost(beta12)
    fv2 = FourVector(E2_rf, p2_rf[0], p2_rf[1], p2_rf[2]).boost(beta12)
    fv3 = FourVector(E3, p3[0], p3[1], p3[2])
    return [fv1, fv2, fv3]
