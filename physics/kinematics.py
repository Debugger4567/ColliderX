import math
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

# -----------------------------
# 4-Vector representation
# -----------------------------

@dataclass
class FourVector:
    """
    Lorentz 4-vector in natural units (c=1).
    All quantities in GeV.
    """
    E: float
    px: float
    py: float
    pz: float

    @property
    def p(self) -> np.ndarray:
        """Return 3-momentum vector."""
        return np.array([self.px, self.py, self.pz])

    @property
    def magnitude(self) -> float:
        """|p| - magnitude of 3-momentum."""
        return np.linalg.norm(self.p)

    @property
    def mass(self) -> float:
        """Invariant mass m = sqrt(E² - |p|²)."""
        m2 = self.E**2 - self.magnitude**2
        return math.sqrt(max(m2, 0.0))

    def beta_vec(self) -> np.ndarray:
        """Return β = p/E (velocity vector in natural units)."""
        if self.E <= 0:
            return np.zeros(3)
        return self.p / self.E

    def boost(self, beta: np.ndarray) -> 'FourVector':
        """Apply Lorentz boost by velocity vector beta."""
        p4 = np.array([self.E, self.px, self.py, self.pz])
        boosted = lorentz_boost_array(p4, beta)
        return FourVector(*boosted)

    def __add__(self, other: 'FourVector') -> 'FourVector':
        return FourVector(
            self.E + other.E,
            self.px + other.px,
            self.py + other.py,
            self.pz + other.pz
        )

    def __sub__(self, other: 'FourVector') -> 'FourVector':
        return FourVector(
            self.E - other.E,
            self.px - other.px,
            self.py - other.py,
            self.pz - other.pz
        )

    def __repr__(self):
        return (f"FourVector(E={self.E:.3f}, px={self.px:.3f}, "
                f"py={self.py:.3f}, pz={self.pz:.3f})")


# -----------------------------------------------------------
# Lorentz boost utilities
# -----------------------------------------------------------

def lorentz_boost_array(p4: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Boost a 4-vector [E, px, py, pz] by velocity beta (3-vector).
    Convention: applying +β maps rest frame → lab frame.
    
    Args:
        p4: 4-momentum [E, px, py, pz]
        beta: velocity vector (v/c in natural units)
    Returns:
        Boosted 4-vector [E', px', py', pz']
    """
    beta = np.asarray(beta, dtype=float)
    p4 = np.asarray(p4, dtype=float)
    beta2 = np.dot(beta, beta)
    
    if beta2 >= 1.0:
        raise ValueError("beta^2 must be < 1 (subluminal)")
    if beta2 <= 1e-18:
        return p4.copy()
    
    gamma = 1.0 / math.sqrt(1.0 - beta2)
    bp = np.dot(beta, p4[1:])
    E_prime = gamma * (p4[0] + bp)
    p_prime = p4[1:] + ((gamma - 1.0) * bp / beta2 + gamma * p4[0]) * beta
    return np.array([E_prime, *p_prime])


def isotropic_direction(rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Generate a random isotropic direction (unit vector) in 3D.
    
    Args:
        rng: Optional numpy random generator for reproducibility
    Returns:
        Unit 3-vector [x, y, z]
    """
    rng = rng or np.random.default_rng()
    costheta = rng.uniform(-1, 1)
    sintheta = math.sqrt(max(1 - costheta**2, 0.0))
    phi = rng.uniform(0, 2 * math.pi)
    return np.array([
        sintheta * math.cos(phi),
        sintheta * math.sin(phi),
        costheta
    ])


# -----------------------------------------------------------
# Relativistic decay kinematics
# -----------------------------------------------------------

def generate_two_body_decay(
    parent_mass: float, 
    daughter_masses: Tuple[float, float],
    rng: Optional[np.random.Generator] = None
) -> List[FourVector]:
    """
    Isotropic two-body decay in parent rest frame.

    Args:
        parent_mass: mass of parent particle (GeV)
        daughter_masses: tuple of (m1, m2) in GeV
        rng: Optional RNG for reproducibility
    Returns:
        List of two FourVectors [daughter1, daughter2]
    """
    rng = rng or np.random.default_rng()
    m0, (m1, m2) = parent_mass, daughter_masses

    if m0 < (m1 + m2):
        raise ValueError(f"Decay not kinematically allowed: {m0} < {m1 + m2}")

    # Relativistic momentum magnitude in rest frame
    p_mag = math.sqrt(
        (m0**2 - (m1 + m2)**2) * (m0**2 - (m1 - m2)**2)
    ) / (2 * m0)

    direction = isotropic_direction(rng)
    p1 = p_mag * direction
    p2 = -p1

    E1 = math.sqrt(m1**2 + p_mag**2)
    E2 = math.sqrt(m2**2 + p_mag**2)

    return [
        FourVector(E1, *p1),
        FourVector(E2, *p2)
    ]


def generate_three_body_decay(
    parent_mass: float, 
    daughter_masses: Tuple[float, float, float],
    rng: Optional[np.random.Generator] = None
) -> List[FourVector]:
    """
    Three-body decay with proper phase space sampling via sequential two-body.
    
    Method:
    1. Parent → (12) + 3  [two-body]
    2. (12) → 1 + 2       [two-body in (12) rest frame]
    3. Boost 1 and 2 to parent rest frame
    
    Args:
        parent_mass: parent rest mass (GeV)
        daughter_masses: (m1, m2, m3) in GeV
        rng: Optional RNG
    Returns:
        List of three FourVectors [d1, d2, d3]
    """
    rng = rng or np.random.default_rng()
    m0 = parent_mass
    m1, m2, m3 = daughter_masses

    if m0 < (m1 + m2 + m3):
        raise ValueError(f"Decay not kinematically allowed: {m0} < {m1+m2+m3}")

    # Randomly pick invariant mass of (1,2) subsystem
    s12_min = (m1 + m2)**2
    s12_max = (m0 - m3)**2
    s12 = rng.uniform(s12_min, s12_max)
    m12 = math.sqrt(s12)

    # Step 1: Decay parent → (12) + 3
    p_mag_3 = math.sqrt(
        (m0**2 - (m12 + m3)**2) * (m0**2 - (m12 - m3)**2)
    ) / (2 * m0)
    
    dir3 = isotropic_direction(rng)
    p3 = p_mag_3 * dir3
    E3 = math.sqrt(m3**2 + p_mag_3**2)
    
    # (12) system has momentum -p3 and energy E12
    E12 = m0 - E3
    p12 = -p3

    # Step 2: Decay (12) → 1 + 2 in (12) rest frame
    p_mag_12 = math.sqrt(
        (m12**2 - (m1 + m2)**2) * (m12**2 - (m1 - m2)**2)
    ) / (2 * m12)
    
    dir12 = isotropic_direction(rng)
    p1_rf = p_mag_12 * dir12
    p2_rf = -p1_rf
    E1_rf = math.sqrt(m1**2 + p_mag_12**2)
    E2_rf = math.sqrt(m2**2 + p_mag_12**2)

    # Step 3: Boost daughters from (12) rest frame to parent rest frame
    beta = p12 / E12  # Corrected: β of (12) system

    fv1_rf = FourVector(E1_rf, *p1_rf)
    fv2_rf = FourVector(E2_rf, *p2_rf)
    
    fv1 = fv1_rf.boost(beta)
    fv2 = fv2_rf.boost(beta)
    fv3 = FourVector(E3, *p3)

    return [fv1, fv2, fv3]
