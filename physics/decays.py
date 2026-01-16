import math
import random
import numpy as np
from typing import List, Optional
from .kinematics import FourVector


def _two_body_decay(parent_mass, m1, m2, rng: Optional[np.random.Generator] = None):
    """
    2-body isotropic decay. Returns [FourVector, FourVector].
    """
    assert rng is not None, "RNG must be provided for reproducibility"
    
    p = math.sqrt(
        max(
            0,
            (parent_mass**2 - (m1 + m2) ** 2) *
            (parent_mass**2 - (m1 - m2) ** 2),
        )
    ) / (2 * parent_mass)

    costheta = rng.uniform(-1, 1)
    sintheta = math.sqrt(1 - costheta**2)
    phi = rng.uniform(0, 2 * math.pi)

    px, py, pz = p * sintheta * math.cos(phi), p * sintheta * math.sin(phi), p * costheta

    E1, E2 = math.sqrt(m1**2 + p**2), math.sqrt(m2**2 + p**2)

    return [
        FourVector(E1, px, py, pz),
        FourVector(E2, -px, -py, -pz),
    ]


def _three_body_decay(parent_mass, m1, m2, m3, rng: Optional[np.random.Generator] = None):
    """
    3-body Raubold–Lynch algorithm. Returns [FourVector, FourVector, FourVector].
    """
    assert rng is not None, "RNG must be provided for reproducibility"
    
    m12_min, m12_max = m1 + m2, parent_mass - m3
    m12 = math.sqrt(rng.uniform(m12_min**2, m12_max**2))

    # first stage: parent -> X + p3
    E3 = (parent_mass**2 + m3**2 - m12**2) / (2 * parent_mass)
    p3 = math.sqrt(max(0, E3**2 - m3**2))
    p3_vec = FourVector(E3, 0, 0, p3)
    X = FourVector(parent_mass - E3, 0, 0, -p3)

    # second stage: X -> p1 + p2
    p = math.sqrt(max(0, (m12**2 - (m1 + m2) ** 2) * (m12**2 - (m1 - m2) ** 2))) / (2 * m12)

    costheta = rng.uniform(-1, 1)
    sintheta = math.sqrt(1 - costheta**2)
    phi = rng.uniform(0, 2 * math.pi)

    px, py, pz = p * sintheta * math.cos(phi), p * sintheta * math.sin(phi), p * costheta
    E1, E2 = math.sqrt(m1**2 + p**2), math.sqrt(m2**2 + p**2)

    p1, p2 = FourVector(E1, px, py, pz), FourVector(E2, -px, -py, -pz)

    # boost p1, p2 into parent frame
    beta = np.array([X.px / X.E, X.py / X.E, X.pz / X.E], dtype=float)
    p1 = p1.boost(beta)
    p2 = p2.boost(beta)

    return [p1, p2, p3_vec]


def decay_particle(parent_mass: float, daughter_masses: List[float], rng: Optional[np.random.Generator] = None) -> List[FourVector]:
    """
    Dispatcher: chooses 2- or 3-body decay.
    Always returns list of FourVectors.
    
    Args:
        parent_mass: Rest mass of parent (MeV)
        daughter_masses: List of daughter rest masses (MeV)
        rng: Random number generator (must be provided for reproducibility)
    
    Returns:
        List of FourVectors for daughters
    """
    assert rng is not None, "RNG must be provided for reproducibility"
    
    if len(daughter_masses) == 2:
        return _two_body_decay(parent_mass, daughter_masses[0], daughter_masses[1], rng)
    elif len(daughter_masses) == 3:
        return _three_body_decay(parent_mass, daughter_masses[0], daughter_masses[1], daughter_masses[2], rng)
    else:
        raise NotImplementedError(f"{len(daughter_masses)}-body decay not supported")
