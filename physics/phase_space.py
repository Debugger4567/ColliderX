"""
Lorentz-invariant phase space generator using Raubold–Lynch algorithm.

Supports arbitrary N-body decays with proper phase-space weighting.

Units: MeV, c = 1 (matches colliderx.db)
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple, Callable, Optional
from .kinematics import FourVector, isotropic_direction


def generate_n_body_decay(
    parent_p4: FourVector,
    masses: List[float],
    rng: Optional[np.random.Generator] = None,
    matrix_element: Optional[Callable] = None
) -> Tuple[List[FourVector], float]:
    """
    Generate an N-body decay using Raubold–Lynch algorithm with phase-space weight.

    Parameters
    ----------
    parent_p4 : FourVector
        Parent four-momentum (can be in any frame)
    masses : list of float
        Final-state particle masses (MeV)
    rng : numpy Generator, optional
        Random number generator
    matrix_element : callable, optional
        Function(final_p4s) -> weight for non-uniform phase space

    Returns
    -------
    (final_particles, event_weight)
        final_particles: list of FourVector (lab frame)
        event_weight: phase-space weight (1.0 for uniform, >1 if ME-weighted)
    """
    rng = rng or np.random.default_rng()
    N = len(masses)

    # Validate inputs
    if N < 2:
        raise ValueError("Need at least two final-state particles.")
    if any(m < 0 for m in masses):
        raise ValueError("All masses must be non-negative.")

    # Boost parent to rest frame
    beta_parent = parent_p4.beta()
    parent_rf = parent_p4.boost(-beta_parent)
    M = parent_rf.mass

    if M <= 0:
        raise ValueError("Parent mass must be positive.")
    if sum(masses) + 1e-6 > M:
        raise ValueError(f"Kinematically forbidden: Σm={sum(masses):.3f} > M={M:.3f}")

    # Generate virtual masses
    virtual_masses = [M]
    remaining_mass = sum(masses)

    for i in range(N - 2):
        m_remain = remaining_mass - masses[i]
        m_max = virtual_masses[-1] - masses[i]
        m_min = m_remain

        if m_min > m_max:
            raise ValueError(f"Phase space violation at step {i}")

        r = rng.random()
        s_min = m_min**2
        s_max = m_max**2
        s = s_min + r * (s_max - s_min)
        m_virtual = math.sqrt(s)
        virtual_masses.append(m_virtual)
        remaining_mass -= masses[i]

    virtual_masses.append(masses[-1])

    # Sequential two-body decays
    final_particles = []
    current_p4 = FourVector(M, 0.0, 0.0, 0.0)
    weight = 1.0  # phase-space weight accumulator

    for i in range(N - 1):
        m_parent = virtual_masses[i]
        m1 = masses[i]
        m2 = virtual_masses[i + 1]

        term1 = m_parent**2 - (m1 + m2)**2
        term2 = m_parent**2 - (m1 - m2)**2
        if term1 * term2 < 0:
            raise ValueError("Kinematic failure in sequential decay")

        p_mag = math.sqrt(max(term1 * term2, 0.0)) / (2.0 * m_parent)

        # ✅ phase-space weighting (Raubold–Lynch factor)
        weight *= p_mag / m_parent

        direction = isotropic_direction(rng)
        p_vec = p_mag * direction

        E1 = math.sqrt(m1**2 + p_mag**2)
        E2 = math.sqrt(m2**2 + p_mag**2)

        p1 = FourVector(E1, p_vec[0], p_vec[1], p_vec[2])
        p2 = FourVector(E2, -p_vec[0], -p_vec[1], -p_vec[2])

        beta = current_p4.beta()
        p1_boosted = p1.boost(beta)
        p2_boosted = p2.boost(beta)

        final_particles.append(p1_boosted)
        current_p4 = p2_boosted

    # Boost to lab frame
    final_lab = [p.boost(beta_parent) for p in final_particles]

    # Apply matrix element weighting if provided
    if matrix_element:
        me_weight = matrix_element(final_lab)
        weight *= me_weight

    return final_lab, weight

'''
def validate_four_momentum_conservation(
    parent_p4: FourVector,
    final_p4s: List[FourVector],
    tolerance: float = 1e-3
) -> bool:
    """Check if 4-momentum is conserved within tolerance (MeV)."""
    total = sum(final_p4s, FourVector(0.0, 0.0, 0.0, 0.0))
    
    dE = abs(parent_p4.E - total.E)
    dp = np.linalg.norm(parent_p4.p - total.p)
    
    if dE > tolerance or dp > tolerance:
        print(f"⚠️  4-momentum NOT conserved:")
        print(f"   ΔE = {dE:.6f} MeV (tolerance: {tolerance})")
        print(f"   Δp = {dp:.6f} MeV (tolerance: {tolerance})")
        return False
    return True

'''
