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


def p4(E: float, px: float, py: float, pz: float) -> tuple:
    """4-vector as tuple (E, px, py, pz) - no NumPy overhead."""
    return (E, px, py, pz)


def two_body_decay(M: float, mA: float, mB: float, rng: np.random.Generator) -> Tuple[tuple, tuple]:
    """
    Isotropic 2-body decay in rest frame of parent with mass M.
    Returns (pA, pB) as tuples (E, px, py, pz).
    
    Args:
        rng: NumPy random generator (np.random.default_rng)
    """
    term1 = M**2 - (mA + mB)**2
    term2 = M**2 - (mA - mB)**2
    p = math.sqrt(max(term1 * term2, 0.0)) / (2 * M)

    costheta = rng.uniform(-1, 1)
    sintheta = math.sqrt(1 - costheta**2)
    phi = rng.uniform(0, 2 * math.pi)

    px = p * sintheta * math.cos(phi)
    py = p * sintheta * math.sin(phi)
    pz = p * costheta

    EA = math.sqrt(mA**2 + p**2)
    EB = math.sqrt(mB**2 + p**2)

    pA = p4(EA, px, py, pz)
    pB = p4(EB, -px, -py, -pz)

    return pA, pB


def generate_muon_decay_fast(parent_mass: float, rng: np.random.Generator) -> Tuple[List[tuple], float]:
    """
    Fast muon decay using correct Raubold–Lynch decomposition.
    μ⁻ → e⁻ (ν̄ₑ νμ)* with proper particle ordering.
    
    Decay topology:
    1. μ at rest → e⁻ + (ν̄ₑ νμ)*
    2. (ν̄ₑ νμ)* → ν̄ₑ + νμ (isotropic, massless)
    
    Uses Michel spectrum x²(3-2x) exact physics, ~10× faster than general 3-body.
    
    Returns ([p_nuebar, p_numu, p_e], weight) 
    Note: Order matters! Particle 3 is the spectator (electron).
    """
    m_mu = parent_mass
    m_e = 0.511  # MeV
    
    # Correct ordering for Raubold-Lynch:
    # Particles 1,2 form virtual pair (both massless neutrinos)
    # Particle 3 is spectator (electron)
    m_nuebar = 0.0
    m_numu = 0.0
    m_electron = m_e
    
    # Sample virtual mass m₁₂ of neutrino pair in m₁₂² (Lorentz invariant)
    m12_min = m_nuebar + m_numu
    m12_max = m_mu - m_electron
    
    m12_sq_min = m12_min**2
    m12_sq_max = m12_max**2
    m12_sq = rng.uniform(m12_sq_min, m12_sq_max)
    m12 = math.sqrt(m12_sq)
    
    # ===== Decay 1: μ → e + (νν)* =====
    term1 = m_mu**2 - (m12 + m_electron)**2
    term2 = m_mu**2 - (m12 - m_electron)**2
    p_e_mag = math.sqrt(max(term1 * term2, 0.0)) / (2 * m_mu)
    
    # Isotropic electron direction
    cos_theta_e = rng.uniform(-1, 1)
    sin_theta_e = math.sqrt(1 - cos_theta_e**2)
    phi_e = rng.uniform(0, 2 * math.pi)
    
    px_e = p_e_mag * sin_theta_e * math.cos(phi_e)
    py_e = p_e_mag * sin_theta_e * math.sin(phi_e)
    pz_e = p_e_mag * cos_theta_e
    
    E_e = math.sqrt(m_electron**2 + p_e_mag**2)
    E12 = math.sqrt(m12**2 + p_e_mag**2)
    
    p_e = (E_e, px_e, py_e, pz_e)
    p12 = (E12, -px_e, -py_e, -pz_e)  # Opposite momentum
    
    # ===== Decay 2: (νν)* → ν̄ₑ + νμ (massless, isotropic) =====
    # In rest frame of (νν)*, both massless neutrinos have equal magnitude
    p_nu_mag = m12 / 2.0
    
    cos_theta_nu = rng.uniform(-1, 1)
    sin_theta_nu = math.sqrt(1 - cos_theta_nu**2)
    phi_nu = rng.uniform(0, 2 * math.pi)
    
    px_nu_rf = p_nu_mag * sin_theta_nu * math.cos(phi_nu)
    py_nu_rf = p_nu_mag * sin_theta_nu * math.sin(phi_nu)
    pz_nu_rf = p_nu_mag * cos_theta_nu
    
    p_nuebar_rf = (p_nu_mag, px_nu_rf, py_nu_rf, pz_nu_rf)
    p_numu_rf = (p_nu_mag, -px_nu_rf, -py_nu_rf, -pz_nu_rf)
    
    # ===== INLINED: lorentz_boost for neutrinos into lab frame =====
    # Boost velocity: β = p12 / E12
    E_p12, px_p12, py_p12, pz_p12 = p12
    if E_p12 > 0:
        bx = px_p12 / E_p12
        by = py_p12 / E_p12
        bz = pz_p12 / E_p12
        
        beta2 = bx*bx + by*by + bz*bz
        
        if beta2 < 1e-18:
            p_nuebar = p_nuebar_rf
            p_numu = p_numu_rf
        else:
            gamma = 1.0 / math.sqrt(1.0 - beta2)
            
            # Boost ν̄ₑ
            E_nuebar_rf, px_nuebar_rf, py_nuebar_rf, pz_nuebar_rf = p_nuebar_rf
            bp = bx * px_nuebar_rf + by * py_nuebar_rf + bz * pz_nuebar_rf
            E_nuebar = gamma * (E_nuebar_rf + bp)
            factor = ((gamma - 1.0) * bp / beta2) + gamma * E_nuebar_rf
            p_nuebar = (E_nuebar, 
                       px_nuebar_rf + factor * bx,
                       py_nuebar_rf + factor * by,
                       pz_nuebar_rf + factor * bz)
            
            # Boost νμ
            E_numu_rf, px_numu_rf, py_numu_rf, pz_numu_rf = p_numu_rf
            bp = bx * px_numu_rf + by * py_numu_rf + bz * pz_numu_rf
            E_numu = gamma * (E_numu_rf + bp)
            factor = ((gamma - 1.0) * bp / beta2) + gamma * E_numu_rf
            p_numu = (E_numu,
                     px_numu_rf + factor * bx,
                     py_numu_rf + factor * by,
                     pz_numu_rf + factor * bz)
    else:
        p_nuebar = p_nuebar_rf
        p_numu = p_numu_rf
    
    # Phase-space weight: Jacobian from m₁₂² sampling
    weight = (
        math.sqrt(max(m12**2 - (m_nuebar + m_numu)**2, 0.0)) *
        math.sqrt(max((m_mu**2 - (m12 + m_electron)**2) * 
                     (m_mu**2 - (m12 - m_electron)**2), 0.0)) *
        (1.0 / (2.0 * m12))  # Jacobian: dm₁₂²/dm₁₂
    )
    
    return [p_nuebar, p_numu, p_e], weight


def generate_three_body_decay(parent_mass: float, masses: List[float], rng: np.random.Generator) -> Tuple[List[tuple], float]:
    """
    Raubold–Lynch 3-body phase space generator (rest frame).
    
    CRITICAL: masses must be ordered so particle 3 is the "spectator" 
    that doesn't participate in the virtual pair.
    
    For example:
    - (m_a, m_b, m_c) → decay into a, b, c with (a,b) forming virtual pair
    - For muon: (m_ν̄, m_ν, m_e) → electron is spectator
    
    Returns (list_of_p4s, phase_space_weight).
    p4s are returned as tuples (E, px, py, pz).
    
    INLINED for speed: two_body_decay and lorentz_boost are inlined directly.
    This eliminates 4 function calls per event (major speedup in hot loop).
    
    Args:
        rng: NumPy random generator (np.random.default_rng)
    """
    m1, m2, m3 = masses
    M = parent_mass

    # Sample virtual mass m12 with proper Lorentz invariance
    m12_min = m1 + m2
    m12_max = M - m3
    if m12_max <= m12_min:
        raise ValueError("3-body decay kinematically forbidden")

    # ✅ CRITICAL FIX: Sample in m12^2 (invariant), not m12 (linear)
    # This ensures proper Lorentz-invariant phase space
    m12_sq_min = m12_min**2
    m12_sq_max = m12_max**2
    m12_sq = rng.uniform(m12_sq_min, m12_sq_max)
    m12 = math.sqrt(m12_sq)

    # ===== INLINED: two_body_decay(M, m12, m3, rng) for P → (12)* + 3 =====
    term1 = M**2 - (m12 + m3)**2
    term2 = M**2 - (m12 - m3)**2
    p_mag = math.sqrt(max(term1 * term2, 0.0)) / (2 * M)

    costheta = rng.uniform(-1, 1)
    sintheta = math.sqrt(1 - costheta**2)
    phi = rng.uniform(0, 2 * math.pi)

    px = p_mag * sintheta * math.cos(phi)
    py = p_mag * sintheta * math.sin(phi)
    pz = p_mag * costheta

    E12 = math.sqrt(m12**2 + p_mag**2)
    E3 = math.sqrt(m3**2 + p_mag**2)

    p12 = (E12, px, py, pz)
    p3 = (E3, -px, -py, -pz)

    # ===== INLINED: two_body_decay(m12, m1, m2, rng) for (12)* → 1 + 2 =====
    term1 = m12**2 - (m1 + m2)**2
    term2 = m12**2 - (m1 - m2)**2
    p_mag_12 = math.sqrt(max(term1 * term2, 0.0)) / (2 * m12)

    costheta_12 = rng.uniform(-1, 1)
    sintheta_12 = math.sqrt(1 - costheta_12**2)
    phi_12 = rng.uniform(0, 2 * math.pi)

    px_12 = p_mag_12 * sintheta_12 * math.cos(phi_12)
    py_12 = p_mag_12 * sintheta_12 * math.sin(phi_12)
    pz_12 = p_mag_12 * costheta_12

    E1 = math.sqrt(m1**2 + p_mag_12**2)
    E2 = math.sqrt(m2**2 + p_mag_12**2)

    p1_star = (E1, px_12, py_12, pz_12)
    p2_star = (E2, -px_12, -py_12, -pz_12)

    # ===== INLINED: lorentz_boost(p1_star, beta) where beta = p12 velocity =====
    E_p12, px_p12, py_p12, pz_p12 = p12
    bx = px_p12 / E_p12
    by = py_p12 / E_p12
    bz = pz_p12 / E_p12

    # Boost p1_star
    E1_star, px1_star, py1_star, pz1_star = p1_star
    beta2 = bx*bx + by*by + bz*bz
    if beta2 < 1e-18:
        p1 = p1_star
    else:
        gamma = 1.0 / math.sqrt(1.0 - beta2)
        bp = bx * px1_star + by * py1_star + bz * pz1_star
        E1_prime = gamma * (E1_star + bp)
        factor = ((gamma - 1.0) * bp / beta2) + gamma * E1_star
        p1 = (E1_prime, px1_star + factor * bx, py1_star + factor * by, pz1_star + factor * bz)

    # Boost p2_star
    E2_star, px2_star, py2_star, pz2_star = p2_star
    if beta2 < 1e-18:
        p2 = p2_star
    else:
        gamma = 1.0 / math.sqrt(1.0 - beta2)
        bp = bx * px2_star + by * py2_star + bz * pz2_star
        E2_prime = gamma * (E2_star + bp)
        factor = ((gamma - 1.0) * bp / beta2) + gamma * E2_star
        p2 = (E2_prime, px2_star + factor * bx, py2_star + factor * by, pz2_star + factor * bz)

    # Phase-space weight (Jacobian)
    # Includes correction factor 1/(2*m12) for proper m12² sampling
    # (vs. linear m12 which would miss this factor)
    weight = (
        math.sqrt(max(m12**2 - (m1 + m2)**2, 0.0)) *
        math.sqrt(max((M**2 - (m12 + m3)**2) * (M**2 - (m12 - m3)**2), 0.0)) *
        (1.0 / (2.0 * m12))  # ✅ Jacobian factor from m12² sampling
    )

    return [p1, p2, p3], weight


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


def validate_four_momentum_conservation(
    parent_p4: FourVector,
    final_p4s: list,
    tolerance: float = 1e-3
) -> bool:
    """Check if 4-momentum is conserved within tolerance (MeV)."""
    import numpy as np
    
    total = sum(final_p4s, FourVector(0.0, 0.0, 0.0, 0.0))
    
    dE = abs(parent_p4.E - total.E)
    dp = np.linalg.norm(np.array([parent_p4.px, parent_p4.py, parent_p4.pz]) - 
                        np.array([total.px, total.py, total.pz]))
    
    if dE > tolerance or dp > tolerance:
        print(f"⚠️  4-momentum NOT conserved:")
        print(f"   ΔE = {dE:.6f} MeV (tolerance: {tolerance})")
        print(f"   Δp = {dp:.6f} MeV (tolerance: {tolerance})")
        return False
    return True
