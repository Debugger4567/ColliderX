"""
Helicity computation for decay products.

Rule: helicity is computed in the PARTICLE'S REST FRAME.
Quantization axis = particle momentum in the PARENT rest frame.

This is the only place in the codebase where helicity
is assigned. Everything downstream just reads SpinState.
"""
import math 
import numpy as np
from .spin import SpinState

def _boost_to_rest_frame(p4_to_boost: tuple, parent_p4: tuple) -> np.ndarray:
    """
    boost p4_to_boost into parent rest frame. 
    returns boosted spatail 3-momentum.
    """
    E_par, px_par, py_par, pz_par = parent_p4
    p_par = np.array([px_par, py_par, pz_par], dtype=float)
    p_par_mag_sq = float(np.dot(p_par, p_par))

    if p_par_mag_sq < 1e-18 or E_par <= 0.0:
        _, px, py, pz = p4_to_boost
        return np.array([px, py, pz], dtype=float)

    beta = p_par / E_par
    beta2 = float(np.dot(beta, beta))
    if beta2 >= 1.0:
        _, px, py, pz = p4_to_boost
        return np.array([px, py, pz], dtype=float)

    gamma = 1.0 / math.sqrt(1.0 - beta2)

    E_in, px_in, py_in, pz_in = p4_to_boost
    p_in = np.array([px_in, py_in, pz_in], dtype=float)


    #Boost into parent rest frame 
    # Boost into parent rest frame => use -beta
    neg_beta = -beta
    bp = float(np.dot(neg_beta, p_in))
    p_out = p_in + neg_beta * (((gamma - 1.0) * bp / beta2) + gamma * E_in)
    return p_out


def compute_tau_helicity(tau_p4: tuple, parent_p4: tuple, rng: np.random.Generator, afb: float = 0.0) -> SpinState:
    """
    Compute helicity of a tau produced in Z → τ⁺τ⁻.
    """

    p_tau_rf = _boost_to_rest_frame(tau_p4, parent_p4)
    p_mag = float(np.linalg.norm(p_tau_rf))
    
    if p_mag < 1e-10:
        return SpinState.unpolarized()  
    
    # Quantization axis = tau direction in Z rest frame
    # (approximately = lab frame direction for high-E Z)
    axis = p_tau_rf / p_mag

    # cos of tau angle w.r.t. beam (z) axis
    cos_theta = float(np.clip(p_tau_rf[2] / p_mag, -1.0, 1.0))


    # Z-pole inspired tau polarization model:
    #   P_tau(c) = - [ A_tau(1+c^2) + 2 A_e c ] / [ (1+c^2) + 2 A_e A_tau c ]
    # where c = cos(theta_tau).
    # Helicity convention here: left-handed tau- => h = -1.
    A_tau = 0.1465
    A_e = afb if abs(afb) > 1e-12 else 0.1516
    A_e = float(np.clip(A_e, -0.95, 0.95))

    numerator = A_tau * (1.0 + cos_theta**2) + 2.0 * A_e * cos_theta
    denominator = (1.0 + cos_theta**2) + 2.0 * A_e * A_tau * cos_theta
    if abs(denominator) < 1e-14:
        P_tau = 0.0
    else:
        P_tau = -numerator / denominator

    P_tau = float(np.clip(P_tau, -1.0, 1.0))
    p_left = 0.5 * (1.0 - P_tau)
    p_left = float(np.clip(p_left, 0.0, 1.0))
    helicity = -1.0 if rng.random() < p_left else 1.0

    return SpinState(helicity=helicity, quantization_axis=axis)


def assign_unpolarized()  -> SpinState:
    """Explicit unpolarixed - used for Phase B regession test"""
    return SpinState.unpolarized()
        
