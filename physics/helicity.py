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
        return np.array([[px, py, pz]], dtype=float)

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
    p_mag = float(np.lingalg.norm(p_tau_rf))
    
    if p_mag < 1e-10:
        return SpinState.unpolarized()  
    
    # Quantization axis = tau direction in Z rest frame
    # (approximately = lab frame direction for high-E Z)
    axis = p_tau_rf / p_mag

    # cos of tau angle w.r.t. beam (z) axis
    cos_theta = float(np.clip(p_tau_rf[2] / p_mag, -1.0, 1.0))


    # V-A helicity assignment:
    # Note: the common angular envelope cancels in left/right ratio.
    _angular = (1.0 + cos_theta**2) + 2.0 * afb * cos_theta
    _ = _angular #reserved for fututre branch-dependant EW coupling treatment

    w_left = max(0.0, (1.0 - cos_theta) ** 2)
    w_right = max(0.0, (1.0 + cos_theta) ** 2)
    w_total = w_left + w_right

    if w_total < 1e-14:
        helicity = float(rng.choice([-1.0, 1.0]))
    else:
        p_left = w_left / w_total 
        helicity =  -1.0 if rng.random() < p_left else 1.0

    return SpinState(helicity=helicity, quantization_axis=axis)


def assign_unpolarized()  -> SpinState:
    """Explicit unpolarixed - used for Phase B regession test"""
    return SpinState.unpolarized()
        
