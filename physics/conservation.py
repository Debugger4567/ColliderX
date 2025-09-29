import numpy as np
from kinematics import FourVector

def check_energy_momentum(initial, final, tol=1e-6):
    """
    Check energy-momentum conservation between initial and final states.

    Parameters
    ----------
    initial : list of FourVector
        Initial 4-momenta (particles before the interaction/decay).
    final : list of FourVector
        Final 4-momenta (particles after the interaction/decay).
    tol : float
        Absolute tolerance for conservation check.

    Returns
    -------
    dict
        Differences (dE, dpx, dpy, dpz) and boolean 'conserved'.
    """
    P_initial = sum(initial[1:], initial[0])
    P_final   = sum(final[1:], final[0])

    diff = P_initial - P_final

    return {
        "dE": diff.E,
        "dpx": diff.px,
        "dpy": diff.py,
        "dpz": diff.pz,
        "conserved": np.allclose(
            [diff.E, diff.px, diff.py, diff.pz],
            [0.0, 0.0, 0.0, 0.0],
            atol=tol
        )
    }


def two_body_decay(parent: FourVector, m1: float, m2: float, tol=1e-6):
    """
    Perform a relativistic two-body decay: parent -> daughter1 + daughter2.

    Parameters
    ----------
    parent : FourVector
        4-momentum of the parent particle in the lab frame.
    m1 : float
        Mass of daughter 1 (MeV).
    m2 : float
        Mass of daughter 2 (MeV).
    tol : float
        Numerical tolerance for kinematic threshold.

    Returns
    -------
    (FourVector, FourVector)
        Daughter four-vectors in the lab frame.
    """
    M = parent.mass()
    if M + tol < (m1 + m2):
        raise ValueError("Decay not kinematically allowed")

    # Magnitude of daughter momentum in parent rest frame (Källén function)
    p_star = np.sqrt(
        max((M**2 - (m1 + m2)**2) * (M**2 - (m1 - m2)**2), 0.0)
    ) / (2.0 * M)

    # Random isotropic direction
    cos_theta = 2 * np.random.rand() - 1
    sin_theta = np.sqrt(1 - cos_theta**2)
    phi = 2 * np.pi * np.random.rand()

    px = p_star * sin_theta * np.cos(phi)
    py = p_star * sin_theta * np.sin(phi)
    pz = p_star * cos_theta

    E1 = np.sqrt(m1**2 + p_star**2)
    E2 = np.sqrt(m2**2 + p_star**2)

    d1_rest = FourVector(E1, px,  py,  pz)
    d2_rest = FourVector(E2, -px, -py, -pz)

    # Boost daughters to lab frame
    if parent.E <= 0:
        raise ValueError("Parent energy must be positive for boost")

    bx = parent.px / parent.E
    by = parent.py / parent.E
    bz = parent.pz / parent.E

    d1_lab = d1_rest.boost(bx, by, bz)
    d2_lab = d2_rest.boost(bx, by, bz)

    return d1_lab, d2_lab
