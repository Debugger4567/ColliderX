# conservation.py
# Core conservation utilities and minimal kinematic decay helper functions.
#
# Design goals:
# - Keep dependencies minimal (pure Python + math) so higher-level modules can build on this.
# - Provide deterministic behavior for tests (fixed decay axis) while leaving clear hooks for
#   future stochastic / phase-space sampling extensions.
from .kinematics import FourVector
import math

def check_energy_conservation(initial_vectors, final_vectors, tol=1e-6):
    """
    Check conservation of energy for any N-body interaction.

    Parameters
    ----------
    initial_vectors : list of FourVector
        List of incoming particles.
    final_vectors : list of FourVector
        List of outgoing particles.
    tol : float
        Numerical tolerance (default 1e-6).

    Returns
    -------
    bool
        True if |E_initial - E_final| < tol, else False.

    Examples
    --------
    # --- 2-body decay ---
    >>> from kinematics import FourVector
    >>> p_initial = FourVector(10, 0, 0, 0)
    >>> p1 = FourVector(5, 3, 0, 0)
    >>> p2 = FourVector(5, -3, 0, 0)
    >>> check_energy_conservation([p_initial], [p1, p2])
    True

    # --- 3-body decay ---
    >>> p_initial = FourVector(12, 0, 0, 0)
    >>> p1 = FourVector(4, 2, 0, 0)
    >>> p2 = FourVector(4, -1, 1, 0)
    >>> p3 = FourVector(4, -1, -1, 0)
    >>> check_energy_conservation([p_initial], [p1, p2, p3])
    True
    """
    E_initial = sum(v.E for v in initial_vectors)
    E_final = sum(v.E for v in final_vectors)
    return abs(E_initial - E_final) < tol


def check_momentum_conservation(initial_vectors, final_vectors, tol=1e-6):
    """
    Check conservation of 3-momentum for any N-body interaction.

    Parameters
    ----------
    initial_vectors : list of FourVector
        List of incoming particles.
    final_vectors : list of FourVector
        List of outgoing particles.
    tol : float
        Numerical tolerance (default 1e-6).

    Returns
    -------
    bool
        True if all components (px, py, pz) are conserved within tol.

    Examples
    --------
    # --- 2-body scattering ---
    >>> from kinematics import FourVector
    >>> p_in1 = FourVector(6, 3, 0, 0)
    >>> p_in2 = FourVector(6, -3, 0, 0)
    >>> p_out1 = FourVector(6, 0, 3, 0)
    >>> p_out2 = FourVector(6, 0, -3, 0)
    >>> check_momentum_conservation([p_in1, p_in2], [p_out1, p_out2])
    True

    # --- 3-body decay ---
    >>> p_initial = FourVector(12, 0, 0, 0)
    >>> p1 = FourVector(4, 2, 0, 0)
    >>> p2 = FourVector(4, -1, 1, 0)
    >>> p3 = FourVector(4, -1, -1, 0)
    >>> check_momentum_conservation([p_initial], [p1, p2, p3])
    True
    """
    px_initial = sum(v.px for v in initial_vectors)
    py_initial = sum(v.py for v in initial_vectors)
    pz_initial = sum(v.pz for v in initial_vectors)

    px_final = sum(v.px for v in final_vectors)
    py_final = sum(v.py for v in final_vectors)
    pz_final = sum(v.pz for v in final_vectors)

    return (
        abs(px_initial - px_final) < tol and
        abs(py_initial - py_final) < tol and
        abs(pz_initial - pz_final) < tol
    )


def check_conservation(initial_vectors, final_vectors, tol=1e-6):
    """
    Check full 4-momentum conservation (energy + momentum).

    Parameters
    ----------
    initial_vectors : list of FourVector
    final_vectors : list of FourVector
    tol : float

    Returns
    -------
    bool
        True if both energy and momentum are conserved.

    Notes
    -----
    - Works for 2-body, 3-body, and general N-body decays/scattering.
    - This is the main conservation check we’ll use everywhere.
    """
    return (
        check_energy_conservation(initial_vectors, final_vectors, tol) and
        check_momentum_conservation(initial_vectors, final_vectors, tol)
    )


def check_energy_momentum(initial_vectors, final_vectors, tol=1e-6):
    """Return diagnostic dict for full 4-momentum conservation.

    Returns dict with deltas for energy and momentum components and a
    boolean 'conserved' key summarizing result within tolerance.
    """
    Ei = sum(v.E for v in initial_vectors)
    Ef = sum(v.E for v in final_vectors)
    pxi = sum(v.px for v in initial_vectors); pxf = sum(v.px for v in final_vectors)
    pyi = sum(v.py for v in initial_vectors); pyf = sum(v.py for v in final_vectors)
    pzi = sum(v.pz for v in initial_vectors); pzf = sum(v.pz for v in final_vectors)
    dE = Ei - Ef; dPx = pxi - pxf; dPy = pyi - pyf; dPz = pzi - pzf
    conserved = (abs(dE) < tol and abs(dPx) < tol and abs(dPy) < tol and abs(dPz) < tol)
    return {
        'conserved': conserved,
        'deltaE': dE,
        'deltaPx': dPx,
        'deltaPy': dPy,
        'deltaPz': dPz,
        'E_initial': Ei,
        'E_final': Ef
    }


def two_body_decay(parent: FourVector, m1: float, m2: float):
    """Deterministic axis-aligned two-body decay helper.

    Uses standard relativistic two-body decay kinematics in the parent
    rest frame, assigning daughter momenta along ±z for reproducibility
    (suitable for tests), then boosting to lab frame if parent moves.
    """
    import math
    M2 = parent.E**2 - parent.momentum()**2
    M = math.sqrt(M2) if M2 > 0 else 0.0
    if m1 + m2 > M + 1e-9:
        raise ValueError("Kinematically forbidden decay: m1+m2 > parent mass")
    if abs(M - (m1 + m2)) < 1e-12:
        p_star = 0.0
        E1 = m1; E2 = m2
    else:
        term1 = M**2 - (m1 + m2)**2
        term2 = M**2 - (m1 - m2)**2
        inside = max(term1 * term2, 0.0)
        p_star = math.sqrt(inside) / (2*M)
        E1 = math.sqrt(m1**2 + p_star**2)
        E2 = math.sqrt(m2**2 + p_star**2)
    d1_rf = FourVector(E1, 0.0, 0.0, p_star)
    d2_rf = FourVector(E2, 0.0, 0.0, -p_star)
    if parent.momentum() < 1e-12:  # parent at rest
        return d1_rf, d2_rf
    # boost components
    bx = parent.px / parent.E
    by = parent.py / parent.E
    bz = parent.pz / parent.E
    d1_lab = d1_rf.boost(bx, by, bz)
    d2_lab = d2_rf.boost(bx, by, bz)
    return d1_lab, d2_lab
