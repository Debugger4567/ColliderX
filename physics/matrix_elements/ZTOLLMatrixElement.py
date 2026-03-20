import numpy as np
from .base import MatrixElement


class ZToLeptonsMatrixElement(MatrixElement):
    """
    Matrix element for Z → ℓ⁺ℓ⁻
    
    Physics:
      Tree-level vector current with optional forward–backward asymmetry
      |M|² ∝ s × [(1 + cos²θ) + A_FB · cosθ]
      
    where:
      s = (p_ℓ⁻ + p_ℓ⁺)² (invariant mass squared)
      θ = decay angle in Z rest frame
    """

    def __init__(self, afb: float = 0.0):
        """
        Parameters
        ----------
        afb : float
            Forward–backward asymmetry coefficient.
            afb = 0 reproduces pure vector current.
        """
        self.afb = float(afb)

    def M2(self, parent_p4: np.ndarray, daughter_p4s: list, context=None) -> float:
        """
        Compute |M|² for Z → ℓ⁺ℓ⁻
        """
        if len(daughter_p4s) != 2:
            return 0.0

        # Ensure NumPy arrays
        # NOTE: daughter_p4s ordering appears to be [ℓ⁺, ℓ⁻].
        # Use ℓ⁻ for cosθ to match z_afb.py.
        p_plus  = np.asarray(daughter_p4s[0], dtype=float)
        p_minus = np.asarray(daughter_p4s[1], dtype=float)

        # Invariant mass squared of dilepton system (Minkowski metric)
        p_total = p_minus + p_plus  # now vector add
        E, px, py, pz = p_total
        s = E*E - px*px - py*py - pz*pz
        if s <= 0.0:
            return 0.0

        # Decay angle (defined using ℓ⁻)
        spatial = p_minus[1:4]
        p_mag = np.linalg.norm(spatial)
        if p_mag < 1e-10:
            return 0.0

        cos_theta = np.clip(spatial[2] / p_mag, -1.0, 1.0)

        # Read afb from context if provided; fallback to self.afb
        afb = self.afb
        if context is not None:
            afb = float(context.get("afb", afb))

        angular = (1.0 + cos_theta**2) + 2.0 * afb * cos_theta
        return s * angular
