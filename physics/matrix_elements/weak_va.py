"""
Weak V-A matrix element for semileptonic decays.

Pure physics: |M|² ∝ (p_parent · p_ν̄)(p_ℓ · p_ν)

No normalization, no Michel spectrum hacks.
If phase space is correct, Michel emerges naturally.
"""
from .base import MatrixElement


class WeakVAMatrixElement(MatrixElement):
    """
    Unpolarized weak V-A interaction matrix element.
    
    Applies to:
        - Muon decay: μ⁻ → e⁻ ν̄ₑ νμ
        - Tau decay: τ⁻ → e⁻ ν̄ₑ ντ
        - Pion decay: π⁺ → μ⁺ νμ (simplified)
    """
    
    name = "Weak V-A (unpolarized)"
    description = "Standard Model weak interaction with V-A coupling"

    def M2(self, parent_p4: tuple, daughter_p4s: list[tuple], context=None) -> float:
        """
        Compute |M|² for unpolarized weak decay.
        
        Args:
            parent_p4: Parent 4-vector (E, px, py, pz)
            daughter_p4s: [charged_lepton, antineutrino, neutrino]
            context: Optional physics parameters (unused)
        
        Returns:
            |M|² ∝ (p_parent · p_ν̄)(p_ℓ · p_ν)
            
        Note:
            Ordering matters! Assumes [ℓ⁻, ν̄ₗ, νparent]
            For muon: [e⁻, ν̄ₑ, νμ]
        """
        p_parent = parent_p4
        p_lepton, p_nubar, p_nu = daughter_p4s

        # V-A matrix element: (p_parent · p_nubar)(p_lepton · p_nu)
        dot1 = self.lorentz_dot(p_parent, p_nubar)
        dot2 = self.lorentz_dot(p_lepton, p_nu)
        
        return dot1 * dot2

    @staticmethod
    def lorentz_dot(p: tuple, q: tuple) -> float:
        """Minkowski inner product with (+,-,-,-) signature."""
        Ep, pxp, pyp, pzp = p
        Eq, pxq, pyq, pzq = q
        return Ep * Eq - (pxp * pxq + pyp * pyq + pzp * pzq)
    


