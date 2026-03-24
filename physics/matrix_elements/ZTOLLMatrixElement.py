import numpy as np
from .base import MatrixElement


class ZToLeptonsMatrixElement(MatrixElement):
    """
    Matrix element for Z → ℓ⁺ℓ⁻

    |M|² ∝ s × [(1 + cos²θ) + 2·A_FB·cosθ]
    Assumes Z is generated at rest in lab for cosθ extraction.
    """

    def __init__(self, afb: float = 0.0):
        self.afb = float(afb)

    def M2(self, parent_p4: np.ndarray, daughter_p4s: list, context=None) -> float:
        if len(daughter_p4s) != 2:
            return 0.0

        # ordering: [ℓ+, ℓ-]
        p_plus = np.asarray(daughter_p4s[0], dtype=float)
        p_minus = np.asarray(daughter_p4s[1], dtype=float)

        p_total = p_plus + p_minus
        E, px, py, pz = p_total
        s = E * E - px * px - py * py - pz * pz
        if s <= 0.0:
            return 0.0

        spatial = p_minus[1:4]
        p_mag = float(np.linalg.norm(spatial))
        if p_mag < 1e-10:
            return 0.0

        cos_theta = float(np.clip(spatial[2] / p_mag, -1.0, 1.0))

        afb = self.afb
        if context is not None:
            afb = float(context.get("afb", afb))

        angular = (1.0 + cos_theta**2) + 2.0 * afb * cos_theta
        return float(s * angular)
