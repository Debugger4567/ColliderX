"""
Tau helicity dependent matrix element for τ → π ν

Physics:
    dΓ/d(cosθ) ∝ 1 + h·cosθ

where:
    h     = tau helicity (±1)
    θ     = angle between pion and tau spin axis
              in the tau rest frame

This is the V-A prediction for τ → π⁻ ντ.
The pion (spin-0) carries the full weak current information,
so its direction directly encodes the tau polarization.
"""

from .base import MatrixElement
import numpy as np


class TauPionHelicityMatrixElement(MatrixElement):
    """
    Helicity-dependent matrix element for τ → π ν.

    Requires context["spin_state"] to be a SpinState instance.
    Falls back to flat (unpolarized) if no spin state provided.
    """

    name = "Tau → π ν (helicity-dependent)"
    description = "V-A weak decay with tau polarization"

    def M2(
        self, parent_p4: tuple, daughter_p4s: list, context: dict | None = None
    ) -> float:
        """
        |M|² = 1 + h·cosθ

        daughter_p4s ordering: [pion, neutrino]
        context["spin_state"]: SpinState of the decaying tau
        """

        if len(daughter_p4s) < 1:
            return 1.0

        # get spin state - fall back to unpolarized
        spin = None
        if context:
            spin = context.get("spin_state", None)

        if spin is None or not spin.is_polarized:
            return 1.0

        # Pion four-vector (index 0 by convention)
        pion_p4 = daughter_p4s[0]
        _, px, py, pz = pion_p4
        p_pion = np.array([px, py, pz], dtype=float)
        p_mag = np.linalg.norm(p_pion)

        if p_mag < 1e-10:
            return 1.0

        # cosθ = angle between pion momentum and tau spin axis
        cos_theta = float(np.dot(p_pion / p_mag, spin.quantization_axis))
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))

        # V-A prediction: 1 + h·cosθ
        # Clamp to non-negative (numerical safety)
        M2 = max(0.0, 1.0 + spin.helicity * cos_theta)
        return M2
