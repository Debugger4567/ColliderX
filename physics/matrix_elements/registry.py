"""
Matrix element registry: maps decay channels to physics models.

Key format: (parent_pdg, tuple(sorted(daughter_pdgs)))
Example: (13, (-11, -12, 14)) for μ⁻ → e⁻ ν̄ₑ νμ

No strings. No ambiguity. No magic.
"""
from .flat import FlatMatrixElement
from .weak_va import WeakVAMatrixElement
from .scalar_2body import ScalarTwoBodyMatrixElement


# Global registry: decay_key -> MatrixElement instance
_REGISTRY: dict = {}


def register(parent_pdg: int, daughters: tuple, model):
    """
    Register a matrix element for a decay channel.
    
    Args:
        parent_pdg: Parent particle PDG ID
        daughters: Tuple of daughter PDG IDs (sorted)
        model: MatrixElement instance
        
    Example:
        >>> register(13, (-11, -12, 14), WeakVAMatrixElement())
    """
    key = (parent_pdg, daughters)
    _REGISTRY[key] = model


def get_matrix_element(decay_key):
    """
    Resolve matrix element for a decay.
    
    Args:
        decay_key: (parent_pdg, tuple(sorted(daughter_pdgs)))
        
    Returns:
        MatrixElement instance (fallback: FlatMatrixElement)
    """
    return _REGISTRY.get(decay_key, FlatMatrixElement())


def list_registered_models():
    """List all registered matrix elements."""
    return {k: v.name for k, v in _REGISTRY.items()}


# ========== AUTO-REGISTER KNOWN PHYSICS ==========
# Muon decay: μ⁻ → e⁻ ν̄ₑ νμ
# PDG IDs: μ⁻=13, e⁻=11, ν̄ₑ=-12, νμ=14
register(13, (-11, -12, 14), WeakVAMatrixElement())

# Antimuon decay: μ⁺ → e⁺ νₑ ν̄μ
# PDG IDs: μ⁺=-13, e⁺=-11, νₑ=12, ν̄μ=-14
register(-13, (11, 12, -14), WeakVAMatrixElement())

# π0 → γ γ
# PDG: π0 = 111, γ = 22
register(111, (22, 22), ScalarTwoBodyMatrixElement())

# Higgs → γ γ (toy)
# PDG: H = 25, γ = 22
register(25, (22, 22), ScalarTwoBodyMatrixElement())