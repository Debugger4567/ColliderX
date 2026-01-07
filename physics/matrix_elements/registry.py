"""
Matrix element registry: maps decay channels to physics models.

Key format: (parent_pdg, tuple(sorted(daughter_pdgs)))
Example: (13, (-11, -12, 14)) for μ⁻ → e⁻ ν̄ₑ νμ

No strings. No ambiguity. No magic.
"""
from .flat import FlatMatrixElement
from .weak_va import WeakVAMatrixElement


# Global registry: decay_key -> MatrixElement instance
_MATRIX_ELEMENTS = {}


def register(decay_key, matrix_element):
    """
    Register a matrix element for a decay channel.
    
    Args:
        decay_key: (parent_pdg, tuple(sorted(daughter_pdgs)))
        matrix_element: MatrixElement instance
        
    Example:
        >>> register((13, (-11, -12, 14)), WeakVAMatrixElement())
    """
    _MATRIX_ELEMENTS[decay_key] = matrix_element


def get_matrix_element(decay_key):
    """
    Resolve matrix element for a decay.
    
    Args:
        decay_key: (parent_pdg, tuple(sorted(daughter_pdgs)))
        
    Returns:
        MatrixElement instance (fallback: FlatMatrixElement)
    """
    return _MATRIX_ELEMENTS.get(decay_key, FlatMatrixElement())


def list_registered_models():
    """List all registered matrix elements."""
    return {k: v.name for k, v in _MATRIX_ELEMENTS.items()}


# ========== AUTO-REGISTER KNOWN PHYSICS ==========
# Muon decay: μ⁻ → e⁻ ν̄ₑ νμ
# PDG IDs: μ⁻=13, e⁻=11, ν̄ₑ=-12, νμ=14
register((13, (-11, -12, 14)), WeakVAMatrixElement())

# Antimuon decay: μ⁺ → e⁺ νₑ ν̄μ
# PDG IDs: μ⁺=-13, e⁺=-11, νₑ=12, ν̄μ=-14
register((-13, (11, 12, -14)), WeakVAMatrixElement())
