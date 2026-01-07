"""
Matrix element library for ColliderX.

Usage:
    from physics.matrix_elements import get_matrix_element
    
    decay_key = (13, (-11, -12, 14))  # μ⁻ → e⁻ ν̄ₑ νμ
    me = get_matrix_element(decay_key)
    M2 = me.M2(parent_p4, daughter_p4s)
"""
from .base import MatrixElement
from .flat import FlatMatrixElement
from .weak_va import WeakVAMatrixElement
from .registry import register, get_matrix_element, list_registered_models

__all__ = [
    "MatrixElement",
    "FlatMatrixElement",
    "WeakVAMatrixElement",
    "register",
    "get_matrix_element",
    "list_registered_models",
]