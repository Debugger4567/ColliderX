"""
Sanity checks for matrix element architecture.

Tests:
    1. Flat fallback for unknown decays
    2. Deterministic M² evaluation
    3. Registry lookup with PDG keys
    4. Weak V-A physics for muon decay
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from physics.matrix_elements import get_matrix_element, FlatMatrixElement, list_registered_models


def test_flat_fallback():
    """Unknown decays should fall back to flat phase space."""
    decay_key = (999, (1, 2, 3))  # Nonsense decay
    me = get_matrix_element(decay_key)
    assert isinstance(me, FlatMatrixElement), f"Expected FlatMatrixElement, got {type(me)}"
    
    parent = (105.0, 0, 0, 0)
    daughters = [(52.5, 10, 0, 0), (52.5, -10, 0, 0)]
    M2 = me.M2(parent, daughters)
    assert M2 == 1.0, f"Flat M² should be 1.0, got {M2}"
    print("✓ Flat fallback works")


def test_deterministic():
    """Matrix elements must be deterministic (no RNG)."""
    decay_key = (13, (-11, -12, 14))  # μ⁻ decay
    me = get_matrix_element(decay_key)
    
    parent = (105.658, 0, 0, 0)
    daughters = [
        (50.0, 20.0, 10.0, 5.0),   # e⁻
        (30.0, -10.0, -5.0, -2.0),  # ν̄ₑ
        (25.658, -10.0, -5.0, -3.0) # νμ
    ]
    
    M2_1 = me.M2(parent, daughters)
    M2_2 = me.M2(parent, daughters)
    assert M2_1 == M2_2, "Matrix element must be deterministic"
    print("✓ Deterministic behavior confirmed")


def test_registry_lookup():
    """PDG-based lookup should resolve muon decay."""
    decay_key = (13, (-11, -12, 14))
    me = get_matrix_element(decay_key)
    assert me.name == "Weak V-A (unpolarized)", f"Expected Weak V-A, got {me.name}"
    print("✓ Registry lookup works")


def test_weak_va_physics():
    """V-A matrix element should give reasonable values."""
    decay_key = (13, (-11, -12, 14))
    me = get_matrix_element(decay_key)
    
    # Muon at rest
    parent = (105.658, 0, 0, 0)
    
    # Typical decay configuration
    daughters = [
        (52.0, 20.0, 10.0, 0.0),   # e⁻
        (28.0, -10.0, -5.0, 0.0),  # ν̄ₑ
        (25.658, -10.0, -5.0, 0.0) # νμ
    ]
    
    M2 = me.M2(parent, daughters)
    assert M2 > 0, f"M² must be positive, got {M2}"
    print(f"✓ Weak V-A physics: M² = {M2:.2e}")


def test_list_models():
    """List registered models."""
    models = list_registered_models()
    print(f"\n📋 Registered models: {len(models)}")
    for key, name in models.items():
        print(f"   {key}: {name}")


if __name__ == "__main__":
    print("=" * 60)
    print("Matrix Element Architecture Sanity Checks")
    print("=" * 60)
    
    test_flat_fallback()
    test_deterministic()
    test_registry_lookup()
    test_weak_va_physics()
    test_list_models()
    
    print("\n" + "=" * 60)
    print("✅ All matrix element tests passed")
    print("=" * 60)
