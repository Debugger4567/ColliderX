"""Unified conservation and kinematics test suite.

This single file consolidates all tests for:
  - Energy conservation
  - Momentum conservation
  - Full four-momentum conservation
  - Two-body decay kinematics (rest, boosted, threshold, forbidden)
  - Numerical precision / tolerance behavior
  - Invariant mass reconstruction
  - Boost consistency

Add new tests here as features expand (multi-body phase space, random orientations, etc.).
"""

import math
import pytest
from .kinematics import FourVector
from .conservation import (
    check_energy_conservation,
    check_momentum_conservation,
    check_conservation,
    check_energy_momentum,
    two_body_decay,
)


# ----------------------------- Utility ------------------------------------
def _assert_close(a, b, tol=1e-9, msg=""):
    assert abs(a - b) < tol, msg or f"Values differ: {a} vs {b} (tol={tol})"


# -------------------------- Energy Conservation ---------------------------
def test_energy_conservation_simple():
    p_in = [FourVector(10, 0, 0, 0)]
    p_out = [FourVector(4, 1, 0, 0), FourVector(6, -1, 0, 0)]
    assert check_energy_conservation(p_in, p_out)


def test_energy_conservation_fail():
    p_in = [FourVector(10, 0, 0, 0)]
    p_out = [FourVector(5.1, 0, 0, 0), FourVector(5.0, 0, 0, 0)]  # 10.1
    assert not check_energy_conservation(p_in, p_out, tol=1e-4)


# ------------------------- Momentum Conservation --------------------------
def test_momentum_conservation_simple():
    p_in = [FourVector(5, 1, 2, -3), FourVector(7, -1, -2, 3)]
    p_out = [FourVector(8, 0.5, 1, -1.5), FourVector(4, -0.5, -1, 1.5)]
    assert check_momentum_conservation(p_in, p_out)


def test_momentum_conservation_fail():
    p_in = [FourVector(5, 1, 0, 0)]
    p_out = [FourVector(5, 0.9, 0, 0)]
    assert not check_momentum_conservation(p_in, p_out, tol=1e-4)


# ------------------------- Combined Conservation --------------------------
def test_combined_conservation_three_body():
    p_in = [FourVector(12, 0, 0, 0)]
    p_out = [FourVector(4, 2, 0, 0), FourVector(4, -1, 1, 0), FourVector(4, -1, -1, 0)]
    assert check_conservation(p_in, p_out)


def test_check_energy_momentum_dict_structure():
    p_in = [FourVector(10, 0, 0, 0)]
    p_out = [FourVector(4, 1, 0, 0), FourVector(6, -1, 0, 0)]
    diag = check_energy_momentum(p_in, p_out)
    assert diag['conserved'] is True
    for key in ['deltaE', 'deltaPx', 'deltaPy', 'deltaPz', 'E_initial', 'E_final']:
        assert key in diag
    _assert_close(diag['deltaE'], 0.0)
    _assert_close(diag['deltaPx'], 0.0)


# --------------------------- Two-Body Decays ------------------------------
@pytest.mark.parametrize(
    "parent,m1,m2",
    [
        (FourVector(1000.0, 0.0, 0.0, 0.0), 139.57, 139.57),  # pion pair
        (FourVector(2000.0, 0.0, 0.0, 1000.0), 105.66, 0.0),  # mu + nu
    ],
)
def test_two_body_decay_conservation(parent, m1, m2):
    d1, d2 = two_body_decay(parent, m1, m2)
    result = check_energy_momentum([parent], [d1, d2])
    assert result["conserved"], f"Conservation violated: {result}"


def test_two_body_decay_rest_equal_masses_properties():
    parent = FourVector(1000.0, 0, 0, 0)
    m_d = 100.0
    d1, d2 = two_body_decay(parent, m_d, m_d)
    _assert_close(d1.E, d2.E)
    _assert_close(d1.pz, -d2.pz)
    diag = check_energy_momentum([parent], [d1, d2])
    assert diag['conserved']


def test_two_body_decay_threshold():
    parent = FourVector(10.0, 0, 0, 0)
    m1, m2 = 4.0, 6.0
    d1, d2 = two_body_decay(parent, m1, m2)
    _assert_close(d1.momentum(), 0.0)
    _assert_close(d2.momentum(), 0.0)


def test_two_body_decay_forbidden_raises():
    parent = FourVector(5.0, 0, 0, 0)
    with pytest.raises(ValueError):
        two_body_decay(parent, 3.0, 3.0)


def test_two_body_decay_boosted_frame():
    parent = FourVector(1200.0, 0.0, 0.0, 600.0)
    d1, d2 = two_body_decay(parent, 100.0, 100.0)
    diag = check_energy_momentum([parent], [d1, d2])
    assert diag['conserved']


def test_two_body_decay_formula_match():
    parent = FourVector(1000.0, 0, 0, 0)
    m1, m2 = 200.0, 300.0
    d1, d2 = two_body_decay(parent, m1, m2)
    M = parent.mass()
    p_expected = math.sqrt(max((M**2 - (m1+m2)**2)*(M**2 - (m1-m2)**2), 0.0)) / (2*M)
    _assert_close(d1.momentum(), p_expected)
    _assert_close(d2.momentum(), p_expected)


def test_two_body_decay_invariant_reconstruction():
    parent = FourVector(1000.0, 0, 0, 0)
    d1, d2 = two_body_decay(parent, 100.0, 200.0)
    total = d1 + d2
    _assert_close(total.E, parent.E)
    _assert_close(total.px, parent.px)
    _assert_close(total.py, parent.py)
    _assert_close(total.pz, parent.pz)


def test_boost_consistency_direct_vs_manual():
    m_parent = 500.0
    parent_rest = FourVector(m_parent, 0, 0, 0)
    m1, m2 = 50.0, 100.0
    d1_rf, d2_rf = two_body_decay(parent_rest, m1, m2)
    # Boost components
    bx, by, bz = 0.0, 0.0, 0.6
    parent_boosted = parent_rest.boost(bx, by, bz)
    d1_b = d1_rf.boost(bx, by, bz)
    d2_b = d2_rf.boost(bx, by, bz)
    # Direct generation from boosted parent
    d1_direct, d2_direct = two_body_decay(parent_boosted, m1, m2)
    # Energies sum to parent energy & conservation holds
    _assert_close(d1_direct.E + d2_direct.E, parent_boosted.E)
    diag = check_energy_momentum([parent_boosted], [d1_direct, d2_direct])
    assert diag['conserved']


# ------------------------- Precision / Tolerance --------------------------
def test_precision_within_tolerance_passes():
    p_in = [FourVector(10.0000001, 0, 0, 0)]
    p_out = [FourVector(4.0, 1, 0, 0), FourVector(6.0, -1, 0, 0)]
    assert check_energy_conservation(p_in, p_out, tol=1e-6)


def test_precision_outside_tolerance_fails():
    p_in = [FourVector(10.001, 0, 0, 0)]
    p_out = [FourVector(4.0, 1, 0, 0), FourVector(6.0, -1, 0, 0)]
    assert not check_energy_conservation(p_in, p_out, tol=1e-6)


# ------------------------- Regression / Edge Cases ------------------------
def test_zero_vectors_conservation():
    # Trivial edge: all zero four-vectors
    z = FourVector(0.0, 0.0, 0.0, 0.0)
    assert check_conservation([z], [z])


def test_asymmetric_three_body_energy_only():
    # Ensure failure when only energy mismatched
    p_in = [FourVector(9.9, 0, 0, 0)]
    p_out = [FourVector(3, 1, 0, 0), FourVector(3, -1, 0, 0), FourVector(4, 0, 0, 0)]  # 10 total
    assert not check_energy_conservation(p_in, p_out, tol=1e-6)


def test_large_values_scale():
    # High-energy stability
    p_in = [FourVector(1e6, 1e5, -2e5, 3e5)]
    p_out = [FourVector(4e5, 5e4, -1e5, 1e5), FourVector(6e5, 5e4, -1e5, 2e5)]
    assert check_conservation(p_in, p_out, tol=1e-6)


def test_two_body_decay_massless_daughter():
    parent = FourVector(500.0, 0, 0, 0)
    d1, d2 = two_body_decay(parent, 0.0, 100.0)
    diag = check_energy_momentum([parent], [d1, d2])
    assert diag['conserved']
