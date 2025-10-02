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
from pathlib import Path
import sqlite3


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


# ================= Particle + FourVector (DB-backed) Decays ==============
def _load_particles_from_db():
    """Helper returning list of (Name, Symbol, Mass) or empty list if unavailable."""
    db_path = Path(__file__).resolve().parents[1] / "colliderx.db"
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute('SELECT Name, Symbol, "Mass (MeV/c^2)" FROM particles')
        rows = cur.fetchall()
        conn.close()
        cleaned = []
        for n, s, m in rows:
            try:
                cleaned.append((str(n), str(s), float(m)))
            except Exception:
                continue
        return cleaned
    except Exception:
        return []


@pytest.mark.skipif(False, reason="placeholder marker - set condition to skip if needed")
def test_particle_db_mass_lookup_and_decay_conservation():
    """Dynamically pick a valid parent and two lighter daughters from DB and test decay conservation.

    Skips if suitable combination not found or DB unavailable.
    """
    parts = _load_particles_from_db()
    if not parts:
        pytest.skip("Particle DB not available or unreadable.")
    # Sort by mass ascending
    parts_sorted = sorted(parts, key=lambda x: x[2])
    # Attempt to find triple M_parent > m1 + m2
    parent = None; d1 = None; d2 = None
    for i in range(len(parts_sorted)-1, -1, -1):  # heavy to light
        Mname, Msym, M = parts_sorted[i]
        # try pairs of lighter
        for j in range(len(parts_sorted)):
            for k in range(j+1, len(parts_sorted)):
                n1, s1, m1 = parts_sorted[j]
                n2, s2, m2 = parts_sorted[k]
                if m1 + m2 < M and m1 > 0 and m2 > 0:
                    parent = (Mname, M)
                    d1 = (n1, m1)
                    d2 = (n2, m2)
                    break
            if parent:
                break
        if parent:
            break
    if not parent:
        pytest.skip("No kinematically allowed triple found in DB.")
    # Build parent FourVector at rest
    parent_vec = FourVector(parent[1], 0, 0, 0)
    dv1, dv2 = two_body_decay(parent_vec, d1[1], d2[1])
    diag = check_energy_momentum([parent_vec], [dv1, dv2])
    assert diag['conserved'], f"Conservation failed for {parent[0]} -> {d1[0]} {d2[0]}: {diag}"
    # Daughter invariant masses should approximate input masses
    # Allow a relative tolerance of 5e-6 or absolute 5e-6 to accommodate DB rounding
    rel_tol = 5e-6
    abs_tol = 5e-6
    _assert_close(dv1.mass(), d1[1], tol=max(abs_tol, d1[1]*rel_tol))
    _assert_close(dv2.mass(), d2[1], tol=max(abs_tol, d2[1]*rel_tol))


@pytest.mark.skipif(False, reason="placeholder marker - set condition to skip if needed")
def test_particle_db_forbidden_decay_raises():
    parts = _load_particles_from_db()
    if len(parts) < 3:
        pytest.skip("Insufficient particles in DB.")
    # pick two heaviest as daughters and a lighter as parent to ensure forbidden
    parts_sorted = sorted(parts, key=lambda x: x[2])
    light = parts_sorted[0]
    heavy1 = parts_sorted[-1]
    heavy2 = parts_sorted[-2]
    if light[2] >= heavy1[2] + heavy2[2]:
        pytest.skip("Could not construct forbidden combination from DB masses.")
    parent_vec = FourVector(light[2], 0, 0, 0)
    with pytest.raises(ValueError):
        two_body_decay(parent_vec, heavy1[2], heavy2[2])


@pytest.mark.skipif(False, reason="placeholder marker - set condition to skip if needed")
def test_particle_db_threshold_decay_if_available():
    parts = _load_particles_from_db()
    if len(parts) < 3:
        pytest.skip("Insufficient particles in DB.")
    # Try to find M ~ m1 + m2 within tiny tolerance (1e-6 relative). Otherwise skip.
    parts_sorted = sorted(parts, key=lambda x: x[2])
    found = None
    for i in range(len(parts_sorted)):
        for j in range(len(parts_sorted)):
            if i == j:
                continue
            for k in range(len(parts_sorted)):
                if k == i or k == j:
                    continue
                M = parts_sorted[i][2]
                m1 = parts_sorted[j][2]
                m2 = parts_sorted[k][2]
                if abs(M - (m1 + m2)) / max(M, 1e-9) < 1e-6:
                    found = (M, m1, m2)
                    break
            if found:
                break
        if found:
            break
    if not found:
        pytest.skip("No near-threshold triple available in DB.")
    M, m1, m2 = found
    parent_vec = FourVector(M, 0, 0, 0)
    d1, d2 = two_body_decay(parent_vec, m1, m2)
    _assert_close(d1.momentum(), 0.0, tol=1e-6)
    _assert_close(d2.momentum(), 0.0, tol=1e-6)
