import importlib
import runpy
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from db import get_conn
from physics.spin import SpinState
from physics.helicity import compute_tau_helicity, assign_unpolarized
from physics.collision import simulate_events, simulate_chain
from physics.matrix_elements.registry import get_matrix_element
from physics.matrix_elements.weak_va import WeakVAMatrixElement
from physics.matrix_elements.ZTOLLMatrixElement import ZToLeptonsMatrixElement
from physics.matrix_elements.tau_helicity import TauPionHelicityMatrixElement


def _db_available() -> bool:
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        cur.close()
        conn.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def db_ready():
    if not _db_available():
        pytest.skip("Database is not reachable for integration tests")
    return True


@pytest.fixture(autouse=True)
def _no_gui_show(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    yield
    plt.close("all")


def test_spin_and_helicity_core_math():
    unpol = assign_unpolarized()
    assert isinstance(unpol, SpinState)
    assert not unpol.is_polarized

    rng = np.random.default_rng(123)
    tau_p4 = (45.0, 10.0, 5.0, 20.0)
    z_p4 = (91.2, 0.0, 0.0, 0.0)
    spin = compute_tau_helicity(tau_p4=tau_p4, parent_p4=z_p4, rng=rng, afb=0.1)

    assert isinstance(spin, SpinState)
    assert spin.helicity in (-1.0, 1.0)
    assert np.isclose(np.linalg.norm(spin.quantization_axis), 1.0, atol=1e-9)


def test_matrix_element_registry_resolution():
    mu_minus = get_matrix_element((13, (-11, -12, 14)))
    mu_plus = get_matrix_element((-13, (-14, 11, 12)))
    zll = get_matrix_element((23, (-13, 13)))
    tau_pi = get_matrix_element((15, (-211, 16)))
    neutron_beta = get_matrix_element((2112, (-12, 11, 2212)))

    assert isinstance(mu_minus, WeakVAMatrixElement)
    assert isinstance(mu_plus, WeakVAMatrixElement)
    assert isinstance(zll, ZToLeptonsMatrixElement)
    assert isinstance(tau_pi, TauPionHelicityMatrixElement)
    assert isinstance(neutron_beta, WeakVAMatrixElement)


def test_main_module_import_and_cli_wiring(monkeypatch):
    module = importlib.import_module("main")
    assert hasattr(module, "main")
    assert hasattr(module, "parse_args")

    monkeypatch.setattr(sys, "argv", ["main.py", "Muon", "1"])
    args = module.parse_args()
    assert args.particle == "Muon"
    assert args.events == 1


@pytest.mark.usefixtures("db_ready")
def test_simulate_events_muon_and_antimuon_smoke():
    r_mu = simulate_events(parent_name="Muon", n_events=2, seed=1, verbose=False)
    assert r_mu["total"] == 2
    assert r_mu["success"] >= 1

    r_amu = simulate_events(parent_name="Antimuon", n_events=2, seed=2, verbose=False)
    assert r_amu["total"] == 2
    assert r_amu["success"] >= 1


@pytest.mark.usefixtures("db_ready")
def test_simulate_chain_smoke():
    result = simulate_chain(parent_name="Muon", n_events=1, seed=7, verbose=False)
    assert result["total"] == 1
    assert result["success"] + result["failed"] == 1


@pytest.mark.usefixtures("db_ready")
def test_all_plot_modules_smoke():
    modules_with_main = [
        "visualization.dalitz_muon_decay",
        "visualization.michel_spectrum",
        "visualization.michel_overlay",
        "visualization.z_costheta",
        "visualization.z_invariant_mass",
        "visualization.michel_flat_phase_space",
    ]

    for mod_name in modules_with_main:
        mod = importlib.import_module(mod_name)
        assert hasattr(mod, "main")
        mod.main()

    # This module executes at import time; run as script path for smoke coverage.
    runpy.run_module("visualization.electron_energy_distribution", run_name="__main__")
