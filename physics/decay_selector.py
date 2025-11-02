import sqlite3
import random
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict

DB_PATH = Path(__file__).resolve().parents[1] / "colliderx.db"

# Simple in-memory cache: pdg_id -> List[(mode_text, br)]
_CACHE: Dict[int, List[Tuple[str, float]]] = {}

# Updated canonicalization map matching your actual DB particle names
_CANON = {
    # Leptons
    "e−": "Electron",
    "e-": "Electron",
    "e+": "Positron",
    "μ−": "Muon",
    "μ-": "Muon",
    "μ+": "Antimuon",
    "τ−": "Tau",
    "τ-": "Tau",
    "τ+": "Antitau",
    
    # Neutrinos
    "νe": "Electron neutrino",
    "ν̄e": "Electron antineutrino",
    "νμ": "Muon neutrino",
    "ν̄μ": "Muon antineutrino",
    "ντ": "Tau neutrino",
    "ν̄τ": "Tau antineutrino",
    
    # Quarks
    "u": "Up quark",
    "ū": "AntiUp quark",
    "d": "Down quark",
    "d̄": "AntiDown quark",
    "s": "Strange quark",
    "s̄": "AntiStrange quark",
    "c": "Charm quark",
    "c̄": "AntiCharm quark",
    "b": "Bottom quark",
    "b̄": "AntiBottom quark",
    "bbar": "AntiBottom quark",
    "t": "Top quark",
    "t̄": "AntiTop quark",
    
    # Gauge bosons
    "γ": "Photon",
    "gamma": "Photon",
    "g": "Gluon",
    "Z0": "Z boson",
    "Z": "Z boson",
    "W+": "W+ boson",
    "W−": "W- boson",
    "W-": "W- boson",
    "H0": "Higgs boson",
    "H": "Higgs boson",
    
    # Baryons
    "p": "Proton",
    "p̄": "Antiproton",
    "n": "Neutron",
    "n̄": "Antineutron",
    
    # Mesons
    "π+": "Pion+",
    "π−": "Pion-",
    "π-": "Pion-",
    "π0": "Pion0",
    "K+": "Kaon+",
    "K−": "Kaon-",
    "K-": "Kaon-",
    "K0": "Kaon0",
    "K̄0": "Antikaon0",
    "Λ": "Lambda",
    "Λ̄": "Antilambda",
}


def _canonicalize_mode(mode_text: str) -> List[str]:
    """
    Convert a CSV mode like 'μ+ νμ' into particle names from your DB.
    Examples:
      'μ+ νμ' → ['Antimuon', 'Muon neutrino']
      'γ γ' → ['Photon', 'Photon']
    """
    # Remove parenthetical notes and collapse spaces
    mode = re.sub(r"\s*\(.*?\)\s*", " ", mode_text).strip()
    mode = re.sub(r"\s+", " ", mode)
    
    if not mode or mode.lower().startswith("stable"):
        return []
    
    tokens = mode.split(" ")
    canonicalized = []
    
    for token in tokens:
        # Try direct lookup in canon map
        if token in _CANON:
            canonicalized.append(_CANON[token])
        else:
            # If not found, keep original (might be a full name already)
            canonicalized.append(token)
    
    return canonicalized


def get_decay_modes(pdg_id: int) -> List[Tuple[str, float]]:
    """Fetch (decay_mode, branching_fraction) for the given PDG ID (skips 'stable')."""
    if pdg_id in _CACHE:
        return _CACHE[pdg_id]

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT decay_mode, branching_fraction FROM decays WHERE pdg_id = ?", (pdg_id,))
        rows = cur.fetchall()

    modes: List[Tuple[str, float]] = []
    for r in rows:
        mode = (r["decay_mode"] or "").strip()
        if not mode or "stable" in mode.lower():
            continue
        try:
            br = float(r["branching_fraction"])
        except Exception:
            continue
        modes.append((mode, br))

    _CACHE[pdg_id] = modes
    return modes


def choose_decay_mode(pdg_id: int, rng: Optional[random.Random] = None) -> str:
    """
    Choose a decay mode string using branching fractions as probabilities.
    Returns 'stable' if no usable modes/weights.
    """
    rng = rng or random.Random()
    modes = get_decay_modes(pdg_id)
    if not modes:
        return "stable"

    names, weights = zip(*modes)
    # Clamp negatives to zero
    weights = [w if w > 0.0 else 0.0 for w in weights]
    total = sum(weights)
    if total <= 0.0:
        return "stable"

    # Normalize and sample
    norm = [w / total for w in weights]
    return rng.choices(names, weights=norm, k=1)[0]


def choose_decay_daughters(pdg_id: int, rng: Optional[random.Random] = None) -> List[str]:
    """
    Pick a decay and return canonicalized daughter particle names from your DB.
    Examples:
      PDG 211 (Pion+) → ['Antimuon', 'Muon neutrino']
      PDG 111 (Pion0) → ['Photon', 'Photon']
    Returns [] if 'stable' (no decay).
    """
    mode = choose_decay_mode(pdg_id, rng)
    if mode == "stable":
        return []
    return _canonicalize_mode(mode)
