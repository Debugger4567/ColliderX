import random
import re
import numpy as np
from typing import List, Tuple, Optional, Dict
from db import get_conn

def _conn():
    # Alias to avoid refactor churn; uses shared db.get_conn (psycopg2)
    return get_conn()

# Simple in-memory cache: pdg_id -> List[(mode_text, br)]
_CACHE: Dict[int, List[Tuple[str, float]]] = {}

# NEW: Cache for decay products
_DECAY_PRODUCTS_CACHE: Dict[tuple, List[str]] = {}

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

    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT decay_mode, branching_fraction FROM decays WHERE pdg_id = %s",
            (pdg_id,),
        )
        rows = cur.fetchall()

    modes: List[Tuple[str, float]] = []
    for r in rows:
        mode = (r[0] or "").strip()
        if not mode or "stable" in mode.lower():
            continue
        try:
            br = float(r[1])
        except Exception:
            continue
        modes.append((mode, br))

    _CACHE[pdg_id] = modes
    return modes


def choose_decay_mode(pdg_id: int, rng=None) -> str:
    """
    Choose a decay mode string using branching fractions as probabilities.
    Returns 'stable' if no usable modes/weights.
    """
    rng = rng or np.random.default_rng()
    modes = get_decay_modes(pdg_id)
    if not modes:
        return "stable"

    names = [m[0] for m in modes]
    brs = np.array([m[1] for m in modes], dtype=float)
    norm = brs / brs.sum()

    # numpy Generator supports .choice with probabilities
    return rng.choice(names, p=norm)


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


def get_decay_products(pdg_id: int, decay_mode: str) -> List[str]:
    """
    Return canonicalized daughter names for a decay mode.
    Uses in-memory cache to avoid repeated DB queries.
    """
    key = (pdg_id, decay_mode)
    
    # Check cache first
    if key in _DECAY_PRODUCTS_CACHE:
        return _DECAY_PRODUCTS_CACHE[key]
    
    # Query DB only on cache miss
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT product_index, product
            FROM decay_products
            WHERE pdg_id = %s AND decay_mode = %s
            ORDER BY product_index
            """,
            (pdg_id, decay_mode),
        )
        rows = cur.fetchall()

    if not rows:
        raise RuntimeError(
            f"No decay products for PDG {pdg_id} mode '{decay_mode}'"
        )

    indices = [r[0] for r in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate product_index detected")

    products = [r[1] for r in rows]
    
    # Store in cache
    _DECAY_PRODUCTS_CACHE[key] = products
    return products
