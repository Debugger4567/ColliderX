import os
import math
import numpy as np
from datetime import datetime
from pathlib import Path
from .particles import Particle
from .decay_selector import choose_decay_mode, get_decay_modes, get_decay_products
from .phase_space import generate_three_body_decay
from db import get_conn


def _conn():
    return get_conn()


# ========== CACHES (computed once, reused forever) ==========
_PARTICLE_CACHE = {}
_PDG_CACHE = {}


def get_particle(name: str) -> dict:
    """Cache particle properties (mass, name) to avoid repeated Particle() construction."""
    if name not in _PARTICLE_CACHE:
        p = Particle(name)
        _PARTICLE_CACHE[name] = {
            "mass": p.mass,
            "name": p.name,
        }
    return _PARTICLE_CACHE[name]


def get_pdg_id_cached(particle_name: str) -> int | None:
    """Cache PDG ID lookups."""
    if particle_name not in _PDG_CACHE:
        with _conn() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT "PDG ID" FROM particles
                   WHERE LOWER("Name") = LOWER(%s) OR LOWER("Symbol") = LOWER(%s)
                   LIMIT 1""",
                (particle_name, particle_name),
            )
            row = cur.fetchone()
        _PDG_CACHE[particle_name] = int(row[0]) if row else None
    return _PDG_CACHE[particle_name]


def p4(E: float, px: float, py: float, pz: float) -> tuple:
    """4-vector as tuple (E, px, py, pz) - no NumPy overhead."""
    return (E, px, py, pz)


def _reorder_daughters_for_matrix_element(
    daughter_names: list[str], 
    p4s: list[tuple], 
    parent_pdg: int
) -> list[tuple]:
    """
    Reorder daughter momenta to match matrix element expectations.
    
    For weak V-A decays: [charged_lepton, antineutrino, neutrino]
    
    Args:
        daughter_names: Particle names in generation order
        p4s: 4-momenta in same order as daughter_names
        parent_pdg: Parent PDG ID (for context)
        
    Returns:
        Reordered list of 4-momenta matching matrix element conventions
    """
    # Get PDG IDs for all daughters
    daughter_pdgs = [get_pdg_id_cached(name) for name in daughter_names]
    
    # Build (pdg, p4, name) tuples
    daughters = list(zip(daughter_pdgs, p4s, daughter_names))
    
    # For 3-body leptonic decays, enforce V-A ordering
    if len(daughters) == 3:
        charged_lepton_p4 = None
        antineutrino_p4 = None
        neutrino_p4 = None
        
        for pdg, p4, name in daughters:
            # Charged leptons: e-, mu-, tau- (PDG: 11, 13, 15)
            if pdg in [11, 13, 15]:
                charged_lepton_p4 = p4
            # Charged antileptons: e+, mu+, tau+ (PDG: -11, -13, -15)
            elif pdg in [-11, -13, -15]:
                charged_lepton_p4 = p4
            # Antineutrinos: nu_e_bar, nu_mu_bar, nu_tau_bar (PDG: -12, -14, -16)
            elif pdg in [-12, -14, -16]:
                antineutrino_p4 = p4
            # Neutrinos: nu_e, nu_mu, nu_tau (PDG: 12, 14, 16)
            elif pdg in [12, 14, 16]:
                neutrino_p4 = p4
        
        # If we identified all three components, return ordered
        if charged_lepton_p4 and antineutrino_p4 and neutrino_p4:
            return [charged_lepton_p4, antineutrino_p4, neutrino_p4]
    
    # Fallback: return original order
    # WARNING: Only FlatMatrixElement tolerates ambiguous ordering.
    # If you add a physics-sensitive ME that doesn't enforce ordering, it will silently fail.
    return p4s


def validate_event_inline(parent_E: float, final_states: list, tol: float = 1e-6) -> bool:
    """Inlined conservation check (faster, no tuple unpacking)."""
    E_f = sum(p[0] for p in final_states)
    px_f = sum(p[1] for p in final_states)
    py_f = sum(p[2] for p in final_states)
    pz_f = sum(p[3] for p in final_states)
    
    # Parent is at rest: px=py=pz=0
    return (abs(parent_E - E_f) < tol and 
            abs(px_f) < tol and 
            abs(py_f) < tol and 
            abs(pz_f) < tol)


def init_event_db():
    """Initialize tables for event storage (Postgres)."""
    with _conn() as conn, conn.cursor() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS events(
                id SERIAL PRIMARY KEY,
                parent TEXT,
                decay_mode TEXT,
                energy DOUBLE PRECISION,
                timestamp TEXT,
                event_weight DOUBLE PRECISION,
                phase_space_weight DOUBLE PRECISION
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS final_states(
                id SERIAL PRIMARY KEY,
                event_id INTEGER REFERENCES events(id) ON DELETE CASCADE,
                particle TEXT,
                px DOUBLE PRECISION,
                py DOUBLE PRECISION,
                pz DOUBLE PRECISION,
                E DOUBLE PRECISION
            )
        """)
        conn.commit()


def get_pdg_id(particle_name: str) -> int | None:
    """Resolve PDG ID from particle name."""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT "PDG ID" FROM particles
            WHERE LOWER("Name") = LOWER(%s) OR LOWER("Symbol") = LOWER(%s)
            LIMIT 1
        """, (particle_name, particle_name))
        row = cur.fetchone()
    return int(row[0]) if row else None


def _is_kinematically_allowed(parent_mass: float, daughter_names: list[str]) -> bool:
    """Check m_parent >= sum m_daughters (using cached masses)."""
    try:
        masses = [get_particle(name)["mass"] for name in daughter_names]
    except Exception:
        return False
    return parent_mass + 1e-9 >= sum(masses)


def _generate_decay_fourvectors(parent_mass: float, daughter_names: list[str], rng: np.random.Generator, apply_weights: bool = True, parent_pdg: int = 0):
    """
    Generate decay in rest frame. Returns (list of p4 tuples, total weight).
    
    Clean architecture:
        1. Phase space generation (kinematic sampling)
        2. Matrix element lookup (physics weighting)
        3. Total weight = ps_weight * |M|²
    
    No physics hacks. No special cases. Pure pipeline.
    """
    from .matrix_elements import get_matrix_element
    
    masses = [get_particle(name)["mass"] for name in daughter_names]
    N = len(masses)
    
    # Get PDG IDs for matrix element lookup
    daughter_pdgs = tuple(sorted([get_pdg_id_cached(name) or 0 for name in daughter_names]))
    decay_key = (parent_pdg, daughter_pdgs)
    
    # General 2-body decay
    if N == 2:
        m1, m2 = masses
        if parent_mass <= 0 or parent_mass + 1e-9 < (m1 + m2):
            raise ValueError("Kinematically forbidden 2-body decay")

        term1 = parent_mass**2 - (m1 + m2) ** 2
        term2 = parent_mass**2 - (m1 - m2) ** 2
        p_star = math.sqrt(max(term1 * term2, 0.0)) / (2 * parent_mass) if parent_mass > 0 else 0.0
        
        E1 = math.sqrt(m1**2 + p_star**2)
        E2 = math.sqrt(m2**2 + p_star**2)

        d1 = p4(E1, 0.0, 0.0, +p_star)
        d2 = p4(E2, 0.0, 0.0, -p_star)
        return [d1, d2], 1.0

    # General 3-body decay
    elif N == 3:
        p4s_rest, ps_weight = generate_three_body_decay(parent_mass, masses, rng)
        
        if apply_weights:
            parent_p4 = (parent_mass, 0.0, 0.0, 0.0)
            me = get_matrix_element(decay_key)
            
            # CRITICAL: Reorder daughters to match matrix element expectations
            # V-A expects: [charged_lepton, antineutrino, neutrino]
            ordered_p4s = _reorder_daughters_for_matrix_element(
                daughter_names, p4s_rest, parent_pdg
            )
            
            M2 = me.M2(parent_p4, ordered_p4s, context=None)
            total_weight = ps_weight * M2
        else:
            total_weight = ps_weight
            
        return p4s_rest, total_weight

    else:
        raise NotImplementedError("Only 2- and 3-body decays supported")


def _simulate_event_inmemory(parent_name: str, 
                             parent_mass: float,
                             parent_pdg: int,
                             fixed_decay_mode: str | None,
                             event_weight: float, 
                             rng: np.random.Generator, 
                             M2_max: dict, 
                             event_index: int = 0, 
                             warmup_events: int = 500,
                             run_timestamp: str = ""):
    """
    Generate event with all parent info pre-resolved. No DB lookups in loop.
    
    Args:
        parent_name: Particle name (cached)
        parent_mass: Pre-resolved mass
        parent_pdg: Pre-resolved PDG ID
        fixed_decay_mode: If single channel, bypass RNG
        run_timestamp: Computed once per run
        M2_max: UNUSED - accept-reject unweighting not yet implemented
        warmup_events: UNUSED - accept-reject unweighting not yet implemented
    """
    # Bypass decay mode selection if single channel
    if fixed_decay_mode:
        decay_mode = fixed_decay_mode
    else:
        decay_mode = choose_decay_mode(parent_pdg, rng)
    
    if decay_mode == "stable":
        return None
    
    try:
        daughters = get_decay_products(parent_pdg, decay_mode)
    except Exception:
        return None
    
    if not _is_kinematically_allowed(parent_mass, daughters):
        return None
    
    try:
        p4s_rest, total_weight = _generate_decay_fourvectors(parent_mass, daughters, rng, apply_weights=True, parent_pdg=parent_pdg)
    except Exception:
        return None
    
    # Store weighted events (no accept-reject for now)
    stored_weight = event_weight * total_weight
    
    event_row = (
        parent_name,
        decay_mode,
        parent_mass,
        run_timestamp,  # Use pre-computed timestamp
        stored_weight,
        total_weight,
    )
    
    # Store final states as a list
    final_state_rows = [
        (name, float(fv[1]), float(fv[2]), float(fv[3]), float(fv[0]))
        for name, fv in zip(daughters, p4s_rest)
    ]
    
    return event_row, final_state_rows


def _flush_batch(event_rows, final_state_groups, batch_size: int = 5000):
    """
    Bulk insert events and their final states in safe batches.
    batch_size: Max events per transaction (Postgres handles 5k-20k easily)
    final_state_groups: list of lists, one per event.
    """
    if not event_rows:
        return
    
    total_events = len(event_rows)
    
    try:
        for batch_start in range(0, total_events, batch_size):
            batch_end = min(batch_start + batch_size, total_events)
            batch_events = event_rows[batch_start:batch_end]
            batch_final_states = final_state_groups[batch_start:batch_end]
            
            with _conn() as conn, conn.cursor() as cur:
                # Insert events and collect IDs
                event_ids = []
                for row in batch_events:
                    cur.execute(
                        "INSERT INTO events(parent, decay_mode, energy, timestamp, event_weight, phase_space_weight) "
                        "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                        row
                    )
                    event_ids.append(cur.fetchone()[0])
                
                # Build final_states with correct FK linkage
                fs_rows = []
                for event_id, fs_group in zip(event_ids, batch_final_states):
                    for name, px, py, pz, E in fs_group:
                        fs_rows.append((event_id, name, px, py, pz, E))
                
                # Bulk insert final_states
                if fs_rows:
                    cur.executemany(
                        "INSERT INTO final_states(event_id, particle, px, py, pz, E) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        fs_rows
                    )
                
                conn.commit()
    except Exception as e:
        print(f"[ERROR] Batch flush failed: {e}")


def simulate_events(parent_name: str,
                    events: int = 10,
                    n_events: int | None = None,
                    seed: int | None = None,
                    event_weight: float = 1.0,
                    verbose: bool = False,
                    warmup_events: int = 500,
                    use_accept_reject: bool = False,
                    store_neutrinos: bool = False) -> dict:
    """
    High-performance batch event generator: pure RAM generation → single DB dump.
    
    Args:
        use_accept_reject: Apply dynamic unweighting (False = weighted events, True = unweighted)
        store_neutrinos: Include neutrinos in final_states (False for leaner output)
    
    Philosophy: Generate weighted events first, optimize unweighting later.
    Note: accept-reject disabled by default (mathematically requires bounded M2).
    """
    if verbose:
        assert events <= 100, "Verbose mode only for debugging (≤100 events)"
    
    total = n_events if n_events is not None else events
    rng = np.random.default_rng(seed)
    
    # ========== PRE-RESOLUTION: Everything computed once ==========
    print(f"\n[SETUP] Resolving {parent_name}...")
    parent_info = get_particle(parent_name)
    parent_mass = parent_info["mass"]
    
    parent_pdg = get_pdg_id_cached(parent_name)
    if parent_pdg is None:
        raise RuntimeError(f"Unknown particle: {parent_name}")
    
    # Pre-fetch decay modes; bypass RNG if single channel
    decay_modes = get_decay_modes(parent_pdg)
    if len(decay_modes) == 1:
        fixed_decay_mode = decay_modes[0][0]
        print(f"[SETUP] Single decay mode: {fixed_decay_mode}")
    else:
        fixed_decay_mode = None
        print(f"[SETUP] {len(decay_modes)} decay modes available")
    
    # Timestamp once per run
    run_timestamp = datetime.utcnow().isoformat()
    
    success = 0
    failed = 0
    rejected = 0
    
    # All events collected in RAM (NO DB access during generation)
    events_out = []
    final_states_out = []
    
    # Track max M² per decay mode
    M2_max = {}
    
    # Progress
    show_progress = total > 100
    last_progress = 0
    
    # ========== GENERATION PHASE (RAM ONLY) ==========
    print(f"[GEN] Generating {total} events for {parent_name}...")
    start_gen = datetime.now()
    
    for i in range(total):
        result = _simulate_event_inmemory(
            parent_name, 
            parent_mass,
            parent_pdg,
            fixed_decay_mode,
            event_weight, 
            rng,
            M2_max,
            event_index=i,
            warmup_events=warmup_events if use_accept_reject else -1,
            run_timestamp=run_timestamp
        )
        
        if result is not None:
            event_row, final_state_rows = result
            
            # Optional: filter neutrinos
            if not store_neutrinos:
                final_state_rows = [
                    fs for fs in final_state_rows 
                    if "nu" not in fs[0].lower()
                ]
            
            events_out.append(event_row)
            final_states_out.append(final_state_rows)
            success += 1
        else:
            failed += 1
        
        # Progress
        if show_progress:
            progress_pct = (i + 1) / total
            if progress_pct - last_progress >= 0.05:
                print(f"[GEN] {i+1:6d}/{total} ({progress_pct*100:5.1f}%) | Success: {success:6d}")
                last_progress = progress_pct
    
    gen_time = (datetime.now() - start_gen).total_seconds()
    print(f"[GEN] ✓ Complete: {success} events in {gen_time:.2f}s ({success/gen_time:.0f} evt/sec)")
    
    # ========== PERSISTENCE PHASE (ONE TRANSACTION) ==========
    store_time = 0.0
    if events_out:
        print(f"\n[STORE] Writing {success} events + {sum(len(fs) for fs in final_states_out)} particles...")
        start_store = datetime.now()
        _flush_batch(events_out, final_states_out)
        store_time = (datetime.now() - start_store).total_seconds()
        print(f"[STORE] ✓ Complete in {store_time:.2f}s")
    
    return {
        "success": success,
        "failed": failed,
        "rejected": rejected,
        "total": total,
        "gen_time": gen_time,
        "store_time": store_time,
        "run_timestamp": run_timestamp,
    }
