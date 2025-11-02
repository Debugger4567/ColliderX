from .particles import Particle
from .decay_selector import choose_decay_mode, choose_decay_daughters
from .kinematics import FourVector, generate_two_body_decay, generate_three_body_decay
import sqlite3
from datetime import datetime
from pathlib import Path
import random

DB_PATH = Path(__file__).resolve().parents[1] / "colliderx.db"


def init_event_db():
    """Initialize tables for event storage"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Main event metadata
    c.execute('''
        CREATE TABLE IF NOT EXISTS events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent TEXT,
            decay_mode TEXT,
            energy REAL,
            timestamp TEXT,
            event_weight REAL
        )
    ''')

    # Final-state particles (daughters)
    c.execute('''
        CREATE TABLE IF NOT EXISTS final_states(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER,
            particle TEXT,
            px REAL,
            py REAL,
            pz REAL,
            E REAL,
            FOREIGN KEY(event_id) REFERENCES events(id)
        )
    ''')

    conn.commit()
    conn.close()


def get_pdg_id(particle_name: str) -> int | None:
    """
    Resolve PDG ID from particle name using your particles table schema.
    Handles quoted column names like "PDG ID", "Name", "Symbol".
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Query using quoted column names
    cur.execute('''
        SELECT "PDG ID" FROM particles 
        WHERE LOWER("Name") = LOWER(?) OR LOWER("Symbol") = LOWER(?)
        LIMIT 1
    ''', (particle_name, particle_name))
    
    row = cur.fetchone()
    conn.close()
    
    return int(row["PDG ID"]) if row else None


def _is_kinematically_allowed(parent_mass: float, daughter_names: list[str]) -> bool:
    """Check m_parent >= sum m_daughters using Particle masses (GeV)."""
    try:
        dm = [Particle(name).mass for name in daughter_names]
        return parent_mass + 1e-12 >= sum(dm)
    except Exception as e:
        print(f"[DEBUG] Mass lookup failed for {daughter_names}: {e}")
        return True  # Assume allowed if lookup fails


def _generate_decay_fourvectors(parent_mass: float, daughter_names: list[str]) -> list[FourVector]:
    """Generate daughter 4-vectors in parent rest frame for 2- or 3-body decays."""
    try:
        dm = [Particle(name).mass for name in daughter_names]
    except Exception as e:
        raise ValueError(f"Cannot resolve masses for {daughter_names}: {e}")
    
    if len(dm) == 2:
        return generate_two_body_decay(parent_mass, (dm[0], dm[1]))
    if len(dm) == 3:
        return generate_three_body_decay(parent_mass, (dm[0], dm[1], dm[2]))
    raise NotImplementedError(f"Only 2- and 3-body decays are supported (got {len(dm)} daughters)")


def simulate_event(parent_name: str, event_weight: float = 1.0, rng: random.Random | None = None):
    """
    Simulate a single decay event based on PDG branching ratios.

    Steps:
      1. Fetch parent particle info (including PDG ID)
      2. Randomly choose decay mode using branching fractions
      3. Check kinematic feasibility (m_parent >= Σm_daughters)
      4. Generate rest-frame 4-vectors for daughters
      5. Store event and final states in colliderx.db
    
    Returns:
        event_id (int) if successful, None otherwise
    """
    rng = rng or random.Random()
    init_event_db()  # Ensure DB tables exist

    try:
        parent = Particle(parent_name)
    except Exception as e:
        print(f"[ERROR] Cannot load particle '{parent_name}': {e}")
        return None

    # Get PDG ID from DB (handles quoted column names)
    parent_pdg = get_pdg_id(parent.name)
    if parent_pdg is None:
        print(f"[ERROR] Cannot resolve PDG ID for '{parent.name}'")
        return None

    # Pick decay mode and daughters
    decay_mode = choose_decay_mode(parent_pdg, rng)
    daughters = choose_decay_daughters(parent_pdg, rng)

    if not daughters or decay_mode == "stable":
        # Particle is stable or no decay channels
        return None

    # Retry up to 10 times if kinematically forbidden
    attempts = 0
    while not _is_kinematically_allowed(parent.mass, daughters) and attempts < 10:
        decay_mode = choose_decay_mode(parent_pdg, rng)
        daughters = choose_decay_daughters(parent_pdg, rng)
        if not daughters or decay_mode == "stable":
            return None
        attempts += 1

    if not _is_kinematically_allowed(parent.mass, daughters):
        print(f"[WARN] Decay forbidden after retries: {parent.name} → {' '.join(daughters)}")
        return None

    # --- Kinematics ---
    try:
        fvs = _generate_decay_fourvectors(parent.mass, daughters)
    except Exception as e:
        print(f"[ERROR] Kinematics failed: {e}")
        return None
    
    decay_products = list(zip(daughters, fvs))

    # --- Store in DB ---
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        # Insert event info
        c.execute('''
            INSERT INTO events(parent, decay_mode, energy, timestamp, event_weight)
            VALUES(?, ?, ?, ?, ?)
        ''', (parent.name, decay_mode, parent.mass, datetime.now().isoformat(), event_weight))
        event_id = c.lastrowid

        # Insert each daughter as final state
        for name, fv in decay_products:
            c.execute('''
                INSERT INTO final_states(event_id, particle, px, py, pz, E)
                VALUES(?, ?, ?, ?, ?, ?)
            ''', (event_id, name, fv.px, fv.py, fv.pz, fv.E))

        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"[ERROR] DB insertion failed: {e}")
        event_id = None
    finally:
        conn.close()

    return event_id


def simulate_events(
    parent_name: str, 
    n_events: int, 
    event_weight: float = 1.0, 
    seed: int | None = None,
    verbose: bool = True
) -> dict:
    """
    Generate multiple decay events for the given parent particle.
    
    Args:
        parent_name: Name of parent particle (e.g., 'pion', 'pi+', 'π+')
        n_events: Number of events to generate
        event_weight: Weight assigned to each event
        seed: Random seed for reproducibility
        verbose: Print progress updates
    
    Returns:
        Dict with keys: event_ids, success, failed, total, parent, success_rate
    """
    rng = random.Random(seed)
    event_ids = []
    failed = 0
    
    for i in range(n_events):
        try:
            eid = simulate_event(parent_name, event_weight, rng)
            if eid is not None:
                event_ids.append(eid)
            else:
                failed += 1
        except Exception as e:
            if verbose:
                print(f"[WARN] Event {i+1}/{n_events} error: {e}")
            failed += 1
        
        # Progress tracking
        if verbose and ((n_events > 100 and (i + 1) % 100 == 0) or (n_events <= 100 and (i + 1) % 10 == 0)):
            print(f"[PROGRESS] {i+1}/{n_events} processed ({len(event_ids)} success, {failed} failed)")
    
    stats = {
        "event_ids": event_ids,
        "success": len(event_ids),
        "failed": failed,
        "total": n_events,
        "parent": parent_name,
        "success_rate": len(event_ids) / n_events if n_events > 0 else 0.0
    }
    
    if verbose:
        print(f"[INFO] ✅ Generated {stats['success']}/{n_events} events for {parent_name}. "
              f"Failed: {failed} ({stats['success_rate']:.1%} success rate)")
    
    return stats
