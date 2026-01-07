import json
import random
import logging
from typing import List, Dict, Optional
from db import get_conn
from .particles import Particle
from .kinematics import generate_two_body_decay, generate_three_body_decay, FourVector
from .events import EventDB

logger = logging.getLogger(__name__)

def load_decay_channels(parent_name: str) -> List[Dict]:
    """
    Load decay channels from colliderx.db.
    
    Returns:
        List of dicts with 'daughters' and 'branching_fraction'
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT daughter1, daughter2, branching_fraction
            FROM decays
            WHERE parent = %s
            """,
            (parent_name,),
        )
        rows = cur.fetchall()
    
    if not rows:
        raise ValueError(f"No decay channels found for '{parent_name}'")
    
    channels = []
    for row in rows:
        daughters = [row[0]]
        if row[1]:  # Two-body decay
            daughters.append(row[1])
        
        channels.append({
            "daughters": daughters,
            "branching_fraction": row[2]
        })
    
    return channels


def select_decay_channel(channels: List[Dict]) -> Dict:
    """
    Randomly select a decay channel weighted by branching fraction.
    
    Args:
        channels: List of decay channel dicts
    Returns:
        Selected channel dict
    """
    if not channels:
        raise ValueError("No channels provided")
    
    total_br = sum(ch["branching_fraction"] for ch in channels)
    rand = random.uniform(0, total_br)
    cumulative = 0.0
    
    for ch in channels:
        cumulative += ch["branching_fraction"]
        if rand <= cumulative:
            return ch
    
    return channels[-1]  # Fallback to last channel


def validate_kinematics(parent_mass: float, daughter_masses: List[float]) -> bool:
    """Check if decay is kinematically allowed."""
    return parent_mass >= sum(daughter_masses)


def validate_conservation(parent_mass: float, daughters: List[FourVector], 
                         energy_tol: float = 1e-6, 
                         momentum_tol: float = 1e-6) -> Dict[str, bool]:
    """
    Verify energy and momentum conservation.
    
    Returns:
        Dict with 'energy' and 'momentum' boolean flags
    """
    total = sum(daughters[1:], start=daughters[0])
    
    return {
        "energy": abs(total.E - parent_mass) < energy_tol,
        "momentum": total.magnitude < momentum_tol
    }


def simulate_event(parent_name: str, channel: Optional[Dict] = None, 
                   rng: Optional[random.Random] = None) -> int:
    """
    Generate one decay event and store in the DB.

    Args:
        parent_name: Particle name (e.g., 'Higgs boson', 'pion')
        channel: Optional specific decay channel dict; if None, randomly selects
        rng: Optional RNG for reproducibility
    
    Returns:
        event_id of stored event
    """
    rng = rng or random.Random()
    db = EventDB()
    parent = Particle(parent_name)
    
    # Load and select decay channel
    if channel is None:
        channels = load_decay_channels(parent.name)
        channel = select_decay_channel(channels)
    
    daughters = channel["daughters"]
    daughter_masses = [Particle(d).mass for d in daughters]
    
    # Validate kinematics
    if not validate_kinematics(parent.mass, daughter_masses):
        raise ValueError(
            f"Decay not allowed: {parent.name} ({parent.mass:.3f} GeV) "
            f"→ {daughters} ({sum(daughter_masses):.3f} GeV)"
        )
    
    # Generate four-vectors
    if len(daughters) == 2:
        fvs = generate_two_body_decay(
            parent.mass, 
            tuple(daughter_masses)
        )
    elif len(daughters) == 3:
        fvs = generate_three_body_decay(
            parent.mass, 
            tuple(daughter_masses)
        )
    else:
        raise NotImplementedError(
            f"Only 2- and 3-body decays supported, got {len(daughters)}"
        )
    
    # Validate conservation
    conserved = validate_conservation(parent.mass, fvs)
    
    # Pair names with 4-vectors
    daughter_data = list(zip(daughters, fvs))
    
    # Store in DB
    db.store_event(
        parent_name=parent.name,
        parent_mass=parent.mass,
        daughters=daughter_data,
        event_type=f"{len(daughters)}_body_decay",
        process_tag=f"{parent.name} → {' '.join(daughters)}",
        energy_tolerance=1e-6,
        momentum_tolerance=1e-6
    )
    
    # Get last inserted event_id
    with db.get_connection() as conn:
        event_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    
    logger.info(
        f"✅ Event {event_id}: {parent.name} → {' '.join(daughters)} "
        f"[E: {'✓' if conserved['energy'] else '✗'}, "
        f"p: {'✓' if conserved['momentum'] else '✗'}]"
    )
    
    return event_id


def simulate_batch(parent_name: str, n: int = 10, 
                   seed: Optional[int] = None) -> Dict[str, int]:
    """
    Generate multiple decay events.
    
    Args:
        parent_name: Particle to decay
        n: Number of events
        seed: Optional random seed for reproducibility
    
    Returns:
        Stats dict with 'success', 'failed', 'total'
    """
    rng = random.Random(seed)
    success = 0
    failed = 0
    
    for i in range(n):
        try:
            simulate_event(parent_name, rng=rng)
            success += 1
        except Exception as e:
            logger.warning(f"Event {i+1}/{n} failed: {e}")
            failed += 1
    
    logger.info(f"\n✅ Batch complete: {success}/{n} succeeded, {failed} failed")
    
    return {
        "success": success,
        "failed": failed,
        "total": n
    }


def get_available_parents() -> List[str]:
    """Return list of particles with defined decay channels."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT parent FROM decays")
        parents = [row[0] for row in cur.fetchall()]
    return sorted(parents)
