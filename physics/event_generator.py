import logging
import numpy as np
import warnings
import math
from typing import List, Dict, Optional, Tuple
from db import get_conn
from .particles import Particle
from .kinematics import FourVector
from .events import EventDB
from .decays import decay_particle
from .decay_selector import choose_decay_mode, get_decay_products
from .unweighting import UnweightingController



logger = logging.getLogger(__name__)


def sort_daughters_for_matrix_element(
    parent_pdg_id: int,
    daughter_particles: List[Particle],
    daughter_p4s: List[FourVector]
) -> Tuple[List[Particle], List[FourVector]]:
    """
    Reorder daughters to match matrix element conventions.
    
    This is matrix-element-specific and MUST match the order
    assumed by get_matrix_element().
    
    For now: lexicographic by PDG ID (stable, reproducible).
    Future: consult a registry for each decay.
    """
    # Sort by PDG ID for reproducibility
    sorted_pairs = sorted(zip(daughter_particles, daughter_p4s), key=lambda x: x[0].pdg_id)
    sorted_particles, sorted_p4s = zip(*sorted_pairs) if sorted_pairs else ([], [])
    
    logger.debug(
        f"Daughter ordering for PDG {parent_pdg_id}: "
        f"{[p.pdg_id for p in sorted_particles]}"
    )
    
    return list(sorted_particles), list(sorted_p4s)


def validate_kinematics(parent_mass: float, daughter_masses: List[float]) -> bool:
    """Check if decay is kinematically allowed."""
    return parent_mass >= sum(daughter_masses)


def validate_conservation(parent_p4: FourVector,
                          daughters: List[FourVector],
                          energy_tol: float = 1e-6,
                          momentum_tol: float = 1e-6) -> Dict[str, bool]:
    """
    Verify energy and momentum conservation against the provided parent four-vector.
    """
    total = sum(daughters[1:], start=daughters[0])
    dE = abs(total.E - parent_p4.E)
    dpx = total.px - parent_p4.px
    dpy = total.py - parent_p4.py
    dpz = total.pz - parent_p4.pz
    dp = math.sqrt(dpx * dpx + dpy * dpy + dpz * dpz)
    return {
        "energy": dE < energy_tol,
        "momentum": dp < momentum_tol,
    }


def simulate_event(parent_name: str,
                   rng: Optional[np.random.Generator] = None,
                   use_matrix_element: bool = True,
                   unweighting_controller: Optional[UnweightingController] = None) -> Optional[int]:
    """
    Generate one decay event and store in the DB.

    Args:
        parent_name: Particle name (e.g., 'Higgs boson', 'pion')
        rng: Optional RNG for reproducibility
        use_matrix_element: Include |M|² weighting (default True)
    
    Returns:
        event_id of stored event, or None if decay is stable
    """
    rng = rng or np.random.default_rng()
    db = EventDB()
    parent = Particle(parent_name)
    
    # Step 1: Get decay mode from decay_selector (DB authoritative)
    mode = choose_decay_mode(parent.pdg_id, rng=rng)
    if mode == "stable":
        logger.info(f"⚛️  {parent.name} is stable, no decay generated")
        return None
    
    # Step 2: Get daughters from decay_selector
    daughters = get_decay_products(parent.pdg_id, mode)
    daughter_particles = [Particle(d) for d in daughters]
    daughter_masses = [p.mass for p in daughter_particles]
    
    # Step 3: Validate kinematics
    if not validate_kinematics(parent.mass, daughter_masses):
        logger.error(
            f"Decay not allowed: {parent.name} ({parent.mass:.3f} MeV) "
            f"→ {daughters} ({sum(daughter_masses):.3f} MeV)"
        )
        raise ValueError("Decay kinematically forbidden")
    
    # Step 4: Generate four-vectors (pure kinematics)
    fvs = decay_particle(parent.mass, daughter_masses, rng=rng)
    
    # Step 5a: Set parent 4-vector explicitly (at rest in decay frame)
    parent.fourvec = FourVector(parent.mass, 0.0, 0.0, 0.0)
    
    # Step 5b: Validate conservation
    conserved = validate_conservation(parent.fourvec, fvs)
    
    # Step 6: Reorder daughters for matrix element convention
    daughter_particles, fvs = sort_daughters_for_matrix_element(
        parent.pdg_id, daughter_particles, fvs
    )
    
    # Step 7: Calculate matrix element weight (optional)
    weight = 1.0
    if use_matrix_element:
        try:
            from physics.matrix_elements import get_matrix_element
            
            decay_key = (
                parent.pdg_id,
                tuple([p.pdg_id for p in daughter_particles])  # NOW IN CORRECT ORDER
            )
            me = get_matrix_element(decay_key)
            if me:
                weight = me.M2(
                    parent_p4=parent.fourvec.to_tuple(),
                    daughter_p4s=[fv.to_tuple() for fv in fvs],
                )
            logger.debug(f"Matrix element weight: {weight:.4f}")
        except Exception as e:
            logger.warning(f"Matrix element lookup failed: {e}, using weight=1.0")
            weight = 1.0
    
    # Step 7b: Accept–reject unweighting (if enabled)
    if unweighting_controller is not None:
        accepted = unweighting_controller.accept(weight, rng)
        if not accepted:
            logger.debug("Event rejected by unweighting controller")
            return None
        weight = 1.0
    
    # Step 8: Pair names with 4-vectors (NOW ORDERED)
    daughter_names = [p.name for p in daughter_particles]
    daughter_data = list(zip(daughter_names, fvs))
    
    # Step 9: Build decay_mode string for DB
    decay_mode = f"{parent.name} → {' '.join(daughter_names)}"
    energy = parent.fourvec.E
    
    # Step 10: Store in DB
    db.store_event(
        parent_name=parent.name,
        decay_mode=decay_mode,
        energy=energy,
        daughters=daughter_data,
        event_type=f"{len(daughters)}_body_decay",
        process_tag=decay_mode,
        weight=weight,
    )
    
    # Get last inserted event_id
    with db.cursor() as cur:
        cur.execute("SELECT MAX(id) FROM events")
        event_id = cur.fetchone()[0]
    
    logger.info(
        f"✅ Event {event_id}: {parent.name} → {' '.join(daughter_names)} "
        f"[w={weight:.4f}]"
    )
    
    return event_id


def simulate_batch(parent_name: str, n: int = 10, 
                   seed: Optional[int] = None,
                   use_matrix_element: bool = True,
                   unweight: bool = False) -> Dict[str, int]:
    """
    Generate multiple decay events.
    """
    rng = np.random.default_rng(seed)
    success = 0
    failed = 0
    rejected = 0
    
    controller = None
    if unweight and use_matrix_element:
        w_max = estimate_w_max(parent_name, rng=rng)
        controller = UnweightingController(w_max)
        logger.info(f"Estimated w_max = {w_max:.3e}")
     
    for i in range(n):
        try:
            eid = simulate_event(
                parent_name,
                rng=rng,
                use_matrix_element=use_matrix_element,
                unweighting_controller=controller,
            )
            if eid is not None:
                success += 1
            else:
                rejected += 1
        except Exception as e:
            logger.warning(f"Event {i+1}/{n} failed: {e}")
            failed += 1
     
    logger.info(f"\n✅ Batch complete: {success}/{n} succeeded, {failed} failed, {rejected} rejected")
    if controller is not None:
        logger.info(f"Unweighting efficiency: {controller.efficiency:.3f}")
     
    return {
        "success": success,
        "failed": failed,
        "rejected": rejected,
        "total": n
    }


def get_available_parents() -> List[str]:
    """Return list of particles with defined decay channels."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT p.\"Name\" FROM particles p "
            "INNER JOIN decays d ON p.\"PDG ID\" = d.pdg_id"
        )
        parents = sorted([row[0] for row in cur.fetchall()])
    return parents


def generate_decay_kinematics(parent_particle, daughter_names):
    """
    ⚠️  LEGACY: Bypasses event pipeline (no decay_selector, matrix elements, weighting).
    
    Given a parent Particle and ordered daughter names,
    generate FourVectors for daughters.
    
    Use simulate_event() instead for production workflows.
    """
    warnings.warn(
        "generate_decay_kinematics is legacy and bypasses decay_selector, "
        "matrix elements, and conservation checks. Use simulate_event() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    daughter_particles = [Particle(name) for name in daughter_names]
    daughter_masses = [p.mass for p in daughter_particles]

    p4s = decay_particle(parent_particle.mass, daughter_masses)

    # Assign four-vectors back to particles
    for p, fv in zip(daughter_particles, p4s):
        p.fourvec = fv

    return daughter_particles



def estimate_w_max(parent_name: str, n_trials: int = 5000, rng=None) -> float:
    rng = rng or np.random.default_rng()
    parent = Particle(parent_name)

    w_max = 0.0

    for _ in range(n_trials):
        mode = choose_decay_mode(parent.pdg_id, rng=rng)
        if mode == "stable":
            continue

        daughters = get_decay_products(parent.pdg_id, mode)
        daughter_particles = [Particle(d) for d in daughters]
        daughter_masses = [p.mass for p in daughter_particles]

        if not validate_kinematics(parent.mass, daughter_masses):
            continue

        fvs = decay_particle(parent.mass, daughter_masses, rng=rng)

        daughter_particles, fvs = sort_daughters_for_matrix_element(
            parent.pdg_id, daughter_particles, fvs
        )

        from physics.matrix_elements import get_matrix_element
        decay_key = (parent.pdg_id, tuple(p.pdg_id for p in daughter_particles))
        me = get_matrix_element(decay_key)
        if not me:
            continue

        w = me.M2(
            parent_p4=(parent.mass, 0.0, 0.0, 0.0),
            daughter_p4s=[fv.to_tuple() for fv in fvs],
        )

        w_max = max(w_max, w)

    if w_max <= 0:
        raise RuntimeError("Failed to estimate w_max (no valid decays or matrix elements)")
    return w_max