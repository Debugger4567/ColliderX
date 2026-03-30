"""
Validate π⁰ → γγ decay via opening angle distribution.

At truth level, a π⁰ decaying at rest produces two photons that are:
  • Equal energy
  • Exactly back-to-back
  • Opening angle θ = π radians (180°)

This script verifies that prediction by plotting the opening angle histogram.
A razor-sharp spike at π = success. Anything else = physics bug.
"""

import numpy as np
import matplotlib.pyplot as plt
from db import get_conn


def load_pi0_events():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT 
            e.id,
            fs.particle,
            fs.px,
            fs.py,
            fs.pz
        FROM events e
        JOIN final_states fs ON fs.event_id = e.id
        WHERE e.parent = 'Pion0'
        ORDER BY e.id
    """)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    events = {}
    for event_id, particle, px, py, pz in rows:
        events.setdefault(event_id, []).append((particle, px, py, pz))
    return events


def opening_angle(p1, p2):
    """Return opening angle between two 3-vectors in radians."""
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)

    dot = np.dot(p1, p2)
    mag = np.linalg.norm(p1) * np.linalg.norm(p2)
    if mag == 0:
        return np.nan

    cos_theta = np.clip(dot / mag, -1.0, 1.0)
    return np.arccos(cos_theta)


def main():
    events = load_pi0_events()

    angles = []
    skipped = 0

    for event_id, particles in events.items():
        photons = [p for p in particles if p[0] == "Photon"]
        if len(photons) != 2:
            skipped += 1
            continue

        _, px1, py1, pz1 = photons[0]
        _, px2, py2, pz2 = photons[1]

        theta = opening_angle((px1, py1, pz1), (px2, py2, pz2))
        if not np.isnan(theta):
            angles.append(theta)

    angles = np.array(angles)

    print(f"[π0 → γγ] Used events   : {len(angles)}")
    print(f"[π0 → γγ] Skipped events: {skipped}")

    assert len(angles) > 0, "No valid π0 → γγ events found!"

    plt.figure(figsize=(7, 5))
    plt.hist(angles, bins=60, range=(0, np.pi), density=True, alpha=0.8)
    plt.axvline(np.pi, color="r", linestyle="--", linewidth=2, label=r"$\theta = \pi$")
    plt.xlabel("Opening angle θ (radians)")
    plt.ylabel("Normalized counts")
    plt.title(r"$\pi^0 \to \gamma\gamma$ opening angle (truth level)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
