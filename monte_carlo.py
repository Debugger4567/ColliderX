#!/usr/bin/env python3
"""
Monte Carlo driver script for ColliderX

Examples:
    python monte_carlo.py --particle "Pion+" --events 1000
    python monte_carlo.py --particle "Z boson" --events 100 --seed 42 --output z_events.csv
"""

import argparse
import sqlite3
import csv
from pathlib import Path
from physics.collision import simulate_events

DB_PATH = Path(__file__).resolve().parent / "colliderx.db"


def print_event_stats():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM events")
    total_events = cur.fetchone()[0]

    cur.execute("""
        SELECT parent, COUNT(*) AS count, AVG(energy) AS avg_energy
        FROM events
        GROUP BY parent
        ORDER BY count DESC
    """)
    parent_stats = cur.fetchall()

    cur.execute("""
        SELECT decay_mode, COUNT(*) AS count
        FROM events
        GROUP BY decay_mode
        ORDER BY count DESC
        LIMIT 10
    """)
    decay_modes = cur.fetchall()

    cur.execute("""
        SELECT e.id, e.parent, e.energy, SUM(f.E) AS daughter_sum
        FROM events e
        JOIN final_states f ON e.id = f.event_id
        GROUP BY e.id
        LIMIT 200
    """)
    energy_violations = 0
    for event_id, parent, parent_E, daughter_sum in cur.fetchall():
        if abs(parent_E - daughter_sum) > 1e-3:
            energy_violations += 1

    conn.close()

    print("\n📊 Database Statistics")
    print("=" * 60)
    print(f"Total events stored          : {total_events}")
    print("\nEvents by parent particle:")
    for parent, count, avg_E in parent_stats:
        print(f"  • {parent:20s}: {count:6d} events (avg E = {avg_E:.3f} MeV)")

    print("\nTop decay modes:")
    for mode, count in decay_modes:
        print(f"  • {mode:20s}: {count:6d} events")

    print("\nEnergy conservation (sample):")
    print(f"  Violations: {energy_violations}/{min(200, total_events)} (tolerance 1 keV)")
    print("=" * 60 + "\n")


def export_events_to_csv(event_ids, filename):
    """Export selected events to CSV."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    rows_written = 0

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "parent", "decay_mode", "daughter", "E", "px", "py", "pz"])
        for eid in event_ids:
            cur.execute("""
                SELECT e.id, e.parent, e.decay_mode, f.particle, f.E, f.px, f.py, f.pz
                FROM events e
                JOIN final_states f ON e.id = f.event_id
                WHERE e.id = ?
                ORDER BY f.id
            """, (eid,))
            ev_rows = cur.fetchall()
            if ev_rows:
                writer.writerows(ev_rows)
                rows_written += len(ev_rows)

    conn.close()
    print(f"📄 Exported {len(event_ids)} events ({rows_written} rows) to {filename}")


def build_parser():
    return argparse.ArgumentParser(
        description="ColliderX Monte Carlo Event Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python monte_carlo.py --particle "Pion+" --events 1000
  python monte_carlo.py --particle "Pion0" --events 500 --seed 42
  python monte_carlo.py --particle "Z boson" --events 100 --verbose --output z.csv"""
    )


def main():
    parser = build_parser()
    parser.add_argument("--particle", required=True, help='Parent particle name (e.g. "Pion+", "Z boson")')
    parser.add_argument("--events", type=int, default=10, help="Number of events (default 10)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    parser.add_argument("--weight", type=float, default=1.0, help="Event weight (default 1.0)")
    parser.add_argument("--verbose", action="store_true", help="Show progress output")
    parser.add_argument("--stats", action="store_true", help="Print DB statistics after generation")
    parser.add_argument("--output", type=str, help="Export generated events to CSV file")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🔥 ColliderX Monte Carlo Event Generator")
    print("=" * 60)
    print(f"Parent Particle  : {args.particle}")
    print(f"Number of Events : {args.events}")
    print(f"Random Seed      : {args.seed if args.seed is not None else 'None'}")
    print(f"Event Weight     : {args.weight}")
    if args.output:
        print(f"CSV Output       : {args.output}")
    print("=" * 60 + "\n")

    results = simulate_events(
        parent_name=args.particle,
        n_events=args.events,
        event_weight=args.weight,
        seed=args.seed,
        verbose=args.verbose
    )

    print("\n" + "=" * 60)
    print("✅ Generation Complete")
    print("=" * 60)
    print(f"Successful events : {results['success']}/{results['total']}")
    print(f"Failed events     : {results['failed']}")
    print(f"Success rate      : {results['success_rate']:.2%}")
    if results["event_ids"]:
        print(f"Event ID range    : {min(results['event_ids'])} - {max(results['event_ids'])}")
        if len(results["event_ids"]) <= 20:
            print(f"Event IDs         : {results['event_ids']}")
    else:
        print("No events generated.")
    print("=" * 60 + "\n")

    if args.output:
        export_events_to_csv(results["event_ids"], args.output)

    if args.stats and results["success"] > 0:
        print_event_stats()


if __name__ == "__main__":
    main()
