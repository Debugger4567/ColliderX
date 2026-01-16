#!/usr/bin/env python3
"""
Monte Carlo driver script for ColliderX

Examples:
    python monte_carlo.py --particle "Pion+" --events 1000
    python monte_carlo.py --particle "Z boson" --events 100 --seed 42 --output z_events.csv
"""

import argparse
import csv
from physics.collision import simulate_events, init_event_db
from db import get_conn


def print_event_stats():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM events")
        total_events = cur.fetchone()[0]

        cur.execute(
            """
            SELECT parent, COUNT(*) AS count, AVG(energy) AS avg_energy
            FROM events
            GROUP BY parent
            ORDER BY count DESC
            """
        )
        parent_stats = cur.fetchall()

        cur.execute(
            """
            SELECT decay_mode, COUNT(*) AS count
            FROM events
            GROUP BY decay_mode
            ORDER BY count DESC
            LIMIT 10
            """
        )
        decay_modes = cur.fetchall()

        cur.execute(
            """
            SELECT e.id, e.parent, e.energy, SUM(f.E) AS daughter_sum
            FROM events e
            JOIN final_states f ON e.id = f.event_id
            GROUP BY e.id
            LIMIT 200
            """
        )
        energy_violations = 0
        for event_id, parent, parent_E, daughter_sum in cur.fetchall():
            if abs(parent_E - daughter_sum) > 1e-3:
                energy_violations += 1

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
    rows_written = 0
    with get_conn() as conn, conn.cursor() as cur, open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "parent", "decay_mode", "daughter", "E", "px", "py", "pz"])
        for eid in event_ids:
            cur.execute(
                """
                SELECT e.id, e.parent, e.decay_mode, f.particle, f.E, f.px, f.py, f.pz
                FROM events e
                JOIN final_states f ON e.id = f.event_id
                WHERE e.id = %s
                ORDER BY f.id
                """,
                (eid,),
            )
            ev_rows = cur.fetchall()
            if ev_rows:
                writer.writerows(ev_rows)
                rows_written += len(ev_rows)

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
    parser = argparse.ArgumentParser(description="ColliderX Monte Carlo Event Generator")
    parser.add_argument("--particle", required=True, help="Parent particle name")
    parser.add_argument("--events", type=int, default=1000, help="Number of events")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--weight", type=float, default=1.0, help="Event weight")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--output", default=None, help="Output file")
    parser.add_argument("--store-neutrinos", action="store_true", help="Store neutrinos in final_states table (needed for Dalitz plots)")
    parser.add_argument("--unweighted", action="store_true", help="Enable accept-reject unweighting")
    parser.add_argument("--warmup", type=int, default=800, help="Warm-up events for w_max")

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

    result = simulate_events(
        parent_name=args.particle,
        n_events=args.events,
        seed=args.seed,
        event_weight=args.weight,
        verbose=args.verbose,
        store_neutrinos=args.store_neutrinos,
        use_accept_reject=args.unweighted,     # NEW
        warmup_events=args.warmup,             # NEW
    )

    print("\n" + "=" * 60)
    print("✅ Pipeline Complete")
    print("=" * 60)
    print(f"Successful events : {result['success']}/{result['total']}")
    print(f"Failed events     : {result['failed']}")
    success_rate = (result['success'] / result['total']) if result['total'] > 0 else 0.0
    print(f"Success rate      : {success_rate:.2%}")
    print(f"\nTiming:")
    print(f"  Generation : {result['gen_time']:.3f}s ({result['success']/result['gen_time']:.0f} evt/sec)")
    print(f"  Storage    : {result['store_time']:.3f}s")
    print(f"  Total      : {result['gen_time'] + result['store_time']:.3f}s")
    
    # Enhanced statistics from DB
    if result['success'] > 0:
        print("\n📊 Physics Summary")
        print("-" * 60)
        with get_conn() as conn, conn.cursor() as cur:
            # Decay mode distribution (filtered by current run timestamp)
            cur.execute("""
                SELECT decay_mode, COUNT(*) as count
                FROM events
                WHERE parent = %s AND timestamp = %s
                GROUP BY decay_mode
                ORDER BY count DESC
            """, (args.particle, result['run_timestamp']))
            modes = cur.fetchall()
            print("\nDecay modes sampled:")
            for mode, count in modes:
                pct = 100 * count / result['success']
                print(f"  {mode:30s}: {count:6d} ({pct:5.2f}%)")
            
            # Final state particle counts (filtered by current run timestamp)
            cur.execute("""
                SELECT particle, COUNT(*) as count
                FROM final_states fs
                JOIN events e ON fs.event_id = e.id
                WHERE e.parent = %s AND e.timestamp = %s
                GROUP BY particle
                ORDER BY count DESC
            """, (args.particle, result['run_timestamp']))
            particles = cur.fetchall()
            print("\nFinal state particles produced:")
            for particle, count in particles:
                print(f"  {particle:30s}: {count:6d}")
            
            # Energy statistics (filtered by current run timestamp)
            cur.execute("""
                SELECT 
                    AVG(E) as avg_E,
                    MIN(E) as min_E,
                    MAX(E) as max_E,
                    STDDEV(E) as std_E
                FROM final_states fs
                JOIN events e ON fs.event_id = e.id
                WHERE e.parent = %s AND e.timestamp = %s
            """, (args.particle, result['run_timestamp']))
            avg_E, min_E, max_E, std_E = cur.fetchone()
            print(f"\nDaughter energy distribution (MeV):")
            print(f"  Average  : {avg_E:.3f}")
            print(f"  Std Dev  : {std_E:.3f}")
            print(f"  Min      : {min_E:.3f}")
            print(f"  Max      : {max_E:.3f}")
    
    print("=" * 60 + "\n")

    if args.output:
        print("⚠️ Export skipped: event IDs not available from simulate_events().")

    if args.stats and result["success"] > 0:
        print_event_stats()


if __name__ == "__main__":
    main()
