import argparse
from physics.collision import simulate_events

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ColliderX event simulation entrypoint")
    parser.add_argument("--parent", default="Z boson", help='Parent particle name (default: "Z boson")')
    parser.add_argument("--n-events", type=int, default=1000, help="Number of events to simulate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--afb", type=float, default=0.0, help="Forward-backward asymmetry parameter")
    parser.add_argument("--fixed-decay-mode", default=None, help="Optional fixed decay mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    result = simulate_events(
        parent_name=args.parent, 
        n_events = args.n_events, 
        seed=args.seed,
        afb=args.afb,
        fixed_decay_mode=args.fixed_decay_mode, 
        verbose=args.verbose,
    )
    print(result)



if __name__ == "__main__":
    main()