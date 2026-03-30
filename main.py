import argparse
from datetime import datetime

from visualization.flow import run_full_flow


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ColliderX MVP CLI (one command flow)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py Muon 1000\n"
            "  python main.py -p \"Z boson\" -n 500 -s 7\n"
            "  python main.py -p Muon -n 1000 --save --no-show"
        ),
    )

    parser.add_argument(
        "particle_pos",
        nargs="?",
        default=None,
        help="Particle name (positional shortcut)",
    )
    parser.add_argument(
        "events_pos",
        nargs="?",
        type=int,
        default=None,
        help="Number of events (positional shortcut)",
    )
    parser.add_argument("-p", "--particle", default=None, help="Particle name")
    parser.add_argument("-n", "--events", type=int, default=None, help="Number of events")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Random seed")
    parser.add_argument("-E", "--energy", type=float, default=None, help="Parent energy override in MeV")
    parser.add_argument("--save", action="store_true", help="Save generated diagram/graphs to output directory")
    parser.add_argument("--no-show", action="store_true", help="Do not open plot windows")
    parser.add_argument("-o", "--out", default=None, help="Output directory (default: artifacts/run_<timestamp>)")
    parser.add_argument("--decay", default=None, help="Optional fixed root decay mode")
    parser.add_argument("--afb", type=float, default=0.0, help="Forward-backward asymmetry parameter")

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.particle = args.particle or args.particle_pos or "Muon"
    args.events = args.events or args.events_pos or 1000
    return args


def main() -> None:
    args = parse_args()
    output_dir = args.out or f"artifacts/run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    result = run_full_flow(
        parent_name=args.particle,
        n_events=args.events,
        seed=args.seed,
        output_dir=output_dir,
        afb=args.afb,
        fixed_decay_mode=args.decay,
        parent_energy=args.energy,
        diagram_mode="sample",
        diagram_max_depth=3,
        diagram_max_nodes=500,
        diagram_rankdir="LR",
        render_format="png",
        show_images=not args.no_show,
        save_images=args.save,
    )
    print(result)


if __name__ == "__main__":
    main()
