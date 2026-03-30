from physics.collision import simulate_events


def main() -> None:
    result = simulate_events(
        parent_name="Pion0",
        n_events=5000,
        seed=42,
        use_accept_reject=True,
        warmup_events=500,
        verbose=False,
    )

    accepted = result["success"]
    rejected = result["rejected"]
    total_attempted = accepted + rejected
    efficiency = (accepted / total_attempted) if total_attempted > 0 else 0.0

    print("Accepted:", accepted)
    print("Rejected:", rejected)
    print("Efficiency:", efficiency)


if __name__ == "__main__":
    main()
