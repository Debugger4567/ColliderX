from physics.collision import simulate_events


def main():
    result = simulate_events(
        parent_name="Higgs",
        n_events=1,
        seed=42,
    )
    print(result)


if __name__ == "__main__":
    main()
