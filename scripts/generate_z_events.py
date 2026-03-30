"""
Generate Z → ℓ⁺ℓ⁻ events at truth level for Phase B validation.

Constraints:
  • Z at rest in lab frame
  • Fixed mass (no width smearing)
  • 2-body decay only
  • Full daughter 4-vectors stored for downstream analysis
"""

from physics.collision import simulate_events


def main():
    print("\n" + "=" * 70)
    print("PHASE B: Z → μ⁺μ⁻ Event Generation (Breit-Wigner)")
    print("=" * 70)

    # Generate Z → μ⁺μ⁻ with Breit-Wigner (weighted events)
    result = simulate_events(
        parent_name="Z0",
        n_events=10000,
        seed=42,
        event_weight=1.0,
        verbose=False,
        warmup_events=500,  # Learn M²_max during first 500 events
        use_accept_reject=False,  # MUST be False with BW
        store_neutrinos=False,
        ar_inflate=1.2,  # 20% safety margin on M²_max
        fixed_decay_mode="μ+ μ−",
        use_breit_wigner=True,  # Enable mass sampling
        bw_window=10.0,
        afb=-0.1,  # ← Change this to 0.0, +0.1, -0.1 per test
    )

    print("\n[GENERATION SUMMARY]")
    print(f"  Events accepted:   {result['success']}")
    print(f"  Events rejected:   {result['rejected']}")
    print(f"  Events failed:     {result['failed']}")
    if result["rejected"] > 0:
        print(
            f"  Efficiency:        {result['success']/(result['success']+result['rejected'])*100:.1f}%"
        )
    print(f"  Generation time:   {result['gen_time']:.2f}s")
    print(f"  Storage time:      {result['store_time']:.2f}s")
    print(f"  Total time:        {result['gen_time'] + result['store_time']:.2f}s")


if __name__ == "__main__":
    main()
