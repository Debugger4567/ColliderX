"""
Validate tau helicity effect:
Plot pion cosθ distribution for Z→τ⁺τ⁻→(πνν̄)(πνν̄).
"""

import numpy as np
import matplotlib.pyplot as plt
from physics.collision import simulate_chain


def main():
    print("Generating Z→τ⁺τ⁻→(πνν̄)(πνν̄) chain events...")

    mode_candidates = ["τ+ τ−", "tau+ tau-"]
    result = None
    last_err = None
    for mode in mode_candidates:
        try:
            result = simulate_chain(
                parent_name="Z boson",
                n_events=10000,
                seed=42,
                fixed_decay_mode=mode,
                force_tau_pion_only=True,
            )
            break
        except Exception as exc:
            last_err = exc

    if result is None:
        raise RuntimeError(
            f"Could not run tau chain for any decay mode: {mode_candidates}"
        ) from last_err

    cos_thetas = np.array(result.get("tau_pion_cos_theta", []), dtype=float)
    print("N cosθ:", cos_thetas.size, "mean:", float(np.mean(cos_thetas)))

    by_charge = result.get("tau_pion_by_charge", {})
    for k, vals in by_charge.items():
        arr = np.array(vals, dtype=float)
        if arr.size:
            print(f"{k}: N={arr.size}, mean={arr.mean():.4f}")

    if cos_thetas.size == 0:
        raise RuntimeError("No tau->pi events found. Check tau decay modes in DB.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: raw cosθ distribution
    axes[0].hist(
        cos_thetas,
        bins=50,
        range=(-1, 1),
        density=True,
        alpha=0.8,
        color="#7b8cf7",
        label="Simulation",
    )

    x = np.linspace(-1, 1, 200)
    axes[0].plot(x, np.ones_like(x) * 0.5, "r--", lw=2, label="Flat (unpolarized)")
    axes[0].set_xlabel(r"$\cos\theta_\pi$")
    axes[0].set_ylabel("Normalized events")
    axes[0].set_title(
        r"Pion angular distribution: $Z \to \tau^+\tau^- \to (\pi\nu)(\pi\bar\nu)$"
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Right: reference helicity shapes
    hist, edges = np.histogram(cos_thetas, bins=20, range=(-1, 1), density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])

    axes[1].scatter(centres, hist, color="#7b8cf7", s=30, zorder=5, label="Simulation")
    axes[1].plot(
        x,
        0.5 * (1 + x),
        color="#e87070",
        lw=2,
        linestyle="--",
        label=r"$\frac{1}{2}(1+\cos\theta)$ [h=+1]",
    )
    axes[1].plot(
        x,
        0.5 * (1 - x),
        color="#4ecb7a",
        lw=2,
        linestyle="--",
        label=r"$\frac{1}{2}(1-\cos\theta)$ [h=-1]",
    )
    axes[1].set_xlabel(r"$\cos\theta_\pi$")
    axes[1].set_ylabel("Probability density")
    axes[1].set_title("Spin correlation signal")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
