import numpy as np
import matplotlib.pyplot as plt


M_MU = 105.66  # Muon mass in MeV
N = 200_000

def main():
    rng = np.random.default_rng(42)

    #Simple flat 3-body energy sharing
    E1 = rng.random(N)
    E2 = rng.random(N)
    E3 = rng.random(N)

    S = E1+ E2 + E3
    E1 /= S
    E1 *= M_MU / 2

    x=2 * E1 / M_MU 

    plt.figure(figsize=(7,5))
    plt.hist(x, bins=60, density=True, alpha=0.8, label="Flat phase space")
    plt.xlabel(r"$x = 2E_e/m_\mu$")
    plt.ylabel("Normalized counts")
    plt.title("Flat phase space (no matrix element)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()