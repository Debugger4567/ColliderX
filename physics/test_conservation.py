import numpy as np
from kinematics import FourVector
from conservation import two_body_decay, check_energy_momentum

def test_decay(parent, m1, m2, n_trials=100):
    print(f"Testing decay: Parent mass={parent.mass():.2f} → {m1} + {m2}")

    for i in range(n_trials):
        d1, d2 = two_body_decay(parent, m1, m2)
        result = check_energy_momentum([parent], [d1, d2])

        if not result["conserved"]:
            print(" ❌ Conservation violated:", result)
            return False

    print(f" ✅ Passed {n_trials} trials (conserved)")
    return True


if __name__ == "__main__":
    # Example: Parent at rest decaying into two pions
    parent_rest = FourVector(1000.0, 0.0, 0.0, 0.0)  # ~1 GeV mass
    m_pi = 139.57  # charged pion mass (MeV)

    test_decay(parent_rest, m_pi, m_pi)

    # Example: Boosted parent (moving along z)
    parent_boosted = FourVector(2000.0, 0.0, 0.0, 1000.0)  # E=2000, pz=1000
    m_mu = 105.66  # muon
    m_nu = 0.0     # neutrino approx massless

    test_decay(parent_boosted, m_mu, m_nu)

    print("All tests done.")
