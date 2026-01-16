import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from physics.particles import Particle
from physics.decay_selector import get_decay_products
from physics.event_generator import generate_decay_kinematics



def main():
    # Parent particle
    pi0 = Particle("Pion0")

    # Authoritative daughters from DB
    daughters = get_decay_products(111, "γ γ")

    print("Decay mode: π0 →", daughters)

    # Generate kinematics
    final_particles = generate_decay_kinematics(pi0, daughters)

    p1, p2 = final_particles

    # Invariant mass check
    inv_mass = (p1.fourvec + p2.fourvec).mass

    print("Daughter 1:", p1)
    print("Daughter 2:", p2)
    print(f"Invariant mass = {inv_mass:.4f} MeV")
    print(f"Expected mass  = {pi0.mass:.4f} MeV")

if __name__ == "__main__":
    main()


