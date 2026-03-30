import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from physics.particles import Particle
from physics.decay_selector import get_decay_products
from physics.decays import decay_particle
import numpy as np


def main():
    # Parent particle
    pi0 = Particle("Pion0")

    # Authoritative daughters from DB
    daughters = get_decay_products(111, "γ γ")

    print("Decay mode: π0 →", daughters)

    # Generate kinematics directly from decay kernel
    daughter_particles = [Particle(name) for name in daughters]
    daughter_masses = [particle.mass for particle in daughter_particles]
    p4s = decay_particle(pi0.mass, daughter_masses, rng=np.random.default_rng())

    for particle, four_vector in zip(daughter_particles, p4s):
        particle.fourvec = four_vector

    p1, p2 = daughter_particles

    # Invariant mass check
    inv_mass = (p1.fourvec + p2.fourvec).mass

    print("Daughter 1:", p1)
    print("Daughter 2:", p2)
    print(f"Invariant mass = {inv_mass:.4f} MeV")
    print(f"Expected mass  = {pi0.mass:.4f} MeV")


if __name__ == "__main__":
    main()
