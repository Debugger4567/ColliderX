# main.py
from physics.collsion import MonteCarloCollider
from physics.particles import Particles
from physics.decays import Decays


if __name__ == "__main__":
    sim = MonteCarloCollider()

    # Simulate proton-proton collision at 8 TeV (LHC scale)
    sim.collide("p", "p", energy=8000000)
'''  

def main():
    particle_db = Particles()
    decay_db = Decays()

    # Example: pion (PDG ID = 211)
    pdg_id = 211
    particle = particle_db.get_particle(pdg_id)

    if particle:
        print(f"=== Particle: {particle[0]} ({particle[1]}) ===")
        print(f"Type: {particle[2]}")
        print(f"Mass: {particle[3]} MeV/c²")
        print(f"Charge: {particle[4]} e")
        print(f"Spin: {particle[5]}")
        print()

        # Decays
        decays = decay_db.get_decay(pdg_id)
        if decays:
            print("--- Decays ---")
            for mode, fraction in decays:
                percent = round(fraction * 100, 4) if fraction is not None else None
                if percent is not None:
                    print(f"{mode} ({percent}%)")
                else:
                    print(mode)
        else:
            print("No decay modes found.")

    else:
        print(f"No particle found for PDG ID {pdg_id}")

if __name__ == "__main__":
    main()
'''