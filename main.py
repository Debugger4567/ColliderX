# main.py
from physics.collision import MonteCarloCollider
from physics.particles import Particles
from physics.decays import Decays

'''
def main():
    collider = MonteCarloCollider()

    # Example: proton-proton collision at 14 TeV
    stable_products = collider.collide("p", "p", energy=14_000_000)

    print("\n=== Final Detector Output ===")
    for p in stable_products:
        print(f"- {p[0]} ({p[1]})")


if __name__ == "__main__":
    main()
'''


'''
collider = MonteCarloCollider(default_energy=8_000_000)

# Run 10 million collisions (FAST with NumPy)
results = collider.collide("p", "p", n=10_000_000, verbose=True)

# Filter just Higgs events
higgs_hits = results[results == "higgs"]
print(f"Higgs events found: {len(higgs_hits)}")

'''

from physics.kinematics import FourVector

p1 = FourVector(7000, 0, 0, 7000)   # ~massless particle moving +z
p2 = FourVector(7000, 0, 0, -7000)  # ~massless particle moving -z

total = p1 + p2
print("Total:", total)
print("Invariant mass √s =", total.mass())