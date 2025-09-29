import sqlite3
import random

class MonteCarloCollider:
    def __init__(self, db_path="colliderx.db"):
        self.conn = sqlite3.connect(db_path)

    def get_particle(self, name_or_symbol):
        cur = self.conn.cursor()
        cur.execute("""
            SELECT Name, Symbol, `Mass (MeV/c^2)`, `Charge (e)`, Spin, `PDG ID`
            FROM particles
            WHERE Name = ? OR Symbol = ?
        """, (name_or_symbol, name_or_symbol))
        return cur.fetchone()

    def get_decays(self, pdg_id):
        cur = self.conn.cursor()
        cur.execute("""
            SELECT decay_mode, branching_fraction
            FROM decays
            WHERE pdg_id = ?
        """, (pdg_id,))
        return cur.fetchall()

    def is_stable(self, pdg_id):
        return len(self.get_decays(pdg_id)) == 0

    def simulate_decay(self, particle, depth=0):
        """Simulate decay recursively and print decay tree."""
        name, sym, mass, charge, spin, pdg_id = particle
        decays = self.get_decays(pdg_id)

        indent = "  " * depth

        if not decays:  # stable
            print(f"{indent}↳ {name} ({sym}) [stable]")
            return [particle]

        # Weighted random decay
        products_str, _ = random.choices(decays, weights=[d[1] for d in decays], k=1)[0]
        product_names = products_str.split()

        print(f"{indent}↳ {name} ({sym}) decays into {', '.join(product_names)}")

        products = []
        for pname in product_names:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT Name, Symbol, `Mass (MeV/c^2)`, `Charge (e)`, Spin, `PDG ID`
                FROM particles
                WHERE Name = ? OR Symbol = ?
            """, (pname, pname))
            res = cur.fetchone()
            if res:
                products.extend(self.simulate_decay(res, depth+1))
        return products

    def collide(self, particle1, particle2, energy):
        """Run a full collision event with decay chains until all stable."""
        p1 = self.get_particle(particle1)
        p2 = self.get_particle(particle2)

        if not p1 or not p2:
            return "Error: One or both particles not found."

        name1, sym1, m1, _, _, _ = p1
        name2, sym2, m2, _, _, _ = p2

        print(f"\n=== Collision Event ===")
        print(f"{name1} ({sym1}) + {name2} ({sym2})")
        print(f"√ Center-of-mass energy: {energy:.2f} MeV")
        print(f"√ Total rest mass: {m1+m2:.2f} MeV")

        # Event starts with the two colliding particles
        event_particles = [p1, p2]

        # If energy is high enough, maybe create a random heavy particle
        if energy > 1.2e5:  # ~120 GeV, enough for Higgs
            cur = self.conn.cursor()
            cur.execute("""
                SELECT Name, Symbol, `Mass (MeV/c^2)`, `Charge (e)`, Spin, `PDG ID`
                FROM particles
                WHERE `Mass (MeV/c^2)` <= ?
            """, (energy,))
            candidates = cur.fetchall()
            if candidates:
                newp = random.choice(candidates)
                event_particles.append(newp)
                print(f"✨ New particle created: {newp[0]} ({newp[1]})")

        # Run decay chains
        stable_particles = []
        for particle in event_particles:
            stable_particles.extend(self.simulate_decay(particle, depth=1))

        # Final detector output
        print("\n=== Final Stable Particles (Detector Output) ===")
        for sp in stable_particles:
            print(f"- {sp[0]} ({sp[1]})")

        return stable_particles
    


'''
import numpy as np

class MonteCarloCollider:
    def __init__(self, default_energy=8_000_000):
        """
        Monte Carlo Proton-Proton Collider
        :param default_energy: Default center-of-mass collision energy (MeV).
        """
        self.default_energy = default_energy

        # Very rough/fictional probabilities for demo
        self.particle_probs = {
            "higgs": 1e-7,      # Higgs boson is extremely rare
            "z_boson": 1e-5,
            "w_boson": 1e-5,
            "top_quark": 1e-6,
            "muon": 1e-3,
            "pion": 1e-2,
            "proton": 0.1,
            "neutron": 0.1,
            "photon": 0.2,
            "gluon": 0.2,
        }

        # Normalize probabilities to sum = 1
        total = sum(self.particle_probs.values())
        for k in self.particle_probs:
            self.particle_probs[k] /= total

        self.particles = np.array(list(self.particle_probs.keys()))
        self.probabilities = np.array(list(self.particle_probs.values()))

    def collide(self, p1="p", p2="p", energy=None, n=1, verbose=True):
        """
        Simulate n proton-proton collisions using vectorized NumPy sampling.
        :param p1: first particle (unused, kept for API consistency)
        :param p2: second particle (unused, kept for API consistency)
        :param energy: collision energy in MeV
        :param n: number of collisions
        :param verbose: if True, prints results
        :return: list of produced particles (NumPy array)
        """
        if energy is None:
            energy = self.default_energy

        # Draw all collisions at once
        results = np.random.choice(self.particles, size=n, p=self.probabilities)

        if verbose:
            unique, counts = np.unique(results, return_counts=True)
            summary = dict(zip(unique, counts))
            print(f"Simulated {n:,} collisions at {energy:,} MeV")
            print("Event summary:", summary)

        return results
'''