import sqlite3
import random

class MonteCarloCollider:
    def __init__(self, db_path="colliderx.db"):
        self.conn = sqlite3.connect(db_path)

    def get_particle(self, name_or_symbol):
        cur = self.conn.cursor()
        cur.execute("SELECT name, symbol, mass, charge, spin, pdg_id FROM particles WHERE name = ? OR symbol = ?", 
                    (name_or_symbol, name_or_symbol))
        return cur.fetchone()

    def get_decays(self, pdg_id):
        cur = self.conn.cursor()
        cur.execute("SELECT products, branching_ratio FROM decays WHERE parent_pdg_id = ?", (pdg_id,))
        return cur.fetchall()

    def is_stable(self, pdg_id):
        return len(self.get_decays(pdg_id)) == 0

    def simulate_decay(self, particle):
        """Simulate decay of a single particle (randomly by branching ratio)."""
        name, sym, mass, charge, spin, pdg_id = particle
        decays = self.get_decays(pdg_id)

        if not decays:  # stable
            return [particle]

        # Weighted random decay
        products_str, _ = random.choices(decays, weights=[d[1] for d in decays], k=1)[0]

        products = []
        for prod in products_str.split():
            cur = self.conn.cursor()
            cur.execute("SELECT name, symbol, mass, charge, spin, pdg_id FROM particles WHERE name=? OR symbol=?", (prod, prod))
            res = cur.fetchone()
            if res:
                products.append(res)

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
        if energy > 2000:  # arbitrary cutoff
            cur = self.conn.cursor()
            cur.execute("SELECT name, symbol, mass, charge, spin, pdg_id FROM particles WHERE mass <= ?", (energy,))
            candidates = cur.fetchall()
            if candidates:
                newp = random.choice(candidates)
                event_particles.append(newp)
                print(f"✨ New particle created: {newp[0]} ({newp[1]})")

        # Run decay chains
        stable_particles = []
        queue = event_particles

        while queue:
            current = queue.pop()
            if self.is_stable(current[5]):
                stable_particles.append(current)
            else:
                products = self.simulate_decay(current)
                print(f"⚡ {current[0]} decayed into: {', '.join([p[0] for p in products])}")
                queue.extend(products)

        # Final detector output
        print("\n=== Final Stable Particles ===")
        for sp in stable_particles:
            print(f"- {sp[0]} ({sp[1]})")

        return stable_particles
