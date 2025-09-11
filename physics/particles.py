import sqlite3

class Particles:
    def __init__(self, db_path="colliderx.db"):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def list_particles(self):
        """Return all particles with key properties."""
        query = """
        SELECT Name, Symbol, "Mass (MeV/c^2)", "Charge (e)", Spin, "PDG ID"
        FROM particles
        """
        return self.cursor.execute(query).fetchall()

    def get_particle(self, pdg_id):
        """Return a single particle by PDG ID."""
        query = """
        SELECT * FROM particles WHERE "PDG ID" = ?
        """
        return self.cursor.execute(query, (pdg_id,)).fetchone()
