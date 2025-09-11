import sqlite3

class Decays:
    def __init__(self, db_path="colliderx.db"):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def get_decay(self, pdg_id):
        """Return decay modes for a particle by PDG ID."""
        query = "SELECT decay_mode, branching_fraction FROM decays WHERE pdg_id = ?"
        return self.cursor.execute(query, (pdg_id,)).fetchall()
