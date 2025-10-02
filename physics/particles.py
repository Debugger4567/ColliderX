import sqlite3
from pathlib import Path
from .kinematics import FourVector

DB_PATH = Path(__file__).resolve().parents[1] / "colliderx.db"

class Particle:
    """
    Minimal Particle interface that integrates with SQLite DB.
    Provides mass lookup and a constructor for FourVectors.
    """

    def __init__(self, name: str, px=0.0, py=0.0, pz=0.0):
        self.name = name
        self.mass = self.lookup_mass(name)
        self.fourvec = self.make_fourvector(px, py, pz)

    @staticmethod
    def lookup_mass(name: str) -> float:
        """Fetch particle mass (MeV) from SQLite DB."""
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT mass FROM particles WHERE name = ?", (name,))
        row = cur.fetchone()
        conn.close()
        if not row:
            raise ValueError(f"Particle '{name}' not found in DB")
        return float(row[0])

    def make_fourvector(self, px, py, pz) -> FourVector:
        """Construct a FourVector from mass + momenta."""
        E = (self.mass**2 + px**2 + py**2 + pz**2) ** 0.5
        return FourVector(E, px, py, pz)

    def __repr__(self):
        return f"Particle(name={self.name}, mass={self.mass:.2f}, fv={self.fourvec})"
