import sqlite3
from pathlib import Path
from .kinematics import FourVector

# Path to the colliderx.db file
DB_PATH = Path(__file__).resolve().parents[1] / "colliderx.db"


class Particle:
    """
    Particle interface that integrates with the colliderx.db PDG-style schema.
    Includes in-memory caching for fast lookups.
    """

    _cache = {}  # Cache particle data to avoid repeated DB reads

    def __init__(self, name_or_symbol: str, px=0.0, py=0.0, pz=0.0):
        data = self.lookup_particle(name_or_symbol)
        self.name = data["Name"]
        self.symbol = data.get("Symbol", "")
        self.mass = data.get("Mass (MeV/c^2)", 0.0)
        self.charge = data.get("Charge (e)", 0.0)
        self.spin = data.get("Spin", 0.0)
        self.quantum_numbers = {
            "Baryon Number": data.get("Baryon Number", 0),
            "Le": data.get("Le", 0),
            "Lmu": data.get("Lmu", 0),
            "Ltau": data.get("Ltau", 0),
            "Strangeness": data.get("Strangeness", 0),
            "Charm": data.get("Charm", 0),
            "Bottomness": data.get("Bottomness", 0),
            "Topness": data.get("Topness", 0),
        }

        self.fourvec = self.make_fourvector(px, py, pz)

    # -------------------- Database Lookup --------------------

    @classmethod
    def lookup_particle(cls, name_or_symbol: str) -> dict:
        """Fetch all particle info from cache or DB by name or symbol."""
        key = name_or_symbol.lower()

        # Use cache if available
        if key in cls._cache:
            return cls._cache[key]

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Search by Name or Symbol, case-insensitive
        cur.execute("""
            SELECT * FROM particles
            WHERE LOWER("Name") = LOWER(?) OR LOWER("Symbol") = LOWER(?)
        """, (name_or_symbol, name_or_symbol))

        row = cur.fetchone()
        conn.close()

        if not row:
            raise ValueError(f"❌ Particle '{name_or_symbol}' not found in database at {DB_PATH}")

        data = dict(row)
        cls._cache[key] = data  # Store in cache
        return data

    # -------------------- Physics Methods --------------------

    def make_fourvector(self, px, py, pz) -> FourVector:
        """Construct a FourVector using rest mass and momentum components."""
        E = (self.mass**2 + px**2 + py**2 + pz**2) ** 0.5
        return FourVector(E, px, py, pz)

    # -------------------- Representation --------------------

    def __repr__(self):
        qnums = ", ".join(f"{k}={v}" for k, v in self.quantum_numbers.items() if v != 0)
        return (
            f"Particle(name={self.name}, symbol={self.symbol}, mass={self.mass:.2f} MeV/c², "
            f"charge={self.charge:+.0f}e, spin={self.spin}, {qnums}, fv={self.fourvec})"
        )
