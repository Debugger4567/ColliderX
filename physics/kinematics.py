# kinematics.py
import math

class FourVector:
    def __init__(self, E, px, py, pz):
        """
        Four-vector representation of a particle.
        :param E: energy (MeV)
        :param px, py, pz: momentum components (MeV/c)
        """
        self.E = float(E)
        self.px = float(px)
        self.py = float(py)
        self.pz = float(pz)

    def __repr__(self):
        return f"FourVector(E={self.E:.2f}, px={self.px:.2f}, py={self.py:.2f}, pz={self.pz:.2f})"

    # --- Magnitudes ---
    def momentum(self):
        """Return |p| (3-momentum magnitude)."""
        return math.sqrt(self.px**2 + self.py**2 + self.pz**2)

    def mass(self):
        """Invariant mass (m^2 = E^2 - |p|^2)."""
        m2 = self.E**2 - self.momentum()**2
        return math.sqrt(m2) if m2 > 0 else 0.0

    # --- Operations ---
    def __add__(self, other):
        return FourVector(
            self.E + other.E,
            self.px + other.px,
            self.py + other.py,
            self.pz + other.pz,
        )

    def __sub__(self, other):
        return FourVector(
            self.E - other.E,
            self.px - other.px,
            self.py - other.py,
            self.pz - other.pz,
        )

    def dot(self, other):
        """Minkowski dot product: E1E2 - p1·p2"""
        return self.E * other.E - (
            self.px * other.px + self.py * other.py + self.pz * other.pz
        )

    def boost(self, bx, by, bz):
        """
        Apply a Lorentz boost (for later, not needed in Phase 1).
        (bx, by, bz) = velocity vector (v/c).
        """
        b2 = bx**2 + by**2 + bz**2
        gamma = 1.0 / math.sqrt(1 - b2)
        bp = bx*self.px + by*self.py + bz*self.pz
        gamma2 = (gamma - 1.0) / b2 if b2 > 0 else 0.0

        px = self.px + gamma2*bp*bx + gamma*bx*self.E
        py = self.py + gamma2*bp*by + gamma*by*self.E
        pz = self.pz + gamma2*bp*bz + gamma*bz*self.E
        E  = gamma*(self.E + bp)

        return FourVector(E, px, py, pz)
