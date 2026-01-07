from abc import ABC, abstractmethod

class MatrixElement(ABC):
    """
    Base class for all matrix elements.

    Computes |M|^2 for a given decay configuration.
    All implementations must be pure functions (no RNG, no DB access).
    """

    name: str = "abstract"
    description: str = ""

    @abstractmethod
    def M2(
        self, 
        parent_p4: tuple,
        daughter_p4s: list[tuple],
        context: dict | None = None
    ) -> float:
        """
        Return |M|^2 for the given momenta.

        Args:
            parent_p4: 4-vector (E, px, py, pz) of parent
            daughter_p4s: list of daughter 4-vectors (E, px, py, pz)
            context: optional physics parameters (polarization, couplings, etc)

        Returns:
            Non-negative float proportional to |M|^2
        """