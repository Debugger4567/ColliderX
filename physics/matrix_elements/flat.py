from .base import MatrixElement

class FlatMatrixElement(MatrixElement):
    """Uniform phase space (no dynamics)."""
    
    name = "Flat Phase Space"
    description = "Returns |M|^2 = 1.0 for all configurations"

    def M2(self, parent_p4: tuple, daughter_p4s: list[tuple], context=None) -> float:
        """Always returns 1.0 (uniform phase space)."""
        return 1.0