from.base import MatrixElement

class ScalarTwoBodyMatrixElement(MatrixElement):
    """
    Scalar → 2-body decay
    Examples:
        π0 → γ γ
        H → γ γ (toy)
    """
    name = "Scalar → 2-body"
    description = "Constant |M|^2 for scalar two-body decays"

    def __init__(self, coupling: float = 1.0):
        self.coupling = coupling

    def M2(self, parent_p4, daughter_p4s, context=None) -> float:
        return self.coupling ** 2
    
    