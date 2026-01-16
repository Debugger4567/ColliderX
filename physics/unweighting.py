import numpy as np

class UnweightingController:
    def __init__(self, w_max: float, safety_factor: float = 1.2):
        self.w_max = w_max * safety_factor
        self.accepted = 0
        self.rejected = 0

    def accept(self, weight: float, rng: np.random.Generator) -> bool:
        r = rng.uniform(0.0, self.w_max)
        if r < weight:
            self.accepted += 1
            return True
        else:
            self.rejected += 1
            return False

    @property
    def efficiency(self) -> float:
        total = self.accepted + self.rejected
        return self.accepted / total if total > 0 else 0.0
