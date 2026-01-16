from physics.event_generator import simulate_event, estimate_w_max
from physics.unweighting import UnweightingController
import numpy as np

rng = np.random.default_rng(42)

wmax = estimate_w_max("Pion0", n_trials=2000, rng=rng)
uw = UnweightingController(wmax)

accepted = 0
for _ in range(5000):
    eid = simulate_event(
        "Pion0",
        rng=rng,
        use_matrix_element=True,
        unweighting_controller=uw
    )
    if eid:
        accepted += 1

print("Accepted:", accepted)
print("Efficiency:", uw.efficiency)