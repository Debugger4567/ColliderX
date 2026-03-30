import numpy as np
import matplotlib.pyplot as plt
from db import get_conn

conn = get_conn()
cur = conn.cursor()

cur.execute("""
SELECT fs.E, COALESCE(ev.event_weight, ev.weight, 1.0) AS event_weight
FROM final_states fs
JOIN events ev ON fs.event_id = ev.id
WHERE fs.particle = 'Electron';
""")

data = cur.fetchall()
E = np.array([row[0] for row in data])
W = np.array([row[1] for row in data])

# Histogram
bins = 100  # smoother
hist, edges = np.histogram(E, bins=bins, weights=W, density=True)
centres = 0.5 * (edges[:-1] + edges[1:])

plt.plot(centres, hist, drawstyle="steps-mid", alpha=0.7, label="V-A MC")

# Add Michel spectrum (analytic)
m_mu = 105.658
Emax = m_mu / 2

x = centres / Emax
michel = x**2 * (3 - 2 * x)
michel[x > 1] = 0
# Normalize the analytic Michel curve over the x grid.
integral = np.trapezoid(michel, x)
if integral == 0 or not np.isfinite(integral):
    michel = np.zeros_like(michel)
else:
    michel /= integral

plt.plot(centres, michel, label="Michel spectrum", linestyle="--")

plt.xlabel("Electron energy (MeV)")
plt.ylabel("Normalized distribution")
plt.title("Muon decay: electron energy spectrum")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

cur.close()
conn.close()
