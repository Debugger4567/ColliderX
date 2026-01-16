import psycopg2
import numpy as np
import matplotlib.pyplot as plt

# ---------- DB CONNECTION ----------
def get_conn():
    return psycopg2.connect(
        dbname="colliderx",
        user="postgres",
        password="Soccer@21",
        host="localhost",
        port=5432,
    )

# ---------- LOAD EVENTS ----------
def load_muon_events():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            e.id,
            fs.particle,
            fs.E,
            fs.px,
            fs.py,
            fs.pz,
            e.event_weight
        FROM events e
        JOIN final_states fs ON fs.event_id = e.id
        WHERE e.parent = 'Muon'
        ORDER BY e.id
    """)

    rows = cur.fetchall()
    conn.close()

    # Group by event
    events = {}
    weights = {}
    
    for event_id, particle, E, px, py, pz, weight in rows:
        if event_id not in events:
            events[event_id] = {}
        events[event_id][particle] = (E, px, py, pz)
        weights[event_id] = weight

    return events, weights

# ---------- MAIN ----------
def main():
    events, weights = load_muon_events()

    # Debug: show actual particle names
    all_particles = set()
    for event_particles in events.values():
        all_particles.update(event_particles.keys())
    print(f"[DEBUG] Particle names in DB: {sorted(all_particles)}")

    x_e = []
    x_nubar = []
    W = []
    m_mu = 105.66  # MeV

    used = 0
    skipped = 0

    required_particles = ["Electron", "Muon neutrino", "Electron antineutrino"]

    for event_id, particles in events.items():
        if not all(p in particles for p in required_particles):
            skipped += 1
            continue

        E_e = particles["Electron"][0]
        E_nubar = particles["Electron antineutrino"][0]
        
        xe = 2 * E_e / m_mu
        xn = 2 * E_nubar / m_mu
        
        x_e.append(xe)
        x_nubar.append(xn)
        W.append(weights[event_id])
        used += 1

    print(f"[Dalitz] Used events: {used}")
    print(f"[Dalitz] Skipped events: {skipped}")

    if used == 0:
        print("\n[ERROR] No complete 3-body events found!")
        print("Did you regenerate with --store-neutrinos?")
        return

    x_e = np.array(x_e)
    x_nubar = np.array(x_nubar)
    W = np.array(W)

    # ---------- PLOT ----------
    plt.figure(figsize=(8, 7))
    bins = 60
    h = plt.hist2d(x_e, x_nubar, bins=bins, weights=W, cmap='viridis', range=[[0, 1], [0, 1]])

    plt.xlabel(r"$x_e = 2E_e / m_\mu$", fontsize=12)
    plt.ylabel(r"$x_{\bar{\nu}_e} = 2E_{\bar{\nu}_e} / m_\mu$", fontsize=12)
    plt.title(r"Dalitz plot: $\mu^- \to e^- \bar{\nu}_e \nu_\mu$", fontsize=14)
    plt.colorbar(h[3], label="Event weight")

    # Kinematic boundary inside [0,1]^2: x_e + x_nubar = 1 (x_{νμ} = 1)
    x_boundary = np.linspace(0, 1, 200)
    y_boundary = 1 - x_boundary
    plt.plot(x_boundary, y_boundary, 'r--', linewidth=2, label=r'$x_e + x_{\bar{\nu}_e} = 1$')

    plt.legend(fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()