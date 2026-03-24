import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from scipy.optimize import curve_fit

def get_conn():
    return psycopg2.connect(
        dbname="colliderx",
        user="postgres",
        password="Soccer@21",
        host="localhost",
        port=5432,
    )

def main():
    conn = get_conn()
    cur = conn.cursor()

    # Fetch both daughters per event (no fragile name filters)
    cur.execute("""
        SELECT
            e.id,
            fs.px,
            fs.py,
            fs.pz,
            fs.e AS e_fs,
            COALESCE(e.event_weight, e.weight, 1.0) AS event_weight
        FROM final_states fs
        JOIN events e ON fs.event_id = e.id
        WHERE e.parent = 'Z0'
        ORDER BY e.id
    """)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    # Group final states by event id
    events = {}
    for eid, px, py, pz, E, w in rows:
        events.setdefault(eid, {"p4": [], "w": w})
        events[eid]["p4"].append(np.array([E, px, py, pz], dtype=float))

    masses = []
    weights = []
    for evt in events.values():
        if len(evt["p4"]) != 2:
            continue
        p_tot = evt["p4"][0] + evt["p4"][1]
        w_evt = evt["w"]
        E, px, py, pz = p_tot
        m2 = E*E - px*px - py*py - pz*pz
        if m2 > 0:
            masses.append(np.sqrt(m2) / 1000.0)  # MeV → GeV
            weights.append(float(w_evt))

    masses = np.array(masses)
    weights = np.array(weights)

    print(f"[INFO] Loaded {len(masses)} invariant masses")
    if len(masses) == 0:
        print("[WARN] No events found for Z0 → μ+ μ−")
        return

    # Basic stats
    mean_m = np.average(masses, weights=weights)
    min_m, max_m = masses.min(), masses.max()
    print(f"[INFO] Mass range: {min_m:.2f} – {max_m:.2f} GeV")
    print(f"[INFO] Mean mass:  {mean_m:.2f} GeV")

    plt.figure(figsize=(7, 5))
    hist, edges, _ = plt.hist(
        masses,
        bins=80,
        range=(65, 115),
        weights=weights,
        density=False,     # Match fit units
        alpha=0.8,
        label="Simulation"
    )

    def breit_wigner(m, M, Gamma, A):
        # CORRECT formula: A × (m² Γ) / [(m² - M²)² + (m Γ)²]
        return A * (m*m * Gamma) / ((m*m - M*M)**2 + (m*Gamma)**2)

    centers = 0.5 * (edges[:-1] + edges[1:])

    # Initial guesses
    p0 = [91.2, 2.5, hist.max()]

    try:
        # Fit only in the peak region (80-100 GeV)
        mask = (centers > 80) & (centers < 100)
        popt, pcov = curve_fit(
            breit_wigner,
            centers[mask],
            hist[mask],
            p0=p0,
            bounds=([80, 0.0, 0], [100, 10.0, np.inf])  # Constrain Γ ≥ 0
        )
        M_fit, Gamma_fit, A_fit = popt

        print(f"[FIT] M_Z     = {M_fit:.3f} GeV")
        print(f"[FIT] Γ_Z     = {Gamma_fit:.3f} GeV")

        # Plot fit
        x = np.linspace(65, 115, 800)
        plt.plot(x, breit_wigner(x, *popt),
                 "r--", lw=2, label="Breit–Wigner fit")
    except RuntimeError as e:
        print(f"[WARN] Fit failed: {e}")

    plt.xlabel(r"Invariant mass $m_{\mu\mu}$ [GeV]")
    plt.ylabel("Events")
    plt.title(r"$Z \rightarrow \mu^+ \mu^-$ invariant mass")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()






