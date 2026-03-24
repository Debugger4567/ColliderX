import numpy as np
import matplotlib.pyplot as plt
import psycopg2

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

    # Get mu- momenta only (define θ using ℓ⁻)
    cur.execute("""
        SELECT fs.px, fs.py, fs.pz, COALESCE(e.event_weight, e.weight, 1.0)
        FROM final_states fs
        JOIN events e ON fs.event_id = e.id
        WHERE e.parent = 'Z0'
            AND e.decay_mode = 'μ+ μ−'
            AND fs.particle = 'Muon'
    """)
    
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if len(rows) == 0:
        raise RuntimeError("No events found")
    
    cos_theta = []
    weights = []
    for px, py, pz, w in rows:
        p_mag = np.sqrt(px**2 + py**2 + pz**2)
        if p_mag > 0:
            cos_theta.append(np.clip(pz / p_mag, -1.0, 1.0))
            weights.append(w)
    
    cos_theta = np.array(cos_theta)
    weights = np.array(weights)

    print(f"[INFO] Loaded {len(cos_theta)} cosθ values")

    plt.figure(figsize=(7,5))
    plt.hist(
        cos_theta, 
        bins=60, 
        range=(-1, 1),
        weights=weights,     # NEW: use event weights
        density=True, 
        alpha=0.8,
        label="Simulation"
    )

    #Analytical expectation(normalized by hand later)
    x = np.linspace(-1, 1, 400)
    y= 1 + x**2
    y /= np.trapezoid(y, x)

    plt.plot(x, y, "r--", lw=2, label = r"$1 + \cos^2\theta$")
    plt.xlabel(r"$\cos\theta$")
    plt.ylabel("Normalized events")
    plt.title(r"$Z \rightarrow \mu^+ \mu^-$ angular distribution")
    plt.legend()
    plt.grid(alpha = 0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


