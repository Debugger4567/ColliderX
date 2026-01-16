import psycopg2
import numpy as np
import matplotlib.pyplot as plt

M_MU = 105.66
N = 200_000

def get_conn():
    return psycopg2.connect(
        dbname="colliderx",
        user="postgres",
        password="Soccer@21",
        host="localhost",
        port=5432,
    )

def load_real():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT fs.E
        FROM events e
        JOIN final_states fs ON fs.event_id = e.id
        WHERE e.parent = 'Muon'
        AND fs.particle = 'Electron'
    """)
    rows = cur.fetchall()
    conn.close()
    return np.array([r[0] for r in rows])

def generate_flat():
    rng = np.random.default_rng(0)
    E = rng.random((N,3))
    E /= E.sum(axis=1)[:,None]
    return E[:,0] * (M_MU / 2)

def main():
    E_real = load_real()
    x_real = 2 * E_real / M_MU
    x_real = x_real[(x_real > 0) & (x_real < 1)]

    E_flat = generate_flat()
    x_flat = 2 * E_flat / M_MU

    plt.figure(figsize=(7,5))
    plt.hist(x_flat, bins=60, density=True, alpha=0.6, label="Flat phase space")
    plt.hist(x_real, bins=60, density=True, histtype="step", linewidth=2,
             label="ColliderX (V–A)")
    plt.xlabel(r"$x = 2E_e/m_\mu$")
    plt.ylabel("Normalized counts")
    plt.title("Michel spectrum: dynamics vs kinematics")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()