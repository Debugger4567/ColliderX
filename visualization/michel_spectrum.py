import numpy as np
import matplotlib.pyplot as plt
from db import get_conn

M_MU = 105.66  # Muon mass in MeV


def get_db_connection():
    return get_conn()


def load_electron_energies():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT fs.E
        FROM events e
        JOIN final_states fs ON fs.event_id = e.id
        WHERE e.parent = 'Muon' AND fs.particle = 'Electron'
    """)
    rows = cur.fetchall()
    conn.close()
    return np.array([r[0] for r in rows], dtype=float)


def michel_pdf(x):
    return x**2 * (3 - 2 * x)


def main():
    E = load_electron_energies()  # call the function
    x = 2 * E / M_MU
    x = x[(x > 0) & (x < 1)]

    plt.figure(figsize=(7, 5))
    plt.hist(x, bins=60, density=True, alpha=0.8, label="ColliderX (V-A)")

    # Overlay Michel spectrum (normalized)
    x_grid = np.linspace(0, 1, 200)
    y = michel_pdf(x_grid)
    # normalize using trapezoidal rule (np.trapz is deprecated)
    integral = np.trapezoid(y, x_grid)
    if integral == 0 or not np.isfinite(integral):
        y = np.zeros_like(y)
    else:
        y /= integral
    plt.plot(x_grid, y, "r--", label="Michel spectrum")

    plt.xlabel(r"$x = 2E_e/m_\mu$")
    plt.ylabel("Normalized counts / PDF")
    plt.title(r"Michel Spectrum: $\mu^- \to e^- \bar{\nu}_e \nu_\mu$")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
