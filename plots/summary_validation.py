"""
Summary validation plot — all physics in one figure.
Panels:
1) Michel spectrum
2) Z invariant mass peak
3) Z cosθ distribution
4) Tau pion asymmetry
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from db import get_conn


def main():
    conn = get_conn()
    cur = conn.cursor()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ColliderX — Physics Validation Summary", fontsize=15, fontweight="500", y=1.01)

    # Panel 1: Michel spectrum
    cur.execute(
        """
        SELECT fs.E
        FROM final_states fs
        JOIN events e ON fs.event_id = e.id
        WHERE e.parent = 'Muon' AND fs.particle = 'Electron'
        LIMIT 50000
        """
    )
    E_e = np.array([r[0] for r in cur.fetchall()], dtype=float)
    x_e = 2 * E_e / 105.658
    x_e = x_e[(x_e > 0) & (x_e < 1)]

    axes[0, 0].hist(x_e, bins=60, density=True, alpha=0.8, color="#e87070", label="ColliderX")
    x_grid = np.linspace(0, 1, 200)
    michel = x_grid**2 * (3 - 2 * x_grid)
    michel /= np.trapezoid(michel, x_grid)
    axes[0, 0].plot(x_grid, michel, "k--", lw=2, label=r"$x^2(3-2x)$")
    axes[0, 0].set_xlabel(r"$x = 2E_e/m_\mu$")
    axes[0, 0].set_ylabel("Normalized")
    axes[0, 0].set_title(r"$\mu^- \to e^- \bar\nu_e \nu_\mu$ Michel spectrum")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Panel 2: Z invariant mass
    cur.execute(
        """
        SELECT e.id, fs.px, fs.py, fs.pz, fs.E
        FROM final_states fs
        JOIN events e ON fs.event_id = e.id
        WHERE e.parent = 'Z boson'
        ORDER BY e.id
        """
    )
    rows = cur.fetchall()
    events_z = {}
    for eid, px, py, pz, E in rows:
        events_z.setdefault(eid, []).append(np.array([E, px, py, pz], dtype=float))

    masses = []
    for ev in events_z.values():
        if len(ev) == 2:
            p = ev[0] + ev[1]
            m2 = p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2
            if m2 > 0:
                masses.append(np.sqrt(m2) / 1000.0)  # MeV -> GeV
    masses = np.array(masses, dtype=float)

    if len(masses) > 0:
        hist, edges, _ = axes[0, 1].hist(masses, bins=80, range=(85, 97), alpha=0.8, color="#4ecba0", label="ColliderX")
        centres = 0.5 * (edges[:-1] + edges[1:])

        def bw(m, M, G, A):
            return A * (m**2 * G) / ((m**2 - M**2) ** 2 + (m * G) ** 2)

        try:
            mask = (centres > 88) & (centres < 95)
            popt, _ = curve_fit(bw, centres[mask], hist[mask], p0=[91.2, 2.5, max(hist.max(), 1e-9)])
            x_fit = np.linspace(85, 97, 500)
            axes[0, 1].plot(x_fit, bw(x_fit, *popt), "r--", lw=2, label=f"BW fit: M={popt[0]:.2f} GeV")
        except Exception:
            pass

    axes[0, 1].set_xlabel(r"$m_{\ell\ell}$ [GeV]")
    axes[0, 1].set_ylabel("Events")
    axes[0, 1].set_title(r"$Z \to \mu^+\mu^-$ invariant mass")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Panel 3: Z cosθ
    cur.execute(
        """
        SELECT fs.px, fs.py, fs.pz
        FROM final_states fs
        JOIN events e ON fs.event_id = e.id
        WHERE e.parent = 'Z boson'
          AND e.decay_mode = 'μ+ μ−'
          AND fs.particle = 'Muon'
        """
    )
    rows = cur.fetchall()
    cos_z = []
    for px, py, pz in rows:
        pmag = np.sqrt(px**2 + py**2 + pz**2)
        if pmag > 0:
            cos_z.append(pz / pmag)
    cos_z = np.array(cos_z, dtype=float)

    if len(cos_z) > 0:
        axes[1, 0].hist(cos_z, bins=50, range=(-1, 1), density=True, alpha=0.8, color="#7b8cf7", label="ColliderX")
        x = np.linspace(-1, 1, 200)
        y = 1 + x**2
        y /= np.trapezoid(y, x)
        axes[1, 0].plot(x, y, "k--", lw=2, label=r"$1+\cos^2\theta$")

    axes[1, 0].set_xlabel(r"$\cos\theta$")
    axes[1, 0].set_ylabel("Normalized")
    axes[1, 0].set_title(r"$Z \to \mu^+\mu^-$ angular distribution")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Panel 4: Tau pion asymmetry
    cur.execute(
        """
        SELECT fs.pz, fs.px, fs.py
        FROM final_states fs
        JOIN events e ON fs.event_id = e.id
        WHERE e.parent = 'Z boson'
          AND e.decay_mode IN ('τ+ τ−', 'tau+ tau-')
          AND fs.particle IN ('Pion-', 'Pion+')
        """
    )
    rows = cur.fetchall()
    cos_tau = []
    for pz, px, py in rows:
        pmag = np.sqrt(px**2 + py**2 + pz**2)
        if pmag > 0:
            cos_tau.append(pz / pmag)
    cos_tau = np.array(cos_tau, dtype=float)

    if len(cos_tau) > 0:
        axes[1, 1].hist(cos_tau, bins=50, range=(-1, 1), density=True, alpha=0.8, color="#a78bfa", label="ColliderX")
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            'Run simulate_chain("Z boson", fixed_decay_mode="τ+ τ−")\n(or "tau+ tau-") to populate this panel',
            transform=axes[1, 1].transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="gray",
        )

    axes[1, 1].set_xlabel(r"$\cos\theta_\pi$")
    axes[1, 1].set_ylabel("Normalized")
    axes[1, 1].set_title(r"$\tau$ spin correlation: pion angular distribution")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    cur.close()
    conn.close()

    plt.tight_layout()
    plt.savefig("plots/summary_validation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: plots/summary_validation.png")


if __name__ == "__main__":
    main()