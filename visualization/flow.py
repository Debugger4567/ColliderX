from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from db import get_conn
from physics.collision import simulate_chain, simulate_events
from .feynman import generate_diagram_files


def _fetch_run_statistics(
    run_timestamp: str, parent_name: str
) -> tuple[List[tuple], List[tuple], List[float]]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT decay_mode, COUNT(*) AS count
            FROM events
            WHERE parent = %s AND timestamp = %s
            GROUP BY decay_mode
            ORDER BY count DESC
            """,
            (parent_name, run_timestamp),
        )
        decay_mode_counts = cur.fetchall()

        cur.execute(
            """
            SELECT fs.particle, COUNT(*) AS count
            FROM final_states fs
            JOIN events e ON fs.event_id = e.id
            WHERE e.parent = %s AND e.timestamp = %s
            GROUP BY fs.particle
            ORDER BY count DESC
            """,
            (parent_name, run_timestamp),
        )
        particle_counts = cur.fetchall()

        cur.execute(
            """
            SELECT fs.E
            FROM final_states fs
            JOIN events e ON fs.event_id = e.id
            WHERE e.parent = %s AND e.timestamp = %s
            """,
            (parent_name, run_timestamp),
        )
        energies = [float(row[0]) for row in cur.fetchall() if row[0] is not None]

    return decay_mode_counts, particle_counts, energies


def _fetch_run_rows(run_timestamp: str, parent_name: str) -> List[tuple]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                e.id,
                e.decay_mode,
                fs.particle,
                fs.E,
                fs.px,
                fs.py,
                fs.pz,
                COALESCE(e.event_weight, e.weight, 1.0) AS event_weight
            FROM events e
            JOIN final_states fs ON fs.event_id = e.id
            WHERE e.parent = %s AND e.timestamp = %s
            ORDER BY e.id, fs.id
            """,
            (parent_name, run_timestamp),
        )
        return cur.fetchall()


def _build_event_groups(rows: List[tuple]) -> Dict[int, Dict[str, object]]:
    events: Dict[int, Dict[str, object]] = {}
    for event_id, decay_mode, particle, E, px, py, pz, weight in rows:
        payload = events.setdefault(
            int(event_id),
            {"decay_mode": decay_mode, "weight": float(weight), "particles": {}},
        )
        payload["particles"][str(particle)] = (
            float(E),
            float(px),
            float(py),
            float(pz),
        )
    return events


def _finalize_figure(
    fig: plt.Figure, output_path: Optional[Path], show_images: bool, save_images: bool
) -> Optional[str]:
    saved_path = None
    if save_images and output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        saved_path = str(output_path)
    if not show_images:
        plt.close(fig)
    return saved_path


def _plot_decay_modes(
    decay_mode_counts: List[tuple],
    output_path: Optional[Path],
    show_images: bool,
    save_images: bool,
) -> Optional[str]:
    if not decay_mode_counts:
        return None

    labels = [row[0] for row in decay_mode_counts]
    values = [int(row[1]) for row in decay_mode_counts]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(labels)), values, color="#7f8cff")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Events")
    ax.set_title("Decay Mode Distribution")
    fig.tight_layout()
    return _finalize_figure(fig, output_path, show_images, save_images)


def _plot_particle_counts(
    particle_counts: List[tuple],
    output_path: Optional[Path],
    show_images: bool,
    save_images: bool,
) -> Optional[str]:
    if not particle_counts:
        return None

    labels = [row[0] for row in particle_counts]
    values = [int(row[1]) for row in particle_counts]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(labels)), values, color="#9ece6a")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Counts")
    ax.set_title("Final State Particle Counts")
    fig.tight_layout()
    return _finalize_figure(fig, output_path, show_images, save_images)


def _plot_energy_histogram(
    energies: List[float],
    output_path: Optional[Path],
    show_images: bool,
    save_images: bool,
) -> Optional[str]:
    if not energies:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(energies, bins=40, color="#f6c177", alpha=0.85)
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("Entries")
    ax.set_title("Final State Energy Spectrum")
    fig.tight_layout()
    return _finalize_figure(fig, output_path, show_images, save_images)


def _plot_michel_suite(
    parent_name: str,
    event_groups: Dict[int, Dict[str, object]],
    graphs_dir: Path,
    show_images: bool,
    save_images: bool,
) -> Dict[str, Optional[str]]:
    output: Dict[str, Optional[str]] = {
        "michel_spectrum": None,
        "michel_overlay": None,
        "dalitz_muon_decay": None,
    }

    lepton_name = "Electron" if parent_name.lower() == "muon" else "Positron"
    m_mu = 105.66

    electron_energies = []
    for evt in event_groups.values():
        particles = evt["particles"]
        if lepton_name in particles:
            electron_energies.append(float(particles[lepton_name][0]))

    if electron_energies:
        x = 2.0 * np.array(electron_energies, dtype=float) / m_mu
        x = x[(x > 0) & (x < 1)]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(x, bins=60, density=True, alpha=0.8, label="ColliderX")
        x_grid = np.linspace(0.0, 1.0, 200)
        y = x_grid**2 * (3.0 - 2.0 * x_grid)
        y /= np.trapezoid(y, x_grid)
        ax.plot(x_grid, y, "r--", label="Michel spectrum")
        ax.set_xlabel(r"$x = 2E_e/m_\mu$")
        ax.set_ylabel("Normalized counts / PDF")
        ax.set_title("Michel Spectrum")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        output["michel_spectrum"] = _finalize_figure(
            fig,
            graphs_dir / "michel_spectrum.png",
            show_images,
            save_images,
        )

        rng = np.random.default_rng(0)
        flat = rng.random((200000, 3))
        flat /= flat.sum(axis=1)[:, None]
        x_flat = 2.0 * (flat[:, 0] * (m_mu / 2.0)) / m_mu

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(x_flat, bins=60, density=True, alpha=0.6, label="Flat phase space")
        ax.hist(
            x,
            bins=60,
            density=True,
            histtype="step",
            linewidth=2,
            label="ColliderX (V-A)",
        )
        ax.set_xlabel(r"$x = 2E_e/m_\mu$")
        ax.set_ylabel("Normalized counts")
        ax.set_title("Michel: dynamics vs kinematics")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        output["michel_overlay"] = _finalize_figure(
            fig,
            graphs_dir / "michel_overlay.png",
            show_images,
            save_images,
        )

    # Dalitz
    dalitz_x = []
    dalitz_y = []
    dalitz_w = []
    for evt in event_groups.values():
        particles = evt["particles"]
        weight = float(evt["weight"])

        if parent_name.lower() == "muon":
            needed = ("Electron", "Electron antineutrino", "Muon neutrino")
            if not all(name in particles for name in needed):
                continue
            e_e = particles["Electron"][0]
            e_nubar = particles["Electron antineutrino"][0]
        else:
            needed = ("Positron", "Electron neutrino", "Muon antineutrino")
            if not all(name in particles for name in needed):
                continue
            e_e = particles["Positron"][0]
            e_nubar = particles["Electron neutrino"][0]

        dalitz_x.append(2.0 * float(e_e) / m_mu)
        dalitz_y.append(2.0 * float(e_nubar) / m_mu)
        dalitz_w.append(weight)

    if dalitz_x:
        x_arr = np.array(dalitz_x, dtype=float)
        y_arr = np.array(dalitz_y, dtype=float)
        w_arr = np.array(dalitz_w, dtype=float)

        fig, ax = plt.subplots(figsize=(8, 7))
        h = ax.hist2d(
            x_arr, y_arr, bins=60, weights=w_arr, cmap="viridis", range=[[0, 1], [0, 1]]
        )
        boundary = np.linspace(0, 1, 200)
        ax.plot(
            boundary,
            1.0 - boundary,
            "r--",
            linewidth=2,
            label=r"$x_e + x_{\bar\nu_e} = 1$",
        )
        ax.set_xlabel(r"$x_e = 2E_e / m_\mu$")
        ax.set_ylabel(r"$x_{\bar{\nu}_e} = 2E_{\bar{\nu}_e} / m_\mu$")
        ax.set_title("Dalitz plot")
        fig.colorbar(h[3], ax=ax, label="Event weight")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        output["dalitz_muon_decay"] = _finalize_figure(
            fig,
            graphs_dir / "dalitz_muon_decay.png",
            show_images,
            save_images,
        )

    return output


def _plot_z_suite(
    event_groups: Dict[int, Dict[str, object]],
    graphs_dir: Path,
    show_images: bool,
    save_images: bool,
) -> Dict[str, Optional[str]]:
    output: Dict[str, Optional[str]] = {
        "z_costheta": None,
        "z_invariant_mass": None,
    }

    cos_theta = []
    cos_w = []
    masses = []
    mass_w = []

    for evt in event_groups.values():
        particles = evt["particles"]
        weight = float(evt["weight"])

        if "Muon" in particles:
            _, px, py, pz = particles["Muon"]
            p_mag = float(np.sqrt(px**2 + py**2 + pz**2))
            if p_mag > 0:
                cos_theta.append(float(np.clip(pz / p_mag, -1.0, 1.0)))
                cos_w.append(weight)

        if "Muon" in particles and "Antimuon" in particles:
            E1, px1, py1, pz1 = particles["Muon"]
            E2, px2, py2, pz2 = particles["Antimuon"]
            E = E1 + E2
            px = px1 + px2
            py = py1 + py2
            pz = pz1 + pz2
            m2 = E * E - px * px - py * py - pz * pz
            if m2 > 0:
                masses.append(float(np.sqrt(m2) / 1000.0))
                mass_w.append(weight)

    if cos_theta:
        cos_arr = np.array(cos_theta, dtype=float)
        w_arr = np.array(cos_w, dtype=float)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(
            cos_arr,
            bins=60,
            range=(-1, 1),
            weights=w_arr,
            density=True,
            alpha=0.8,
            label="Simulation",
        )
        x = np.linspace(-1, 1, 300)
        y = 1.0 + x**2
        y /= np.trapezoid(y, x)
        ax.plot(x, y, "r--", lw=2, label=r"$1+\cos^2\theta$")
        ax.set_xlabel(r"$\cos\theta$")
        ax.set_ylabel("Normalized events")
        ax.set_title(r"$Z \to \mu^+\mu^-$ angular distribution")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        output["z_costheta"] = _finalize_figure(
            fig, graphs_dir / "z_costheta.png", show_images, save_images
        )

    if masses:
        m_arr = np.array(masses, dtype=float)
        w_arr = np.array(mass_w, dtype=float)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(m_arr, bins=80, range=(65, 115), weights=w_arr, alpha=0.8)
        ax.set_xlabel(r"Invariant mass $m_{\mu\mu}$ [GeV]")
        ax.set_ylabel("Events")
        ax.set_title(r"$Z \to \mu^+\mu^-$ invariant mass")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        output["z_invariant_mass"] = _finalize_figure(
            fig, graphs_dir / "z_invariant_mass.png", show_images, save_images
        )

    return output


def _plot_tau_suite(
    n_events: int,
    seed: Optional[int],
    graphs_dir: Path,
    show_images: bool,
    save_images: bool,
) -> Dict[str, Optional[str]]:
    output: Dict[str, Optional[str]] = {"tau_pion_angular": None}

    mode_candidates = ["τ+ τ−", "tau+ tau-"]
    result = None
    for mode in mode_candidates:
        try:
            result = simulate_chain(
                parent_name="Z boson",
                n_events=max(1000, min(n_events, 8000)),
                seed=seed,
                fixed_decay_mode=mode,
                force_tau_pion_only=True,
            )
            break
        except Exception:
            continue

    if result is None:
        return output

    cos_thetas = np.array(result.get("tau_pion_cos_theta", []), dtype=float)
    if cos_thetas.size == 0:
        return output

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(
        cos_thetas,
        bins=50,
        range=(-1, 1),
        density=True,
        alpha=0.8,
        color="#7b8cf7",
        label="Simulation",
    )
    x = np.linspace(-1, 1, 200)
    axes[0].plot(x, np.ones_like(x) * 0.5, "r--", lw=2, label="Flat (unpolarized)")
    axes[0].set_xlabel(r"$\cos\theta_\pi$")
    axes[0].set_ylabel("Normalized events")
    axes[0].set_title("Tau pion angular distribution")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    hist, edges = np.histogram(cos_thetas, bins=20, range=(-1, 1), density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    axes[1].scatter(centres, hist, color="#7b8cf7", s=30, zorder=5, label="Simulation")
    axes[1].plot(
        x,
        0.5 * (1 + x),
        "--",
        lw=2,
        color="#e87070",
        label=r"$\frac{1}{2}(1+\cos\theta)$",
    )
    axes[1].plot(
        x,
        0.5 * (1 - x),
        "--",
        lw=2,
        color="#4ecb7a",
        label=r"$\frac{1}{2}(1-\cos\theta)$",
    )
    axes[1].set_xlabel(r"$\cos\theta_\pi$")
    axes[1].set_ylabel("Probability density")
    axes[1].set_title("Spin correlation signal")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    output["tau_pion_angular"] = _finalize_figure(
        fig, graphs_dir / "tau_pion_angular.png", show_images, save_images
    )
    return output


def run_full_flow(
    parent_name: str,
    n_events: int,
    seed: Optional[int],
    output_dir: str | Path,
    afb: float = 0.0,
    fixed_decay_mode: Optional[str] = None,
    parent_energy: Optional[float] = None,
    diagram_mode: str = "sample",
    diagram_max_depth: int = 3,
    diagram_max_nodes: int = 500,
    diagram_rankdir: str = "LR",
    render_format: str = "png",
    show_images: bool = True,
    save_images: bool = False,
) -> Dict[str, object]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    simulation_result = simulate_events(
        parent_name=parent_name,
        n_events=n_events,
        seed=seed,
        afb=afb,
        fixed_decay_mode=fixed_decay_mode,
        parent_energy=parent_energy,
    )

    run_timestamp = simulation_result.get("run_timestamp")
    if not run_timestamp:
        raise RuntimeError("Simulation did not return run_timestamp")

    temp_dir: Optional[Path] = None
    if save_images:
        dot_path = out_dir / "feynman.dot"
        render_path = out_dir / f"feynman.{render_format}"
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="colliderx_flow_"))
        dot_path = temp_dir / "feynman.dot"
        render_path = temp_dir / f"feynman.{render_format}"

    diagram_warning = None
    try:
        diagram_result = generate_diagram_files(
            parent_name=parent_name,
            dot_out=dot_path,
            render_out=render_path if show_images or save_images else None,
            mode=diagram_mode,
            max_depth=diagram_max_depth,
            seed=seed,
            fixed_decay_mode=fixed_decay_mode,
            max_nodes=diagram_max_nodes,
            rankdir=diagram_rankdir,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "Graphviz 'dot' executable not found" not in message:
            raise
        diagram_result = generate_diagram_files(
            parent_name=parent_name,
            dot_out=dot_path,
            render_out=None,
            mode=diagram_mode,
            max_depth=diagram_max_depth,
            seed=seed,
            fixed_decay_mode=fixed_decay_mode,
            max_nodes=diagram_max_nodes,
            rankdir=diagram_rankdir,
        )
        diagram_warning = message

    decay_mode_counts, particle_counts, energies = _fetch_run_statistics(
        run_timestamp=run_timestamp,
        parent_name=parent_name,
    )

    run_rows = _fetch_run_rows(run_timestamp=run_timestamp, parent_name=parent_name)
    event_groups = _build_event_groups(run_rows)

    graphs_dir = out_dir / "graphs"
    if save_images:
        graphs_dir.mkdir(parents=True, exist_ok=True)

    generic_graphs = {
        "decay_modes": _plot_decay_modes(
            decay_mode_counts,
            graphs_dir / "decay_modes.png" if save_images else None,
            show_images,
            save_images,
        ),
        "final_state_counts": _plot_particle_counts(
            particle_counts,
            graphs_dir / "final_state_counts.png" if save_images else None,
            show_images,
            save_images,
        ),
        "energy_spectrum": _plot_energy_histogram(
            energies,
            graphs_dir / "energy_spectrum.png" if save_images else None,
            show_images,
            save_images,
        ),
    }

    auto_graphs: Dict[str, Optional[str]] = {}
    decay_mode_text = " ".join(str(mode) for mode, _ in decay_mode_counts).lower()
    particle_names = {str(name).lower() for name, _ in particle_counts}
    parent_lower = parent_name.lower()

    if parent_lower in {"muon", "antimuon"}:
        auto_graphs.update(
            _plot_michel_suite(
                parent_name=parent_name,
                event_groups=event_groups,
                graphs_dir=graphs_dir,
                show_images=show_images,
                save_images=save_images,
            )
        )

    if "z" in parent_lower and (
        "muon" in particle_names
        or "antimuon" in particle_names
        or "μ" in decay_mode_text
    ):
        auto_graphs.update(
            _plot_z_suite(
                event_groups=event_groups,
                graphs_dir=graphs_dir,
                show_images=show_images,
                save_images=save_images,
            )
        )

    if "z" in parent_lower and ("τ" in decay_mode_text or "tau" in decay_mode_text):
        auto_graphs.update(
            _plot_tau_suite(
                n_events=n_events,
                seed=seed,
                graphs_dir=graphs_dir,
                show_images=show_images,
                save_images=save_images,
            )
        )

    if show_images:
        plt.show()
        plt.close("all")

    summary_path = out_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Parent: {parent_name}\n")
        f.write(f"Events requested: {n_events}\n")
        f.write(f"Run timestamp: {run_timestamp}\n")
        f.write(f"Success: {simulation_result.get('success', 0)}\n")
        f.write(f"Failed: {simulation_result.get('failed', 0)}\n")
        f.write(f"Rejected: {simulation_result.get('rejected', 0)}\n")
        if parent_energy is not None:
            f.write(f"Parent energy override (MeV): {parent_energy}\n")
        if save_images:
            f.write(f"Diagram DOT: {diagram_result.get('dot_path')}\n")
        else:
            f.write("Diagram DOT: [ephemeral, not saved]\n")
        if diagram_result.get("rendered_path"):
            if save_images:
                f.write(f"Diagram Rendered: {diagram_result.get('rendered_path')}\n")
            else:
                f.write("Diagram Rendered: [ephemeral, not saved]\n")
        if diagram_warning:
            f.write(f"Diagram Warning: {diagram_warning}\n")

    if temp_dir is not None:
        shutil.rmtree(temp_dir, ignore_errors=True)

    result = {
        "simulation": simulation_result,
        "diagram": diagram_result,
        "graphs": {
            "generic": generic_graphs,
            "auto": auto_graphs,
        },
        "summary_path": str(summary_path),
        "output_dir": str(out_dir),
        "show_images": show_images,
        "save_images": save_images,
    }
    if diagram_warning:
        result["diagram_warning"] = diagram_warning
    return result
