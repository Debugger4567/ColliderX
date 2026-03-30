from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np

from physics.decay_selector import get_decay_modes, get_decay_products
from physics.particles import Particle

DiagramMode = Literal["sample", "all"]


@dataclass
class DiagramNode:
    node_id: str
    label: str
    depth: int
    kind: Literal["particle", "mode"] = "particle"


@dataclass
class DiagramEdge:
    source: str
    target: str
    label: str = ""
    style: str = "solid"
    color: str = "#d6deeb"


@dataclass
class FeynmanDiagram:
    parent_name: str
    mode: DiagramMode
    max_depth: int
    nodes: List[DiagramNode] = field(default_factory=list)
    edges: List[DiagramEdge] = field(default_factory=list)


def _particle_edge_style(particle_name: str) -> tuple[str, str]:
    name = particle_name.lower()

    fermion_tokens = [
        "electron",
        "muon",
        "tau",
        "neutrino",
        "proton",
        "neutron",
        "quark",
        "lambda",
        "sigma",
        "xi",
        "omega",
    ]
    boson_tokens = ["photon", "gluon", "boson", "higgs"]
    meson_tokens = ["pion", "kaon", "meson"]

    if any(token in name for token in boson_tokens):
        return ("dashed", "#7aa2f7")
    if any(token in name for token in meson_tokens):
        return ("solid", "#f6c177")
    if any(token in name for token in fermion_tokens):
        return ("solid", "#9ece6a")
    return ("solid", "#d6deeb")


class _Builder:
    def __init__(
        self,
        parent_name: str,
        mode: DiagramMode,
        max_depth: int,
        rng: np.random.Generator,
        fixed_decay_mode: Optional[str],
        max_nodes: int,
    ) -> None:
        self.parent_name = parent_name
        self.mode = mode
        self.max_depth = max_depth
        self.rng = rng
        self.fixed_decay_mode = fixed_decay_mode
        self.max_nodes = max_nodes

        self.nodes: List[DiagramNode] = []
        self.edges: List[DiagramEdge] = []
        self._counter = 0

    def _new_node(
        self, label: str, depth: int, kind: Literal["particle", "mode"] = "particle"
    ) -> str:
        if len(self.nodes) >= self.max_nodes:
            raise RuntimeError(
                f"Diagram node budget exceeded ({self.max_nodes}). Reduce max_depth or use mode='sample'."
            )
        self._counter += 1
        node_id = f"n{self._counter}"
        self.nodes.append(
            DiagramNode(node_id=node_id, label=label, depth=depth, kind=kind)
        )
        return node_id

    @staticmethod
    def _pdg(name: str) -> Optional[int]:
        try:
            particle = Particle(name)
            return int(particle.pdg_id) if particle.pdg_id is not None else None
        except Exception:
            return None

    def _expand_particle(
        self,
        particle_name: str,
        depth: int,
        parent_node: Optional[str] = None,
        edge_label: str = "",
    ) -> str:
        particle_node = self._new_node(
            label=particle_name, depth=depth, kind="particle"
        )

        if parent_node is not None:
            style, color = _particle_edge_style(particle_name)
            self.edges.append(
                DiagramEdge(
                    source=parent_node,
                    target=particle_node,
                    label=edge_label,
                    style=style,
                    color=color,
                )
            )

        if depth >= self.max_depth:
            return particle_node

        pdg_id = self._pdg(particle_name)
        if pdg_id is None:
            return particle_node

        modes = get_decay_modes(pdg_id)
        if not modes:
            return particle_node

        valid_modes: list[tuple[str, float, list[str]]] = []
        for mode_name, branching_fraction in modes:
            try:
                daughters = get_decay_products(pdg_id, mode_name)
            except Exception:
                continue
            valid_modes.append((mode_name, float(branching_fraction), daughters))

        if not valid_modes:
            return particle_node

        if self.mode == "sample":
            if depth == 0 and self.fixed_decay_mode:
                chosen_mode = self.fixed_decay_mode
                try:
                    daughters = get_decay_products(pdg_id, chosen_mode)
                except Exception:
                    return particle_node
            else:
                names = [mode_name for mode_name, _, _ in valid_modes]
                weights = np.array(
                    [branching_fraction for _, branching_fraction, _ in valid_modes],
                    dtype=float,
                )
                weights_sum = float(weights.sum())
                if weights_sum <= 0.0:
                    index = int(self.rng.integers(0, len(valid_modes)))
                else:
                    weights = weights / weights_sum
                    index = int(self.rng.choice(len(valid_modes), p=weights))
                chosen_mode, _, daughters = valid_modes[index]

            mode_node = self._new_node(label=chosen_mode, depth=depth + 1, kind="mode")
            self.edges.append(
                DiagramEdge(
                    source=particle_node,
                    target=mode_node,
                    label="",
                    style="dotted",
                    color="#7dcfff",
                )
            )
            for daughter_name in daughters:
                self._expand_particle(daughter_name, depth + 1, parent_node=mode_node)
            return particle_node

        # mode == "all"
        for mode_name, branching_fraction, daughters in sorted(
            valid_modes, key=lambda item: item[1], reverse=True
        ):
            mode_label = f"{mode_name}\\nBR={branching_fraction:.4g}"
            mode_node = self._new_node(label=mode_label, depth=depth + 1, kind="mode")
            self.edges.append(
                DiagramEdge(
                    source=particle_node,
                    target=mode_node,
                    label="",
                    style="dotted",
                    color="#7dcfff",
                )
            )
            for daughter_name in daughters:
                self._expand_particle(daughter_name, depth + 1, parent_node=mode_node)

        return particle_node


def build_feynman_diagram(
    parent_name: str,
    mode: DiagramMode = "sample",
    max_depth: int = 3,
    seed: Optional[int] = None,
    fixed_decay_mode: Optional[str] = None,
    max_nodes: int = 500,
) -> FeynmanDiagram:
    if mode not in ("sample", "all"):
        raise ValueError("mode must be one of: 'sample', 'all'")
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")

    rng = np.random.default_rng(seed)
    builder = _Builder(
        parent_name=parent_name,
        mode=mode,
        max_depth=max_depth,
        rng=rng,
        fixed_decay_mode=fixed_decay_mode,
        max_nodes=max_nodes,
    )
    builder._expand_particle(parent_name, depth=0, parent_node=None)

    return FeynmanDiagram(
        parent_name=parent_name,
        mode=mode,
        max_depth=max_depth,
        nodes=builder.nodes,
        edges=builder.edges,
    )


def to_dot(diagram: FeynmanDiagram, rankdir: Literal["LR", "TB"] = "LR") -> str:
    if rankdir not in ("LR", "TB"):
        raise ValueError("rankdir must be 'LR' or 'TB'")

    lines: List[str] = [
        "digraph ColliderXFeynman {",
        f"  rankdir={rankdir};",
        '  graph [bgcolor="#0f1220", fontname="Helvetica", fontsize=12];',
        '  node [fontname="Helvetica", fontsize=11, color="#c0caf5", fontcolor="#e5e9f0"];',
        '  edge [fontname="Helvetica", fontsize=10, color="#c0caf5", fontcolor="#c0caf5", arrowsize=0.8];',
    ]

    for node in diagram.nodes:
        if node.kind == "particle":
            shape = "ellipse"
            fill = "#1f2335"
            peripheries = 1
        else:
            shape = "box"
            fill = "#2a2f45"
            peripheries = 1

        safe_label = node.label.replace('"', '\\"')
        lines.append(
            f'  {node.node_id} [label="{safe_label}", shape={shape}, style="filled", fillcolor="{fill}", peripheries={peripheries}];'
        )

    for edge in diagram.edges:
        safe_label = edge.label.replace('"', '\\"') if edge.label else ""
        label_clause = f', label="{safe_label}"' if safe_label else ""
        lines.append(
            f'  {edge.source} -> {edge.target} [style={edge.style}, color="{edge.color}"{label_clause}];'
        )

    lines.append("}")
    return "\n".join(lines)


def save_dot(
    diagram: FeynmanDiagram, dot_path: str | Path, rankdir: Literal["LR", "TB"] = "LR"
) -> Path:
    path = Path(dot_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_dot(diagram, rankdir=rankdir), encoding="utf-8")
    return path


def render_dot(
    dot_path: str | Path, output_path: str | Path, fmt: Optional[str] = None
) -> Path:
    dot_file = Path(dot_path)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if fmt is None:
        suffix = out_file.suffix.lower().lstrip(".")
        fmt = suffix if suffix else "svg"

    cmd = ["dot", f"-T{fmt}", str(dot_file), "-o", str(out_file)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Graphviz 'dot' executable not found. Install Graphviz or use DOT output only."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"Graphviz rendering failed: {stderr}") from exc

    return out_file


def generate_diagram_files(
    parent_name: str,
    dot_out: str | Path,
    render_out: Optional[str | Path] = None,
    mode: DiagramMode = "sample",
    max_depth: int = 3,
    seed: Optional[int] = None,
    fixed_decay_mode: Optional[str] = None,
    max_nodes: int = 500,
    rankdir: Literal["LR", "TB"] = "LR",
) -> Dict[str, str]:
    diagram = build_feynman_diagram(
        parent_name=parent_name,
        mode=mode,
        max_depth=max_depth,
        seed=seed,
        fixed_decay_mode=fixed_decay_mode,
        max_nodes=max_nodes,
    )

    dot_path = save_dot(diagram, dot_out, rankdir=rankdir)
    result = {
        "dot_path": str(dot_path),
        "nodes": str(len(diagram.nodes)),
        "edges": str(len(diagram.edges)),
        "mode": diagram.mode,
    }

    if render_out is not None:
        rendered = render_dot(dot_path, render_out)
        result["rendered_path"] = str(rendered)

    return result
