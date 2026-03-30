from pathlib import Path

from visualization.feynman import build_feynman_diagram, generate_diagram_files, to_dot


def test_build_feynman_diagram_sample_mode():
    diagram = build_feynman_diagram(
        parent_name="Muon",
        mode="sample",
        max_depth=2,
        seed=123,
    )

    assert len(diagram.nodes) > 0
    assert len(diagram.edges) > 0

    dot_text = to_dot(diagram)
    assert "digraph ColliderXFeynman" in dot_text
    assert "Muon" in dot_text


def test_generate_diagram_files_writes_dot(tmp_path: Path):
    dot_path = tmp_path / "muon.dot"

    result = generate_diagram_files(
        parent_name="Muon",
        dot_out=dot_path,
        mode="sample",
        max_depth=2,
        seed=7,
    )

    assert dot_path.exists()
    assert result["dot_path"] == str(dot_path)
    assert int(result["nodes"]) >= 1
    assert int(result["edges"]) >= 1
