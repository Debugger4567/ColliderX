from __future__ import annotations

def apply_detector(final_states: list[dict]) -> dict:
    """
    Minimal detector model:
    - Neutrinos are invisible
    - Everything else is visible
    - MET is computed from invisible transverse momentum
    """
    invisible_names = {"Electron neutrino", "Electron antineutrino", "Muon neutrino", "Muon antineutrino", "Tau neutrino", "Tau antineutrino"}

    visible = []
    met_px  = 0.0
    met_py = 0.0

    for fs in final_states:
        name = fs.get("name", "")
        p4 = fs.get("p4", (0.0, 0.0, 0.0, 0.0))
        if name in invisible_names:
            met_px += float(p4[1])
            met_py += float(p4[2])
        else:
            visible.append(fs)

    met_mag = (met_px**2 + met_py**2) ** 0.5

    return {
        "visible": visible, 
        "met_px": met_px,
        "met_py": met_py,
        "visible_count": len(visible),
        "invisible_count": len(final_states) - len(visible) 
    }