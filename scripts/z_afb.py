import numpy as np
from db import get_conn


def get_db_connection():
    return get_conn()


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

    # Compute forward-backward asymmetry
    forward = weights[cos_theta > 0].sum()
    backward = weights[cos_theta < 0].sum()
    total = forward + backward

    A_FB = (forward - backward) / total if total > 0 else 0.0

    # estimate statistical unvectainity
    sigma_afb = (
        np.sqrt(1.0 - A_FB**2) / np.sqrt(len(cos_theta)) if len(cos_theta) > 0 else 0.0
    )

    if sigma_afb > 0:
        significance = abs(A_FB) / sigma_afb
    else:
        significance = 0.0

    print("\n[RESULT] Forward-Backward Asymmetry")
    print(f"  N_forward  = {forward:.1f}")
    print(f"  N_backward = {backward:.1f}")
    print(f"  A_FB       = {A_FB:.6f} ± {sigma_afb:.6f}")
    print("\n[INTERPRETATION] Expected A_FB ≈ 0 (parity symmetric)")
    print(f"  |A_FB| / σ = {significance:.2f}σ")


if __name__ == "__main__":
    main()
