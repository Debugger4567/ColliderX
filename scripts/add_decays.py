import sqlite3

def add_decays(db_path="colliderx.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Dictionary: PDG ID → [(decay_mode, branching_fraction), ...]
    decays = {
        # Higgs boson (H0, 25)
        25: [
            ("b b̄", 58.2),
            ("W+ W−", 21.5),
            ("Z0 Z0", 2.6),
            ("τ+ τ−", 6.3),
            ("γ γ", 0.23),       # famous diphoton decay
            ("gg", 8.5),
            ("μ+ μ−", 0.022),
        ],

        # Z boson (Z0, 23)
        23: [
            ("e+ e−", 3.37),
            ("μ+ μ−", 3.37),
            ("τ+ τ−", 3.36),
            ("νe ν̄e", 6.7),
            ("νμ ν̄μ", 6.7),
            ("ντ ν̄τ", 6.7),
            ("qq̄", 69.9),  # hadronic decays
        ],

        # W+ boson (24)
        24: [
            ("e+ νe", 10.7),
            ("μ+ νμ", 10.6),
            ("τ+ ντ", 11.3),
            ("qq̄'", 67.4),
        ],

        # W− boson (-24)
        -24: [
            ("e− ν̄e", 10.7),
            ("μ− ν̄μ", 10.6),
            ("τ− ν̄τ", 11.3),
            ("qq̄'", 67.4),
        ],
    }

    for pdg_id, modes in decays.items():
        for mode, br in modes:
            cur.execute("""
                INSERT INTO decays (pdg_id, decay_mode, branching_fraction)
                VALUES (?, ?, ?)
            """, (pdg_id, mode, br))

    conn.commit()
    conn.close()
    print("✅ New decays added successfully!")

if __name__ == "__main__":
    add_decays()
