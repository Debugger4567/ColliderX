import pandas as pd
from db import get_conn


def main():
    conn = get_conn()
    cur = conn.cursor()

    particles_df = pd.read_csv('data/particles.csv')
    decays_df = pd.read_csv('data/decays.csv')

    decays_df = decays_df.rename(columns={
        "PDG ID": "pdg_id",
        "Decay mode": "decay_mode",
        "Branching fraction": "branching_fraction",
    })
    decays_df = decays_df.loc[:, ["pdg_id", "decay_mode", "branching_fraction"]]

    cur.execute("TRUNCATE TABLE decays RESTART IDENTITY")
    cur.execute("TRUNCATE TABLE particles RESTART IDENTITY")

    particle_cols = [
        "PDG ID", "Name", "Symbol", "Mass (MeV/c^2)", "Charge (e)", "Spin",
        "Baryon Number", "Le", "Lmu", "Ltau", "Strangeness", "Charm",
        "Bottomness", "Topness",
    ]

    for row in particles_df.to_dict(orient="records"):
        values = [row.get(col) for col in particle_cols]
        cur.execute(
            """
            INSERT INTO particles (
                "PDG ID", "Name", "Symbol", "Mass (MeV/c^2)", "Charge (e)", "Spin",
                "Baryon Number", "Le", "Lmu", "Ltau", "Strangeness", "Charm",
                "Bottomness", "Topness"
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT ("PDG ID") DO UPDATE SET
                "Name" = EXCLUDED."Name",
                "Symbol" = EXCLUDED."Symbol",
                "Mass (MeV/c^2)" = EXCLUDED."Mass (MeV/c^2)",
                "Charge (e)" = EXCLUDED."Charge (e)",
                "Spin" = EXCLUDED."Spin",
                "Baryon Number" = EXCLUDED."Baryon Number",
                "Le" = EXCLUDED."Le",
                "Lmu" = EXCLUDED."Lmu",
                "Ltau" = EXCLUDED."Ltau",
                "Strangeness" = EXCLUDED."Strangeness",
                "Charm" = EXCLUDED."Charm",
                "Bottomness" = EXCLUDED."Bottomness",
                "Topness" = EXCLUDED."Topness"
            """,
            values,
        )

    for row in decays_df.to_dict(orient="records"):
        cur.execute(
            """
            INSERT INTO decays (pdg_id, decay_mode, branching_fraction)
            VALUES (%s, %s, %s)
            ON CONFLICT (pdg_id, decay_mode) DO UPDATE SET
                branching_fraction = EXCLUDED.branching_fraction
            """,
            (row["pdg_id"], row["decay_mode"], row["branching_fraction"]),
        )

    conn.commit()
    conn.close()
    print("Migration complete: Postgres tables refreshed with CSV data.")


if __name__ == "__main__":
    main()
