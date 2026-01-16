from db import get_conn


def list_decays(pdg_id):
    with get_conn() as conn, conn.cursor() as cursor:
        cursor.execute(
            "SELECT decay_mode, branching_fraction FROM decays WHERE pdg_id=%s",
            (pdg_id,),
        )
        rows = cursor.fetchall()

    print(f"=== Decays of PDG ID {pdg_id} ===")
    for mode, br in rows:
        print(f"{mode} ({br*100:.2f}%)")


if __name__ == "__main__":
    list_decays(25)  # Higgs boson
    list_decays(23)  # Z boson
    list_decays(24)  # W boson