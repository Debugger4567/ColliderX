'''
import sqlite3

# Connect to your DB
conn = sqlite3.connect("colliderx.db")
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

for table in tables:
    table_name = table[0]
    print(f"\n=== {table_name.upper()} TABLE ===")

    # Get column info
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    for col in columns:
        print(col[1])  # col[1] = column name

conn.close()
'''


'''
import sqlite3

def list_all_particles(db_path="colliderx.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Select all particle names + symbols
    cursor.execute("SELECT `Name`, `Symbol`, `PDG ID` FROM particles")
    rows = cursor.fetchall()

    conn.close()

    print("=== Particles in Database ===")
    for name, symbol, pdg in rows:
        print(f"{name} ({symbol}) | PDG ID: {pdg}")

if __name__ == "__main__":
    list_all_particles()
'''

import sqlite3

def list_decays(pdg_id, db_path="colliderx.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT decay_mode, branching_fraction FROM decays WHERE pdg_id=?", (pdg_id,))
    rows = cursor.fetchall()
    conn.close()

    print(f"=== Decays of PDG ID {pdg_id} ===")
    for mode, br in rows:
        print(f"{mode} ({br*100:.2f}%)")

if __name__ == "__main__":
    # Higgs boson
    list_decays(25)
    # Z boson
    list_decays(23)
    # W boson
    list_decays(24)