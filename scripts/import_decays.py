import sqlite3
import csv
from pathlib import Path

# === Paths ===
BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "colliderx.db"
CSV_PATH = BASE_DIR / "data" / "decays.csv"  # adjust path if your file is elsewhere

# === Connect to DB ===
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# === Create table if it doesn't exist ===
cur.execute("""
CREATE TABLE IF NOT EXISTS decays (
    pdg_id INTEGER,
    decay_mode TEXT,
    branching_fraction REAL,
    UNIQUE(pdg_id, decay_mode)
);
""")

# === Read CSV and insert data ===
inserted = 0
skipped = 0

with open(CSV_PATH, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        try:
            pdg_id = int(row["PDG ID"].strip())
            decay_mode = row["Decay mode"].strip()
            br_str = row["Branching fraction"].strip()

            if not br_str:
                skipped += 1
                continue

            branching_fraction = float(br_str)

            cur.execute("""
                INSERT OR IGNORE INTO decays (pdg_id, decay_mode, branching_fraction)
                VALUES (?, ?, ?)
            """, (pdg_id, decay_mode, branching_fraction))

            if cur.rowcount > 0:
                inserted += 1
            else:
                skipped += 1

        except Exception as e:
            print(f"⚠️ Skipping row {row} due to error: {e}")
            skipped += 1

conn.commit()
conn.close()

print(f"✅ Done! Inserted {inserted} new rows. Skipped {skipped} (duplicates or invalid).")
print(f"📦 Database: {DB_PATH}")
print(f"📊 Table: decays")
