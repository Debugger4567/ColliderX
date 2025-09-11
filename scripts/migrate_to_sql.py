import sqlite3
import pandas as pd

conn = sqlite3.connect('colliderx.db')

# Load CSVs
particles_df = pd.read_csv('data/particles.csv')
decays_df = pd.read_csv('data/decays.csv')

# Clean up decays_df column names
decays_df = decays_df.rename(columns={
    "PDG ID": "pdg_id",
    "Decay mode": "decay_mode",
    "Branching fraction": "branching_fraction"
})

# Drop unwanted unnamed columns if they exist
decays_df = decays_df.loc[:, ["pdg_id", "decay_mode", "branching_fraction"]]

# Save clean tables to SQLite
particles_df.to_sql('particles', conn, if_exists='replace', index=False)
decays_df.to_sql('decays', conn, if_exists='replace', index=False)

print("Migration complete: colliderx.db refreshed with clean tables.")

conn.commit()
conn.close()
