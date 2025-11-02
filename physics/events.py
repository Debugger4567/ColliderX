import sqlite3
import json
import math
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
import numpy as np
from .kinematics import FourVector

DB_PATH = Path(__file__).resolve().parents[1] / "colliderx.db"


class EventDB:
    """
    Handles creation and storage of simulated decay/collision events in colliderx.db.
    Each event stores the parent name, daughter particles with 4-vectors,
    and automatically validates conservation laws.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.create_table()

    @contextmanager
    def get_connection(self):
        """Context manager for safe DB access."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_table(self):
        """Create the 'events' table with conservation tracking."""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT,
                    process_tag TEXT,
                    parent_name TEXT,
                    parent_mass REAL,
                    daughter_names TEXT,
                    E_values TEXT,
                    px_values TEXT,
                    py_values TEXT,
                    pz_values TEXT,
                    energy_conserved INTEGER,
                    momentum_conserved INTEGER,
                    invariant_mass REAL,
                    timestamp TEXT
                )
            """)

    def store_event(
        self,
        parent_name: str,
        parent_mass: float,
        daughters: List[tuple[str, FourVector]],
        event_type: str = "decay",
        process_tag: Optional[str] = None,
        energy_tolerance: float = 1e-6,
        momentum_tolerance: float = 1e-6
    ):
        """
        Store a single event with automatic conservation validation.
        
        Args:
            parent_name: Name of parent particle
            parent_mass: Rest mass of parent (GeV)
            daughters: List of (name, FourVector) tuples
            event_type: "two_body_decay", "three_body_decay", "collision", etc.
            process_tag: Optional physics label, e.g., "H → γγ"
            energy_tolerance: Threshold for energy conservation check
            momentum_tolerance: Threshold for momentum conservation check
        """
        if not process_tag:
            daughter_symbols = " ".join(name for name, _ in daughters)
            process_tag = f"{parent_name} → {daughter_symbols}"

        # Extract FourVectors
        daughter_names = json.dumps([name for name, _ in daughters])
        fvs = [fv for _, fv in daughters]
        
        # Serialize 4-momenta
        E_vals = json.dumps([fv.E for fv in fvs])
        px_vals = json.dumps([fv.px for fv in fvs])
        py_vals = json.dumps([fv.py for fv in fvs])
        pz_vals = json.dumps([fv.pz for fv in fvs])
        
        # Calculate totals
        total_E = sum(fv.E for fv in fvs)
        total_p = np.sum([fv.p for fv in fvs], axis=0)
        
        # Check conservation
        energy_conserved = abs(total_E - parent_mass) < energy_tolerance
        momentum_conserved = np.linalg.norm(total_p) < momentum_tolerance
        
        # Calculate invariant mass of daughters
        p_squared = np.dot(total_p, total_p)
        invariant_mass = math.sqrt(max(total_E**2 - p_squared, 0.0))
        
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO events (
                    event_type, process_tag, parent_name, parent_mass,
                    daughter_names, E_values, px_values, py_values, pz_values,
                    energy_conserved, momentum_conserved, invariant_mass, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_type, process_tag, parent_name, parent_mass,
                daughter_names, E_vals, px_vals, py_vals, pz_vals,
                int(energy_conserved), int(momentum_conserved),
                invariant_mass, datetime.now().isoformat(timespec="seconds")
            ))

    def parse_event(self, event_id: int) -> Optional[Dict[str, Any]]:
        """
        Reconstruct event with FourVector objects.
        Returns None if event_id not found.
        """
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM events WHERE event_id = ?", (event_id,))
            row = cur.fetchone()
        
        if not row:
            return None
        
        # Deserialize arrays
        E_vals = json.loads(row["E_values"])
        px_vals = json.loads(row["px_values"])
        py_vals = json.loads(row["py_values"])
        pz_vals = json.loads(row["pz_values"])
        daughter_names = json.loads(row["daughter_names"])
        
        # Rebuild FourVectors
        daughters = [
            (name, FourVector(E, px, py, pz))
            for name, E, px, py, pz in zip(daughter_names, E_vals, px_vals, py_vals, pz_vals)
        ]
        
        return {
            "event_id": row["event_id"],
            "event_type": row["event_type"],
            "process_tag": row["process_tag"],
            "parent_name": row["parent_name"],
            "parent_mass": row["parent_mass"],
            "daughters": daughters,
            "energy_conserved": bool(row["energy_conserved"]),
            "momentum_conserved": bool(row["momentum_conserved"]),
            "invariant_mass": row["invariant_mass"],
            "timestamp": row["timestamp"]
        }

    def list_events(
        self, 
        limit: int = 10, 
        event_type: Optional[str] = None,
        conserved_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Return recent events, optionally filtered.
        
        Args:
            limit: Max number of events to return
            event_type: Filter by type (e.g., "two_body_decay")
            conserved_only: Only return events with perfect conservation
        """
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            
            query = "SELECT * FROM events WHERE 1=1"
            params = []
            
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)
            
            if conserved_only:
                query += " AND energy_conserved = 1 AND momentum_conserved = 1"
            
            query += " ORDER BY event_id DESC LIMIT ?"
            params.append(limit)
            
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]

    def stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            
            cur.execute("SELECT COUNT(*) FROM events")
            total = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM events WHERE energy_conserved = 1")
            e_conserved = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM events WHERE momentum_conserved = 1")
            p_conserved = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM events WHERE energy_conserved = 1 AND momentum_conserved = 1")
            both_conserved = cur.fetchone()[0]
            
            cur.execute("SELECT AVG(invariant_mass) FROM events WHERE invariant_mass IS NOT NULL")
            avg_mass = cur.fetchone()[0] or 0.0
            
        return {
            "total_events": total,
            "energy_conserved": e_conserved,
            "momentum_conserved": p_conserved,
            "both_conserved": both_conserved,
            "average_invariant_mass": avg_mass,
            "conservation_rate": both_conserved / total if total > 0 else 0.0
        }

    def fetch_event(self, event_id: int) -> Optional[Dict[str, Any]]:
        """Alias for parse_event (backward compatibility)."""
        return self.parse_event(event_id)

    def clear_events(self):
        """Delete all events (use with caution!)."""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM events")