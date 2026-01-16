import json
import math
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
import numpy as np
from .kinematics import FourVector
from db import get_conn


class EventDB:
    """
    Handles creation and storage of simulated decay/collision events in colliderx.db.
    Uses a persistent connection for performance.
    """

    def __init__(self):
        self.conn = get_conn()
        self.create_table()

    @contextmanager
    def cursor(self):
        """Context manager for cursor with auto-commit."""
        cur = self.conn.cursor()
        try:
            yield cur
            self.conn.commit()
        finally:
            cur.close()

    def create_table(self):
        """Create the 'events' table with conservation tracking."""
        with self.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id SERIAL PRIMARY KEY,
                    event_type TEXT,
                    process_tag TEXT,
                    parent TEXT,
                    decay_mode TEXT,
                    energy DOUBLE PRECISION CHECK (energy > 0),
                    weight DOUBLE PRECISION DEFAULT 1.0,
                    timestamp TEXT
                )
                """
            )

    def store_event(
        self,
        parent_name: str,
        decay_mode: str,
        energy: float,
        daughters: List[tuple[str, FourVector]],
        event_type: str = "decay",
        process_tag: Optional[str] = None,
        weight: float = 1.0,
    ):
        """
        Store a single event in the database.
        
        Args:
            parent_name: Name of parent particle (maps to DB 'parent')
            decay_mode: Decay channel string (e.g., "π0 → γ γ")
            energy: Parent energy in MeV (from parent.fourvec.E)
            daughters: List of (name, FourVector) tuples
            event_type: Type of decay process
            process_tag: Optional physics label
            weight: Matrix element weight (default 1.0 for unweighted)
        """
        if not process_tag:
            process_tag = f"{parent_name} → {' '.join(name for name, _ in daughters)}"

        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO events (
                    parent, decay_mode, energy, event_type, process_tag, weight, timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    parent_name,
                    decay_mode,
                    energy,
                    event_type,
                    process_tag,
                    weight,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def parse_event(self, event_id: int) -> Optional[Dict[str, Any]]:
        """
        Legacy: columns don't match current DB schema. Use raw SQL queries instead.
        """
        return None

    # def parse_event_old(self, event_id: int) -> Optional[Dict[str, Any]]:

    def list_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch recent events from database."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, parent, decay_mode, energy, event_type, weight, timestamp "
                    "FROM events ORDER BY id DESC LIMIT %s",
                    (limit,)
                )
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]

    def stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM events")
                total = cur.fetchone()[0]

                cur.execute("SELECT AVG(weight) FROM events")
                avg_weight = cur.fetchone()[0] or 1.0
            
        return {
            "total_events": total,
            "average_weight": avg_weight,
        }

    def fetch_event(self, event_id: int) -> Optional[Dict[str, Any]]:
        """Alias for parse_event (backward compatibility)."""
        return self.parse_event(event_id)

    def clear_events(self):
        """Delete all events (use with caution!)."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM events")