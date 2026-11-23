"""Content-addressed response cache.

Keyed on a hash of (provider identity, messages, generation params). Re-running an eval after
changing only a *scorer* therefore costs zero API calls, which makes iterating on metrics cheap.
Only deterministic requests (temperature == 0) are cached by default: caching sampled outputs
would silently collapse the variance you are trying to measure.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from evalharness.types import ModelResponse


class ResponseCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            " key TEXT PRIMARY KEY, response TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> ModelResponse | None:
        row = self._conn.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        response = ModelResponse.model_validate_json(row[0])
        return response.model_copy(update={"cached": True})

    def put(self, key: str, response: ModelResponse) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO responses (key, response, created_at) VALUES (?, ?, ?)",
            (key, response.model_copy(update={"cached": False}).model_dump_json(), time.time()),
        )
        self._conn.commit()

    def __len__(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0])

    def close(self) -> None:
        self._conn.close()
