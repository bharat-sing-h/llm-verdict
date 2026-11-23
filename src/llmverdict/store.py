"""SQLite results store.

Every sample result is persisted the moment it is scored. That gives crash-safe *resume*
(re-running a run id only evaluates what's missing), and one queryable file that the CLI,
reports and dashboard all read from. SQLite is the right tool at this scale: zero-ops,
transactional, and trivially copied into CI artifacts.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llmverdict.types import ModelResponse, SampleResult, Score

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT PRIMARY KEY,
    created_at     REAL NOT NULL,
    task           TEXT NOT NULL,
    model          TEXT NOT NULL,
    provider       TEXT NOT NULL,
    status         TEXT NOT NULL,
    primary_metric TEXT,
    config         TEXT NOT NULL,
    summary        TEXT
);
CREATE TABLE IF NOT EXISTS results (
    run_id    TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    sample_id TEXT NOT NULL,
    response  TEXT,
    scores    TEXT NOT NULL,
    error     TEXT,
    tags      TEXT NOT NULL,
    PRIMARY KEY (run_id, sample_id)
);
"""


class RunRecord(BaseModel):
    run_id: str
    created_at: float
    task: str
    model: str
    provider: str
    status: str
    primary_metric: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] | None = None


class RunNotFoundError(KeyError):
    pass


class ResultStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)

    # -- runs -------------------------------------------------------------------------------
    def create_run(
        self,
        run_id: str,
        *,
        task: str,
        model: str,
        provider: str,
        config: dict[str, Any],
        primary_metric: str | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO runs (run_id, created_at, task, model, provider, status, primary_metric,"
            " config) VALUES (?, ?, ?, ?, ?, 'running', ?, ?)"
            " ON CONFLICT(run_id) DO UPDATE SET status = 'running'",
            (run_id, time.time(), task, model, provider, primary_metric, json.dumps(config)),
        )
        self._conn.commit()

    def finish_run(self, run_id: str, summary: dict[str, Any], status: str = "completed") -> None:
        self._conn.execute(
            "UPDATE runs SET status = ?, summary = ? WHERE run_id = ?",
            (status, json.dumps(summary), run_id),
        )
        self._conn.commit()

    def _record(self, row: tuple[Any, ...]) -> RunRecord:
        return RunRecord(
            run_id=row[0],
            created_at=row[1],
            task=row[2],
            model=row[3],
            provider=row[4],
            status=row[5],
            primary_metric=row[6],
            config=json.loads(row[7]),
            summary=json.loads(row[8]) if row[8] else None,
        )

    _RUN_COLS = "run_id, created_at, task, model, provider, status, primary_metric, config, summary"

    def get_run(self, run_id: str) -> RunRecord:
        row = self._conn.execute(
            f"SELECT {self._RUN_COLS} FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise RunNotFoundError(run_id)
        return self._record(row)

    def list_runs(self, task: str | None = None) -> list[RunRecord]:
        query = f"SELECT {self._RUN_COLS} FROM runs"
        args: tuple[Any, ...] = ()
        if task:
            query += " WHERE task = ?"
            args = (task,)
        rows = self._conn.execute(query + " ORDER BY created_at DESC", args).fetchall()
        return [self._record(r) for r in rows]

    def delete_run(self, run_id: str) -> None:
        self._conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        self._conn.commit()

    # -- results ----------------------------------------------------------------------------
    def save_result(self, result: SampleResult) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO results (run_id, sample_id, response, scores, error, tags)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                result.run_id,
                result.sample_id,
                result.response.model_dump_json() if result.response else None,
                json.dumps([s.model_dump() for s in result.scores]),
                result.error,
                json.dumps(result.tags),
            ),
        )
        self._conn.commit()

    def completed_ids(self, run_id: str) -> set[str]:
        """Samples that finished without a provider error (errored ones are retried on resume)."""
        rows = self._conn.execute(
            "SELECT sample_id FROM results WHERE run_id = ? AND error IS NULL", (run_id,)
        ).fetchall()
        return {r[0] for r in rows}

    def get_results(self, run_id: str) -> list[SampleResult]:
        rows = self._conn.execute(
            "SELECT sample_id, response, scores, error, tags FROM results WHERE run_id = ?"
            " ORDER BY sample_id",
            (run_id,),
        ).fetchall()
        return [
            SampleResult(
                run_id=run_id,
                sample_id=r[0],
                response=ModelResponse.model_validate_json(r[1]) if r[1] else None,
                scores=[Score.model_validate(s) for s in json.loads(r[2])],
                error=r[3],
                tags=json.loads(r[4]),
            )
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()
