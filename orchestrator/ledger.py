"""SQLite run ledger (SPEC §6): one row per run, append-only event log."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

from .models import TERMINAL

DEFAULT_DB = Path(__file__).resolve().parents[1] / "state" / "ledger.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id     TEXT PRIMARY KEY,
  kind       TEXT NOT NULL,            -- 'train' | 'smoke'
  spec_json  TEXT NOT NULL,
  state      TEXT NOT NULL,
  provider   TEXT DEFAULT '',
  instance_id TEXT DEFAULT '',
  base_url   TEXT DEFAULT '',
  artifact_uri TEXT DEFAULT '',
  error      TEXT DEFAULT '',
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  id      INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id  TEXT NOT NULL,
  ts      REAL NOT NULL,
  state   TEXT NOT NULL,
  note    TEXT DEFAULT ''
);
"""


class Ledger:
    def __init__(self, path: Path | str = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)

    def create(self, kind: str, spec: dict, provider: str = "",
               run_id: str | None = None) -> str:
        run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        ts = time.time()
        self.db.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, kind, json.dumps(spec, default=str), "pending", provider,
             "", "", "", "", ts, ts))
        self.event(run_id, "pending", "created")
        self.db.commit()
        return run_id

    def get(self, run_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def list(self, state: str | None = None, include_terminal: bool = True) -> list[dict]:
        q, args = "SELECT * FROM runs", []
        if state:
            q += " WHERE state=?"
            args.append(state)
        elif not include_terminal:
            q += " WHERE state NOT IN (%s)" % ",".join("?" * len(TERMINAL))
            args.extend(TERMINAL)
        rows = self.db.execute(q + " ORDER BY created_at", args).fetchall()
        return [dict(r) for r in rows]

    def update(self, run_id: str, state: str | None = None, **fields) -> None:
        sets, args = [], []
        if state:
            sets.append("state=?")
            args.append(state)
        for key in ("provider", "instance_id", "base_url", "artifact_uri", "error"):
            if key in fields:
                sets.append(f"{key}=?")
                args.append(fields[key])
        sets.append("updated_at=?")
        args.append(time.time())
        args.append(run_id)
        self.db.execute(f"UPDATE runs SET {','.join(sets)} WHERE run_id=?", args)
        if state:
            self.event(run_id, state, fields.get("note", ""))
        self.db.commit()

    def event(self, run_id: str, state: str, note: str = "") -> None:
        self.db.execute("INSERT INTO events (run_id, ts, state, note) VALUES (?,?,?,?)",
                        (run_id, time.time(), state, note[:2000]))

    def live_instances(self, provider: str) -> set[str]:
        """Instance ids the ledger believes are not finished."""
        settled = sorted(TERMINAL | {"verified", "delivered", "lost"})
        q = ("SELECT instance_id FROM runs WHERE provider=? "
             "AND state NOT IN (%s) AND instance_id != ''"
             % ",".join("?" * len(settled)))
        rows = self.db.execute(q, (provider, *settled)).fetchall()
        return {r["instance_id"] for r in rows}
