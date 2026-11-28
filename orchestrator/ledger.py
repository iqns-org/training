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
-- EVAL HISTORY (SPEC 13am): one row per bench recording, keyed by
-- (model, host, artifact run). The cross-round/cross-target table we
-- never had: round 2's "loss improved, behavior flattened" was invisible
-- because nothing persisted a score to compare against.
CREATE TABLE IF NOT EXISTS evals (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  ts        REAL NOT NULL,
  run_id    TEXT DEFAULT '',
  model     TEXT NOT NULL,
  host      TEXT DEFAULT '',
  target    TEXT DEFAULT '',
  summary   TEXT DEFAULT '',
  rows_json TEXT NOT NULL
);
"""

# additive columns via ALTER (idempotent): budget model 13q/14 — est_cost
# is the PRE-LAUNCH estimate (drives the aggregate gate + `qtrain budget`
# report), price_hr the launched card's rate (actual spend math).
_ADDITIVE = (("est_cost", "REAL DEFAULT 0"), ("price_hr", "REAL DEFAULT 0"))


class Ledger:
    def __init__(self, path: Path | str = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)
        cols = {r["name"] for r in self.db.execute(
            "PRAGMA table_info(runs)")}
        for name, decl in _ADDITIVE:
            if name not in cols:
                self.db.execute(f"ALTER TABLE runs ADD COLUMN {name} {decl}")
        self.db.commit()

    def create(self, kind: str, spec: dict, provider: str = "",
               run_id: str | None = None) -> str:
        run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        ts = time.time()
        self.db.execute(
            "INSERT INTO runs (run_id, kind, spec_json, state, provider,"
            " instance_id, base_url, artifact_uri, error, created_at,"
            " updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
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

    def merge_spec(self, run_id: str, **fields) -> None:
        """Amend spec_json AFTER create(). The ledger row must register
        early (async callers grab the run_id before staging finishes),
        but provenance — dataset sha/lines, exact image — only exists
        once staging is done. Never rewrite existing keys' semantics;
        additive fields only (SPEC 13ak)."""
        row = self.get(run_id)
        if not row:
            raise KeyError(f"no run {run_id}")
        spec = json.loads(row["spec_json"])
        spec.update(fields)
        self.db.execute("UPDATE runs SET spec_json=?, updated_at=? WHERE run_id=?",
                        (json.dumps(spec, default=str), time.time(), run_id))
        self.db.commit()

    def update(self, run_id: str, state: str | None = None, **fields) -> None:
        sets, args = [], []
        if state:
            sets.append("state=?")
            args.append(state)
        for key in ("provider", "instance_id", "base_url", "artifact_uri", "error",
                    "est_cost", "price_hr"):
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

    # -- EVAL HISTORY (SPEC 13am) ----------------------------------------
    def record_eval(self, model: str, rows: list[dict], *, run_id: str = "",
                    host: str = "", target: str = "") -> None:
        """Persist one bench recording (rows = bench.run() output)."""
        graded = [r for r in rows if r.get("pass") != "—"]
        ok = sum(1 for r in graded if r.get("pass") == "PASS")
        self.db.execute(
            "INSERT INTO evals (ts, run_id, model, host, target, summary,"
            " rows_json) VALUES (?,?,?,?,?,?,?)",
            (time.time(), run_id, model, host, target,
             f"{ok}/{len(graded)}", json.dumps(rows, default=str)))
        self.db.commit()

    def list_evals(self, model: str | None = None,
                   limit: int = 20) -> list[dict]:
        q = "SELECT * FROM evals"
        args: list = []
        if model:
            q += " WHERE model=?"
            args.append(model)
        q += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        out = []
        for r in self.db.execute(q, args).fetchall():
            d = dict(r)
            d["rows"] = json.loads(d.pop("rows_json"))
            out.append(d)
        return out

    def live_instances(self, provider: str) -> set[str]:
        """Instance ids the ledger believes are not finished."""
        settled = sorted(TERMINAL | {"verified", "delivered", "lost",
                                     "unchanged"})
        q = ("SELECT instance_id FROM runs WHERE provider=? "
             "AND state NOT IN (%s) AND instance_id != ''"
             % ",".join("?" * len(settled)))
        rows = self.db.execute(q, (provider, *settled)).fetchall()
        return {r["instance_id"] for r in rows}
