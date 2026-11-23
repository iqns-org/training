"""Run-lifecycle event bus (SPEC §14) — one emit(), three sinks.

  1. JOURNAL   state/notifications.jsonl — always. The offline-visible
               source of truth; anything that can tail a file ("and
               others") can consume it.
  2. WEBHOOK   env QTRAIN_WEBHOOK_URL — JSON POST per event, 5s timeout,
               fired on a daemon thread so a slow/dead receiver can
               never stall or kill a run. The "others" channel: gatus,
               ntfy, slack bridges, whatever speaks POST.
  3. LISTENERS in-process callables — for hosts that want push without
               polling. mcp-trainer registers one and bridges events to
               MCP clients via notifications/message + progress tokens.

Stdlib-only: the orchestrator venv is deliberately lean (no httpx here;
urllib in a thread is enough for best-effort fire-and-forget).
"""
from __future__ import annotations

import atexit
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

from .config import ROOT, load_dotenv

JOURNAL = ROOT / "state" / "notifications.jsonl"

_LISTENERS: list = []
_LOCK = threading.Lock()
_PENDING: set = set()          # live webhook threads (short-lived CLI
                               # processes must not drop events at exit)


def webhook_url() -> str:
    load_dotenv()
    return os.environ.get("QTRAIN_WEBHOOK_URL", "").strip()


def register_listener(fn) -> None:
    with _LOCK:
        _LISTENERS.append(fn)


def unregister_listener(fn) -> None:
    with _LOCK:
        if fn in _LISTENERS:
            _LISTENERS.remove(fn)


def _post(url: str, body: bytes) -> None:
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5):
            pass
    except Exception:                                  # noqa: BLE001
        pass        # best-effort by contract: receivers are never critical


def emit(kind: str, run_id: str = "", **fields) -> dict:
    """Fan out one event. NEVER raises — a broken sink cannot break a run."""
    event = {"ts": round(time.time(), 3), "kind": kind, "run_id": run_id,
             **fields}
    try:
        JOURNAL.parent.mkdir(parents=True, exist_ok=True)
        with open(JOURNAL, "a") as f:
            f.write(json.dumps(event) + "\n")
    except OSError:
        pass
    url = webhook_url()
    if url:
        def _send():
            try:
                _post(url, json.dumps(event).encode())
            finally:
                _PENDING.discard(_t)
        _t = threading.Thread(target=_send, daemon=True, name="qtrain-webhook")
        _PENDING.add(_t)
        _t.start()
    with _LOCK:
        fns = list(_LISTENERS)
    for fn in fns:
        try:
            fn(event)
        except Exception:                              # noqa: BLE001
            pass
    return event


def flush(timeout: float = 5.0) -> None:
    """Join in-flight webhook POSTs before the process dies — otherwise a
    short-lived CLI exit (capacity refusal -> SystemExit) silently drops
    the very failure events receivers most want. Daemon threads make this
    a bounded wait, never a hang."""
    deadline = time.time() + timeout
    for t in list(_PENDING):
        t.join(max(0.05, deadline - time.time()))


atexit.register(flush)


def deliver_sync(event: dict) -> tuple[bool, str]:
    """Synchronous webhook probe for `qtrain notify-test` — the only path
    that surfaces delivery errors instead of swallowing them."""
    url = webhook_url()
    if not url:
        return False, "QTRAIN_WEBHOOK_URL not set"
    req = urllib.request.Request(
        url, data=json.dumps(event).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return True, f"HTTP {r.status}"
    except Exception as e:                             # noqa: BLE001
        return False, str(e)[:200]


def recent(limit: int = 20, run_id: str | None = None) -> list[dict]:
    """Tail the journal (newest last). Malformed lines are skipped."""
    if not JOURNAL.is_file():
        return []
    rows = []
    for line in JOURNAL.read_text(errors="replace").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if run_id and e.get("run_id") != run_id:
            continue
        rows.append(e)
    return rows[-limit:]
