"""wandb-lite (SPEC §14): progress snapshots + history from lane artifacts.

No new data plane: the in-pod agent already parses run.log live and
heartbeats status.json onto the lane (state/step/total/loss/epoch/lr),
and finalize uploads run.log + manifest. This module joins those into
one view — adding orchestrator-side context (cost, ETA) the pod cannot
know — and reads the durable history trail (progress.jsonl) that the
last-writer-wins heartbeat overwrites.
"""
from __future__ import annotations

import ast
import json
import time
from pathlib import Path

from . import storage
from .config import ROOT
from .ledger import Ledger

TERMINAL = ("done", "failed", "cancelled")

# axolotl logs one python-dict per step: {'loss': '...', 'epoch': ...}
_METRIC_KEYS = ("loss", "epoch", "learning_rate", "ppl",
                "tokens/trainable", "tokens/train_per_sec_per_gpu",
                "tokens/total")


def lane_getter(provider: str):
    """Read side of the lane that carried this run (stage/train both)."""
    if provider == "local":
        return storage.host_get
    vol = storage.vol_cfg()
    if vol:
        return storage.vol_get
    return storage.get_object


def _loads(raw):
    if raw is None:
        return None
    try:
        return json.loads(raw.decode() if isinstance(raw, (bytes, bytearray))
                          else raw)
    except (ValueError, UnicodeDecodeError):
        return None


def status(run_id: str, provider: str = "runpod"):
    return _loads(lane_getter(provider)(f"runs/{run_id}/status.json"))


def history(run_id: str, provider: str = "runpod", limit: int = 0):
    blob = lane_getter(provider)(f"runs/{run_id}/progress.jsonl")
    if not blob:
        return []
    if isinstance(blob, (bytes, bytearray)):
        blob = blob.decode(errors="replace")
    rows = []
    for line in blob.splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows[-limit:] if limit else rows


def snapshot(run_id: str) -> dict:
    """One joined view: ledger truth + live heartbeat + cost context."""
    led = Ledger(ROOT / "state" / "ledger.db")
    row = led.get(run_id)
    if not row:
        return {"run_id": run_id, "error": "unknown run"}
    st = status(run_id, row["provider"]) or {}
    hist = history(run_id, row["provider"], limit=1)
    out = {"run_id": run_id, "kind": row["kind"], "state": row["state"],
           "live": st, "last": hist[-1] if hist else None}
    step, total = st.get("step"), st.get("total")
    if step is not None and total:
        out["pct"] = round(100.0 * step / total, 1)
    if row["state"] in TERMINAL:
        man = _loads(lane_getter(row["provider"])(
            f"runs/{run_id}/manifest.json")) or {}
        dur = (man.get("finished_at", 0) or 0) - (man.get("started_at", 0) or 0)
        if dur:
            out["pod_seconds"] = round(dur)
    return out


def parse_runlog(text: str) -> list[dict]:
    """Full loss curve from a run.log (axolotl dict lines)."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if "'loss'" not in line and "'train_runtime'" not in line:
            continue
        try:
            d = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            continue
        if isinstance(d, dict):
            rec = {k: d[k] for k in _METRIC_KEYS if k in d}
            if "loss" not in rec and "train_loss" in d:
                rec["loss"] = d["train_loss"]   # final summary line
            if rec:
                out.append(rec)
    return out


def runlog_local(run_id: str):
    p = ROOT / "runs" / run_id / "run.log"
    return p.read_text(errors="replace") if p.is_file() else None


def fmt_line(snap: dict) -> str:
    """One human line for `qtrain watch`."""
    lv = snap.get("live") or {}
    last = snap.get("last") or {}
    seg = [f"[{snap.get('state') or '?'}]"]
    if lv.get("step") is not None:
        pct = snap.get("pct")
        seg.append(f"step {lv['step']}/{lv.get('total') or '?'}"
                   + (f" ({pct}%)" if pct is not None else ""))
    if lv.get("loss") is not None:
        seg.append(f"loss {lv['loss']}")
    if last.get("cost") is not None:
        seg.append(f"${last['cost']}")
    return "  ".join(str(x) for x in seg)
