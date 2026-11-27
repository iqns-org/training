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


def ping_age(run_id: str, provider: str) -> float | None:
    """Seconds since the last orchestrator ping on this run's lane, None
    if absent. THE deadman tell (r2 law, 2026-11-26): the pod SIGTERMs its
    own container when this exceeds the grace — and the pod's default
    runtime RESTARTS the container, so a silently-dead watcher becomes an
    infinite zombie-attempt billing loop, not an honest stop. Stale ping +
    `running` status = the watch hung while the pod trains on borrowed
    time. Surfacing it turns that death spiral into one glance."""
    try:
        raw = lane_getter(provider)(f"runs/{run_id}/ping")
    except Exception:
        return None
    if not raw:
        return None
    try:
        return round(time.time() - float(raw.decode().strip()), 1)
    except (ValueError, UnicodeDecodeError):
        return None


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
    if row["state"] not in TERMINAL and st.get("state") == "running":
        out["ping_age"] = ping_age(run_id, row["provider"])
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


# ---------------------------------------------------------------------------
# runs/index.jsonl — the UI-facing flat rollup (SPEC §14c). One line per
# finished run joining pod truth (manifest) with orchestrator truth the pod
# CANNOT know (provider, cost, spec, parsed log metrics). A front end lists
# runs with ONE list + one GET — no bucket scan, no N+1, no local sqlite.

INDEX_KEY = "runs/index.jsonl"


def lane_putter(provider: str):
    """Write side of the lane that carries run artifacts (mirror of
    lane_getter). Tests/CLI pass a real provider; best-effort callers
    swallow whatever the lane raises."""
    if provider == "local":
        return lambda k, d, ct="": storage.host_put(k, d)
    vol = storage.vol_cfg()
    if vol:
        return lambda k, d, ct="": storage.vol_put(k, d)
    return lambda k, d, ct="": storage.put_object(k, d, ct or
                                                  "application/x-ndjson")


def metrics_summary(log_text: str | None) -> dict:
    """First/last/min loss + point count from a run.log — so the terminal
    summary exists even when the (old) agent left manifest metrics null."""
    pts = parse_runlog(log_text or "")
    if not pts:
        return {"n_points": 0}
    out: dict = {"n_points": len(pts)}
    losses = [float(p["loss"]) for p in pts if p.get("loss") is not None]
    if losses:
        out.update(first_loss=losses[0], last_loss=losses[-1],
                   min_loss=min(losses))
    return out


def build_record(run_id: str, manifest: dict, row: dict | None = None,
                 *, provider: str = "runpod", seconds: float | None = None,
                 cost: float | None = None, price_hr: float | None = None,
                 log_text: str | None = None) -> dict:
    """One flat, UI-queryable rollup. Fixed keys, no nested blobs; the
    metric parse runs here so `reindex` of old runs yields the same shape
    as live-harvested ones."""
    spec: dict = {}
    if row and row.get("spec_json"):
        try:
            spec = json.loads(row["spec_json"])
        except (ValueError, TypeError):
            spec = {}
    man = manifest or {}
    tr = man.get("training") or {}
    hw = (man.get("hardware") or {}).get("gpus") or []
    started = man.get("started_at")
    fin = man.get("finished_at")
    if seconds is None and started and fin:
        seconds = fin - started
    kind = (row or {}).get("kind") or "train"
    if kind == "stage":
        kind = spec.get("kind") or "stage"     # awq/onnx/merge/int8 is truth
    rec = {
        "ts": round(time.time(), 1),
        "run_id": run_id,
        "kind": kind,
        "provider": (row or {}).get("provider") or provider,
        "status": man.get("status", "unknown"),
        "exit_code": tr.get("exit_code"),
        "image_tag": man.get("image_tag", ""),
        "agent_version": man.get("agent_version", ""),
        "started_at": started,
        "finished_at": fin,
        "seconds": round(seconds) if seconds is not None else None,
        "cost": cost,
        "est_cost": (row or {}).get("est_cost"),
        "price_hr": price_hr,
        "base_model": spec.get("base_model"),
        "from_run": spec.get("from_run"),
        "domain": spec.get("domain"),
        "recipe": spec.get("recipe"),
        "corpora_sha": spec.get("corpora"),
        "recipe_sha": spec.get("recipe_sha"),
        "gpu": (str(hw[0]).split(",")[0].strip() if hw else ""),
        "n_artifacts": len(man.get("artifacts") or {}),
        "artifacts_sha": man.get("artifacts_tar_sha256", ""),
        "n_problems": len(man.get("problems") or []),
    }
    rec.update(metrics_summary(log_text))
    return {k: v for k, v in rec.items()
            if v is not None and v != "" and v != []}


def append_index(record: dict, get_run, put_run) -> None:
    """RMW append (S3 has no append) — KB-small by construction: one line
    per finished run. Best-effort BY DESIGN: the index is the UI view, the
    lane objects remain the authoritative trail; a broken index sink must
    never affect a run (same law as _progress_tick)."""
    try:
        prev = get_run(INDEX_KEY) or b""
        if isinstance(prev, str):
            prev = prev.encode()
        put_run(INDEX_KEY, prev + (json.dumps(record) + "\n").encode())
    except Exception:                              # noqa: BLE001
        pass


def read_index(provider: str = "runpod", limit: int = 0) -> list[dict]:
    try:
        blob = lane_getter(provider)(INDEX_KEY)
    except Exception:                              # noqa: BLE001
        return []
    if not blob:
        return []
    if isinstance(blob, (bytes, bytearray)):
        blob = blob.decode(errors="replace")
    rows = []
    for line in blob.splitlines():
        try:
            d = json.loads(line)
            if isinstance(d, dict) and d.get("run_id"):
                rows.append(d)
        except ValueError:
            continue
    return rows[-limit:] if limit else rows


def reindex(provider: str = "runpod") -> dict:
    """Rebuild the lane index from locally kept copies (runs/*/manifest +
    run.log + ledger). Merge semantics: locally REBUILT records replace
    same-run_id lines (they carry parsed metrics); lane lines for runs this
    controller never held are preserved — idempotent, safe on repeat."""
    led = Ledger(ROOT / "state" / "ledger.db")
    get_run = lane_getter(provider)
    on_lane = {r["run_id"]: r for r in read_index(provider)}
    rebuilt: dict = {}
    for mpath in sorted((ROOT / "runs").glob("*/manifest.json")):
        rid = mpath.parent.name
        try:
            man = json.loads(mpath.read_text())
        except (ValueError, OSError):
            continue
        try:
            row = led.get(rid)
        except Exception:                          # noqa: BLE001
            row = None
        prov = (row or {}).get("provider") or provider
        try:
            price = float((row or {}).get("price_hr") or 0) or None
        except (TypeError, ValueError):
            price = None
        rebuilt[rid] = build_record(
            rid, man, row, provider=prov,
            price_hr=price, log_text=runlog_local(rid))
    merged = {**on_lane, **rebuilt}
    ordered = sorted(merged.values(),
                     key=lambda r: r.get("finished_at") or r.get("ts") or 0)
    blob = "".join(json.dumps(r) + "\n" for r in ordered).encode()
    try:
        lane_putter(provider)(INDEX_KEY, blob, "application/x-ndjson")
    except Exception as e:                         # noqa: BLE001
        return {"status": "error", "error": str(e)[:200],
                "rebuilt": len(rebuilt)}
    return {"status": "ok", "rebuilt": len(rebuilt),
            "preserved_from_lane": len(on_lane) - len(
                set(on_lane) & set(rebuilt)), "total": len(ordered)}


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
    pa = snap.get("ping_age")
    if pa is not None and pa > 420:      # 70% of the 600s grace — act now
        seg.append(f"PING-STALE {pa:.0f}s (deadman in {max(0, 600 - pa):.0f}s)")
    return "  ".join(str(x) for x in seg)
