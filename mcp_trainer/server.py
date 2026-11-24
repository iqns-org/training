"""mcp-trainer — cluster MCP server exposing the qtrain capability set
(SPEC §4 mcp layer, v0.3.0). Fleet port 3022, bearer-gated.

Tools mirror `qtrain` 1:1 so any MCP client (desktop agents, the mcp-agent
fleet service) can drive training without shell access. submit_training is
NON-BLOCKING: it returns the run_id the moment the ledger registers it and
the full cycle proceeds on a worker thread; job_status reads truth from
the ledger + lane heartbeat.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

from fastmcp import FastMCP

from orchestrator import dispatch, notify, progress, storage
from orchestrator.config import ROOT, load_dotenv
from orchestrator.ledger import Ledger
from orchestrator.providers import get_provider

from . import __version__
from .auth import BearerGate

mcp = FastMCP(
    "trainer",
    instructions=(
        "Automated LoRA/QLoRA fine-tuning on rented GPUs. Typical flow: "
        "list_recipes -> gpu_catalog/preflight_report -> submit_training "
        "(returns run_id) -> run_progress/job_status (poll; heartbeat "
        "mid-run) -> artifacts land in runs/<id>/ on the PVC lane + one "
        "rollup line in runs/index.jsonl (list_runs reads that index — the "
        "UI-facing table). bench_models scores any OpenAI-compatible "
        "endpoint. terminate_pod / reap_pods / janitor_pass manage "
        "capacity; budget_report watches spend."),
)

LEDGER = "state/ledger.db"


def _led() -> Ledger:
    return Ledger(ROOT / LEDGER)


def _lane_get(key: str, provider: str = "runpod") -> bytes | None:
    if provider == "local":
        return storage.host_get(key)
    vol = storage.vol_cfg()
    return storage.vol_get(key) if vol else storage.get_object(key)


# ------------------------------------------------------------------ tools

@mcp.tool
def gpu_catalog(gpus: list[str] | None = None, tier: str | None = None,
                provider: str = "runpod") -> dict:
    """Live GPU offers, cheapest first. gpus: name filters (regex-ish
    substring); tier: 'community'|'secure'."""
    offers = get_provider(provider).catalog(gpus, tier)
    return {"offers": [{"gpu_id": o.gpu_id, "vram_gb": o.vram_gb,
                        "price_hr": o.price_hr, "secure": o.secure,
                        "availability": o.availability}
                       for o in offers[:20]]}


@mcp.tool
def list_recipes(domain: str | None = None) -> dict:
    """Training recipes in the repo (axolotl/<domain>/<model>.yaml)."""
    base = ROOT / "axolotl"
    out = []
    if base.is_dir():
        for d in sorted(base.iterdir()):
            if not d.is_dir() or (domain and d.name != domain):
                continue
            for f in sorted(d.glob("*.yaml")):
                out.append({"domain": d.name, "model": f.stem,
                            "recipe": str(f.relative_to(ROOT))})
    return {"recipes": out}


@mcp.tool
def submit_training(domain: str, recipe: str, mini: bool = True,
                    overrides: dict | None = None,
                    gpus: list[str] | None = None, tier: str | None = None,
                    max_runtime: int = 3600, max_cost: float | None = None,
                    dry: bool = False, provider: str = "runpod") -> dict:
    """Launch a full training cycle in the background; returns as soon as
    the run is registered (seconds). mini=True is a cheap lane-validation
    run (48 rows, 1 epoch). provider: runpod | local ($0 K8s Jobs).
    Budget: max_cost None = derived (runtime x rate); offers above the
    rate ceiling are never launched — waiting beats overbuying.
    Track with job_status(run_id)."""
    recipe_path = Path(recipe)
    if not recipe_path.is_absolute():
        recipe_path = ROOT / recipe_path
    if not recipe_path.is_file():
        return {"error": f"recipe not found: {recipe}"}
    holder: dict = {}
    gate = threading.Event()

    def _on_run_id(rid: str) -> None:
        holder["run_id"] = rid
        gate.set()

    def _bg() -> None:
        try:
            holder["result"] = dispatch.submit(
                domain, recipe_path, mini=mini, overrides=overrides or {},
                gpus=gpus, tier=tier, max_runtime=max_runtime,
                max_cost=max_cost, dry=dry, provider=provider,
                on_run_id=_on_run_id)
        except Exception as e:                          # noqa: BLE001
            holder["error"] = f"{type(e).__name__}: {e}"
        finally:
            gate.set()

    threading.Thread(target=_bg, daemon=True,
                     name="submit-training").start()
    deadline = time.time() + 240
    while time.time() < deadline:
        if "run_id" in holder:
            return {"run_id": holder["run_id"], "state": "submitted",
                    "note": "full cycle continues in background; poll "
                            "job_status()"}
        if "error" in holder or "result" in holder:      # failed pre-ledger
            return {"error": holder.get("error"),
                    "result": holder.get("result")}
        time.sleep(0.25)
    return {"error": "staging did not register a run within 240s"}


@mcp.tool
def job_status(run_id: str) -> dict:
    """Truth for one run: ledger state + live heartbeat (mid-run) +
    terminal manifest summary when done."""
    row = next((r for r in _led().list()
                if r["run_id"] == run_id), None)
    if row is None:
        return {"error": f"no such run: {run_id}"}
    out = {"run_id": run_id, "state": row["state"], "kind": row["kind"],
           "instance_id": row["instance_id"],
           "artifact_uri": row["artifact_uri"], "error": row["error"],
           "updated_at": row["updated_at"]}
    try:
        prov = row["provider"] or "runpod"
        hb = _lane_get(f"runs/{run_id}/status.json", prov)
        if hb:
            out["heartbeat"] = json.loads(hb)
        man = _lane_get(f"runs/{run_id}/manifest.json", prov)
        if man:
            m = json.loads(man)
            out["manifest"] = {
                "status": m.get("status"), "training": m.get("training"),
                "artifacts": sorted(m.get("artifacts", {}))[:12],
                "problems": m.get("problems")}
    except Exception as e:                              # noqa: BLE001
        out["lane_error"] = str(e)[:200]
    return out


@mcp.tool
def submit_stage(kind: str, from_run: str, base_model: str | None = None,
                 provider: str = "runpod", max_runtime: int = 3600,
                 max_cost: float | None = None, dry: bool = False) -> dict:
    """Post-process a finished run: kind merge (needs base_model) | onnx
    | int8 | awq (GPU; calibrates on the source run's dataset). Cloud
    (runpod) by default; provider=local is the opt-in $0 fallback (cpu
    kinds only). Budget defaults are derived per stage (estimate = rate x
    est-minutes x margin); the rate ceiling refuses oversized cards.
    Non-blocking; track with job_status(). Stage outputs chain as the
    next stage's from_run."""
    holder: dict = {}
    gate = threading.Event()

    def _on_run_id(rid: str) -> None:
        holder["run_id"] = rid
        gate.set()

    def _bg() -> None:
        try:
            holder["result"] = dispatch.submit_stage(
                kind, from_run, base_model=base_model, provider=provider,
                max_runtime=max_runtime, max_cost=max_cost, dry=dry,
                on_run_id=_on_run_id)
        except Exception as e:                          # noqa: BLE001
            holder["error"] = f"{type(e).__name__}: {e}"
        finally:
            gate.set()

    threading.Thread(target=_bg, daemon=True, name="submit-stage").start()
    deadline = time.time() + 60
    while time.time() < deadline:
        if "run_id" in holder:
            return {"run_id": holder["run_id"], "state": "submitted"}
        if "error" in holder or "result" in holder:
            return {"error": holder.get("error"), "result": holder.get("result")}
        time.sleep(0.25)
    return {"error": "stage did not register within 60s"}


def row_provider(run_id: str) -> str:
    row = next((r for r in _led().list() if r["run_id"] == run_id), None)
    return (row or {}).get("provider") or "runpod"


@mcp.tool
def job_logs(run_id: str, tail: int = 100) -> dict:
    """Training log: prefer local harvest (runs/<id>/run.log), else the
    lane copy. tail = last N lines."""
    local = ROOT / "runs" / run_id / "run.log"
    raw = local.read_bytes() if local.is_file() else _lane_get(
        f"runs/{run_id}/run.log", row_provider(run_id))
    if raw is None:
        return {"error": "no run.log on disk or lane (still running on an "
                         "older image? heartbeat has progress meanwhile)"}
    lines = raw.decode(errors="replace").splitlines()
    return {"run_id": run_id, "lines": lines[-tail:]}


@mcp.tool
def list_runs(limit: int = 20) -> dict:
    """Recent runs: lane index first (UI-facing, cross-controller truth),
    local ledger as fallback for runs the lane never saw."""
    try:
        from orchestrator import progress as prog
        rows = prog.read_index("runpod", limit=limit)
        if rows:
            return {"source": "lane_index", "runs": list(reversed(rows))}
    except Exception:                              # noqa: BLE001
        pass
    rows = _led().list()[-limit:]
    return {"source": "ledger", "runs": [{"run_id": r["run_id"], "state": r["state"],
                      "provider": r["provider"], "instance_id": r["instance_id"],
                      "artifact_uri": r["artifact_uri"], "error": r["error"]}
                     for r in rows]}


@mcp.tool
def list_pods(provider: str = "runpod") -> dict:
    """Live provider instances (capacity census; compare with list_runs)."""
    return {"pods": [{"id": i.id, "status": i.status, "name": i.name,
                      "gpu": i.gpu_type, "price_hr": i.price_hr}
                     for i in get_provider(provider).list_instances()]}


@mcp.tool
def terminate_pod(instance_id: str, provider: str = "runpod") -> dict:
    """Force-terminate one instance (budget emergency / orphan cleanup)."""
    return {"id": instance_id,
            "terminated": get_provider(provider).terminate(instance_id)}


@mcp.tool
def doctor(provider: str = "runpod") -> dict:
    """Orphan sweep: live pods with no ledger claim (billable leaks)."""
    live = {i.id: i for i in get_provider(provider).list_instances()}
    claimed = {r["instance_id"] for r in _led().list()
               if r["instance_id"]}
    orphans = [{"id": i, "gpu": p.gpu_type, "price_hr": p.price_hr}
               for i, p in live.items() if i not in claimed]
    return {"live_pods": len(live), "orphans": orphans}


@mcp.tool
def refresh_corpora() -> dict:
    """git pull the corpora checkout (new data -> fresh expansion next
    submit). CORPORA_DIR must point at a git clone."""
    d = os.environ.get("CORPORA_DIR", "")
    if not d or not (Path(d) / ".git").is_dir():
        return {"error": f"CORPORA_DIR={d!r} is not a git checkout"}
    r = subprocess.run(["git", "-C", d, "pull", "--ff-only"],
                       capture_output=True, text=True, timeout=120)
    return {"rc": r.returncode, "out": (r.stdout or r.stderr).strip()[:300]}


@mcp.tool
def reconcile(run_id: str | None = None) -> dict:
    """Re-attach the orchestrator to an OPEN run (state not terminal) and
    finish it: poll lane -> harvest -> terminate instance. Use after a
    crash/restart dropped the watch loop; run_id None = most recent open."""
    holder: dict = {}
    gate = threading.Event()

    def _bg() -> None:
        try:
            holder["result"] = dispatch.resume(run_id=run_id)
        except Exception as e:                          # noqa: BLE001
            holder["error"] = f"{type(e).__name__}: {e}"
        finally:
            gate.set()

    threading.Thread(target=_bg, daemon=True, name="reconcile").start()
    gate.wait(15)
    if "result" in holder and holder["result"].get("resumed") is None:
        return holder["result"]                          # no open runs: fast
    if "error" in holder:
        return {"error": holder["error"]}
    return {"state": "reconciling",
            "note": "watch resumed in background; poll job_status()",
            "run_id": run_id or "most-recent-open"}


def resume_open_runs() -> int:
    """Boot hook: fleet restarts mid-job must not orphan runs. Any ledger
    row in an open state gets a watcher thread (harvest + terminate)."""
    try:
        rows = [r for r in _led().list() if r["state"] in dispatch.OPEN_STATES]
    except Exception:                                    # noqa: BLE001
        return 0
    for row in rows:
        threading.Thread(target=dispatch.resume,
                         kwargs={"run_id": row["run_id"]}, daemon=True,
                         name=f"resume-{row['run_id'][-6:]}").start()
        print(f"mcp-trainer: resumed watch for open run {row['run_id']} "
              f"(state={row['state']})", flush=True)
    return len(rows)


@mcp.tool
def run_progress(run_id: str) -> dict:
    """wandb-lite view (SPEC 14): ledger state + live heartbeat + durable
    progress history + recent lifecycle events. Cheap to poll; the
    heartbeat (status.json) is in-pod truth, progress.jsonl is the trail."""
    snap = progress.snapshot(run_id)
    snap["events"] = notify.recent(limit=12, run_id=run_id)
    return snap


@mcp.tool
def wait_for_events(since: float = 0.0, timeout: int = 300,
                   run_id: str | None = None) -> dict:
    """Long-poll the lifecycle feed: blocks until events newer than `since`
    (a prior response's cursor) appear, or timeout. MCP clients get
    notification semantics without busy-polling job_status; webhooks
    (QTRAIN_WEBHOOK_URL) carry the same events across hosts."""
    deadline = time.time() + min(max(timeout, 0), 600)
    while True:
        fresh = [e for e in notify.recent(limit=100, run_id=run_id)
                 if e["ts"] > since]
        if fresh or time.time() >= deadline:
            return {"events": fresh,
                    "cursor": fresh[-1]["ts"] if fresh else since,
                    "timed_out": not fresh}
        time.sleep(2)


@mcp.resource("trainer://runs/{run_id}/progress")
def progress_resource(run_id: str) -> str:
    """Progress snapshot as a subscribable resource: clients may
    resources/subscribe and receive updated notifications when this
    server's watch loop emits them."""
    return json.dumps(progress.snapshot(run_id), default=str)


@mcp.tool
def reap_pods(provider: str = "runpod", dry: bool = True) -> dict:
    """Idempotency hygiene: terminate live GPUs with no active task
    behind them (terminal claims, stale run-suffix pods). DRY by
    default; only trainer-* named pods are ever candidates; unclaimed
    pods are kept unless QTRAIN_REAP_UNCLAIMED=1 declares a single
    controller."""
    return dispatch.reap(provider, dry=dry)


@mcp.tool
def janitor_pass(provider: str = "runpod", dry: bool = True) -> dict:
    """One full idempotency sweep: reap stray pods + relaunch stale
    pending-but-never-launched runs (crashed dispatchers)."""
    return dispatch.janitor_once(provider, dry=dry)


@mcp.tool
def preflight_report(kind: str = "train", provider: str = "runpod",
                     max_price_hr: float | None = None,
                     max_runtime: int = 3600) -> dict:
    """Read-only launch decision (SPEC 13p): fitting cards, cost estimate,
    kill threshold, budget gate verdicts. Launches nothing, writes
    nothing — answers 'what would this cost, on what' before committing."""
    return dispatch.preflight(kind, provider=provider,
                              max_price_hr=max_price_hr,
                              max_runtime=max_runtime)


@mcp.tool
def budget_report(provider: str = "runpod") -> dict:
    """Spend visibility: live balance, open-run projections vs the cap,
    per-kind observed medians (self-calibrated from real runs)."""
    from orchestrator import budget as bud
    led = _led()
    bal = None
    if provider == "runpod":
        try:
            bal = get_provider(provider).balance()
        except Exception:                          # noqa: BLE001
            pass
    opens = [r for r in led.list() if r["state"] in bud.BILLED]
    obs = bud._obs()
    import statistics
    return {"balance": bal, "reserve": bud.RESERVE, "budget_total": bud.TOTAL,
            "margin": bud.MARGIN,
            "open_runs": [{"run_id": r["run_id"], "kind": r["kind"],
                           "state": r["state"],
                           "est_cost": round(r.get("est_cost") or 0, 3)}
                          for r in opens],
            "open_projected": round(sum(r.get("est_cost") or 0
                                        for r in opens), 2),
            "observed_median_minutes": {k: round(statistics.median(v), 1)
                                        for k, v in obs.items() if v}}


@mcp.tool
def bench_models(urls: list[str], repeats: int = 2, timeout: int = 90,
                 max_tokens: int = 64) -> dict:
    """Question-matrix benchmark against OpenAI-compatible endpoints
    (identity/feeling/voice/instruction/json/world + rule checks, TTFT +
    tok/s from the stream, models auto-discovered via /v1/models).
    Returns a markdown table + raw rows — run it on the facade URL to
    cover every model behind the router in one pass."""
    from orchestrator import bench
    cfg = {"urls": urls, "repeats": repeats, "timeout": timeout,
           "max_tokens": max_tokens}
    rows = bench.run(cfg)
    return {"table": bench.table(rows), "rows": rows}


@mcp.tool
def rebuild_run_index(provider: str = "runpod") -> dict:
    """Rebuild runs/index.jsonl on the PVC lane from locally kept copies
    (manifests + run.log + ledger). Idempotent; preserves index lines for
    runs this controller never held. The UI's list view is that one
    object — see SPEC 14c."""
    return progress.reindex(provider)


# ------------------------------------------------------------------ serve

def build_app():
    app = mcp.http_app(path="/mcp")
    # gate ALWAYS wraps: refuse-all is the safe default without a token
    app.add_middleware(BearerGate)
    return app


def _port() -> int:
    raw = os.environ.get("MCP_TRAINER_PORT", "3022")
    try:
        return int(raw)
    except ValueError:      # e.g. k8s service-link tcp://... if re-enabled
        return 3022


def _janitor_loop() -> None:
    """Periodic reap+relaunch sweep for controllers that own the lane.
    OFF unless QTRAIN_JANITOR=1: the server ledger is pod-local, so
    enabling it is a declaration that this process is THE janitor."""
    interval = int(os.environ.get("QTRAIN_JANITOR_INTERVAL", "900"))
    while True:
        try:
            out = dispatch.janitor_once(dry=False)
            interesting = {k: out.get(k) for k in ("reaped", "relaunched")
                           if out.get(k)}
            if interesting:
                print("qtrain-janitor "
                      + json.dumps(interesting, default=str)[:600],
                      flush=True)
        except Exception as e:                              # noqa: BLE001
            print(f"qtrain-janitor error: {e}", flush=True)
        time.sleep(interval)


def main(argv=None) -> int:
    import uvicorn
    load_dotenv()
    # events -> container log as machine-parseable lines: `kubectl logs -f
    # mcp-trainer` becomes a live feed for humans and cluster tooling
    notify.register_listener(
        lambda e: print("qtrain-event "
                        + json.dumps(e, default=str), flush=True))
    if not os.environ.get("MCP_TRAINER_TOKEN"):
        print("WARNING: MCP_TRAINER_TOKEN unset — REFUSING all tool traffic "
              "except /healthz (set MCP_TRAINER_ALLOW_OPEN=1 for local dev)",
              flush=True)
    port = _port()
    n = resume_open_runs()
    if n:
        print(f"mcp-trainer: {n} open run(s) re-attached", flush=True)
    if os.environ.get("QTRAIN_JANITOR", "") == "1":
        threading.Thread(target=_janitor_loop, daemon=True,
                         name="janitor").start()
        print("mcp-trainer: janitor loop active "
              f"(every {os.environ.get('QTRAIN_JANITOR_INTERVAL', '900')}s)",
              flush=True)
    print(f"mcp-trainer {__version__} listening :{port} (/mcp, /healthz)",
          flush=True)
    uvicorn.run(build_app(), host="0.0.0.0", port=port, log_level="info")
    return 0
