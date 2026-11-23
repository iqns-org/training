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

from orchestrator import dispatch, storage
from orchestrator.config import ROOT, load_dotenv
from orchestrator.ledger import Ledger
from orchestrator.providers import get_provider

from . import __version__
from .auth import BearerGate

mcp = FastMCP(
    "trainer",
    instructions=(
        "Automated LoRA/QLoRA fine-tuning on rented GPUs. Typical flow: "
        "list_recipes -> gpu_catalog -> submit_training (returns run_id) -> "
        "job_status (poll; heartbeat mid-run) -> artifacts land in runs/<id>/ "
        "and the ledger. terminate_pod / doctor manage capacity."),
)

LEDGER = "state/ledger.db"


def _led() -> Ledger:
    return Ledger(ROOT / LEDGER)


def _lane_get(key: str) -> bytes | None:
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
                    max_runtime: int = 3600, max_cost: float = 2.50,
                    dry: bool = False) -> dict:
    """Launch a full training cycle in the background; returns as soon as
    the run is registered (seconds). mini=True is a cheap lane-validation
    run (48 rows, 1 epoch). Track with job_status(run_id)."""
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
                max_cost=max_cost, dry=dry, on_run_id=_on_run_id)
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
        hb = _lane_get(f"runs/{run_id}/status.json")
        if hb:
            out["heartbeat"] = json.loads(hb)
        man = _lane_get(f"runs/{run_id}/manifest.json")
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
def job_logs(run_id: str, tail: int = 100) -> dict:
    """Training log: prefer local harvest (runs/<id>/run.log), else the
    lane copy. tail = last N lines."""
    local = ROOT / "runs" / run_id / "run.log"
    raw = local.read_bytes() if local.is_file() else _lane_get(
        f"runs/{run_id}/run.log")
    if raw is None:
        return {"error": "no run.log on disk or lane (still running on an "
                         "older image? heartbeat has progress meanwhile)"}
    lines = raw.decode(errors="replace").splitlines()
    return {"run_id": run_id, "lines": lines[-tail:]}


@mcp.tool
def list_runs(limit: int = 20) -> dict:
    """Recent ledger rows."""
    rows = _led().list()[-limit:]
    return {"runs": [{"run_id": r["run_id"], "state": r["state"],
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


# ------------------------------------------------------------------ serve

def build_app():
    from fastmcp.server.http import StarletteWithLifespan
    app: StarletteWithLifespan = mcp.http_app(path="/mcp")
    if os.environ.get("MCP_TRAINER_TOKEN"):
        app.add_middleware(BearerGate)
    return app


def main(argv=None) -> int:
    import uvicorn
    load_dotenv()
    if not os.environ.get("MCP_TRAINER_TOKEN"):
        print("WARNING: MCP_TRAINER_TOKEN unset — server is UNAUTHENTICATED "
              "(dev mode only)", flush=True)
    port = int(os.environ.get("MCP_TRAINER_PORT", "3022"))
    print(f"mcp-trainer {__version__} listening :{port} (/mcp, /healthz)",
          flush=True)
    uvicorn.run(build_app(), host="0.0.0.0", port=port, log_level="info")
    return 0
