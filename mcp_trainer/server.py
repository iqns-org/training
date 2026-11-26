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
from orchestrator.config import ROOT, corpora_dir, load_dotenv, resolve_recipe
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
    """Training recipes — task config lives in corpora (SPEC §4b boundary):
    $CORPORA_DIR/recipes/<domain>/<model>.yaml, with the legacy in-repo
    axolotl/<domain>/ as read-fallback until fully retired."""
    out = []
    bases = [(corpora_dir() / "recipes", ""), (ROOT / "axolotl", "axolotl/")]
    for base, prefix in bases:
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir() or (domain and d.name != domain):
                continue
            for f in sorted(d.glob("*.yaml")):
                out.append({"domain": d.name, "model": f.stem,
                            "recipe": prefix + f.relative_to(base).as_posix()})
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
    recipe_path = resolve_recipe(recipe)
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
def verify(run_id: str | None = None, live: bool = False,
           recompute: bool = False) -> dict:
    """Sanity check one run (default: most recent): does what the pod
    TRAINED ON match what we THINK it trained on? Cross-checks ledger
    record -> lane payload -> agent-attested hashes (dataset sha/lines,
    config, exact image), plus corpora/recipe drift. 'ok' is False iff a
    leg disagrees. recompute re-expands the domain first (slow, exact).
    """
    from orchestrator import provenance
    return provenance.verify_run(run_id, live=live, recompute=recompute)


@mcp.tool
def families(base_model: str | None = None) -> dict:
    """Family capability registry (SPEC 13al): what serving targets each
    base-model family supports, as gated data with run-id evidence. With
    base_model: that family's verdicts; without: the whole registry."""
    from orchestrator import families as fam
    if base_model:
        f = fam.family_for(base_model)
        return {"base_model": base_model, "family": f[0] if f else None,
                "targets": f[1].get("targets") if f else None,
                "notes": f[1].get("notes") if f else None,
                "refusals": {k: v for k, v in (f[1].get("targets") or {}).items()
                             if str(v).lower() == "false"} if f else {}}
    return {"families": fam.registry()}


@mcp.tool
def eval_history(model: str | None = None, limit: int = 20) -> dict:
    """Recorded bench scores (SPEC 13am): the cross-round table — every
    `qtrain bench --record` lands here keyed by (model, host, run)."""
    return {"evals": _led().list_evals(model=model, limit=limit)}


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
    default; only pods OUR naming scheme made (kind-first train-/awq/…,
    or legacy trainer-) are ever candidates; unclaimed
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


# ------------------------------------------------------- synthesis (FRAMEWORK §10)

_SYNTH = None
_SYNTH_JOBS: dict = {}


def _synth():
    """Lazy-load src/synth.py (the deterministic IR synthesis engine —
    zero LLM, every artifact oracle-gated before it lands)."""
    global _SYNTH
    if _SYNTH is None:
        import importlib.machinery
        import importlib.util
        path = ROOT / "src" / "synth.py"
        spec = importlib.util.spec_from_file_location(
            "synth", str(path),
            loader=importlib.machinery.SourceFileLoader("synth", str(path)))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SYNTH = mod
    return _SYNTH


def _synth_run(argv: list[str]) -> dict:
    """Drive synth.main() and return its JSON report (stdout contract)."""
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        _synth().main(argv)
    out = buf.getvalue().strip()
    try:
        return json.loads(out)
    except Exception:                                    # noqa: BLE001
        return {"raw": out[:2000]}


@mcp.tool
def synth_derive(codes: list[str] | None = None, systems: list[str] | None = None,
                 per_code: int = 4, limit: int = 0, background: bool = True,
                 out: str | None = None, tmp: str | None = None) -> dict:
    """P2/P23 mutation synthesis: valid IR → exactly-one-error mutants
    (15-rule rulebook, E1–E7), emitted as derived invalid-program/repair
    templates + dpo.jsonl preference pairs. Every template is preflighted
    by the Pass-1 checker + fixture executor before it lands — yield
    report in stats. background=True (default): full-corpus runs take
    minutes; poll synth_status(job_id)."""
    argv = ["derive", "--per-code", str(per_code), "--limit", str(limit)]
    if codes:
        argv += ["--codes", ",".join(codes)]
    if systems:
        argv += ["--systems", ",".join(systems)]
    if out:
        argv += ["--out", out]
    if tmp:
        argv += ["--tmp", tmp]
    if not background:
        return _synth_run(argv)
    job = f"synth-{int(time.time()) % 100000:05d}"

    def _bg() -> None:
        try:
            _SYNTH_JOBS[job] = {"state": "done", "stats": _synth_run(argv)}
        except Exception as e:                           # noqa: BLE001
            _SYNTH_JOBS[job] = {"state": "error", "error": str(e)[:400]}

    _SYNTH_JOBS[job] = {"state": "running", "argv": argv}
    threading.Thread(target=_bg, daemon=True, name=job).start()
    return {"job": job, "state": "running",
            "note": "poll synth_status(job)"}


@mcp.tool
def synth_abstain(per_code: int = 2, out: str | None = None) -> dict:
    """P12: NOT-DERIVABLE rows — drop one fact from nl-to-ir graphs and
    emit a refusal row only where the fixture executor proves the program
    can no longer derive the answer (0 rows). Anti-hallucination data."""
    argv = ["abstain", "--per-code", str(per_code)]
    if out:
        argv += ["--out", out]
    return _synth_run(argv)


@mcp.tool
def synth_graph(ttl_path: str, domain: str | None = None,
                max_facts: int = 50, out: str | None = None) -> dict:
    """P4/P6/P12 over REAL knowledge: a .ttl file (or directory of them,
    e.g. rmcp-extract funnel output) becomes graph QA / hop-ladder /
    abstention scenario templates — every answer is RECOMPUTED by the
    closed-world fixture executor, never typed. This is the knowledge↔
    cognition bridge: the same graph teaches facts and reasoning."""
    if not Path(ttl_path).exists():
        return {"error": f"no such ttl path: {ttl_path}"}
    argv = ["graph", ttl_path, "--max-facts", str(max_facts)]
    if domain:
        argv += ["--domain", domain]
    if out:
        argv += ["--out", out]
    return _synth_run(argv)


@mcp.tool
def synth_audit() -> dict:
    """P28/P29/P31 corpus health: template coverage by task family,
    phrasing-entropy ratios (combinatorial-vs-semantic variety),
    final-IR near-duplicate pairs, derived inventory."""
    return _synth_run(["audit"])


@mcp.tool
def synth_manifest(out: str | None = None) -> dict:
    """Phase-0 kernel lineage: content-address every derived template +
    the oracle source digests (FRAMEWORK L3: checker changes invalidate
    downstream rows). Writes derived/manifest.json, returns derived@h."""
    argv = ["manifest"]
    if out:
        argv += ["--out", out]
    return _synth_run(argv)


@mcp.tool
def synth_bundle(domain: str = "cognition", out: str | None = None) -> dict:
    """Close the synthesis→training lane: expand all derived templates into
    $CORPORA_DIR/datasets/<domain>/*.jsonl (diagnosis/repair/abstain/graph)
    + preference.jsonl, and report the dataset@hash ids the qtrain pipeline
    consumes. Then submit_training(domain=..., recipe=...)."""
    argv = ["bundle", "--domain", domain]
    if out:
        argv += ["--out", out]
    return _synth_run(argv)


@mcp.tool
def synth_status(job: str) -> dict:
    """Result of a background synth job (synth_derive). Also lists live
    jobs when job is empty."""
    if not job:
        return {"jobs": {k: v.get("state") for k, v in _SYNTH_JOBS.items()}}
    return {"job": job, **_SYNTH_JOBS.get(job, {"state": "unknown"})}


@mcp.tool
def synth_sample(url: str, model: str = "kandi", k: int = 4, limit: int = 50,
                 task: str | None = None, dataset: str | None = None,
                 temperature: float = 0.8, max_tokens: int = 512) -> dict:
    """R4 student-side RFT: prompts from derived templates, completions
    from a served OpenAI-compatible endpoint, kept only if the fleet
    oracle accepts them (check_record per task family). The recursion
    half: student output becomes data only through the verifier."""
    return _synth_run(["sample", "--url", url, "--model", model,
                       "--k", str(k), "--limit", str(limit),
                       "--temperature", str(temperature),
                       "--max-tokens", str(max_tokens)]
                      + (["--task", task] if task else [])
                      + (["--dataset", dataset] if dataset else []))


@mcp.tool
def synth_mix(sources: str, name: str, rows: int = 4000, seed: int = 7) -> dict:
    """R5 data-mixing regression primitive: weighted merge of PREPARED
    datasets (run prepare-domain per source) into datasets/<name>/ with a
    seeded, content-addressed manifest. The mixing-law experiment tool."""
    return _synth_run(["mix", "--sources", sources, "--rows", str(rows),
                       "--seed", str(seed), "--name", name])


@mcp.tool
def synth_grade(path: str, out: str | None = None) -> dict:
    """R7-ready RLVR reward oracle: grade a jsonl of {messages} rows
    against the derived templates' fleet contract — (s, r) pairs WITHOUT
    model weights. This is the exact reward function the GRPO loop will
    reuse."""
    argv = ["grade", path]
    if out:
        argv += ["--out", out]
    return _synth_run(argv)


@mcp.tool
def synth_gaps(bench_path: str) -> dict:
    """P7/R10 gap analysis: failed bench questions -> targeted synthesis
    proposals (exact bin/synth commands). The outer loop's propose step."""
    return _synth_run(["gaps", "--bench", bench_path])


@mcp.tool
def synth_gbnf(out: str | None = None) -> dict:
    """R3: emit the Graph IR GBNF grammar (one rule per line, llama.cpp/
    vLLM compatible) for guided decoding. Syntax becomes a decode-time
    guarantee; serve with --guided-decoding-backend or sampling_params
    extra_body guided_grammar."""
    return _synth_run(["gbnf", "--out", out] if out else ["gbnf"])


@mcp.tool
def synth_merge(base: str, adapters: str, method: str = "ties",
                density: float = 0.5) -> dict:
    """R6 merge-algebra config generator (mergekit: ties/dare_ties/
    slerp): compose persona x domain x cognition LoRA tracks into ONE
    merged adapter = single-LoRA serving. Returns the config; the
    execution rides the stage/pod lane."""
    return _synth_run(["merge", "--base", base, "--adapters", adapters,
                       "--method", method, "--density", str(density)])


@mcp.tool
def synth_contrasts(domain: str = "queer", behaviors: str | None = None,
                    limit: int = 400) -> dict:
    """R14 contrast-set factory (ORACLES.md §2): labeled doing/not-doing
    pairs from verified artifacts (ir-valid from DPO couples, abstain from
    refusal rows) and persona datasets via bench-grammar checks
    (self-name/emoji/json). This is the fuel the ablate/steering stage
    consumes — behaviors to remove, add, or dial."""
    argv = ["contrasts", "--domain", domain, "--limit", str(limit)]
    if behaviors:
        argv += ["--behaviors", behaviors]
    return _synth_run(argv)


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
