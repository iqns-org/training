"""Job dispatch loop (SPEC §4/§6, P1): expand -> upload -> launch -> watch.

Two storage lanes, chosen per submit (SPEC §13 g):
  volume   — RunPod network volume mounted into the pod at RUNPOD_VOLUME_MOUNT
             (/data). Pod writes files, orchestrator reads them via the
             per-DC S3 API. Zero pod credentials, zero public cluster route,
             zero presigning. Manifest appearing == job finished.
  presign  — rustfs + presigned URLs: the portable lane for providers
             without volumes (gpuai, local). Same push semantics.

In both lanes the orchestrator learns completion by polling
runs/<id>/manifest.json; the return lane IS the ledger.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml

from . import budget, notify, storage
from .config import ROOT
from .ledger import Ledger
from .models import ProviderError
from .providers import get_provider

MINI_LINES = 48


def _tolerant_get(get_run, key: str, state: dict, limit: int = 5):
    """Lane reads during the watch loop can blip (network, provider API);
    tolerate a few consecutive failures before declaring the lane dead.
    `state` tracks the streak across loop iterations."""
    try:
        data = get_run(key)
        state["errors"] = 0
        return data
    except Exception:
        state["errors"] = state.get("errors", 0) + 1
        if state["errors"] >= limit:
            raise
        return None


def _progress_tick(run_id: str, get_run, put_run, price_hr, started: float,
                   memo: dict) -> None:
    """wandb-lite (SPEC 14): join the live heartbeat into a durable history
    row (progress.jsonl — S3 has no append, so read-modify-write a file
    that stays KB-small), and emit progress events on state change or a
    >=10% step jump. Everything here is best-effort BY DESIGN: a broken
    history sink must never affect the run; the heartbeat + manifest
    remain the authoritative signals."""
    sstate = memo.setdefault("status", {})
    raw = _tolerant_get(get_run, f"runs/{run_id}/status.json", sstate)
    if raw is None:
        return
    try:
        st = json.loads(raw.decode() if isinstance(raw, (bytes, bytearray))
                        else raw)
    except (ValueError, UnicodeDecodeError):
        return
    if not isinstance(st, dict):
        return
    row = {"ts": round(time.time(), 3)}
    for k in ("state", "step", "total", "loss", "epoch"):
        if st.get(k) is not None:
            row[k] = st[k]
    if price_hr and started:
        row["cost"] = round(price_hr * (time.time() - started) / 3600, 4)
    sig = (row.get("state"), row.get("step"), row.get("cost"))
    if sig == memo.get("sig"):
        return
    memo["sig"] = sig
    if put_run:
        try:
            key = f"runs/{run_id}/progress.jsonl"
            prev = get_run(key) or b""
            if isinstance(prev, str):
                prev = prev.encode()
            put_run(key, prev + (json.dumps(row) + "\n").encode())
        except Exception:                              # noqa: BLE001
            pass
    pct = (100.0 * row["step"] / row["total"]) if row.get("total") else None
    prev_pct = memo.get("pct")
    jumped = pct is not None and (prev_pct is None or abs(pct - prev_pct) >= 10)
    if row.get("state") != memo.get("state") or jumped:
        memo["pct"], memo["state"] = pct, row.get("state")
        notify.emit("progress", run_id,
                    pct=round(pct, 1) if pct is not None else None,
                    **{k: v for k, v in row.items() if k != "ts"})


def _keepalive_public_lane() -> None:
    """Quick tunnels idle-expire; during training the pod sends no traffic,
    so the orchestrator pings the public endpoint each poll cycle (presign
    lane only; no-op when public == LAN or volume lane)."""
    import httpx
    pub = os.environ.get("S3_PUBLIC_ENDPOINT", "").rstrip("/")
    lan = os.environ.get("S3_ENDPOINT", "").rstrip("/")
    if not pub or pub == lan:
        return
    try:
        httpx.get(f"{pub}/minio/health/live", timeout=10)
    except Exception:
        pass  # keepalive only — the presigned calls surface real errors


def _watch(run_id: str, get_run, result: dict, led: Ledger, poll: int,
           started: float, max_runtime: int, prov=None, inst_id: str = "",
           price_hr: float | None = None, max_cost: float = 0.0,
           keepalive: bool = False, put_run=None) -> dict:
    """Shared watch loop (submit + resume): poll the lane for the manifest,
    tolerate transient blips, fail fast when the instance died without one,
    respect budget + wall-clock ceilings. Returns `result` after harvest;
    raises ProviderError on budget/timeout/run-lost. Side channel (SPEC 14):
    heartbeat -> progress.jsonl history + events; failures are emitted by
    the CALLER's except (single choke point — no double events)."""
    lane_state: dict = {}
    memo: dict = {}
    deadline = started + max_runtime
    inst = result.get("instance")
    inst_fields = dict(inst) if isinstance(inst, dict) else (
        {"instance": inst} if inst else {})
    notify.emit("launched", run_id, job=result.get("kind") or "train",
                lane=result.get("lane", ""), **inst_fields,
                max_cost=round(max_cost, 4) if max_cost else None)
    while time.time() < deadline:
        if max_cost and price_hr:
            hours = (time.time() - started) / 3600
            if hours * price_hr > max_cost:
                raise ProviderError(f"budget cap ${max_cost} crossed")
        if keepalive:
            _keepalive_public_lane()
        manifest_raw = _tolerant_get(get_run, f"runs/{run_id}/manifest.json",
                                     lane_state)
        if manifest_raw:
            _harvest(run_id, manifest_raw, get_run, led, result)
            _finish_done(run_id, started, price_hr, get_run, put_run,
                         kind=result.get("kind") or "")
            return result
        if prov and inst_id:
            try:
                st = prov.poll(inst_id).status
            except Exception:
                st = ""          # provider API blip: re-check next cycle
            if st in {"terminated", "deleted", "failed", "error", "stopped",
                      "exited"}:
                time.sleep(poll)  # grace: agent may still be flushing
                manifest_raw = get_run(f"runs/{run_id}/manifest.json")
                if manifest_raw:
                    _harvest(run_id, manifest_raw, get_run, led, result)
                    _finish_done(run_id, started, price_hr, get_run, put_run,
                         kind=result.get("kind") or "")
                    return result
                raise ProviderError(
                    f"instance {inst_id} {st} without manifest — run lost")
        _progress_tick(run_id, get_run, put_run, price_hr, started, memo)
        time.sleep(poll)
    raise ProviderError("max runtime exceeded without manifest")


def _finish_done(run_id: str, started: float, price_hr, get_run=None,
                 put_run=None, kind: str = "") -> None:
    """Terminal signal for the progress channel: final history row (best
    effort) + run_done event."""
    cost = (round(price_hr * (time.time() - started) / 3600, 4)
            if price_hr else None)
    if get_run and put_run:
        try:
            key = f"runs/{run_id}/progress.jsonl"
            prev = get_run(key) or b""
            if isinstance(prev, str):
                prev = prev.encode()
            put_run(key, prev + (json.dumps(
                {"ts": round(time.time(), 3), "state": "done",
                 "cost": cost}) + "\n").encode())
        except Exception:                              # noqa: BLE001
            pass
    if kind:
        budget.record_observed(kind, (time.time() - started) / 60)
    notify.emit("run_done", run_id, cost=cost,
                seconds=round(time.time() - started))


def _sha(path: Path) -> str:
    return storage.sha256_file(path)


def prepare_dataset(domain: str, mini: bool = False,
                    vol: dict | None = None) -> tuple[str, Path, str]:
    """Run corpus expansion (via prepare-domain), upload to the lane, return
    (key, local_path, sha256). vol=None -> presign lane; vol dict -> volume
    lane; vol["local"] set -> host lane."""
    subprocess.run([sys.executable, str(ROOT / "bin/prepare-domain"),
                    domain], check=True, capture_output=True, text=True)
    src = ROOT / "tmp" / f"{domain}.jsonl"
    if not src.is_file():
        raise ProviderError(f"prepare-domain produced no {src}")
    if mini:
        lines = src.read_text().splitlines()[:MINI_LINES]
        mini_path = src.with_name(f"{domain}.mini.jsonl")
        mini_path.write_text("\n".join(lines) + "\n")
        src = mini_path
    sha = _sha(src)
    key = f"datasets/{domain}/{sha[:16]}/{src.name}"
    data = src.read_bytes()
    if vol and vol.get("local"):
        storage.host_put(key, data)
    elif vol:
        storage.vol_put(key, data)
    else:
        storage.ensure_bucket()
        storage.put_object(key, data, "application/x-ndjson")
    return key, src, sha


def render_config(recipe: Path, overrides: dict, dataset_name: str) -> str:
    cfg = yaml.safe_load(recipe.read_text())
    name = f"/work/datasets/{dataset_name}"
    if cfg.get("datasets"):
        cfg["datasets"][0]["path"] = name
        cfg["datasets"] = [cfg["datasets"][0]]
    else:
        cfg["datasets"] = [{"path": name, "type": "chat_template",
                            "ds_type": "json", "field_messages": "messages",
                            "roles_to_train": ["assistant"]}]
    cfg.update({
        "dataset_prepared_path": "/work/prepared",
        "output_dir": "/work/outputs",
        "val_set_size": 0.0,
        "logging_steps": 5,
        "save_total_limit": 1,
        "bf16": cfg.get("bf16", "auto"),
    })
    cfg.update(overrides)
    return yaml.safe_dump(cfg, sort_keys=False)


def _harvest(run_id: str, manifest_raw: bytes, get_run, led: Ledger,
             result: dict) -> None:
    """Terminal-lane logic shared by submit() and resume(): verify, store,
    ledger. Raises on artifact corruption (state stays failed-able)."""
    manifest = json.loads(manifest_raw)
    ok = manifest.get("status") == "succeeded"
    local = ROOT / "runs" / run_id
    local.mkdir(parents=True, exist_ok=True)
    (local / "manifest.json").write_bytes(manifest_raw)
    if ok:
        tar = get_run(f"runs/{run_id}/artifacts.tar.gz")
        assert tar is not None
        (local / "artifacts.tar.gz").write_bytes(tar)
        log = get_run(f"runs/{run_id}/run.log")   # forensic trail
        if log:
            (local / "run.log").write_bytes(log)
        got = hashlib.sha256(tar).hexdigest()
        want = str(manifest.get("artifacts_tar_sha256", "")).removeprefix("sha256:")
        if want and got != want:
            raise ProviderError(f"artifact sha mismatch: {got} != {want}")
        led.update(run_id, state="done",
                   artifact_uri=f"runs/{run_id}/artifacts.tar.gz")
        result.update({"status": "succeeded", "local": str(local),
                       "artifacts": len(manifest.get("artifacts", {}))})
    else:
        log = get_run(f"runs/{run_id}/run.log") or b""
        (local / "run.log").write_bytes(log[-20000:])
        led.update(run_id, state="failed",
                   error=str(manifest.get("problems"))[:400])
        result.update({"status": "failed", "problems": manifest.get("problems")})


OPEN_STATES = ("pending", "verified", "provisioning", "running", "uploading")


def resume(run_id: str | None = None, poll: int = 30,
           max_runtime: int = 4 * 3600) -> dict:
    """Re-attach to an open (non-terminal) run: resume lane polling,
    harvest, ALWAYS terminate the instance. Recovery path for crashed
    orchestrators / fleet restarts mid-job (2026-11-23 incident)."""
    led = Ledger(ROOT / "state" / "ledger.db")
    # 'verified' + no instance = DRY run (staged, never launched): nothing
    # to re-attach to; polling its lane would burn max_runtime for $0 work
    rows = [r for r in led.list() if r["state"] in OPEN_STATES
            and not (r["state"] == "verified" and not r["instance_id"])]
    if run_id:
        rows = [r for r in rows if r["run_id"] == run_id]
    if not rows:
        return {"resumed": None, "note": "no open runs"}
    row = rows[-1]
    run_id = row["run_id"]
    provider = row["provider"] or "runpod"
    inst_id = row["instance_id"]
    result = {"run_id": run_id, "resumed_from": row["state"],
              "kind": row["kind"], "instance": inst_id}
    if provider == "local":
        get_run = storage.host_get
        put_run = lambda k, d, ct="": storage.host_put(k, d)
    else:
        vol = storage.vol_cfg() if provider == "runpod" else None
        get_run = (lambda k: storage.vol_get(k)) if vol else storage.get_object
        put_run = ((lambda k, d, ct="": storage.vol_put(k, d)) if vol
                   else (lambda k, d, ct="": storage.put_object(k, d, ct)))
    result["lane"] = "volume" if (provider == "runpod" and vol) else provider
    try:
        prov = get_provider(provider) if inst_id else None
        _watch(run_id, get_run, result, led, poll, started=time.time(),
               max_runtime=max_runtime, prov=prov, inst_id=inst_id,
               put_run=put_run)
        return result
    except Exception as e:
        led.update(run_id, state="failed", error=str(e)[:400])
        result.update({"status": "error", "error": str(e)})
        notify.emit("run_failed", run_id, error=str(e)[:300])
        return result
    finally:
        if inst_id:
            try:
                get_provider(provider).terminate(inst_id)
                result["terminated"] = True
            except Exception as e:
                result["terminate_error"] = str(e)


def submit(domain: str, recipe: Path, *, model: str = "", mini: bool = False,
           overrides: dict | None = None, provider: str = "runpod",
           gpus: list[str] | None = None, tier: str | None = None,
           dry: bool = False, max_runtime: int = 3600,
           max_cost: float | None = None, poll: int = 20,
           force_lane: str | None = None, max_price_hr: float | None = None,
           on_run_id: "callable | None" = None) -> dict:
    """Full cycle; returns ledger-shaped result dict. NEVER leaves a pod
    behind: every terminal path terminates. on_run_id (if given) fires the
    instant the ledger registers the run — lets async callers (mcp-trainer)
    return before staging finishes."""
    led = Ledger(ROOT / "state" / "ledger.db")  # ROOT-scoped: tests redirect
    run_id = led.create("train", {"domain": domain, "recipe": str(recipe),
                                  "mini": mini, "overrides": overrides or {}},
                        provider=provider)
    if on_run_id:
        on_run_id(run_id)
    result = {"run_id": run_id}
    pod_id = ""
    if provider == "local":
        mount = os.environ.get("LOCAL_MOUNT", "/data")
        vol = {"mount": mount, "local": True}
        lane = "host"
        get_run = storage.host_get
        put_run = lambda k, d, ct="": storage.host_put(k, d)
    else:
        vol = storage.vol_cfg() if provider == "runpod" else None
        if force_lane == "presign":
            vol = None
        elif force_lane == "volume" and not vol:
            led.update(run_id, state="failed", error="volume lane not configured")
            return {"run_id": run_id, "status": "error",
                    "error": "RUNPOD_VOLUME_ID/RUNPOD_S3_* unset — volume lane off"}
        lane = "volume" if vol else "presign"
        get_run = (lambda k: storage.vol_get(k)) if vol else storage.get_object
        put_run = ((lambda k, d, ct="": storage.vol_put(k, d)) if vol
                   else (lambda k, d, ct="": storage.put_object(k, d, ct)))
    result["lane"] = lane
    try:
        dataset_key, dataset_path, dataset_sha = prepare_dataset(domain, mini, vol)
        ov = dict(overrides or {})
        if mini:
            ov.update({"sequence_len": 256, "num_epochs": 1,
                       "micro_batch_size": 2, "gradient_accumulation_steps": 1,
                       "saves_per_epoch": 1})
        config_yaml = render_config(recipe, ov, dataset_path.name)

        if vol:  # pod sees the dataset THROUGH the mount
            pod_ds_path = f"{vol['mount']}/{dataset_key}"
            payload = {"run_id": run_id,
                       "command": "axolotl train /work/job/train.yaml",
                       "config_yaml": config_yaml,
                       "dataset": {"name": dataset_path.name,
                                   "path": pod_ds_path, "sha256": dataset_sha},
                       "outputs_dir": "/work/outputs",
                       "urls": {},
                       "volume": {"mount": vol["mount"],
                                  "prefix": f"runs/{run_id}"}}
            payload_url = ""
            payload_env = {"JOB_PAYLOAD_PATH": f"{vol['mount']}/runs/{run_id}/payload.json"}
        else:
            urls = storage.mint_job_urls(run_id, dataset_key)
            payload = {"run_id": run_id,
                       "command": "axolotl train /work/job/train.yaml",
                       "config_yaml": config_yaml,
                       "dataset": {"name": dataset_path.name,
                                   "url": urls["dataset_get"],
                                   "sha256": dataset_sha},
                       "outputs_dir": "/work/outputs",
                       "urls": urls}
            payload_url = urls["payload_get"]
            payload_env = {"JOB_PAYLOAD_URL": payload_url}
        put_run(f"runs/{run_id}/payload.json", json.dumps(payload).encode(),
                "application/json")
        result.update({"dataset_key": dataset_key, "dataset_sha256": dataset_sha,
                       "payload_key": f"runs/{run_id}/payload.json"})
        if dry:
            led.update(run_id, state="verified", note=f"dry-run: {lane} lane staged")
            result["dry"] = True
            return result

        prov = get_provider(provider)
        offers = prov.catalog(gpus, tier)
        if vol:
            # network volumes exist only in the SECURE cloud; community
            # launches that ask for one 400 as "no instances available"
            offers = [o for o in offers if o.secure]
        # right-sizing (SPEC 13q): a rate ceiling keeps small models off
        # B200/H200-class hardware even when that is all that's in stock —
        # waiting is cheaper than overbuying. Pre-launch gates: aggregate
        # projection + account balance (hard floor) with a reserve.
        ceiling = (max_price_hr if max_price_hr is not None
                   else float(os.environ.get("QTRAIN_MAX_PRICE_HR", "1.50")))
        within = [o for o in offers if (o.price_hr or 0.0) <= ceiling]
        if not within:
            best = min((o.price_hr for o in offers if o.price_hr), default=None)
            raise ProviderError(
                f"no capacity across offers within ceiling ${ceiling:.2f}/hr "
                f"(cheapest in stock: ${best:.2f}/hr) — retry-loop or raise "
                "max_price_hr")
        est = budget.estimate(within[0].price_hr or 0.0, "train")
        why = budget.aggregate_gate(
            [r.get("est_cost") or 0.0 for r in led.list()
             if r["state"] in budget.BILLED and r["run_id"] != run_id], est)
        if not why and provider == "runpod":
            try:
                bal = prov.balance()
            except Exception:
                bal = None                       # scoped key: prepaid anyway
            why = budget.balance_gate(bal, est)
        if why:
            raise ProviderError(why)
        result["estimate"] = round(est, 2)
        token = hashlib.sha256(run_id.encode()).hexdigest()[:24]
        last_err = None
        inst = None
        for o in within[:8]:
            try:
                inst = prov.launch(
                    name=f"trainer-{run_id[-8:]}", image=_image_ref(),
                    gpu_id=o.gpu_id, cloud="SECURE" if o.secure else "COMMUNITY",
                    env={**payload_env, "AGENT_TOKEN": token,
                         "IMAGE_TAG": _image_tag()},
                    **({"volume_id": vol["bucket"], "volume_path": vol["mount"],
                        "data_center": vol["dc"]}
                       if (vol and not vol.get("local")) else {}))
                pod_id = inst.id
                result["instance"] = {"id": inst.id, "gpu": o.gpu_id,
                                      "price_hr": o.price_hr}
                break
            except ProviderError as e:
                last_err = e
        if not inst:
            raise ProviderError(f"no capacity across 8 offers: {last_err}")
        led.update(run_id, state="provisioning", instance_id=inst.id,
                   base_url=inst.base_url, est_cost=round(est, 4),
                   price_hr=inst.price_hr or 0.0)

        # kill threshold: explicit wins; else the runtime cap expressed in
        # dollars (wall-clock already bounds cost at this rate)
        mc = (max_cost if max_cost is not None
              else (inst.price_hr or 0.0) * max_runtime / 60)
        _watch(run_id, get_run, result, led, poll, started=time.time(),
               max_runtime=max_runtime, prov=prov, inst_id=inst.id,
               price_hr=inst.price_hr, max_cost=mc,
               keepalive=(lane == "presign"), put_run=put_run)
        return result
    except Exception as e:
        led.update(run_id, state="failed", error=str(e)[:400])
        result.update({"status": "error", "error": str(e)})
        notify.emit("run_failed", run_id, error=str(e)[:300])
        return result
    finally:
        if pod_id:
            try:
                get_provider(provider).terminate(pod_id)
                result["terminated"] = True
            except Exception as e:
                result["terminate_error"] = str(e)


STAGE_SPECS = {
    "merge": {"script": "merge_adapter.py", "needs_base": True,
              "required": ["config.json"], "max_price_hr": 0.80},
    # preludes now carry ONLY the dependency stack (pip drift-fixes until
    # p-v0.1.4+ images land via CI); the stage scripts + util.py ride along
    # as sha-verified file inputs (see submit_stage), so script fixes NEVER
    # wait on an image rebuild.
    "onnx": {"script": "export_onnx.py", "required": ["model.onnx"],
             "max_price_hr": 0.80,
             # torch 2.6: newer torches dropped the symbolic_opset14
             # symbols optimum 1.x imports; optimum 2.x dropped the
             # exporters extra entirely.
             "prelude": "pip install --no-cache-dir -q --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.*'"
                        " && pip install --no-cache-dir -q 'optimum[exporters]<2' onnx"},
    "int8": {"script": "quantize_onnx.py", "required": ["model.int8.onnx"],
             "max_price_hr": 0.80,
             "prelude": "pip install --no-cache-dir -q onnx"},
    "awq": {"script": "quantize_awq.py", "required": ["config.json"],
            "gpu": True, "needs_dataset": True, "max_price_hr": 0.80,
            # cu128 swap on Blackwell only — datacenter sm_100 (compute
            # 10.x) AND consumer sm_120 (12.x); baked cu124 lacks both.
            # Guard skips the 3.5GB download on Ada/Ampere (530b30's
            # B200 would now be rate-culled before launch anyway).
            "prelude": "if nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -qE '^(10|11|12)'; then pip install --no-cache-dir -q --index-url https://download.pytorch.org/whl/cu128 torch; fi"},
}

POSTPROC_PIN = ("ghcr.io/iqns-org/postproc"
                "@sha256:e61e6254628793f5003b3a5aeb61b37d913065f11c474380e871fcfda5d361ac")
POSTPROC_GPU_PIN = ("ghcr.io/iqns-org/postproc-gpu"
                    "@sha256:905f4fe0bc64cd6eaca90f87f8d9a418b8b84e2dd17b0862eedf2bd0bec12b2d")


def _postproc_image() -> str:
    return os.environ.get("POSTPROC_IMAGE", POSTPROC_PIN)


def _postproc_gpu_image() -> str:
    return os.environ.get("POSTPROC_GPU_IMAGE", POSTPROC_GPU_PIN)


def _trace_dataset(get_run, run_id: str, depth: int = 4) -> dict | None:
    """Walk the artifact lineage upstream (stage -> stage -> train) until a
    payload carries a dataset reference — calibration provenance is
    inherited, never guessed."""
    rid = run_id
    for _ in range(depth):
        raw = get_run(f"runs/{rid}/payload.json")
        if not raw:
            return None
        p = json.loads(raw)
        ds = p.get("dataset") or {}
        if ds.get("path") or ds.get("url"):
            return ds
        nxt = None
        for inp in p.get("inputs", []):
            if inp.get("name") == "model":
                src = inp.get("path") or inp.get("url") or ""
                m = re.search(r"runs/([^/]+)/artifacts", src)
                if m:
                    nxt = m.group(1)
        if nxt is None:
            return None
        rid = nxt
    return None


def submit_stage(kind: str, from_run: str, *, base_model: str | None = None,
                 provider: str = "runpod", max_runtime: int = 3600,
                 max_cost: float | None = None, poll: int = 15,
                 dry: bool = False, max_price_hr: float | None = None,
                 on_run_id: "callable | None" = None) -> dict:
    """Post-process a finished run's artifacts (merge|onnx|int8).

    Routing policy (2026-11-23, user call): cloud boxes are stronger in
    every dimension (RAM/CPU/NVMe/pull speed) — stages default to the
    RunPod volume lane; provider="local" remains as the opt-in $0
    fallback/dev lane. Input = runs/<from_run>/artifacts.tar.gz (the
    volume lane's own copy — every cloud run's output chains there);
    outputs pack through the same finalize contract with a stage gate.
    """
    if kind not in STAGE_SPECS:
        return {"status": "error",
                "error": f"unknown stage {kind!r} (have: "
                         f"{', '.join(STAGE_SPECS)})"}
    spec = STAGE_SPECS[kind]
    if spec.get("needs_base") and not base_model:
        return {"status": "error", "error": "merge stage needs base_model"}
    if provider not in ("local", "runpod"):
        return {"status": "error",
                "error": f"stage provider {provider!r} not wired yet"}
    led = Ledger(ROOT / "state" / "ledger.db")
    run_id = led.create("stage", {"kind": kind, "from_run": from_run,
                                  "base_model": base_model},
                        provider=provider)
    if on_run_id:
        on_run_id(run_id)
    result = {"run_id": run_id, "kind": kind, "from_run": from_run}
    pod_id = ""

    if provider == "local":
        mount = os.environ.get("LOCAL_MOUNT", "/data")
        lane = "host"
        get_run = storage.host_get
        put_run = lambda k, d, ct="": storage.host_put(k, d)
    elif provider == "runpod":
        vol = storage.vol_cfg()
        if not vol:
            led.update(run_id, state="failed", error="volume lane off")
            return {"run_id": run_id, "status": "error",
                    "error": "RUNPOD_VOLUME_ID/RUNPOD_S3_* unset"}
        mount = vol["mount"]
        lane = "volume"
        get_run = storage.vol_get
        put_run = lambda k, d, ct="": storage.vol_put(k, d)
    else:
        return {"status": "error",
                "error": f"stage provider {provider!r} not wired yet"}
    result["lane"] = lane

    in_key = f"runs/{from_run}/artifacts.tar.gz"
    cmd = (f"python /work/inputs/{spec['script']} "
           f"--in /work/inputs/model --out /work/outputs")
    if spec.get("needs_base"):
        cmd += f" --base {shlex.quote(base_model)}"
    inputs: list[dict] = [{"name": "model", "path": f"{mount}/{in_key}",
                           "extract": True}]
    if spec.get("needs_dataset"):
        # calibration must match what the model was TRAINED on: the dataset
        # is INHERITED from the artifact lineage (merge -> train), never
        # guessed or re-supplied
        ds = None
        try:
            ds = _trace_dataset(get_run, from_run)
        except Exception as e:                       # noqa: BLE001
            err = f"lineage read failed: {e}"
            led.update(run_id, state="failed", error=err[:400])
            notify.emit("run_failed", run_id, error=err[:300])
            return {"status": "error", "run_id": run_id, "error": err}
        ds = ds or {}
        ds_ref = {"name": "data"}
        if ds.get("path"):
            ds_ref["path"] = ds["path"]
        elif ds.get("url"):
            ds_ref["url"] = ds["url"]
        else:
            err = f"no dataset in lineage of {from_run}"
            led.update(run_id, state="failed", error=err)
            return {"status": "error", "run_id": run_id, "error": err}
        if ds.get("sha256"):
            ds_ref["sha256"] = ds["sha256"]
        inputs.append(ds_ref)
        cmd += " --dataset /work/inputs/data"
    # Run CURRENT repo scripts, not the frozen /opt copies baked into the
    # p-v images: those are stranded at pre-fix builds (mirror stall), and
    # 6709e3 proved the failure class — llmcompressor's API drifted around
    # the baked script while the repo one was already fixed. Plain file
    # inputs materialize at /work/inputs/<name> (agent _fetch + sha check);
    # util.py rides along so the script's `from util import ...` resolves.
    scripts: dict[str, bytes] = {}
    for fname in (spec["script"], "util.py"):
        fpath = ROOT / "postproc" / fname
        if not fpath.is_file():
            err = f"stage script missing in repo: postproc/{fname}"
            led.update(run_id, state="failed", error=err)
            notify.emit("run_failed", run_id, error=err)
            return {"status": "error", "run_id": run_id, "error": err}
        data = fpath.read_bytes()
        scripts[fname] = data
        inputs.append({"name": fname,
                       "path": f"{mount}/runs/{run_id}/script/{fname}",
                       "sha256": hashlib.sha256(data).hexdigest()})
    if spec.get("prelude"):
        # wrap LAST: the runner execs shlex.split(command) — no shell — so
        # the whole chain must arrive as ONE quoted `sh -c` invocation
        cmd = "sh -c " + shlex.quote(f"{spec['prelude']} && {cmd}")
    payload = {"run_id": run_id, "command": cmd, "config_yaml": "",
               "inputs": inputs,
               "outputs_dir": "/work/outputs",
               "gate": {"required_files": spec["required"]},
               "urls": {},
               "volume": {"mount": mount, "prefix": f"runs/{run_id}"}}
    payload_env = {"JOB_PAYLOAD_PATH":
                   f"{mount}/runs/{run_id}/payload.json"}
    try:
        for fname, data in scripts.items():
            put_run(f"runs/{run_id}/script/{fname}", data)
        put_run(f"runs/{run_id}/payload.json",
                json.dumps(payload).encode())
        result["payload_key"] = f"runs/{run_id}/payload.json"
        if dry:
            led.update(run_id, state="verified", note=f"dry stage {kind}")
            result["dry"] = True
            return result

        prov = get_provider(provider)
        token = hashlib.sha256(run_id.encode()).hexdigest()[:24]
        inst = None
        last_err = None
        if provider == "local":
            offers = prov.catalog()
            lk = {}
        else:
            # the right GPU = cheapest that FITS (budget.fits): SECURE
            # offers only (network volumes don't exist in community),
            # vram >= 10 as the "real RAM + NVMe" proxy, and a RATE
            # CEILING — an oversized card is never the fallback: when
            # only B200-class stock exists the job WAITs (530b30 burned
            # $7.99/hr + a crashloop proving it). arch_exclude per spec.
            pool = [o for o in prov.catalog() if o.secure]
            for pat in spec.get("arch_exclude", []):
                pool = [o for o in pool if pat.lower() not in o.gpu_id.lower()]
            ceiling = (max_price_hr if max_price_hr is not None
                       else spec.get("max_price_hr", 1.50))
            offers, best = budget.fits(pool, spec.get("min_vram", 10), ceiling)
            if not offers:
                cheap = f"${best:.2f}/hr" if best is not None else "none"
                raise ProviderError(
                    f"no capacity across offers within ceiling "
                    f"${ceiling:.2f}/hr (cheapest vram-fit in stock: {cheap})")
            offers = offers[:12]
            lk = {"volume_id": vol["bucket"], "volume_path": vol["mount"],
                  "data_center": vol["dc"]}
        if not offers:
            raise ProviderError("no capacity across offers: catalog empty")
        # normalized budgeting (SPEC 13q): estimate drives BOTH the
        # pre-launch gates and the kill threshold unless an explicit
        # max_cost overrides the latter. No magic numbers anywhere.
        est = budget.estimate(offers[0].price_hr or 0.0, kind)
        mc = max_cost if max_cost is not None else est
        result["estimate"] = round(est, 2)
        why = budget.aggregate_gate(
            [r.get("est_cost") or 0.0 for r in led.list()
             if r["state"] in budget.BILLED and r["run_id"] != run_id], est)
        if not why and provider == "runpod":
            try:
                bal = prov.balance()
            except Exception:
                bal = None                       # scoped key; prepaid anyway
            why = budget.balance_gate(bal, est)
        if why:
            raise ProviderError(why)
        for o in offers:
            try:
                inst = prov.launch(
                    name=f"trainer-{run_id[-8:]}",
                    image=(_postproc_gpu_image() if spec.get("gpu")
                           else _postproc_image()),
                    gpu_id=o.gpu_id,
                    cloud="SECURE" if o.secure else "COMMUNITY",
                    env={**payload_env, "AGENT_TOKEN": token}, **lk)
                result["instance"] = {"id": inst.id, "gpu": o.gpu_id,
                                      "price_hr": o.price_hr}
                break
            except ProviderError as e:
                last_err = e
        if not inst:
            raise ProviderError(f"no capacity across offers: {last_err}")
        pod_id = inst.id
        led.update(run_id, state="provisioning", instance_id=inst.id,
                   est_cost=round(est, 4), price_hr=o.price_hr or 0.0)

        _watch(run_id, get_run, result, led, poll,
               started=time.time(), max_runtime=max_runtime,
               prov=prov, inst_id=inst.id, price_hr=o.price_hr,
               max_cost=mc, keepalive=False, put_run=put_run)
        result["output"] = f"runs/{run_id}/artifacts.tar.gz"
        return result
    except Exception as e:
        led.update(run_id, state="failed", error=str(e)[:400])
        result.update({"status": "error", "error": str(e)})
        notify.emit("run_failed", run_id, error=str(e)[:300])
        return result
    finally:
        if pod_id:
            try:
                get_provider(provider).terminate(pod_id)
                result["terminated"] = True
            except Exception as e:
                result["terminate_error"] = str(e)


def preflight(kind: str = "train", *, provider: str = "runpod",
              max_price_hr: float | None = None,
              max_runtime: int = 3600) -> dict:
    """Read-only decision report (the plan's `--preflight`): which cards
    pass the fit, what the estimate/kill threshold would be, gate verdicts
    — launches NOTHING, writes NOTHING. Answers 'what would this job
    cost, on what' before committing a cent."""
    led = Ledger(ROOT / "state" / "ledger.db")
    prov = get_provider(provider)
    out: dict = {"kind": kind, "provider": provider}
    try:
        raw_catalog = prov.catalog()
    except Exception as e:                             # noqa: BLE001
        out["decision"] = f"refuse: catalog unavailable: {str(e)[:160]}"
        return out
    is_stage = kind in STAGE_SPECS
    if is_stage:
        spec = STAGE_SPECS[kind]
        pool = [o for o in raw_catalog if o.secure]
        for pat in spec.get("arch_exclude", []):
            pool = [o for o in pool if pat.lower() not in o.gpu_id.lower()]
        ceiling = (max_price_hr if max_price_hr is not None
                   else spec.get("max_price_hr", 1.50))
        offers, best = budget.fits(pool, spec.get("min_vram", 10), ceiling)
    else:
        ceiling = (max_price_hr if max_price_hr is not None
                   else float(os.environ.get("QTRAIN_MAX_PRICE_HR", "1.50")))
        all_ = raw_catalog
        if storage.vol_cfg():
            all_ = [o for o in all_ if o.secure]
        offers = sorted((o for o in all_
                         if (o.price_hr or 0.0) <= ceiling),
                        key=lambda o: o.price_hr or 0.0)
        best = min((o.price_hr for o in all_ if o.price_hr), default=None)
    out["ceiling_hr"] = ceiling
    out["candidates"] = [{"gpu": o.gpu_id, "price_hr": o.price_hr,
                          "vram_gb": o.vram_gb} for o in offers[:5]]
    if not offers:
        cheap = f"${best:.2f}/hr" if best is not None else "none"
        out["decision"] = (f"refuse: no in-stock offer within "
                           f"${ceiling:.2f}/hr (cheapest fit: {cheap})")
        return out
    est = budget.estimate(offers[0].price_hr or 0.0,
                          kind if is_stage else "train")
    kill = est if is_stage else (offers[0].price_hr or 0.0) * max_runtime / 60
    opens = [r.get("est_cost") or 0.0 for r in led.list()
             if r["state"] in budget.BILLED]
    why = budget.aggregate_gate(opens, est)
    bal = None
    if provider == "runpod":
        try:
            bal = prov.balance()
        except Exception:                            # noqa: BLE001
            pass
    why = why or budget.balance_gate(bal, est)
    out.update({"balance": bal, "estimate": round(est, 2),
                "kill_threshold": round(kill, 2),
                "open_projected": round(sum(opens), 2),
                "decision": why or
                f"launch {offers[0].gpu_id} @ ${offers[0].price_hr}/hr"})
    return out


IMAGE_PIN = ("ghcr.io/iqns-org/trainer"
             "@sha256:921b545b1b77bca81c4b53602aa1121101e7cc00ad745938d7b69e9c1e0f3c43")
IMAGE_TAG = "v0.2.0"


def _image_ref() -> str:
    return os.environ.get("TRAINER_IMAGE", IMAGE_PIN)


def _image_tag() -> str:
    return os.environ.get("TRAINER_IMAGE_TAG", IMAGE_TAG)
