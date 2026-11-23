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
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml

from . import storage
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
           keepalive: bool = False) -> dict:
    """Shared watch loop (submit + resume): poll the lane for the manifest,
    tolerate transient blips, fail fast when the instance died without one,
    respect budget + wall-clock ceilings. Returns `result` after harvest;
    raises ProviderError on budget/timeout/run-lost."""
    lane_state: dict = {}
    deadline = started + max_runtime
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
                    return result
                raise ProviderError(
                    f"instance {inst_id} {st} without manifest — run lost")
        time.sleep(poll)
    raise ProviderError("max runtime exceeded without manifest")


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
    rows = [r for r in led.list() if r["state"] in OPEN_STATES]
    if run_id:
        rows = [r for r in rows if r["run_id"] == run_id]
    if not rows:
        return {"resumed": None, "note": "no open runs"}
    row = rows[-1]
    run_id = row["run_id"]
    provider = row["provider"] or "runpod"
    inst_id = row["instance_id"]
    result = {"run_id": run_id, "resumed_from": row["state"],
              "instance": inst_id}
    if provider == "local":
        get_run = storage.host_get
    else:
        vol = storage.vol_cfg() if provider == "runpod" else None
        get_run = (lambda k: storage.vol_get(k)) if vol else storage.get_object
    try:
        prov = get_provider(provider) if inst_id else None
        _watch(run_id, get_run, result, led, poll, started=time.time(),
               max_runtime=max_runtime, prov=prov, inst_id=inst_id)
        return result
    except Exception as e:
        led.update(run_id, state="failed", error=str(e)[:400])
        result.update({"status": "error", "error": str(e)})
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
           max_cost: float = 0.75, poll: int = 20,
           force_lane: str | None = None,
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
        token = hashlib.sha256(run_id.encode()).hexdigest()[:24]
        last_err = None
        inst = None
        for o in offers[:8]:
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
                   base_url=inst.base_url)

        _watch(run_id, get_run, result, led, poll, started=time.time(),
               max_runtime=max_runtime, prov=prov, inst_id=inst.id,
               price_hr=inst.price_hr, max_cost=max_cost,
               keepalive=(lane == "presign"))
        return result
    except Exception as e:
        led.update(run_id, state="failed", error=str(e)[:400])
        result.update({"status": "error", "error": str(e)})
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
              "required": ["config.json"]},
    "onnx": {"script": "export_onnx.py", "required": ["model.onnx"]},
    "int8": {"script": "quantize_onnx.py", "required": ["model.int8.onnx"]},
}

POSTPROC_PIN = ("ghcr.io/iqns-org/postproc"
                "@sha256:4e6e6e3bf709805cf45b8ae10df2f9a7d4e6b65e40f194d631a6879217d23399")


def _postproc_image() -> str:
    return os.environ.get("POSTPROC_IMAGE", POSTPROC_PIN)


def submit_stage(kind: str, from_run: str, *, base_model: str | None = None,
                 provider: str = "runpod", max_runtime: int = 3600,
                 max_cost: float = 1.0, poll: int = 15, dry: bool = False,
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
    cmd = (f"python /opt/postproc/{spec['script']} "
           f"--in /work/inputs/model --out /work/outputs")
    if spec.get("needs_base"):
        cmd += f" --base {shlex.quote(base_model)}"
    payload = {"run_id": run_id, "command": cmd, "config_yaml": "",
               "inputs": [{"name": "model", "path": f"{mount}/{in_key}",
                           "extract": True}],
               "outputs_dir": "/work/outputs",
               "gate": {"required_files": spec["required"]},
               "urls": {},
               "volume": {"mount": mount, "prefix": f"runs/{run_id}"}}
    payload_env = {"JOB_PAYLOAD_PATH":
                   f"{mount}/runs/{run_id}/payload.json"}
    try:
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
            # >=10GB-vram proxy for "real RAM + NVMe" boxes; cheapest first
            offers = [o for o in prov.catalog() if o.vram_gb >= 10][:8]
            lk = {"volume_id": vol["bucket"], "volume_path": vol["mount"],
                  "data_center": vol["dc"]}
        for o in offers:
            try:
                inst = prov.launch(
                    name=f"trainer-{run_id[-8:]}", image=_postproc_image(),
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
        led.update(run_id, state="provisioning", instance_id=inst.id)

        _watch(run_id, get_run, result, led, poll,
               started=time.time(), max_runtime=max_runtime,
               prov=prov, inst_id=inst.id, price_hr=inst.price_hr,
               max_cost=max_cost, keepalive=False)
        result["output"] = f"runs/{run_id}/artifacts.tar.gz"
        return result
    except Exception as e:
        led.update(run_id, state="failed", error=str(e)[:400])
        result.update({"status": "error", "error": str(e)})
        return result
    finally:
        if pod_id:
            try:
                get_provider(provider).terminate(pod_id)
                result["terminated"] = True
            except Exception as e:
                result["terminate_error"] = str(e)


IMAGE_PIN = ("ghcr.io/iqns-org/trainer"
             "@sha256:921b545b1b77bca81c4b53602aa1121101e7cc00ad745938d7b69e9c1e0f3c43")
IMAGE_TAG = "v0.2.0"


def _image_ref() -> str:
    return os.environ.get("TRAINER_IMAGE", IMAGE_PIN)


def _image_tag() -> str:
    return os.environ.get("TRAINER_IMAGE_TAG", IMAGE_TAG)
