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
import subprocess
import time
from pathlib import Path

import yaml

from . import storage
from .config import ROOT
from .ledger import Ledger
from .models import ProviderError
from .providers import get_provider

MINI_LINES = 48


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


def _sha(path: Path) -> str:
    return storage.sha256_file(path)


def prepare_dataset(domain: str, mini: bool = False,
                    vol: dict | None = None) -> tuple[str, Path, str]:
    """Run corpus expansion (via prepare-domain), upload to the lane, return
    (key, pod_local_path, sha256)."""
    subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "bin/prepare-domain"),
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
    if vol:
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


def submit(domain: str, recipe: Path, *, model: str = "", mini: bool = False,
           overrides: dict | None = None, provider: str = "runpod",
           gpus: list[str] | None = None, tier: str | None = None,
           dry: bool = False, max_runtime: int = 3600,
           max_cost: float = 0.75, poll: int = 20,
           force_lane: str | None = None) -> dict:
    """Full cycle; returns ledger-shaped result dict. NEVER leaves a pod
    behind: every terminal path terminates."""
    led = Ledger()
    run_id = led.create("train", {"domain": domain, "recipe": str(recipe),
                                  "mini": mini, "overrides": overrides or {}},
                        provider=provider)
    result = {"run_id": run_id}
    pod_id = ""
    vol = storage.vol_cfg() if provider == "runpod" else None
    if force_lane == "presign":
        vol = None
    elif force_lane == "volume" and not vol:
        led.update(run_id, state="failed", error="volume lane not configured")
        return {"run_id": run_id, "status": "error",
                "error": "RUNPOD_VOLUME_ID/RUNPOD_S3_* unset — volume lane off"}
    lane = "volume" if vol else "presign"
    result["lane"] = lane
    get_run = (lambda k: storage.vol_get(k)) if vol else storage.get_object
    put_run = ((lambda k, d, ct="": storage.vol_put(k, d)) if vol
               else (lambda k, d, ct="": storage.put_object(k, d, ct)))
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
                        "data_center": vol["dc"]} if vol else {}))
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

        deadline = time.time() + max_runtime
        while time.time() < deadline:
            mins = (time.time() - deadline + max_runtime) / 3600
            if inst.price_hr and mins * inst.price_hr > max_cost:
                raise ProviderError(f"budget cap ${max_cost} crossed")
            if lane == "presign":
                _keepalive_public_lane()
            manifest_raw = get_run(f"runs/{run_id}/manifest.json")
            if manifest_raw:
                manifest = json.loads(manifest_raw)
                ok = manifest.get("status") == "succeeded"
                local = ROOT / "runs" / run_id
                local.mkdir(parents=True, exist_ok=True)
                (local / "manifest.json").write_bytes(manifest_raw)
                if ok:
                    tar = get_run(f"runs/{run_id}/artifacts.tar.gz")
                    assert tar is not None
                    (local / "artifacts.tar.gz").write_bytes(tar)
                    got = "sha256:" + hashlib.sha256(tar).hexdigest()
                    want = manifest.get("artifacts_tar_sha256", "")
                    if want and got != want:
                        raise ProviderError(
                            f"artifact sha mismatch: {got} != {want}")
                    led.update(run_id, state="done",
                               artifact_uri=f"runs/{run_id}/artifacts.tar.gz")
                    result.update({"status": "succeeded", "local": str(local),
                                   "artifacts": len(manifest.get("artifacts", {}))})
                else:
                    log = get_run(f"runs/{run_id}/run.log") or b""
                    (local / "run.log").write_bytes(log[-20000:])
                    led.update(run_id, state="failed",
                               error=str(manifest.get("problems"))[:400])
                    result.update({"status": "failed",
                                   "problems": manifest.get("problems")})
                return result
            time.sleep(poll)
        raise ProviderError("max runtime exceeded without manifest")
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
             "@sha256:d1f766d24d62274e770fd8913c3ffe612b87c115d0cd59c318ca60d5c5d8bd36")
IMAGE_TAG = "sha-a27a4f8"


def _image_ref() -> str:
    return os.environ.get("TRAINER_IMAGE", IMAGE_PIN)


def _image_tag() -> str:
    return os.environ.get("TRAINER_IMAGE_TAG", IMAGE_TAG)
