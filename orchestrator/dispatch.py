"""Job dispatch loop (SPEC §4/§6, P1): expand -> upload -> launch -> watch.

Zero-inbound design: the pod receives only a presigned payload_get URL; the
orchestrator learns of completion by polling runs/<id>/manifest.json. The
return lane IS the ledger: verified artifacts land under runs/<id>/ locally.
"""

from __future__ import annotations

import hashlib
import json
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
    """Quick tunnels idle-expire; during the training phase the pod sends no
    traffic, so the orchestrator pings the public endpoint each poll cycle to
    keep it alive for the artifact PUTs at the end. No-op when the public
    endpoint IS the LAN endpoint (permanent infra route)."""
    import os

    import httpx
    pub = os.environ.get("S3_PUBLIC_ENDPOINT", "").rstrip("/")
    lan = os.environ.get("S3_ENDPOINT", "").rstrip("/")
    if not pub or pub == lan:
        return
    try:
        httpx.get(f"{pub}/minio/health/live", timeout=10)
    except Exception:
        pass  # keepalive only — the presigned calls will surface real errors


def _sha(path: Path) -> str:
    return storage.sha256_file(path)


def prepare_dataset(domain: str, mini: bool = False) -> tuple[str, Path, str]:
    """Run corpus expansion (via prepare-domain), return (key, path, sha256)."""
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
    storage.put_object(key, src.read_bytes(), "application/x-ndjson")
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
           max_cost: float = 0.75, poll: int = 20) -> dict:
    """Full cycle; returns ledger-shaped result dict. NEVER leaves a pod
    behind: every terminal path terminates."""
    led = Ledger()
    run_id = led.create("train", {"domain": domain, "recipe": str(recipe),
                                  "mini": mini, "overrides": overrides or {}},
                        provider=provider)
    result = {"run_id": run_id}
    pod_id = ""
    try:
        storage.ensure_bucket()
        dataset_key, dataset_path, dataset_sha = prepare_dataset(domain, mini)
        ov = dict(overrides or {})
        if mini:
            ov.update({"sequence_len": 256, "num_epochs": 1,
                       "micro_batch_size": 2, "gradient_accumulation_steps": 1,
                       "saves_per_epoch": 1})
        config_yaml = render_config(recipe, ov, dataset_path.name)

        urls = storage.mint_job_urls(run_id, dataset_key)
        payload = {"run_id": run_id,
                   "command": "axolotl train /work/job/train.yaml",
                   "config_yaml": config_yaml,
                   "dataset": {"name": dataset_path.name,
                               "url": urls["dataset_get"],
                               "sha256": dataset_sha},
                   "outputs_dir": "/work/outputs",
                   "urls": urls}
        storage.put_object(f"runs/{run_id}/payload.json",
                           json.dumps(payload).encode(), "application/json")
        result.update({"dataset_key": dataset_key, "dataset_sha256": dataset_sha,
                       "payload_key": f"runs/{run_id}/payload.json"})
        if dry:
            led.update(run_id, state="verified", note="dry-run: staged only")
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
                    env={"JOB_PAYLOAD_URL": urls["payload_get"],
                         "AGENT_TOKEN": token, "IMAGE_TAG": _image_tag()})
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
        spent_ceiling = max_cost
        while time.time() < deadline:
            mins = (time.time() - deadline + max_runtime) / 3600
            if inst.price_hr and mins * inst.price_hr > spent_ceiling:
                raise ProviderError(f"budget cap ${spent_ceiling} crossed")
            _keepalive_public_lane()
            manifest_raw = storage.get_object(f"runs/{run_id}/manifest.json")
            if manifest_raw:
                manifest = json.loads(manifest_raw)
                ok = manifest.get("status") == "succeeded"
                if ok:
                    tar = storage.get_object(f"runs/{run_id}/artifacts.tar.gz")
                    assert tar is not None
                    local = ROOT / "runs" / run_id
                    local.mkdir(parents=True, exist_ok=True)
                    (local / "artifacts.tar.gz").write_bytes(tar)
                    (local / "manifest.json").write_bytes(manifest_raw)
                    got = "sha256:" + hashlib.sha256(tar).hexdigest()
                    want = manifest.get("artifacts_tar_sha256", "")
                    if want and got != want:
                        raise ProviderError(
                            f"artifact sha mismatch: {got} != {want}")
                    led.update(run_id, state="done",
                               artifact_uri=f"runs/{run_id}/artifacts.tar.gz")
                    result.update({"status": "succeeded",
                                   "local": str(local),
                                   "artifacts": len(manifest.get("artifacts", {}))})
                else:
                    log = storage.get_object(f"runs/{run_id}/run.log") or b""
                    (ROOT / "runs" / run_id).mkdir(parents=True, exist_ok=True)
                    (ROOT / "runs" / run_id / "run.log").write_bytes(log[-20000:])
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
    import os
    return os.environ.get("TRAINER_IMAGE", IMAGE_PIN)


def _image_tag() -> str:
    return IMAGE_TAG
