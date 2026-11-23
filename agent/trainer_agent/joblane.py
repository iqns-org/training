"""Boot job lane (SPEC §4/§5): materialize a compiled payload and run it.

Payload (from the presigned payload_get URL, or in volume mode from the
mounted payload path) contract:
  {"run_id": str,
   "command": "axolotl train /work/job/train.yaml",   # rendered by orchestrator
   "config_yaml": "<fully rendered axolotl config>",
   "dataset": {"name": "queer.jsonl", "url": "<presigned>", "sha256": "..."},
   "outputs_dir": "/work/outputs",
   "urls": {...}                      # presign lane: PUT targets
   or
   "volume": {"mount": "/data", "prefix": "runs/<id>"}}   # volume lane

Lane rule: presigned URLs carry zero credentials but need a public S3 facade;
the volume lane (RunPod network volumes) needs NO network egress for artifacts
at all — pod and orchestrator share a filesystem, and the pod still holds zero
credentials. In both lanes the manifest is written LAST as the completion
signal.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .transfer import download, sha256_file

WORK = Path("/work")


def materialize(payload: dict, work: Path = WORK) -> dict:
    """Write config + dataset under work/, return {run_id, command,
    outputs_dir, log_path, urls, volume}."""
    run_id = payload["run_id"]
    jobdir = work / "job"
    jobdir.mkdir(parents=True, exist_ok=True)
    (jobdir / "train.yaml").write_text(payload["config_yaml"])

    ds = payload.get("dataset") or {}
    dest = work / "datasets" / ds.get("name", "data.jsonl")
    if ds.get("path"):                                  # volume lane: copy
        dest.parent.mkdir(parents=True, exist_ok=True)
        if Path(ds["path"]) != dest:
            shutil.copyfile(ds["path"], dest)
        if ds.get("sha256") and sha256_file(dest) != ds["sha256"]:
            raise ValueError(f"dataset sha mismatch for {dest.name}")
        ds = {**ds, "name": dest.name}                  # agent-local path
    elif ds.get("url"):                                 # presign lane: fetch
        download(ds["url"], dest, sha256=ds.get("sha256"))

    outputs = Path(payload.get("outputs_dir") or work / "outputs" / run_id)
    outputs.mkdir(parents=True, exist_ok=True)
    return {"run_id": run_id, "command": payload["command"],
            "outputs_dir": outputs, "log_path": None,
            "urls": payload.get("urls", {}),
            "volume": payload.get("volume")}


def boot_from_env(work: Path = WORK) -> dict | None:
    """Normal pod path: fetch/parse payload from JOB_PAYLOAD_URL (presign
    lane) or JOB_PAYLOAD_PATH (volume lane); return run dict or None."""
    import os

    payload_url = os.environ.get("JOB_PAYLOAD_URL", "")
    payload_path = os.environ.get("JOB_PAYLOAD_PATH", "")
    if payload_url:
        pfile = work / "job" / "payload.json"
        download(payload_url, pfile)
    elif payload_path:
        pfile = Path(payload_path)
    else:
        return None
    payload = json.loads(pfile.read_text())
    return materialize(payload, work)
