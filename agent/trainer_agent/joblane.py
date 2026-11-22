"""Boot job lane (SPEC §4/§5): materialize a compiled payload and run it.

Payload (fetched from the presigned payload_get URL) contract:
  {"run_id": str,
   "command": "axolotl train /work/job/train.yaml",   # rendered by orchestrator
   "config_yaml": "<fully rendered axolotl config>",
   "dataset": {"name": "queer.jsonl", "url": "<presigned>", "sha256": "..."},
   "outputs_dir": "/work/outputs/<run_id>"}

On boot the agent sets JOB_PAYLOAD_URL to the payload's presigned GET;
nothing else about the job is passed via env — the payload is the single
source of truth, and presigned URLs carry zero long-lived credentials.
"""

from __future__ import annotations

import json
from pathlib import Path

from .transfer import download

WORK = Path("/work")


def materialize(payload: dict, work: Path = WORK) -> dict:
    """Write config + dataset under work/, return {run_id, command,
    outputs_dir, log_path, urls}."""
    run_id = payload["run_id"]
    jobdir = work / "job"
    jobdir.mkdir(parents=True, exist_ok=True)
    (jobdir / "train.yaml").write_text(payload["config_yaml"])

    ds = payload.get("dataset") or {}
    if ds.get("url"):
        dest = work / "datasets" / ds.get("name", "data.jsonl")
        download(ds["url"], dest, sha256=ds.get("sha256"))

    outputs = Path(payload.get("outputs_dir") or work / "outputs" / run_id)
    outputs.mkdir(parents=True, exist_ok=True)
    urls = payload.get("urls", {})
    return {"run_id": run_id, "command": payload["command"],
            "outputs_dir": outputs, "log_path": None, "urls": urls}


def boot_from_env(work: Path = WORK) -> dict | None:
    """If JOB_PAYLOAD_URL is set (the normal pod path), fetch + materialize
    the payload; return the run dict for the caller to start."""
    import os

    payload_url = os.environ.get("JOB_PAYLOAD_URL", "")
    if not payload_url:
        return None
    pfile = work / "job" / "payload.json"
    download(payload_url, pfile)
    payload = json.loads(pfile.read_text())
    return materialize(payload, work)
