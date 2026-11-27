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

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from .transfer import download, sha256_file

WORK = Path("/work")


def _fetch(inp: dict, blob: Path) -> None:
    """Stage input: local path (volume/host lane) or URL (presign lane)."""
    blob.parent.mkdir(parents=True, exist_ok=True)
    if inp.get("path"):
        src = Path(inp["path"])
        if src != blob:
            shutil.copyfile(src, blob)
        if inp.get("sha256") and sha256_file(blob) != inp["sha256"]:
            raise ValueError(f"input sha mismatch for {blob.name}")
    elif inp.get("url"):
        download(inp["url"], blob, sha256=inp.get("sha256"))
    else:
        raise ValueError(f"input {inp.get('name')!r} has neither path nor url")


def materialize(payload: dict, work: Path = WORK) -> dict:
    """Write config + dataset + stage inputs under work/, return
    {run_id, command, outputs_dir, log_path, urls, volume, gate,
    attestation}.

    Attestation (SPEC 13ak): every byte this pod ACCEPTED is hashed here,
    after the verify-else-raise gates — positive proof, not "absence of
    complaint". Written to work/attest.json (live /sanity reads it) and
    carried in the run dict into the manifest at finalize.
    """
    run_id = payload["run_id"]
    attest: dict = {"run_id": run_id}
    jobdir = work / "job"
    jobdir.mkdir(parents=True, exist_ok=True)
    if payload.get("config_yaml"):
        (jobdir / "train.yaml").write_text(payload["config_yaml"])
        attest["config_sha"] = hashlib.sha256(
            payload["config_yaml"].encode()).hexdigest()
    # CO-TENANT law: multiple configs share the one pod (one process tree,
    # one manifest, per-slug output dirs). Each config's exact bytes ride
    # the attestation — same proof standard as single-config runs.
    for cfg in payload.get("configs") or []:
        name = cfg["name"] if cfg["name"].endswith(".yaml") \
            else cfg["name"] + ".yaml"
        (jobdir / name).write_text(cfg["config_yaml"])
        attest.setdefault("configs", []).append(
            {"name": name,
             "sha256": hashlib.sha256(cfg["config_yaml"].encode()).hexdigest()})

    ds = payload.get("dataset") or {}
    if ds:
        dest = work / "datasets" / ds.get("name", "data.jsonl")
        if ds.get("path"):                               # volume lane: copy
            dest.parent.mkdir(parents=True, exist_ok=True)
            if Path(ds["path"]) != dest:
                shutil.copyfile(ds["path"], dest)
            if ds.get("sha256") and sha256_file(dest) != ds["sha256"]:
                raise ValueError(f"dataset sha mismatch for {dest.name}")
        elif ds.get("url"):                              # presign lane: fetch
            download(ds["url"], dest, sha256=ds.get("sha256"))
        if dest.is_file():
            with open(dest, "rb") as f:
                attest["dataset"] = {"name": dest.name,
                                     "sha256": sha256_file(dest),
                                     "lines": sum(1 for _ in f)}

    attest["inputs"] = []
    for inp in payload.get("inputs", []):                # stage jobs
        name = inp.get("name", "in")
        if inp.get("extract"):
            out = work / "inputs" / name
            out.mkdir(parents=True, exist_ok=True)
            if inp.get("path"):
                # read the tar straight off the mount — copying 700MB+
                # through a flaky NFS-style volume before unpacking
                # doubles the I/O and the stall risk (cae537 boot hang)
                with tarfile.open(inp["path"]) as tf:
                    tf.extractall(out, filter="data")
                attest["inputs"].append(
                    {"name": name, "sha256": sha256_file(Path(inp["path"])),
                     "extracted": True})
            else:
                blob = work / "inputs" / f"{name}.tar.gz"
                _fetch(inp, blob)
                with tarfile.open(blob) as tf:
                    tf.extractall(out, filter="data")
                attest["inputs"].append(
                    {"name": name, "sha256": sha256_file(blob),
                     "extracted": True})
                blob.unlink()
        else:
            _fetch(inp, out := work / "inputs" / name)
            attest["inputs"].append({"name": name, "sha256": sha256_file(out)})

    outputs = Path(payload.get("outputs_dir") or work / "outputs" / run_id)
    outputs.mkdir(parents=True, exist_ok=True)
    (work / "attest.json").write_text(json.dumps(attest, indent=1))
    return {"run_id": run_id, "command": payload["command"],
            "outputs_dir": outputs, "log_path": None,
            "urls": payload.get("urls", {}),
            "volume": payload.get("volume"),
            "gate": payload.get("gate"),
            "attestation": attest}


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
