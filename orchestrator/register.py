"""Deliver a verified run/stage artifact to a serving host (nano0 vLLM).

Lane -> host: the artifacts.tar.gz of any succeeded run (usually a
stage: merged fp16 or AWQ int4) is streamed to the target machine under
<dir>/<name>/ and unpacked there. Serving is then one command:

    MODEL=/home/me/models/<name> ./scripts/serve-vllm.sh   # infra, on nano0

vLLM auto-detects compressed-tensors (AWQ/W4A16) and safetensors from
config.json — no --quantization flag needed for llm-compressor outputs.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from pathlib import Path

from . import storage
from .config import ROOT
from .ledger import Ledger


def register(run_id: str, *, name: str | None = None,
             host: str = "me@nano0.lan",
             dest_dir: str = "/home/me/models",
             dry: bool = False) -> dict:
    led = Ledger(ROOT / "state" / "ledger.db")
    row = next((r for r in led.list() if r["run_id"] == run_id), None)
    result = {"run_id": run_id}
    if row is None:
        return {"status": "error", "error": f"no such run: {run_id}"}
    name = name or f"{row['kind']}-{run_id[-6:]}"

    key = f"runs/{run_id}/artifacts.tar.gz"
    local = ROOT / "runs" / run_id / "artifacts.tar.gz"
    tar = local.read_bytes() if local.is_file() else None
    if tar is None:                       # fall back to the lane
        provider = row["provider"] or "runpod"
        tar = (storage.host_get(key) if provider == "local"
               else storage.vol_get(key))
    if tar is None:
        return {"status": "error", "error": "no artifacts.tar.gz (run not done?)"}

    # integrity: trust the manifest, not the bytes alone
    man_raw = (ROOT / "runs" / run_id / "manifest.json").read_bytes() \
        if (ROOT / "runs" / run_id / "manifest.json").is_file() else None
    if man_raw:
        want = str(json.loads(man_raw).get("artifacts_tar_sha256", "")) \
            .removeprefix("sha256:")
        if want and hashlib.sha256(tar).hexdigest() != want:
            return {"status": "error", "error": "artifact tar corrupted "
                    "vs manifest — re-harvest from lane or rerun stage"}

    target = f"{dest_dir.rstrip('/')}/{name}"
    result.update({"target": f"{host}:{target}", "bytes": len(tar),
                   "sha256": hashlib.sha256(tar).hexdigest()[:16]})
    if dry:
        result["dry"] = True
        return result

    remote = f"mkdir -p {shlex.quote(target)} && tar -xzf - -C {shlex.quote(target)}"
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=10", host, remote],
                       input=tar, capture_output=True, timeout=900)
    if r.returncode != 0:
        return {**result, "status": "error",
                "error": f"ssh unpack failed: {r.stderr.decode()[:200]}"}
    result["status"] = "registered"
    result["serve_hint"] = (
        f"on {host.split('@')[-1]}: MODEL={target} "
        "./scripts/serve-vllm.sh  (infra repo)")
    return result
