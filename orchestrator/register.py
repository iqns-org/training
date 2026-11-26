"""Deliver a verified run/stage artifact to a serving host (nano0 vLLM).

Lane -> host: the artifacts.tar.gz of any succeeded run (usually a
stage: merged fp16 or AWQ int4) is streamed to the target machine under
<dir>/<name>/ and unpacked there. Daily serving is the infra backend
catalog: add an apps.vllm entry with `model: /home/me/models/<name>`
(render_vllm mounts that path automatically) and flip `active`; one-off
validation is `MODEL=/home/me/models/<name> ./scripts/serve-vllm.sh`.

vLLM auto-detects compressed-tensors (AWQ/W4A16) and safetensors from
config.json — no --quantization flag needed for llm-compressor outputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

from . import notify, storage
from .config import ROOT
from .ledger import Ledger

# tars above this go via resumable rsync + staged unpack (slow-link
# survival, 2026-11-24: wifi-Orin leg streams ~0.5 MB/s — a 1.2 GB
# single-shot ssh pipe times out and restarts from zero every retry)
RESUME_THRESHOLD = 64 * 1024 * 1024


def register(run_id: str, *, name: str | None = None,
             host: str | None = None,
             dest_dir: str | None = None,
             dry: bool = False) -> dict:
    led = Ledger(ROOT / "state" / "ledger.db")
    row = next((r for r in led.list() if r["run_id"] == run_id), None)
    result = {"run_id": run_id}
    if row is None:
        return {"status": "error", "error": f"no such run: {run_id}"}
    name = name or f"{row['kind']}-{run_id[-6:]}"
    # §4b/§8 boundary: serving topology is deployment config, not code —
    # QTRAIN_SERVE_HOST/QTRAIN_SERVE_DIR come from .env / pod env.
    host = host or os.environ.get("QTRAIN_SERVE_HOST", "").strip()
    dest_dir = dest_dir or os.environ.get("QTRAIN_SERVE_DIR", "").strip()
    if not host or not dest_dir:
        return {"status": "error", "error":
                "no serve target: pass --host/--dest or set "
                "QTRAIN_SERVE_HOST/QTRAIN_SERVE_DIR"}

    key = f"runs/{run_id}/artifacts.tar.gz"
    local = ROOT / "runs" / run_id / "artifacts.tar.gz"
    if not local.is_file():                   # fall back to the lane
        provider = row["provider"] or "runpod"
        blob = (storage.host_get(key) if provider == "local"
                else storage.vol_get(key))
        if blob is None:
            return {"status": "error",
                    "error": "no artifacts.tar.gz (run not done?)"}
        local.parent.mkdir(parents=True, exist_ok=True)
        # cache it: a rerun/resume of register must not re-pull the lane,
        # and 1.2 GB should never live as one in-memory blob
        local.write_bytes(blob)
    size = local.stat().st_size

    # integrity: trust the manifest, not the bytes alone
    man_raw = (ROOT / "runs" / run_id / "manifest.json").read_bytes() \
        if (ROOT / "runs" / run_id / "manifest.json").is_file() else None
    want = ""
    if man_raw:
        want = str(json.loads(man_raw).get("artifacts_tar_sha256", "")) \
            .removeprefix("sha256:")
    h = hashlib.sha256()
    with local.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if want and h.hexdigest() != want:
        return {"status": "error", "error": "artifact tar corrupted "
                "vs manifest — re-harvest from lane or rerun stage"}

    target = f"{dest_dir.rstrip('/')}/{name}"
    result.update({"target": f"{host}:{target}", "bytes": size,
                   "sha256": h.hexdigest()[:16]})
    if dry:
        result["dry"] = True
        return result

    q = shlex.quote
    if size > RESUME_THRESHOLD:
        # big tars rsync to a staging path: --partial --append-verify
        # makes an interrupted delivery RESUMABLE (a 900 s ssh pipe was
        # a coin-flip at ~0.5 MB/s on the wifi-Orin leg, and every retry
        # restarted from zero). Rerun register; it picks up where it
        # died, then unpacks atomically and cleans the stage.
        # keyed by run (not name): a stale --partial from a DIFFERENT
        # artifact registered under the same name must never poison
        # this delivery (observed: killed merged-register made every
        # later adapter-register unpack EOF on the mismatched remote)
        stage = f"/tmp/qtrain-{run_id}-{name}.tar.gz"
        r = subprocess.run(
            ["rsync", "-e", "ssh -o ConnectTimeout=10",
             "--partial", "--append-verify", "--timeout=120",
             str(local), f"{host}:{stage}"],
            capture_output=True,
            timeout=int(os.environ.get("QTRAIN_REGISTER_TIMEOUT", "7200")))
        if r.returncode != 0:
            return {**result, "status": "error",
                    "error": f"rsync failed rc={r.returncode}: "
                             f"{r.stderr.decode()[-200:]} — rerun register "
                             f"to resume"}
        remote = (f"mkdir -p {q(target)} && tar -xzf {q(stage)} "
                  f"-C {q(target)} && rm -f {q(stage)}")
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=10", host, remote],
                           capture_output=True, timeout=900)
    else:
        remote = f"mkdir -p {q(target)} && tar -xzf - -C {q(target)}"
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=10", host, remote],
                           input=local.read_bytes(), capture_output=True,
                           timeout=900)
    if r.returncode != 0:
        return {**result, "status": "error",
                _ssh_cleanup = subprocess.run(          # noqa: F841
                    ["ssh", "-o", "ConnectTimeout=10", host, f"rm -f {q(stage)}"],
                    capture_output=True, timeout=60)
                return {**result, "status": "error",
                        "error": f"ssh unpack failed: {r.stderr.decode()[:200]}"
                                 " (staged tar removed — retry re-streams clean)"}
    result["status"] = "registered"
    notify.emit("delivered", run_id, name=result.get("name", ""),
                host=result.get("host", ""))
    result["serve_hint"] = (
        f"daily: add/flip a backend for {target} in infra quadra.yaml "
        f"(apps.vllm) then ./scripts/vllm-deploy.sh --apply; "
        f"one-off: MODEL={target} ./scripts/serve-vllm.sh")
    return result
