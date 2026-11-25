"""FastAPI control plane for the trainer pod (SPEC §5).

Auth: bearer ``AGENT_TOKEN`` on everything except ``/healthz``.
One active job per pod; ``POST /v1/jobs`` 409s while one runs.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from . import joblane
from .finalize import finalize
from .runner import Runner

_app = FastAPI(title="trainer-agent", version=__version__)
_active: Runner | None = None


def _auth(authorization: str = Header(default="")) -> None:
    token = os.environ.get("AGENT_TOKEN", "")
    if not token:
        return                      # local dev mode: no auth configured
    if authorization != f"Bearer {token}":
        raise HTTPException(401, "bad or missing bearer token")


class JobRequest(BaseModel):
    run_id: str = Field(min_length=1)
    command: str = Field(min_length=1,
                         description="full shell-argv command, e.g. "
                         "'axolotl train /work/job/train.yaml'")
    env: dict[str, str] = {}


class JobInfo(BaseModel):
    state: Literal["running", "succeeded", "failed", "cancelled", "cancelling",
                   "uploading", "done", "unknown"]
    run_id: str = ""
    step: int | None = None
    total: int | None = None
    loss: float | None = None
    epoch: float | None = None
    lr: float | None = None
    exit_code: int | None = None
    started_at: float | None = None
    finished_at: float | None = None
    command: str = ""


@_app.get("/healthz")
def healthz():
    return {"ok": True, "version": __version__}


@_app.get("/v1/info", dependencies=[Depends(_auth)])
def info():
    gpu = ""
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
    except Exception:
        pass
    return {"version": __version__, "python": platform.python_version(),
            "gpu": gpu, "workdir": os.environ.get("AGENT_WORKDIR", "/work")}


@_app.get("/v1/sanity", dependencies=[Depends(_auth)])
def sanity():
    """Live provenance while a job runs (SPEC 13ak): exact image/agent
    bytes + the hashes materialize accepted. `qtrain verify --live` reads
    this and cross-checks against the ledger's staged contract."""
    att = None
    try:
        p = joblane.WORK / "attest.json"
        if p.is_file():
            att = json.loads(p.read_text())
    except Exception:                          # unreadable mid-write
        att = {"error": "attest.json unreadable"}
    return {"agent_version": __version__,
            "image_tag": os.environ.get("IMAGE_TAG", ""),
            "attestation": att}


@_app.post("/v1/jobs", dependencies=[Depends(_auth)])
def start_job(req: JobRequest):
    global _active
    if _active is not None and _active.active:
        raise HTTPException(409, f"job {_active.run_id} already active")
    runner = Runner(req.run_id, req.command, req.env)
    try:
        runner.start()
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    _active = runner
    return {"run_id": req.run_id, "state": "running"}


def _runner_or_404(run_id: str) -> Runner:
    if _active is None or _active.run_id != run_id:
        raise HTTPException(404, f"no job {run_id} on this pod")
    return _active


@_app.get("/v1/jobs/{run_id}", dependencies=[Depends(_auth)])
def job_status(run_id: str):
    return _runner_or_404(run_id).read_state()


@_app.get("/v1/jobs/{run_id}/logs", dependencies=[Depends(_auth)])
def job_logs(run_id: str, tail: int = 200):
    runner = _runner_or_404(run_id)
    return {"run_id": run_id, "logs": runner.logs(tail)}


@_app.post("/v1/jobs/{run_id}/cancel", dependencies=[Depends(_auth)])
def job_cancel(run_id: str):
    runner = _runner_or_404(run_id)
    return {"run_id": run_id, "cancelling": runner.cancel()}


def start_boot_job() -> None:
    """Normal pod path: JOB_PAYLOAD_URL materialized at boot (SPEC §5)."""
    global _active
    run = joblane.boot_from_env()
    if run is None:
        return
    runner = Runner(run["run_id"], run["command"],
                    outputs_dir=run["outputs_dir"],
                    post_hook=lambda state: finalize(
                        run["run_id"], run["outputs_dir"], state,
                        runner.log_path, run["urls"], run.get("volume"),
                        run.get("gate"), attestation=run.get("attestation")))
    runner.start()
    runner.start_heartbeat(volume=run.get("volume"), urls=run["urls"])
    _active = runner


def main() -> int:
    import uvicorn

    start_boot_job()
    port = int(os.environ.get("AGENT_PORT", "8000"))
    uvicorn.run(_app, host="0.0.0.0", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
