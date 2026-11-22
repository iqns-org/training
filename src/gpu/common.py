"""Shared, provider-agnostic helpers.

Borrowed ideas (not code copies) from bin/gpuai-deploy, bin/gpuai-job and
bin/gpuai-download: JSON-payload extraction from chatty CLIs, active vs
terminal status sets, SSH-metadata parsing, sequential local dirs, and the
final-artifact gate for trained models.
"""

import json
import re
import subprocess
from pathlib import Path


class GpuError(Exception):
    """Actionable GPU-operation failure."""


# Statuses borrowed from the gpuai scripts' shell case statements.
ACTIVE_STATUSES = {
    "running", "allocating", "pending", "provisioning", "booting",
    "starting", "restarting", "creating", "unknown",
}

TERMINAL_STATUSES = {
    "error", "failed", "failed_to_create", "failed_to_start",
    "terminated", "deleted", "stopped", "cancelled", "canceled",
    # RunPod-isms: an EXITED pod will never expose SSH again.
    "exited", "dead",
}

# Final artifacts required before an instance may be torn down
# (same set as require_final_artifacts in bin/gpuai-deploy).
REQUIRED_MODEL_ARTIFACTS = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def run(cmd, timeout=None, **kwargs):
    """Run a command, raising GpuError with stderr on failure or timeout."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              check=True, timeout=timeout, **kwargs)
    except FileNotFoundError:
        raise GpuError(f"command not found: {cmd[0]} (is it installed/on PATH?)")
    except subprocess.TimeoutExpired:
        raise GpuError(f"command timed out after {timeout}s: "
                       f"{cmd[0]} {' '.join(cmd[1:2])}…")
    except subprocess.CalledProcessError as e:
        raise GpuError(f"{' '.join(cmd)} failed (rc={e.returncode}): "
                       f"{(e.stderr or e.stdout or '').strip()[:2000]}")


def retry(fn, attempts=3, backoff=5.0, describe="operation"):
    """Retry a GpuError-raising callable with exponential backoff.

    Used for idempotent, flaky operations (list calls, rsync transfers,
    instance deletion). NOT used for provisioning (a retry could create a
    duplicate billable instance) nor for remote commands (generally
    non-idempotent).
    """
    last = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except GpuError as e:
            last = e
            if attempt < attempts:
                import time

                time.sleep(backoff * (2 ** (attempt - 1)))
    raise GpuError(f"{describe} failed after {attempts} attempts: {last}") from last


def extract_json_payload(text):
    """First JSON object embedded in chatty CLI output (bin/gpuai-deploy idea)."""
    start = text.find("{")
    if start == -1:
        raise GpuError("no JSON payload found in command output")
    decoder = json.JSONDecoder()
    for idx in range(start, len(text)):
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            return obj
        except json.JSONDecodeError:
            continue
    raise GpuError("no JSON payload found in command output")


def is_active(status):
    return str(status or "unknown").lower() in ACTIVE_STATUSES


def is_terminal(status):
    return str(status or "").lower() in TERMINAL_STATUSES


_SSH_P = re.compile(r"-p\s+(\d+)\s+([^@\s]+)@([A-Za-z0-9.\-]+)")
_SSH_BARE = re.compile(r"ssh\s+.*?([^@\s]+)@([A-Za-z0-9.\-]+)")


def parse_ssh_connection(record):
    """Normalize provider record → {user, host, port, ssh_command}.

    Mirrors the parsing fallbacks in bin/gpuai-deploy/job/download:
    explicit connection fields first, then ssh_command regexes.
    """
    record = record or {}
    conn = record.get("connection", {}) if isinstance(record, dict) else {}
    user = (conn.get("user") or conn.get("username") or conn.get("app_user") or "")
    host = (conn.get("hostname") or conn.get("host")
            or record.get("hostname") or record.get("host") or "")
    port = str(conn.get("port") or record.get("port") or "")
    ssh_command = conn.get("ssh_command") or ""
    if ssh_command:
        m = _SSH_P.search(ssh_command)
        if m:
            port, user, host = m.group(1), m.group(2), m.group(3)
        else:
            m = _SSH_BARE.search(ssh_command)
            if m:
                user, host = m.group(1), m.group(2)
                pm = re.search(r"-p\s+(\d+)", ssh_command)
                if pm:
                    port = pm.group(1)
    return {"user": user, "host": host, "port": port, "ssh_command": ssh_command}


def next_sequential_dir(parent, stem):
    """tmp/001-<stem>, tmp/002-<stem>, … (bin/gpuai-deploy idea)."""
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    idx = 1
    while True:
        candidate = parent / f"{idx:03d}-{stem}"
        if not candidate.exists():
            return candidate
        idx += 1


def find_config_file(domain_dir, model, strategy):
    """Pick axolotl YAML like bin/gpuai-deploy's find_config_file."""
    domain_dir = Path(domain_dir)
    base = model[:-len("-Instruct")] if model.endswith("-Instruct") else model
    for cand in (f"{model}-{strategy}.yaml", f"{base}-{strategy}.yaml",
                 f"{model}.yaml", f"{base}.yaml"):
        if (domain_dir / cand).is_file():
            return cand
    for path in sorted(domain_dir.glob("*.yaml")):
        n = path.name
        if (n.startswith(model) or n.startswith(base)
                or n.endswith(f"-{strategy}.yaml")):
            return n
    raise GpuError(f"no Axolotl YAML for model={model!r} strategy={strategy!r} "
                   f"under {domain_dir}")


def require_final_artifacts(model_dir):
    """Gate teardown on the final adapter set (bin/gpuai-deploy idea)."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise GpuError(f"expected model directory {model_dir} does not exist")
    if any(model_dir.glob(p) for p in ("checkpoint-*", "*/*/checkpoint-*")):
        sub = [p for p in model_dir.rglob("checkpoint-*") if p.is_dir()]
        if sub:
            raise GpuError("checkpoint snapshots present; refusing to terminate")
    missing = [f for f in REQUIRED_MODEL_ARTIFACTS if not (model_dir / f).is_file()]
    if missing:
        raise GpuError(f"missing final artifacts in {model_dir}: {', '.join(missing)}")
    return True


def load_dotenv(path=None):
    """Load ROOT/.env key=value lines into os.environ (no override)."""
    import os

    path = Path(path) if path else Path(__file__).resolve().parents[2] / ".env"
    if not path.is_file():
        return {}
    loaded = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
            loaded[k] = v
    return loaded


def job_file(provider, app, domain, root=None):
    """Stable path for saved deployment JSON (borrows tmp/gpuai_*_<domain>.job)."""
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    return root / "tmp" / f"gpu_{provider}_{app}_{domain}.job"


def save_job(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def load_job(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
