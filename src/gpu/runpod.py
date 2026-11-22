"""RunPod provider — same :class:`Provider` interface as GPU.ai.

Implemented against the documented ``runpodctl`` surface (runpodctl
``pod list --all`` / ``pod get`` / ``pod create`` / ``pod delete``,
``ssh info`` / ``ssh list-keys`` / ``ssh add-key``; JSON is runpodctl's
default output, ``pod list`` shows running pods only without ``--all``).
Requires runpodctl >= 2.0.0 for ``--output json`` / ``pod get -j``.

Auth follows runpodctl's own config (``runpodctl config --apiKey`` /
``doctor``); an explicit ``api_key`` is exported as ``RUNPOD_API_KEY``
when given but never required. SSH access reuses the account-level keys,
so there is no per-instance key id.

Verified against the public docs; not yet exercised against a live
account — treat first live run as a smoke test.
"""

import json
import os
import re
import shlex

from .base import Provider
from .common import (
    GpuError, is_active, is_terminal, load_job, retry, run, save_job,
)

# Docs: statuses are PROVISIONING | STARTING | RUNNING | EXITED | ERROR |
# TERMINATED. Normalized to lowercase on ingest; EXITED/ERROR/TERMINATED
# are covered by common.TERMINAL_STATUSES.
BIN_ENV_VAR = "RUNPODCTL_BIN"
DEFAULT_BIN = "runpodctl"


def _ssh_from_command(text):
    """Parse ``ssh user@host -p port -i key`` (docs' canonical shape)."""
    m = re.search(r"ssh\s+(\S+)@(\S+)\s+-p\s+(\d+)(?:\s+-i\s+(\S+))?", text)
    if m:
        user, host, port = m.group(1), m.group(2), m.group(3)
        return {"user": user, "host": host, "port": port,
                "key": m.group(4) or ""}
    m = re.search(r"(\S+)@(\S+)", text)
    if m:
        return {"user": m.group(1), "host": m.group(2), "port": "",
                "key": ""}
    return {}


class RunpodProvider(Provider):
    name = "runpod"

    def __init__(self, api_key=None, ssh_key_file="~/.ssh/id_ed25519.pub",
                 max_redeploys=2, binary=None):
        import pathlib

        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY", "")
        self.ssh_key_file = str(pathlib.Path(ssh_key_file).expanduser())
        self.max_redeploys = max_redeploys
        self.binary = binary or os.environ.get(BIN_ENV_VAR, DEFAULT_BIN)
        self._last_provision = None

    # -- internal ------------------------------------------------------
    def _runpod(self, *args):
        import subprocess

        env = dict(os.environ)
        if self.api_key:
            env["RUNPOD_API_KEY"] = self.api_key
        try:
            return subprocess.run([self.binary, *args], capture_output=True,
                                  text=True, check=True, env=env)
        except FileNotFoundError:
            raise GpuError(f"`{self.binary}` not found (install runpodctl: "
                           f"https://docs.runpod.io/runpodctl/overview)")
        except subprocess.CalledProcessError as e:
            raise GpuError(f"{self.binary} {' '.join(args)} failed: "
                           f"{(e.stderr or e.stdout or '').strip()[:2000]}")

    @staticmethod
    def _normalize(pod):
        """Pod JSON → internal record (id/status/name/gpu_type/cost/raw)."""
        gpu = pod.get("gpu") or {}
        return {
            "id": str(pod.get("id", "")),
            "name": pod.get("name", ""),
            "status": str(pod.get("status", "unknown")).lower(),
            "raw_status": pod.get("status", "unknown"),
            "gpu_type": gpu.get("id") or pod.get("gpu_type", ""),
            "cost": pod.get("cost", ""),
            "raw": pod,
        }

    def _pods(self):
        # --all matters: without it only running pods are listed (docker-ps
        # semantics), so dead pods would look "not found".
        def _fetch():
            out = self._runpod("pod", "list", "--all",
                               "--output", "json").stdout.strip()
            try:
                data = json.loads(out or "[]")
            except json.JSONDecodeError:
                raise GpuError("could not parse `runpodctl pod list` output "
                               "(need runpodctl >= 2.0.0?)")
            pods = data if isinstance(data, list) else [data]
            return [self._normalize(p) for p in pods]

        return retry(_fetch, attempts=3, backoff=5.0,
                     describe="`runpodctl pod list`")

    def _pod_get(self, instance_id):
        """Full pod record incl. ``ssh`` object; tries --output json then -j."""
        last_err = None
        for extra in (["--output", "json"], ["-j"]):
            try:
                out = self._runpod("pod", "get", str(instance_id),
                                   *extra).stdout.strip()
                return json.loads(out)
            except (GpuError, json.JSONDecodeError) as e:
                last_err = e
        raise GpuError(f"could not read pod {instance_id} "
                       f"(need runpodctl >= 2.0.0?): {last_err}")

    # -- provision / creation -------------------------------------------
    def provision(self, app=None, name=None, ssh_key_id=None, job_file=None,
                  image="runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
                  gpu_type="NVIDIA GeForce RTX 4090", gpu_count=1,
                  template_id=None, force=False, **kwargs):
        """Create a pod (or reuse the live one recorded in job_file).

        Either ``template_id`` or (``image`` + ``gpu_type``); SSH defaults
        on (``--ssh``). Extra kwargs must be ``--flag value`` pairs and are
        passed through (e.g. cloud_type="SECURE", ports="22/tcp").
        """
        if job_file and not force:
            saved = load_job(job_file) or {}
            pid = saved.get("instance", {}).get("id", "")
            if pid:
                try:
                    rec = self.status(pid)
                    if is_active(rec.get("status")):
                        return {"instance": rec, "reused": True}
                except GpuError:
                    pass
        # NOTE: creation is not retried (see gpuai.provision): a lost
        # response plus a retry could strand a duplicate billable pod.
        self._last_provision = {"app": app, "name": name or "axolotl",
                                "image": image, "gpu_type": gpu_type,
                                "gpu_count": gpu_count,
                                "template_id": template_id, **kwargs}
        cmd = ["pod", "create", "--name", name or "axolotl"]
        if template_id:
            cmd += ["--template-id", template_id]
        else:
            cmd += ["--image", image, "--gpu-id", gpu_type,
                    "--gpu-count", str(gpu_count)]
        for key, value in kwargs.items():
            cmd += ["--" + key.replace("_", "-"), str(value)]
        out = self._runpod(*cmd).stdout.strip()
        try:
            payload = json.loads(out) if out.startswith("{") else {}
        except json.JSONDecodeError:
            payload = {}
        # Legacy create prints plaintext; fall back to listing our new pod.
        pod_id = str(payload.get("id", "") or "")
        if not pod_id:
            m = re.search(r"\b([0-9a-z]{6,})\b", out)
            pod_id = m.group(1) if m else ""
        if not pod_id:
            raise GpuError(f"pod creation returned no id: {out[:300]}")
        inst = {"id": pod_id, "status": "creating",
                "name": name, "app": app}
        if job_file:
            save_job(job_file, {"instance": inst})
        return {"instance": inst, "reused": False}

    # -- inspection / monitoring -----------------------------------------
    def list_instances(self):
        return self._pods()

    def status(self, instance_id):
        try:
            return self._normalize(self._pod_get(instance_id))
        except GpuError:
            for rec in self._pods():
                if rec["id"] == str(instance_id):
                    return rec
            raise GpuError(f"pod {instance_id} not found")

    # -- ssh plumbing ------------------------------------------------------
    def _conn(self, instance_id):
        """SSH connection from ``pod get``'s ``ssh`` object (docs' signal)."""
        pod = self._pod_get(instance_id)
        ssh = pod.get("ssh") or {}
        cmd = ssh.get("ssh_command", "") or ""
        if cmd:
            parsed = _ssh_from_command(cmd)
            if parsed.get("host"):
                parsed["ssh_command"] = cmd
                parsed.setdefault("id", str(instance_id))
                return parsed
        # Fallback: `ssh info` text output.
        try:
            info = self._runpod("ssh", "info", str(instance_id)).stdout
        except GpuError as e:
            raise GpuError(f"no SSH info for pod {instance_id}: {e}")
        parsed = _ssh_from_command(info)
        if not parsed.get("host"):
            raise GpuError(f"pod {instance_id} has no SSH connection yet")
        parsed["ssh_command"] = info.strip().splitlines()[0] if info.strip() else ""
        parsed.setdefault("id", str(instance_id))
        return parsed

    def wait_ssh(self, instance_id, max_wait=1800, poll=15, redeploy=None):
        import time

        redeploy = self.max_redeploys if redeploy is None else redeploy
        attempts, elapsed, current = 0, 0, instance_id
        while True:
            try:
                rec = self.status(current)
            except GpuError:
                rec = {"id": current, "status": "unknown"}
            if is_terminal(rec.get("status")):
                attempts += 1
                if attempts > redeploy or not self._last_provision:
                    raise GpuError(f"pod {current} terminal "
                                   f"({rec.get('status')}) after {attempts - 1} "
                                   f"redeploys")
                self.terminate(current)
                current = self.provision(
                    force=True, job_file=None,
                    **self._last_provision)["instance"]["id"]
                elapsed = 0
                continue
            try:
                conn = self._conn(current)
            except GpuError:
                conn = {"user": "", "host": "", "port": ""}
            if conn["host"] and conn["port"]:
                if self._ssh_ok(conn):
                    return conn
            if elapsed >= max_wait:
                raise GpuError(f"pod {current} exposed no working SSH "
                               f"within {max_wait}s")
            time.sleep(poll)
            elapsed += poll

    def _ssh_ok(self, conn):
        import subprocess

        cmd = ["ssh", "-o", "LogLevel=ERROR", "-o", "StrictHostKeyChecking=no",
               "-o", "UserKnownHostsFile=/dev/null", "-o", "BatchMode=yes",
               "-p", conn["port"], f"{conn['user']}@{conn['host']}", "true"]
        try:
            subprocess.run(cmd, capture_output=True, timeout=20, check=True)
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError):
            return False

    def _rsh(self, conn):
        return (f"ssh -o LogLevel=ERROR -o StrictHostKeyChecking=no "
                f"-o UserKnownHostsFile=/dev/null -p {conn['port']}")

    # -- software installer --------------------------------------------------
    def run_remote(self, instance_id, script, env=None, timeout=None):
        conn = self.wait_ssh(instance_id)
        target = f"{conn['user']}@{conn['host']}"
        prefix = ""
        if env:
            prefix = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items()) + " "
        try:
            proc = run(["ssh", "-o", "LogLevel=ERROR",
                        "-o", "StrictHostKeyChecking=no",
                        "-o", "UserKnownHostsFile=/dev/null",
                        "-p", conn["port"], target, prefix + script],
                       timeout=timeout)
        except GpuError as e:
            return {"rc": 1, "error": str(e)}
        return {"rc": 0, "stdout": proc.stdout, "stderr": proc.stderr}

    # -- data upload / download ----------------------------------------------
    def upload(self, instance_id, src, dst):
        conn = self.wait_ssh(instance_id)
        retry(lambda: run(["rsync", "-aqz", "--partial", "--progress",
                           "-e", self._rsh(conn),
                           src, f"{conn['user']}@{conn['host']}:{dst}"]),
              attempts=3, backoff=10.0, describe=f"upload {src}")
        return {"src": src, "dst": dst, "instance": instance_id}

    def download(self, instance_id, remote, local, exclude=None):
        conn = self.wait_ssh(instance_id)
        cmd = ["rsync", "-aqz", "--partial", "--progress"]
        for pat in exclude or ["- /checkpoint-*", "- /checkpoint-*/**",
                               "- /runs/**", "- /logs/**"]:
            cmd += ["--filter=" + pat]
        cmd += ["-e", self._rsh(conn),
                f"{conn['user']}@{conn['host']}:{remote}", local]
        retry(lambda: run(cmd), attempts=3, backoff=10.0,
              describe=f"download {remote}")
        return {"remote": remote, "local": local, "instance": instance_id}

    # -- job scheduling / monitoring -------------------------------------------
    def run_job(self, instance_id, command, env=None, timeout=86400):
        return self.run_remote(instance_id, command, env=env, timeout=timeout)

    # -- termination -------------------------------------------------------------
    def terminate(self, instance_id, wait_timeout="5m"):
        _ = wait_timeout  # deletion is synchronous; kept for parity
        try:
            retry(lambda: self._runpod("pod", "delete", str(instance_id)),
                  attempts=3, backoff=10.0,
                  describe=f"terminate {instance_id}")
            return {"id": instance_id, "deleted": True}
        except GpuError as e:
            return {"id": instance_id, "deleted": False, "error": str(e)}

    # -- keys ----------------------------------------------------------------------
    def keys_create(self, name, public_key_file):
        _ = name  # runpodctl names keys itself; kept for parity
        out = self._runpod("ssh", "add-key",
                           "--key-file", public_key_file).stdout
        return {"output": out.strip()}

    def keys_list(self):
        out = self._runpod("ssh", "list-keys").stdout
        try:
            data = json.loads(out or "[]")
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            return {"output": out}
