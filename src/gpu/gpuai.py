"""GPU.ai provider — port of bin/gpuai-deploy / job / download / keys.

Talks to the ``gpu`` CLI (JSON in/out parsed in Python instead of jq) and
preserves the hard-won behaviors: stale job-file detection, redeploy on
terminal states (max 2), SSH-metadata wait loop, and key lookup by hostname.
"""

import json
import socket

from .base import Provider
from .common import (
    GpuError, extract_json_payload, is_active, is_terminal,
    load_job, parse_ssh_connection, retry, run, save_job,
)


class GpuAiProvider(Provider):
    name = "gpuai"

    def __init__(self, key_file="secrets/gpuai", max_redeploys=2):
        self.key_file = key_file
        self.max_redeploys = max_redeploys

    # -- internal ------------------------------------------------------
    def _gpu(self, *args):
        return run(["gpu", *args])

    # -- provision / creation -------------------------------------------
    def provision(self, app, name=None, ssh_key_id=None, job_file=None,
                  force=False, **kwargs):
        """Deploy ``app`` (or reuse the live instance recorded in job_file)."""
        if job_file and not force:
            saved = load_job(job_file)
            inst_id = (saved or {}).get("instance", {}).get("id", "")
            if inst_id:
                try:
                    rec = self.status(inst_id)
                    if is_active(rec.get("status")):
                        return {"instance": rec, "reused": True}
                except GpuError:
                    pass
        if ssh_key_id is None:
            ssh_key_id = self._key_id_for_host()
        # NOTE: the deploy call itself is deliberately NOT retried — a
        # retry after a lost response could create a duplicate billable
        # instance. Recovery happens via the saved job file + wait_ssh.
        out = self._gpu("deploy", app, "--name", name or f"{app}",
                        "--ssh-key-id", ssh_key_id, **kwargs).stdout
        payload = extract_json_payload(out)
        if not payload.get("instance", {}).get("id"):
            raise GpuError(f"deploy returned no instance id: {out[:500]}")
        if job_file:
            save_job(job_file, payload)
        return {"instance": payload["instance"], "reused": False}

    def _key_id_for_host(self):
        out = self._gpu("keys", "list", "-o", "table").stdout
        host = socket.gethostname()
        for line in out.splitlines():
            if host and host in line:
                return line.split()[0]
        raise GpuError(f"no GPU key found for host {host!r} "
                       f"(create one: bin/gpu --provider gpuai keys ...)")

    # -- inspection / monitoring -----------------------------------------
    def list_instances(self):
        def _fetch():
            out = self._gpu("instances", "list", "-o", "json").stdout.strip()
            try:
                data = json.loads(out or "[]")
            except json.JSONDecodeError:
                raise GpuError(f"could not parse `gpu instances list`: {out[:500]}")
            return data if isinstance(data, list) else [data]

        # Transient CLI/API blips must not kill a whole workflow.
        return retry(_fetch, attempts=3, backoff=5.0,
                     describe="`gpu instances list`")

    def status(self, instance_id):
        for rec in self.list_instances():
            if str(rec.get("id")) == str(instance_id):
                return rec
        raise GpuError(f"instance {instance_id} not found")

    def wait_ssh(self, instance_id, max_wait=1800, poll=15, redeploy=None):
        """Wait for SSH metadata; redeploy on terminal states (deploy idea)."""
        import time

        redeploy = self.max_redeploys if redeploy is None else redeploy
        attempts, elapsed = 0, 0
        current = instance_id
        while True:
            try:
                rec = self.status(current)
            except GpuError:
                rec = {"id": current, "status": "unknown"}
            if is_terminal(rec.get("status")):
                attempts += 1
                if attempts > redeploy:
                    raise GpuError(f"instance {current} terminal "
                                   f"({rec.get('status')}) after {redeploy} redeploys")
                self.terminate(current)
                current = self.provision(
                    app=rec.get("app", "axolotl-finetuning"),
                    name=rec.get("name", current))["instance"]["id"]
                elapsed = 0
                continue
            conn = parse_ssh_connection(rec)
            if conn["host"] and conn["port"] and conn["user"]:
                conn["id"] = current
                return conn
            if elapsed >= max_wait:
                raise GpuError(f"instance {current} exposed no SSH metadata "
                               f"within {max_wait}s")
            time.sleep(poll)
            elapsed += poll

    # -- ssh plumbing ------------------------------------------------------
    def _ssh_target(self, instance_id):
        conn = self.wait_ssh(instance_id)
        return conn, (f"{conn['user']}@{conn['host']}", conn["port"])

    def _ssh_base(self, port):
        from pathlib import Path

        key = Path(self.key_file)
        if not key.is_file():
            # also try repo-root relative
            root_key = Path(__file__).resolve().parents[2] / self.key_file
            if root_key.is_file():
                key = root_key
            else:
                raise GpuError(f"SSH private key not found: {self.key_file}")
        base = ["-o", "LogLevel=ERROR", "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null", "-o", "IdentitiesOnly=yes",
                "-i", str(key), "-p", str(port)]
        return base

    # -- software installer --------------------------------------------------
    def run_remote(self, instance_id, script, env=None, timeout=None):
        import shlex

        conn = self.wait_ssh(instance_id)
        target, port = f"{conn['user']}@{conn['host']}", conn["port"]
        prefix = ""
        if env:
            prefix = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items()) + " "
        cmd = ["ssh", *self._ssh_base(port), target, prefix + script]
        try:
            proc = run(cmd, timeout=timeout)
        except GpuError as e:
            return {"rc": 1, "error": str(e)}
        return {"rc": 0, "stdout": proc.stdout, "stderr": proc.stderr}

    # -- data upload / download ----------------------------------------------
    def upload(self, instance_id, src, dst):
        conn = self.wait_ssh(instance_id)
        target, port = f"{conn['user']}@{conn['host']}", conn["port"]
        from pathlib import Path

        key = next((p for p in [Path(self.key_file)] if p.is_file()), None)
        rsh = (f"ssh -o LogLevel=ERROR -o StrictHostKeyChecking=no "
               f"-o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes "
               f"-i {key or self.key_file} -p {port}")
        import shlex

        # rsync over the network flakes; transfers are idempotent, so retry.
        # --partial keeps interrupted files so retries resume them.
        retry(lambda: run(["rsync", "-aqz", "--partial", "--progress",
                           "-e", rsh, src, f"{target}:{dst}"]),
              attempts=3, backoff=10.0, describe=f"upload {src}")
        return {"src": src, "dst": dst, "instance": instance_id}

    def download(self, instance_id, remote, local, exclude=None):
        conn = self.wait_ssh(instance_id)
        target, port = f"{conn['user']}@{conn['host']}", conn["port"]
        from pathlib import Path

        key = next((p for p in [Path(self.key_file)] if p.is_file()), None)
        rsh = (f"ssh -o LogLevel=ERROR -o StrictHostKeyChecking=no "
               f"-o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes "
               f"-i {key or self.key_file} -p {port}")
        cmd = ["rsync", "-aqz", "--partial", "--progress"]
        for pat in exclude or ["- /checkpoint-*", "- /checkpoint-*/**",
                               "- /runs/**", "- /logs/**"]:
            cmd += ["--filter=" + pat]
        cmd += ["-e", rsh, f"{target}:{remote}", local]
        retry(lambda: run(cmd), attempts=3, backoff=10.0,
              describe=f"download {remote}")
        return {"remote": remote, "local": local, "instance": instance_id}

    # -- job scheduling / monitoring -------------------------------------------
    def run_job(self, instance_id, command, env=None, timeout=86400):
        import shlex

        prefix = ""
        if env:
            prefix = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items()) + " "
        result = self.run_remote(instance_id, prefix + command, env=None,
                                 timeout=timeout)
        return result

    def fine_tune(self, dataset, model, method="qlora", max_budget=1,
                  wait_timeout="2h", **lora):
        """Thin wrapper over `gpu fine-tune` (bin/gpuai-fine-tune idea)."""
        cmd = ["gpu", "fine-tune", "--dataset", dataset, "--model", model,
               "--method", method, "--max-budget", str(max_budget),
               "--wait-timeout", wait_timeout]
        for k, v in lora.items():
            cmd += [f"--lora-{k.replace('_', '-')}", str(v)]
        out = run(cmd).stdout
        return extract_json_payload(out) if "{" in out else {"output": out}

    # -- termination -------------------------------------------------------------
    def terminate(self, instance_id, wait_timeout="5m"):
        # A failed delete leaks a billable instance, so this is retried;
        # deletion is idempotent ("not found" still ends billed state).
        try:
            retry(lambda: self._gpu("instances", "delete", str(instance_id),
                                    "--wait-timeout", wait_timeout),
                  attempts=3, backoff=10.0,
                  describe=f"terminate {instance_id}")
            return {"id": instance_id, "deleted": True}
        except GpuError as e:
            return {"id": instance_id, "deleted": False, "error": str(e)}

    # -- keys ----------------------------------------------------------------------
    def keys_create(self, name, public_key_file):
        out = self._gpu("keys", "create", "--name", name,
                        "--public-key-file", public_key_file).stdout
        try:
            return extract_json_payload(out)
        except GpuError:
            return {"output": out.strip()}

    def keys_list(self, output="json"):
        out = self._gpu("keys", "list", "-o", output).stdout
        if output == "json":
            try:
                return json.loads(out or "[]")
            except json.JSONDecodeError:
                raise GpuError(f"could not parse keys list: {out[:300]}")
        return {"output": out}
