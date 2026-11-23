"""Local provider — $0 jobs as K8s Jobs on the cluster itself.

A "pod" is a batch Job (one pod, RestartPolicy Never) created via
`k3s kubectl` over SSH on the pi node, with the trainer hostPath lane
(/srv/mcps/trainer) mounted at /data — the exact tree mcp-trainer pods
already read, so the host lane needs zero sync and zero credentials.

v1 targets pi00 CPU (ONNX export, int8 quantize, eval, merge). GPU
jobs on nano0 reuse the same Job path with a nodeSelector + per-node
volume once needed (arm64 CUDA images are a separate build).

Image pull runs on-node (arm64!). LOCAL_SSH_HOST default me@pi00.lan.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess

from ..models import Instance, Offer, ProviderError

NS = os.environ.get("LOCAL_NAMESPACE", "quadra")
LANE_PATH = os.environ.get("LOCAL_WORKDIR", "/srv/mcps/trainer")

_JOB_TMPL = """apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: {ns}
spec:
  backoffLimit: 0
  activeDeadlineSeconds: {deadline}
  template:
    spec:
      restartPolicy: Never
      nodeSelector: {{kubernetes.io/hostname: {node}}}
      containers:
        - name: job
          image: {image}
          {command_block}
          {gpu_block}
          env:
{env_block}
          volumeMounts:
            - {{name: lane, mountPath: /data}}
      volumes:
        - name: lane
          hostPath: {{path: {lane}, type: DirectoryOrCreate}}
"""


class LocalProvider:
    name = "local"

    def __init__(self, ssh_host: str | None = None):
        self.host = (ssh_host or os.environ.get("LOCAL_SSH_HOST",
                                                "me@pi00.lan"))
        self.node = self.host.rsplit("@", 1)[-1].split(".")[0]  # pi00

    # -- ssh/kubectl plumbing --------------------------------------------
    def _kctl(self, remote_args: list[str], stdin: bytes | None = None,
              timeout: int = 120) -> str:
        cmd = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
               self.host, shlex.join(["k3s", "kubectl", *remote_args])]
        r = subprocess.run(cmd, input=stdin, capture_output=True,
                           timeout=timeout)
        if r.returncode != 0:
            raise ProviderError(
                f"local kubectl {' '.join(remote_args[:2])}: "
                f"rc={r.returncode} {r.stderr.decode()[:200]}")
        return r.stdout.decode()

    # -- lifecycle -----------------------------------------------------------
    def launch(self, name: str, image: str, gpu_id: str, *, gpu_count: int = 1,
               cloud: str = "COMMUNITY", env: dict[str, str] | None = None,
               args: str | None = None, deadline_seconds: int = 14400,
               **_ignored) -> Instance:
        env_block = "\n".join(
            f"            - name: {k}\n              value: {shlex.quote(str(v))}"
            for k, v in (env or {}).items())
        gpu_block = ("resources: {limits: {nvidia.com/gpu: 1}}"
                     if "gpu" in gpu_id else "")
        # json.dumps yields a YAML flow sequence with correct quoting
        command_block = (f"command: {json.dumps(['sh', '-c', args])}"
                         if args else "")
        manifest = _JOB_TMPL.format(
            name=name, ns=NS, deadline=deadline_seconds, node=self.node,
            image=image, gpu_block=gpu_block, env_block=env_block,
            command_block=command_block, lane=LANE_PATH)
        self._kctl(["apply", "-f", "-"], stdin=manifest.encode())
        return Instance(provider="local", id=f"{self.host}/{name}",
                        status="provisioning", name=name, gpu_type=gpu_id)

    def poll(self, instance_id: str) -> Instance:
        host, _, name = instance_id.partition("/")
        try:
            raw = self._kctl(["-n", NS, "get", "job", name, "-o", "json"],
                             timeout=30)
        except ProviderError:
            return Instance(provider="local", id=instance_id,
                            status="terminated")        # gone is gone
        j = json.loads(raw)
        st = j.get("status", {})
        if st.get("succeeded"):
            status = "succeeded"          # container exited 0 (rare: agent
        elif st.get("failed"):            # never exits normally)
            status = "failed"
        else:
            pods = self._kctl(
                ["-n", NS, "get", "pods", "-l", f"job-name={name}",
                 "-o", "jsonpath={.items[0].status.phase}"], timeout=30)
            status = {"Running": "running", "Pending": "provisioning",
                      "Succeeded": "succeeded",
                      "Failed": "failed"}.get(pods.strip(), "unknown")
        return Instance(provider="local", id=instance_id, status=status,
                        name=name)

    def terminate(self, instance_id: str) -> bool:
        host, _, name = instance_id.partition("/")
        try:
            self._kctl(["-n", NS, "delete", "job", name, "--wait=false"],
                       timeout=60)
        except ProviderError:
            pass                          # already gone = success
        return True

    # -- discovery -------------------------------------------------------------
    def catalog(self, gpu_ids: list[str] | None = None,
                tier: str | None = None) -> list[Offer]:
        offers = [Offer("local", f"local-cpu-{self.host}", 0, 0.0, True,
                        "HIGH")]
        if os.environ.get("LOCAL_GPU_OFFER") == "1":
            offers.append(Offer("local", f"local-gpu-{self.host}",
                                int(os.environ.get("LOCAL_VRAM_GB", "8")),
                                0.0, True, "HIGH"))
        if gpu_ids:
            offers = [o for o in offers
                      if any(g.lower() in o.gpu_id.lower() for g in gpu_ids)]
        return offers

    def logs(self, instance_id: str, tail: int = 100) -> list[str]:
        _, _, name = instance_id.partition("/")
        try:
            return self._kctl(["-n", NS, "logs", f"job/{name}",
                               "--tail", str(tail)], timeout=60).splitlines()
        except ProviderError as e:
            return [str(e)]

    def list_instances(self) -> list[Instance]:
        try:
            raw = self._kctl(["-n", NS, "get", "jobs", "-o", "json"],
                             timeout=30)
        except ProviderError:
            return []
        out = []
        for j in json.loads(raw).get("items", []):
            name = j["metadata"]["name"]
            if not name.startswith("trainer-"):
                continue
            st = j.get("status", {})
            status = ("succeeded" if st.get("succeeded") else
                      "failed" if st.get("failed") else "running")
            out.append(Instance(provider="local", id=f"{self.host}/{name}",
                                status=status, name=name, gpu_type="local"))
        return out
