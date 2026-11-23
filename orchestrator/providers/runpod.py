"""RunPod provider — REST API v2 (``https://api.runpod.io/v2``).

Validated live 2026-11-22 against the GraphQL-deprecated surface: scoped
``rpa_`` API keys authenticate on REST v2 (``Authorization: Bearer``) even
where GraphQL returns 401, so the legacy ``runpodctl``-shelling
``src/gpu/runpod.py`` is NOT the base for this adapter.

Endpoints used:
  POST   /v2/pods                 create (name + image + exactly one of gpu/cpu)
  GET    /v2/pods                 list (?fields=ALL)
  GET    /v2/pods/{id}            poll
  DELETE /v2/pods/{id}            terminate
  GET    /v2/pods/{id}/logs       provider-native pod logs (?tail=)
  GET    /v2/catalog/gpus         live offers incl. community/secure pricing
"""

from __future__ import annotations

import os
import re

import httpx

from ..config import require_env
from ..models import Instance, Offer, ProviderError

API = "https://api.runpod.io"

# Provider status → normalized lowercase used by models/ledger.
_STATUS = {
    "CREATING": "creating", "QUEUED": "pending", "PROVISIONING": "provisioning",
    "STARTING": "starting", "RUNNING": "running", "SCHEDULED": "provisioning",
    "RESTARTING": "restarting", "EXITED": "stopped", "TERMINATED": "terminated",
    "TERMINATING": "terminating", "ERROR": "error", "FAILED": "failed",
    "DELETED": "deleted",
}


def normalize_pod(pod: dict) -> Instance:
    """Defensive mapping of a Pod object. `ports` shapes observed in the wild:
    - spec-echo list of strings: ["8000/http", "22/tcp"]  (RunPod REST v2 —
      no mapping data: base_url stays empty, ssh_direct is the access lane)
    - list of dicts with ip/port (legacy/GraphQL-ish)
    - dict mapping "port/protocol" -> {ip, port}
    """
    status = _STATUS.get(str(pod.get("status", "")).upper(),
                         str(pod.get("status", "unknown")).lower())
    gpu = pod.get("gpu") or {}
    base_url = ""
    ports = pod.get("ports") or []
    candidates: list[tuple[str, dict]] = []
    if isinstance(ports, dict):
        candidates = [(str(k), v if isinstance(v, dict) else {})
                      for k, v in ports.items()]
    elif isinstance(ports, list):
        candidates = [(str(p.get("type", "")), p) for p in ports
                      if isinstance(p, dict)]
    for kind, port in candidates:
        k = kind.lower()
        if ("http" in k or k == "tcp") and port.get("ip"):
            base_url = f"http://{port['ip']}:{port.get('port')}"
            break
    price = gpu.get("secondPrice") or gpu.get("price") or pod.get("cost")
    try:
        price_hr = float(price) if price is not None else None
    except (TypeError, ValueError):
        price_hr = None
    return Instance(
        provider="runpod",
        id=str(pod.get("id", "")),
        status=status,
        name=pod.get("name", ""),
        gpu_type=gpu.get("id") or pod.get("gpuTypeId") or "",
        price_hr=price_hr,
        base_url=base_url,
        raw=pod,
    )


def map_offers(items: list[dict], gpu_ids: list[str] | None = None,
               tier: str | None = None) -> list[Offer]:
    """Catalog JSON → orderable Offers, cheapest first. Drops entries with no
    stock (availability NONE), a missing per-tier flag, or zero price."""
    offers: list[Offer] = []
    for g in items:
        gid = g.get("id", "")
        if gpu_ids and not any(re.search(re.escape(w), gid, re.I)
                               for w in gpu_ids):
            continue
        avail = str(g.get("availability") or "")
        tiers = [("secure", True), ("community", False)]
        if tier:
            tiers = [t for t in tiers if t[0] == tier.lower()]
        for key, is_secure in tiers:
            if not g.get(key):          # tier not offered for this GPU
                continue
            price = (g.get("price") or {}).get(key)
            if not price or avail.upper() == "NONE":
                continue
            offers.append(Offer(provider="runpod", gpu_id=gid,
                                vram_gb=int(g.get("memory") or 0),
                                price_hr=float(price), secure=is_secure,
                                availability=avail))
    offers.sort(key=lambda o: o.price_hr)
    return offers


class RunpodProvider:
    name = "runpod"

    def __init__(self, api_key: str | None = None, timeout: float = 30.0):
        self.api_key = api_key or require_env("RUNPOD_API_KEY")
        self.timeout = timeout

    # -- http -------------------------------------------------------------
    def _req(self, method: str, path: str, **kwargs) -> dict | list:
        try:
            resp = httpx.request(
                method, f"{API}{path}", timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.api_key}"}, **kwargs)
        except httpx.HTTPError as e:
            raise ProviderError(f"runpod {method} {path}: {e}") from e
        if resp.status_code >= 400:
            detail = ""
            try:
                body = resp.json()
                detail = body.get("detail") or body.get("title") or str(body)[:300]
            except Exception:
                detail = resp.text[:300]
            raise ProviderError(f"runpod {method} {path} → HTTP {resp.status_code}: {detail}")
        if not resp.content:
            return {}
        return resp.json()

    # -- lifecycle ----------------------------------------------------------
    def launch(self, name: str, image: str, gpu_id: str, *, gpu_count: int = 1,
               cloud: str = "COMMUNITY", env: dict[str, str] | None = None,
               ports: tuple[str, ...] = ("8000/http",), disk_gb: int = 30,
               args: str | None = None, ssh: bool = False,
               volume_id: str | None = None, volume_path: str = "/data",
               data_center: str | None = None) -> Instance:
        body: dict = {
            "name": name,
            "image": image,
            "gpu": {"id": gpu_id, "count": gpu_count},
            "cloud": cloud.upper(),
            "ports": list(ports),
            "disk": disk_gb,
        }
        if volume_id:
            # Network volume REPLACES the volume disk; mount at a custom path
            # (/workspace would shadow the base image's axolotl venv!).
            body["mounts"] = {"network": [{"volumeId": volume_id,
                                           "path": volume_path}]}
        if data_center:
            # volume must live in the pod's DC -> pin placement (SPEC §13 g).
            body["dataCenterIds"] = [data_center]
        if ssh:  # debug/exec lane: RunPod v2 exposes NO port mappings for pods,
                  # so pod-side HTTP is reachable via ssh.direct (22/tcp) only
            body["startSsh"] = True
            if not any(p.startswith("22/") for p in body["ports"]):
                body["ports"] = list(body["ports"]) + ["22/tcp"]
        if env:
            body["env"] = dict(env)
        if args:
            body["args"] = args
        # NOTE: no retry on a lost response — a retry could strand a
        # duplicate billable pod (same rule as the legacy gpuai adapter).
        pod = self._req("POST", "/v2/pods", json=body)
        return normalize_pod(pod if isinstance(pod, dict) else {})

    def poll(self, instance_id: str) -> Instance:
        pod = self._req("GET", f"/v2/pods/{instance_id}")
        return normalize_pod(pod if isinstance(pod, dict) else {})

    def terminate(self, instance_id: str) -> bool:
        try:
            self._req("DELETE", f"/v2/pods/{instance_id}")
        except ProviderError as e:
            if "404" in str(e):
                return True
            return False
        try:
            rec = self.poll(instance_id)
            return rec.status in {"terminated", "deleted", "failed"}
        except ProviderError:
            return True   # gone is gone

    # -- discovery -----------------------------------------------------------
    def catalog(self, gpu_ids: list[str] | None = None,
                tier: str | None = None) -> list[Offer]:
        """Live offers. With AVAILABILITY data, unorderable GPUs are filtered
        (NONE / missing tier flag / zero price) so callers stop learning via
        400s at launch time.

        ``tier``: "community" | "secure" | None (both).
        """
        data = self._req("GET", "/v2/catalog/gpus",
                         params={"limit": 100, "include": "AVAILABILITY",
                                 "product": "POD"})
        items = data if isinstance(data, list) else next(
            (v for v in data.values() if isinstance(v, list)), [])
        return map_offers(items, gpu_ids, tier)

    def ssh_direct(self, instance_id: str) -> dict:
        """{'user','host','port'} for the published 22/tcp, or {} if absent."""
        pod = self._req("GET", f"/v2/pods/{instance_id}")
        direct = (pod.get("ssh") or {}).get("direct") or {}
        return {"user": direct.get("username", ""), "host": direct.get("host", ""),
                "port": str(direct.get("port", ""))} if direct else {}

    def logs(self, instance_id: str, tail: int = 100) -> list[str]:
        try:
            data = self._req("GET", f"/v2/pods/{instance_id}/logs",
                             params={"tail": tail})
        except ProviderError as e:
            if "403" in str(e):
                raise ProviderError(
                    "runpod logs denied (scoped API key without that right) — "
                    "training logs live on the agent lane: GET <base_url>/v1/"
                    "jobs/<run_id>/logs") from e
            raise
        entries = data.get("logs") if isinstance(data, dict) else data
        lines = []
        for e in entries or []:
            if isinstance(e, dict):
                lines.append(str(e.get("message", e)))
            else:
                lines.append(str(e))
        return lines

    def list_instances(self) -> list[Instance]:
        data = self._req("GET", "/v2/pods", params={"fields": "ALL"})
        pods = data.get("pods", []) if isinstance(data, dict) else data
        return [normalize_pod(p) for p in pods or []]

    def balance(self) -> float:
        """Prepaid balance — a useful hard ceiling signal for budget logic."""
        data = self._req("GET", "/v2/billing/balance")
        return float(data.get("balanceAmount", 0) or 0)
