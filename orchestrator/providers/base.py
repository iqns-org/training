"""Provider protocol for the agent-oriented cloud lane (SPEC §6).

Deliberately narrower than the legacy ``src/gpu/base.py``: no SSH, no
upload/download. Datasets and artifacts move via pre-signed S3 URLs; the pod
is driven through its own REST API (see ``agent/``).
"""

from __future__ import annotations

from typing import Protocol

from ..models import Instance, Offer


class CloudProvider(Protocol):
    name: str

    def launch(self, name: str, image: str, gpu_id: str, *,
               gpu_count: int = 1, cloud: str = "COMMUNITY",
               env: dict[str, str] | None = None,
               ports: tuple[str, ...] = ("8000/http",),
               disk_gb: int = 30,
               args: str | None = None) -> Instance:
        """Create a pod running ``image``; returns its record (may not be
        running yet). MUST NOT retry a lost creation response (could strand a
        duplicate billable pod)."""
        ...

    def poll(self, instance_id: str) -> Instance:
        """Current record; raises ProviderError when the id is unknown/gone."""
        ...

    def terminate(self, instance_id: str) -> bool:
        """Delete the pod. True when it is gone afterwards."""
        ...

    def catalog(self, gpu_ids: list[str] | None = None,
                tier: str | None = None) -> list[Offer]:
        """Buyable GPU offers (ordered), optionally filtered by (substring)
        id/name and tier ("community"|"secure")."""
        ...

    def logs(self, instance_id: str, tail: int = 100) -> list[str]:
        """Pod stdout/stderr lines (provider-native, not SSH)."""
        ...

    def list_instances(self) -> list[Instance]:
        ...
