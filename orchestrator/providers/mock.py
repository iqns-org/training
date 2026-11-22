"""Deterministic fake provider for CI + local pipeline development (SPEC §6).

State advances one step per ``poll`` call: creating → running (with a
``base_url``). No network. Catalog is a static, realistic subset.
"""

from __future__ import annotations

from ..models import Instance, Offer


class MockProvider:
    name = "mock"

    _CATALOG = [
        ("NVIDIA RTX A5000", 24, 0.16, False),
        ("NVIDIA A40", 48, 0.35, False),
        ("NVIDIA GeForce RTX 4090", 24, 0.34, False),
        ("NVIDIA RTX 5000 Ada Generation", 32, 0.49, False),
        ("NVIDIA A100 SXM", 80, 1.19, True),
    ]

    def __init__(self) -> None:
        self._pods: dict[str, dict] = {}
        self._seq = 0

    def launch(self, name, image, gpu_id, *, gpu_count=1, cloud="COMMUNITY",
               env=None, ports=("8000/http",), disk_gb=30, args=None) -> Instance:
        self._seq += 1
        pid = f"mock-{self._seq:04d}"
        self._pods[pid] = {"polls": 0, "image": image, "gpu": gpu_id, "name": name}
        return Instance(provider="mock", id=pid, status="creating", name=name,
                        gpu_type=gpu_id, raw={"env": env or {}})

    def poll(self, instance_id: str) -> Instance:
        pod = self._pods.get(instance_id)
        if pod is None:
            from ..models import ProviderError
            raise ProviderError(f"mock: unknown pod {instance_id}")
        pod["polls"] += 1
        status = "creating" if pod["polls"] < 2 else "running"
        return Instance(provider="mock", id=instance_id, status=status,
                        name=pod["name"], gpu_type=pod["gpu"],
                        base_url="http://mock.local:8000" if status == "running" else "")

    def terminate(self, instance_id: str) -> bool:
        return self._pods.pop(instance_id, None) is not None

    def catalog(self, gpu_ids=None, tier=None) -> list[Offer]:
        return [Offer("mock", gid, vram, price, secure, "HIGH")
                for gid, vram, price, secure in self._CATALOG
                if not gpu_ids or any(g.lower() in gid.lower() for g in gpu_ids)]

    def logs(self, instance_id: str, tail: int = 100) -> list[str]:
        return [f"mock log line {i}" for i in range(min(tail, 5))]

    def list_instances(self) -> list[Instance]:
        return [Instance(provider="mock", id=k, status="running", name=v["name"])
                for k, v in self._pods.items()]
