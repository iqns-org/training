"""Core data types for the orchestrator (SPEC §4/§6)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# Run state machine (SPEC §6). Terminal: done, failed, cancelled.
STATES = [
    "pending", "provisioning", "running", "uploading",
    "delivered", "verified", "done", "failed", "cancelled", "lost",
]
TERMINAL = {"done", "failed", "cancelled"}


def now() -> float:
    return time.time()


@dataclass
class Offer:
    provider: str
    gpu_id: str
    vram_gb: int
    price_hr: float
    secure: bool
    availability: str = ""     # HIGH | MEDIUM | LOW | NONE | ""


@dataclass
class Instance:
    provider: str
    id: str
    status: str                      # normalized lowercase
    name: str = ""
    gpu_type: str = ""
    price_hr: float | None = None
    base_url: str = ""               # agent API endpoint when known
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def alive(self) -> bool:
        return self.status not in {"terminated", "deleted", "failed",
                                   "stopped", "error", "dead"}

    @property
    def ready(self) -> bool:
        return self.status in {"running", "READY", "RUNNING"} and bool(self.base_url)

    @property
    def created_epoch(self) -> float | None:
        """Best-effort pod creation time (epoch seconds) from the raw
        provider payload; None when the provider does not expose it."""
        raw = self.raw or {}
        for key in ("createdat", "createdAt", "created", "starttime",
                    "startedat", "startTime"):
            v = raw.get(key)
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str) and v:
                try:
                    from datetime import datetime, timezone
                    iso = v.replace("Z", "+00:00")
                    return datetime.fromisoformat(iso).timestamp()
                except ValueError:
                    continue
        return None


class ProviderError(RuntimeError):
    """Actionable provider failure (auth, capacity, bad payload, …)."""
