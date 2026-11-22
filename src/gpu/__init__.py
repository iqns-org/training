"""GPU provider abstraction.

GPU.ai is one provider; RunPod (and future providers) share the same
interface. All providers live under ``src/gpu/`` namespaced as
``src/gpu/<provider>.py`` and register in :data:`PROVIDERS`.

Usage:
    from gpu import get_provider
    provider = get_provider("gpuai")  # or "runpod"
    inst = provider.provision(app="axolotl-finetuning", name="...", ssh_key_id="...")
"""

from .base import Provider
from .common import GpuError

__all__ = ["Provider", "GpuError", "PROVIDERS", "get_provider", "available_providers"]


def _load_providers():
    from . import gpuai, runpod

    return {"gpuai": gpuai.GpuAiProvider, "runpod": runpod.RunpodProvider}


PROVIDERS = _load_providers()


def available_providers():
    """Names of registered providers."""
    return sorted(PROVIDERS)


def get_provider(name, **kwargs):
    """Instantiate a registered provider by name.

    Raises GpuError for unknown providers (lists what is available).
    """
    try:
        cls = PROVIDERS[name]
    except KeyError:
        raise GpuError(
            f"unknown provider {name!r} (available: {', '.join(available_providers())})"
        )
    return cls(**kwargs)
