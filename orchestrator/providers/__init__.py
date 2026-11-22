"""Provider registry."""

from .mock import MockProvider
from .runpod import RunpodProvider

_PROVIDERS = {
    "runpod": RunpodProvider,
    "mock": MockProvider,
}


def available_providers() -> list[str]:
    return sorted(_PROVIDERS)


def get_provider(name: str, **kwargs):
    try:
        cls = _PROVIDERS[name]
    except KeyError:
        raise ValueError(f"unknown provider {name!r} "
                         f"(have: {', '.join(available_providers())})") from None
    return cls(**kwargs)
