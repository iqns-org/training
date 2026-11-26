"""Family capability registry (SPEC 13al): what each base-model family
can serve, as gated DATA rather than tribal knowledge.

The registry lives in CORPORA (recipes/_families.yaml) — it is domain
truth about model architectures, and the training repo ships machinery
only. `refuse_reason` is the enforcement seam: submit_stage asks it
BEFORE creating a ledger row or a pod, so a proven-impossible stage
costs zero dollars and answers with evidence.

Unknown family / missing registry / no entry -> permissive (absence
means the registry hasn't been taught yet). Only an explicit `false` —
backed by evidence — blocks.
"""

from __future__ import annotations

from .config import corpora_dir

REGISTRY_REL = "recipes/_families.yaml"


def registry() -> dict:
    """{family: {"matches": [...], "targets": {...}, "notes": [...]}}"""
    try:
        import yaml
        path = corpora_dir() / REGISTRY_REL
        if not path.is_file():
            return {}
        return (yaml.safe_load(path.read_text()) or {}).get("families", {})
    except Exception:                               # noqa: BLE001
        return {}                                   # never gate on a load bug


def family_for(base_model: str) -> tuple[str, dict] | None:
    if not base_model:
        return None
    for name, ent in registry().items():
        for pat in ent.get("matches", []):
            if str(pat).lower() in base_model.lower():
                return name, ent
    return None


def target_state(base_model: str, kind: str) -> str | None:
    """Registry verdict for (family, stage kind): 'true' | 'untested' |
    'deferred' | 'false' | None (not in registry)."""
    fam = family_for(base_model)
    if not fam:
        return None
    _name, ent = fam
    v = (ent.get("targets") or {}).get(kind)
    return None if v is None else str(v).lower()


def refuse_reason(base_model: str, kind: str) -> str | None:
    """Non-None string = REFUSE this stage, citing the registry. Only an
    explicit false refuses; untested/deferred/unknown proceed."""
    fam = family_for(base_model)
    if fam and target_state(base_model, kind) == "false":
        return (f"family {fam[0]}: target {kind!r} is registry-proven "
                f"IMPOSSIBLE (corpora {REGISTRY_REL} carries the run-id "
                "evidence). Never re-litigate; re-probing requires a "
                "registry change backed by new evidence.")
    return None
