import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# work without an editable install too
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent"))


@pytest.fixture(autouse=True)
def _notify_isolation(tmp_path, monkeypatch):
    """Never write the repo's real journal from tests, and never fire a
    real webhook if .env happens to carry QTRAIN_WEBHOOK_URL."""
    from orchestrator import budget, notify
    monkeypatch.setattr(notify, "JOURNAL", tmp_path / "notifications.jsonl")
    monkeypatch.setattr(notify, "webhook_url", lambda: "")
    monkeypatch.setattr(budget, "OBS_FILE", tmp_path / "budget_obs.json")


@pytest.fixture(autouse=True)
def _cold_schema_cache():
    """SCHEMA-LINTER CACHE RESET (unknown_keys guard): dispatch caches the
    fixture across calls, but its answer depends on module-level ROOT —
    which every test redirects independently. Without this reset, file
    ordering leaks a loaded fixture into tmp-ROOT tests and makes
    synthetic-recipe assertions order-dependent. Every test starts COLD:
    the loader then hits (or misses) the fixture under ITS OWN ROOT,
    exactly like production."""
    from orchestrator import dispatch
    dispatch._SCHEMA_KEYS = None
    dispatch._SCHEMA_KEYS_LOADED = False
    yield
    dispatch._SCHEMA_KEYS = None
    dispatch._SCHEMA_KEYS_LOADED = False
