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
