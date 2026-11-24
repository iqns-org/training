"""orchestrator/notify.py — journal, listeners, webhook best-effort."""
import json

from orchestrator import notify


def test_journal_alone_needs_no_webhook(tmp_path, monkeypatch):
    monkeypatch.setattr(notify, "JOURNAL", tmp_path / "n.jsonl")
    ev = notify.emit("launched", "r1", kind="train", gpu="A5000")
    assert ev["event"] == "launched" and ev["run_id"] == "r1"
    rows = [json.loads(x) for x in
            (tmp_path / "n.jsonl").read_text().splitlines()]
    assert rows == [ev]


def test_event_kind_split(tmp_path, monkeypatch):
    """The §14 contract: 'event' names the transition, 'kind' names the
    resource — and a caller-provided kind can never forge the event."""
    monkeypatch.setattr(notify, "JOURNAL", tmp_path / "n.jsonl")
    ev = notify.emit("reaped", "r9", kind="pod", pod="p-1")
    assert ev["event"] == "reaped" and ev["kind"] == "pod"
    # kind as a positional field name would previously collide; now the
    # fixed keys are written last, so fields cannot masquerade:
    ev2 = notify.emit("run_done", "r1", kind="awq")
    assert ev2["event"] == "run_done" and ev2["kind"] == "awq"
    ev3 = notify.emit("progress", "r1")   # no kind + unknown run => ""
    assert ev3["kind"] == "" and ev3["event"] == "progress"


def test_recent_filters_by_run(tmp_path, monkeypatch):
    monkeypatch.setattr(notify, "JOURNAL", tmp_path / "n.jsonl")
    notify.emit("progress", "r1", pct=10.0)
    notify.emit("progress", "r2", pct=20.0)
    assert [e["run_id"] for e in notify.recent()] == ["r1", "r2"]
    assert [e["run_id"] for e in notify.recent(run_id="r2")] == ["r2"]
    assert len(notify.recent(limit=1)) == 1


def test_listeners_fire_and_survive_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(notify, "JOURNAL", tmp_path / "n.jsonl")
    seen, boom = [], []

    def good(e):
        seen.append(e)

    def bad(e):
        boom.append(e)
        raise RuntimeError("listener crashed")

    notify.register_listener(good)
    notify.register_listener(bad)
    try:
        notify.emit("run_done", "r1")            # must NOT raise
        assert seen and boom
    finally:
        notify.unregister_listener(good)
        notify.unregister_listener(bad)
    notify.emit("run_done", "r2")
    assert len(seen) == 1                          # unregistered for real


def test_webhook_none_is_silent(tmp_path, monkeypatch):
    monkeypatch.setattr(notify, "JOURNAL", tmp_path / "n.jsonl")
    monkeypatch.delenv("QTRAIN_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(notify, "webhook_url", lambda: "")
    ok, msg = notify.deliver_sync({"kind": "x"})
    assert not ok and "QTRAIN_WEBHOOK_URL" in msg
