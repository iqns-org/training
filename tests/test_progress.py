"""orchestrator/progress.py + dispatch._progress_tick (SPEC 14)."""
import json
import time

from orchestrator import dispatch, notify, progress
from orchestrator.ledger import Ledger


def test_parse_runlog_extracts_metric_dicts():
    text = (
        "some banner\n"
        "{'loss': '4.699', 'learning_rate': '4e-05', 'epoch': '0.02326'}\n"
        "noise line\n"
        "{'loss': '1.935', 'epoch': '2'}\n"
        "{'train_runtime': '808.1', 'train_loss': '2.589'}\n")
    pts = progress.parse_runlog(text)
    assert [p["loss"] for p in pts] == ["4.699", "1.935", "2.589"]
    assert pts[-1].get("train_runtime") is None   # only metric keys land


def test_progress_tick_appends_history_and_emits(tmp_path, monkeypatch):
    journal = tmp_path / "n.jsonl"
    monkeypatch.setattr(notify, "JOURNAL", journal)
    lane = {
        "runs/r1/status.json": json.dumps(
            {"state": "running", "step": 10, "total": 100,
             "loss": 4.2}).encode(),
    }
    get_run = lambda k: lane.get(k)
    put_run = lambda k, d, ct="": lane.__setitem__(k, d)

    memo = {}
    t0 = time.time() - 60
    dispatch._progress_tick("r1", get_run, put_run, 0.60, t0, memo)
    # identical heartbeat again: no duplicate row, no duplicate event
    dispatch._progress_tick("r1", get_run, put_run, 0.60, t0, memo)
    assert lane["runs/r1/progress.jsonl"].decode().count("\n") == 1

    lane["runs/r1/status.json"] = json.dumps(
        {"state": "running", "step": 55, "total": 100, "loss": 3.1}).encode()
    dispatch._progress_tick("r1", get_run, put_run, 0.60, t0, memo)
    assert lane["runs/r1/progress.jsonl"].decode().count("\n") == 2

    events = [json.loads(x) for x in journal.read_text().splitlines()]
    kinds = [(e["event"], e.get("pct")) for e in events]
    assert ("progress", 10.0) in kinds        # first tick always emits
    assert ("progress", 55.0) in kinds        # >=10% jump emits
    cost = [e for e in events if e.get("pct") == 55.0][0]["cost"]
    assert cost > 0.6 * 60 / 3600 - 0.01      # wall x rate sanity


def test_progress_tick_tolerates_dead_lane(tmp_path, monkeypatch):
    monkeypatch.setattr(notify, "JOURNAL", tmp_path / "n.jsonl")

    def get_run(k):
        raise RuntimeError("s3 down")
    memo = {}
    for _ in range(4):                        # under the blip limit
        dispatch._progress_tick("r1", get_run, None, 0.5, time.time(), memo)


def test_snapshot_joins_ledger_and_lane(tmp_path, monkeypatch):
    (tmp_path / "state").mkdir()
    led = Ledger(tmp_path / "state" / "ledger.db")
    rid = led.create("train", {"domain": "d"})
    led.update(rid, state="provisioning", provider="runpod")
    monkeypatch.setattr(progress, "ROOT", tmp_path)
    monkeypatch.setattr(progress, "status",
                        lambda r, p="runpod": {"state": "running",
                                               "step": 3, "total": 10,
                                               "loss": 4.1})
    monkeypatch.setattr(progress, "history",
                        lambda r, p="runpod", limit=0: [{"state": "running",
                                                         "cost": 0.01}])
    snap = progress.snapshot(rid)
    assert snap["state"] == "provisioning"     # ledger truth
    assert snap["pct"] == 30.0 and snap["live"]["loss"] == 4.1
    assert progress.fmt_line(snap).count("step 3/10") == 1
