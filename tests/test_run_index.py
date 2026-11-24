"""runs/index.jsonl — the UI-facing lane rollup (SPEC 14c).

Covers the three gaps the audit found: (1) no cross-run index on the PVC,
(2) terminal metrics absent from the manifest, (3) image_tag empty. The
index joins manifest + ledger + parsed run.log so a front end renders a
run table with one GET; reindex() backfills historical runs.
"""
from __future__ import annotations

import json

import pytest

from orchestrator import dispatch, progress


@pytest.fixture
def lane(monkeypatch):
    """In-memory dict lane replacing storage on both read+write sides."""
    store: dict = {}

    def get_run(key):
        return store.get(key)

    def put_run(key, data, ct=""):
        store[key] = data if isinstance(data, (bytes, bytearray)) \
            else str(data).encode()
        store.setdefault("_ct", {})[key] = ct

    # progress.read_index / reindex go through lane_getter/lane_putter
    monkeypatch.setattr(progress, "lane_getter", lambda provider: get_run)
    monkeypatch.setattr(progress, "lane_putter",
                        lambda provider: put_run)
    return store


def _manifest():
    return {
        "schema": 2, "run_id": "r1", "status": "succeeded",
        "agent_version": "0.3.1", "image_tag": "ghcr.io/iqns-org/trainer@x",
        "started_at": 1000.0, "finished_at": 1600.0,
        "training": {"state": "succeeded", "step": 200, "total": 200,
                     "epoch": 2.0, "loss": 1.9, "exit_code": 0},
        "hardware": {"gpus": ["NVIDIA RTX 2000 Ada Generation, 16380 MiB"]},
        "artifacts": {"adapter_model.safetensors": {"size": 1, "sha256": "a"},
                      "config.json": {"size": 2, "sha256": "b"}},
        "problems": [], "artifacts_tar_sha256": "deadbeef",
    }


def test_build_record_flat_and_joined():
    row = {"kind": "train", "provider": "runpod",
           "spec_json": json.dumps({"domain": "queer", "recipe": "r.yaml"})}
    log = ("{'loss': 4.699, 'epoch': 0.02, 'step': 5}\n"
           "{'loss': 2.589, 'epoch': 2.0, 'step': 200}\n"
           "{'train_loss': 2.59, 'train_runtime': 60.0}\n")
    rec = progress.build_record("r1", _manifest(), row, log_text=log)
    assert rec["run_id"] == "r1"
    assert rec["kind"] == "train" and rec["domain"] == "queer"
    assert rec["status"] == "succeeded" and rec["seconds"] == 600
    assert rec["gpu"] == "NVIDIA RTX 2000 Ada Generation"
    assert rec["n_artifacts"] == 2 and rec["artifacts_sha"] == "deadbeef"
    assert rec["image_tag"].startswith("ghcr.io")   # gap #3: tag present
    # last curve point is the HF trainer's own train_loss summary line
    assert rec["first_loss"] == 4.699 and rec["last_loss"] == 2.59  # gap #2
    assert rec["min_loss"] == 2.589 and rec["n_points"] == 3
    # flat: no nested dicts (UI renders without schema archaeology)
    assert all(not isinstance(v, dict) for v in rec.values())
    # None/empty dropped
    assert "cost" not in rec


def test_build_record_stage_spec():
    row = {"kind": "stage", "provider": "runpod",
           "est_cost": 0.04,
           "spec_json": json.dumps({"kind": "awq", "from_run": "r0",
                                    "base_model": None})}
    rec = progress.build_record("r2", _manifest(), row)
    assert rec["kind"] == "awq" and rec["from_run"] == "r0"
    assert rec["est_cost"] == 0.04              # ledger estimate surfaces
    assert "base_model" not in rec              # None dropped


def test_append_index_rmw(lane):
    a = progress.build_record("ra", _manifest(),
                              {"kind": "train", "spec_json": "{}",
                               "provider": "runpod"})
    b = progress.build_record("rb", _manifest(),
                              {"kind": "train", "spec_json": "{}",
                               "provider": "runpod"})
    get = progress.lane_getter("runpod")
    put = progress.lane_putter("runpod")
    progress.append_index(a, get, put)
    progress.append_index(b, get, put)
    rows = progress.read_index("runpod")
    assert [r["run_id"] for r in rows] == ["ra", "rb"]
    assert len(lane[progress.INDEX_KEY].decode().splitlines()) == 2


def test_append_index_best_effort(lane):
    # a broken sink must never raise into the run
    def boom(*a, **k):
        raise RuntimeError("lane down")
    progress.append_index({"run_id": "x"}, boom, boom)   # no raise


def test_read_index_skips_garbage(lane):
    lane[progress.INDEX_KEY] = b'{"run_id":"ok"}\nnotjson\n{"no_id":1}\n'
    assert [r["run_id"] for r in progress.read_index("runpod")] == ["ok"]


def test_reindex_backfills_local_runs(tmp_path, monkeypatch, lane):
    rid = "20260101-000000-abc123"
    d = tmp_path / "runs" / rid
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(_manifest()))
    (d / "run.log").write_text("{'loss': 3.0, 'epoch': 1}\n"
                               "{'loss': 1.5, 'epoch': 2}\n")
    monkeypatch.setattr(progress, "ROOT", tmp_path)
    led_db = tmp_path / "state"
    led_db.mkdir()
    monkeypatch.setattr(dispatch, "ROOT", tmp_path)
    out = progress.reindex("runpod")
    assert out["status"] == "ok" and out["rebuilt"] == 1
    rows = progress.read_index("runpod")
    assert rows[0]["run_id"] == rid
    assert rows[0]["last_loss"] == 1.5 and rows[0]["first_loss"] == 3.0


def test_reindex_preserves_lane_only_runs(tmp_path, monkeypatch, lane):
    # a run another controller indexed must survive this controller's rebuild
    lane[progress.INDEX_KEY] = (json.dumps({"run_id": "elsewhere",
                                            "kind": "train"}) + "\n").encode()
    monkeypatch.setattr(progress, "ROOT", tmp_path)   # no local runs/
    out = progress.reindex("runpod")
    assert out["rebuilt"] == 0 and out["preserved_from_lane"] == 1
    assert [r["run_id"] for r in progress.read_index("runpod")] == \
        ["elsewhere"]


def test_index_write_hook_wires(lane, monkeypatch):
    # _index_write reads manifest+log from the lane and appends via
    # progress.append_index; end-to-end join (cost/price/seconds).
    rid = "run-join"
    lane[f"runs/{rid}/manifest.json"] = json.dumps(_manifest()).encode()
    lane[f"runs/{rid}/run.log"] = b"{'loss': 2.0, 'epoch': 1.0}\n"
    get = progress.lane_getter("runpod")
    put = progress.lane_putter("runpod")
    dispatch._index_write(rid, get, put, cost=0.42, price_hr=0.25,
                          started=1700.0, led=None)
    rows = progress.read_index("runpod")
    assert len(rows) == 1 and rows[0]["run_id"] == rid
    assert rows[0]["cost"] == 0.42 and rows[0]["price_hr"] == 0.25
    assert rows[0]["last_loss"] == 2.0
    assert rows[0]["status"] == "succeeded"
