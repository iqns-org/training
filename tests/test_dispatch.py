import json
from pathlib import Path

import pytest
import yaml

from orchestrator import dispatch, storage


@pytest.fixture()
def fake_s3(monkeypatch):
    calls = {"put": [], "keys": []}
    monkeypatch.setattr(storage, "ensure_bucket", lambda: None)
    monkeypatch.setattr(dispatch.storage, "ensure_bucket", lambda: None)
    monkeypatch.setattr(dispatch.storage, "put_object",
                        lambda k, d, ct="": calls["put"].append(k))
    monkeypatch.setattr(dispatch.storage, "sha256_file",
                        lambda p: "0" * 64)
    monkeypatch.setattr(dispatch.storage, "mint_job_urls",
                        lambda rid, key=None, hours=48: {
                            "payload_get": f"g/{rid}", "dataset_get": f"d/{key}",
                            "artifacts_put": "a", "manifest_put": "m",
                            "log_put": "l"})
    return calls


@pytest.fixture()
def recipe(tmp_path):
    r = tmp_path / "rec.yaml"
    r.write_text(yaml.safe_dump({
        "base_model": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "load_in_4bit": True,
        "datasets": [{"path": "./datasets/queer.jsonl", "type": "chat_template"}],
        "sequence_len": 1024, "num_epochs": 2,
    }))
    return r


def test_render_config_rewrites_paths(recipe):
    out = yaml.safe_load(dispatch.render_config(recipe, {"learning_rate": 0.0002},
                                                "queer.mini.jsonl"))
    assert out["datasets"][0]["path"] == "/work/datasets/queer.mini.jsonl"
    assert out["output_dir"] == "/work/outputs"
    assert out["dataset_prepared_path"] == "/work/prepared"
    assert out["val_set_size"] == 0.0
    assert out["learning_rate"] == 0.0002
    assert out["base_model"] == "HuggingFaceTB/SmolLM2-360M-Instruct"
    assert out["load_in_4bit"] is True


def test_submit_dry_stages_objects(tmp_path, monkeypatch, fake_s3, recipe):
    monkeypatch.setattr(dispatch, "ROOT", tmp_path)
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp/queer.jsonl").write_text("\n".join(['{"m":1}'] * 60))
    monkeypatch.setattr(dispatch.subprocess, "run", lambda *a, **k: None)
    res = dispatch.submit("queer", recipe, mini=True, dry=True)
    assert not res.get("error"), res
    assert res.get("dry")
    assert any(k.startswith("datasets/queer/") for k in fake_s3["put"])
    assert any(k.endswith("payload.json") for k in fake_s3["put"])


def test_submit_volume_lane_dry(tmp_path, monkeypatch, recipe):
    """Volume lane selected automatically when RunPod volume env is set;
    payload carries volume mount + pod dataset path, no presigned urls."""
    monkeypatch.setenv("RUNPOD_VOLUME_ID", "vol1")
    monkeypatch.setenv("RUNPOD_S3_ACCESS_KEY", "user_x")
    monkeypatch.setenv("RUNPOD_S3_SECRET_KEY", "rps_y")
    monkeypatch.setenv("RUNPOD_S3_ENDPOINT", "https://s3api-eu-ro-1.runpod.io")
    monkeypatch.setattr(dispatch, "ROOT", tmp_path)
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp/queer.jsonl").write_text("\n".join(['{"m":1}'] * 60))
    monkeypatch.setattr(dispatch.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(dispatch.storage, "sha256_file", lambda p: "0" * 64)
    puts = []
    monkeypatch.setattr(dispatch.storage, "vol_put",
                        lambda k, d: puts.append((k, d)))
    res = dispatch.submit("queer", recipe, mini=True, dry=True)
    assert res["lane"] == "volume" and res.get("dry")
    keys = [k for k, _ in puts]
    assert any(k.startswith("datasets/queer/") for k in keys)
    payload = json.loads(dict(puts)[f"runs/{res['run_id']}/payload.json"])
    assert payload["volume"] == {"mount": "/data",
                                 "prefix": f"runs/{res['run_id']}"}
    assert payload["dataset"]["path"].startswith("/data/datasets/queer/")
    assert payload["urls"] == {}


def test_submit_lane_force_presign(tmp_path, monkeypatch, recipe, fake_s3):
    monkeypatch.setenv("RUNPOD_VOLUME_ID", "vol1")
    monkeypatch.setenv("RUNPOD_S3_ACCESS_KEY", "user_x")
    monkeypatch.setattr(dispatch, "ROOT", tmp_path)
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp/queer.jsonl").write_text("\n".join(['{"m":1}'] * 60))
    monkeypatch.setattr(dispatch.subprocess, "run", lambda *a, **k: None)
    res = dispatch.submit("queer", recipe, mini=True, dry=True,
                          force_lane="presign")
    assert res["lane"] == "presign"


def _open_row(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch, "ROOT", tmp_path)
    led = dispatch.Ledger(tmp_path / "state" / "ledger.db")
    rid = led.create("train", {"domain": "queer"}, provider="runpod")
    led.update(rid, state="provisioning", instance_id="i1")
    return led, rid


def test_submit_local_lane_dry(tmp_path, monkeypatch, recipe):
    monkeypatch.setattr(dispatch, "ROOT", tmp_path)
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp/queer.jsonl").write_text("\n".join(['{"m":1}'] * 60))
    monkeypatch.setattr(dispatch.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(dispatch.storage, "sha256_file", lambda p: "0" * 64)
    puts = []
    monkeypatch.setattr(dispatch.storage, "host_put",
                        lambda k, d: puts.append(k))
    res = dispatch.submit("queer", recipe, mini=True, dry=True,
                          provider="local")
    assert res["lane"] == "host" and res.get("dry")
    payload = None
    for k in puts:
        if k.endswith("payload.json"):
            payload = k
    assert payload and any(k.startswith("datasets/queer/") for k in puts)


def test_resume_tolerates_lane_blips(tmp_path, monkeypatch):
    """Transient lane errors (network blip) must NOT kill a healthy run:
    _watch tolerates a streak, then harvests normally."""
    import hashlib
    from orchestrator.models import Instance
    led, rid = _open_row(tmp_path, monkeypatch)
    tar = b"fake-tar"
    manifest = {"status": "succeeded", "artifacts": {"a": {"size": 1}},
                "artifacts_tar_sha256": hashlib.sha256(tar).hexdigest()}
    monkeypatch.setattr(dispatch.storage, "vol_cfg", lambda: {"bucket": "v"})
    calls = {"n": 0}

    def flaky(k):
        calls["n"] += 1
        if calls["n"] <= 2:               # two blips up front
            raise ConnectionError("transient")
        if k.endswith("manifest.json"):
            return json.dumps(manifest).encode()
        if k.endswith("artifacts.tar.gz"):
            return tar
        return b""
    monkeypatch.setattr(dispatch.storage, "vol_get", flaky)
    term = []

    class Stub:
        def poll(self, i):
            return Instance(provider="runpod", id=i, status="running")

        def terminate(self, i):
            term.append(i)
            return True
    monkeypatch.setattr(dispatch, "get_provider", lambda name: Stub())
    res = dispatch.resume(rid, poll=1)
    assert res["status"] == "succeeded" and term == ["i1"]
    assert calls["n"] > 2                  # blips happened, loop survived


def test_resume_harvests_and_terminates(tmp_path, monkeypatch):
    import hashlib
    from orchestrator.models import Instance
    led, rid = _open_row(tmp_path, monkeypatch)
    tar = b"fake-tar-bytes"
    manifest = {"status": "succeeded", "artifacts": {"a": {"size": 1}},
                "artifacts_tar_sha256": hashlib.sha256(tar).hexdigest()}
    monkeypatch.setattr(dispatch.storage, "vol_cfg", lambda: {"bucket": "v"})

    def vget(k):
        if k.endswith("manifest.json"):
            return json.dumps(manifest).encode()
        if k.endswith("artifacts.tar.gz"):
            return tar
        return b"log line\n"
    monkeypatch.setattr(dispatch.storage, "vol_get", vget)
    term = []

    class Stub:
        def poll(self, i):
            return Instance(provider="runpod", id=i, status="running")

        def terminate(self, i):
            term.append(i)
            return True
    monkeypatch.setattr(dispatch, "get_provider", lambda name: Stub())
    res = dispatch.resume(rid, poll=1)
    assert res["status"] == "succeeded"
    assert term == ["i1"] and res.get("terminated")
    assert (tmp_path / "runs" / rid / "artifacts.tar.gz").read_bytes() == tar
    assert (tmp_path / "runs" / rid / "run.log").is_file()
    row = next(r for r in led.list() if r["run_id"] == rid)
    assert row["state"] == "done" and row["artifact_uri"]


def test_resume_instance_gone_without_manifest(tmp_path, monkeypatch):
    from orchestrator.models import Instance
    led, rid = _open_row(tmp_path, monkeypatch)
    monkeypatch.setattr(dispatch.storage, "vol_cfg", lambda: {"bucket": "v"})
    monkeypatch.setattr(dispatch.storage, "vol_get", lambda k: None)
    term = []

    class Stub:
        def poll(self, i):
            return Instance(provider="runpod", id=i, status="terminated")

        def terminate(self, i):
            term.append(i)
            return True
    monkeypatch.setattr(dispatch, "get_provider", lambda name: Stub())
    res = dispatch.resume(rid, poll=1)
    assert res["status"] == "error" and "without manifest" in res["error"]
    assert term == ["i1"]
    row = next(r for r in led.list() if r["run_id"] == rid)
    assert row["state"] == "failed"
