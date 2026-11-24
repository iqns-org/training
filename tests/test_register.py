import hashlib
import json

import pytest

from orchestrator import register
from orchestrator.ledger import Ledger

RID = "2026-rid-awq001"


@pytest.fixture()
def staged(tmp_path, monkeypatch):
    """A harvested stage run: ledger row + local manifest + artifacts tar."""
    monkeypatch.setattr(register, "ROOT", tmp_path)
    monkeypatch.setenv("QTRAIN_SERVE_HOST", "me@nano0.lan")
    monkeypatch.setenv("QTRAIN_SERVE_DIR", "/home/me/models")
    led = Ledger(tmp_path / "state" / "ledger.db")
    led.db.execute("INSERT INTO runs (run_id, kind, spec_json, state, "
                   "provider, created_at, updated_at) VALUES (?,?,?,?,?,0,0)",
                   (RID, "stage", '{"kind": "awq"}', "done", "runpod"))
    led.db.commit()
    d = tmp_path / "runs" / RID
    d.mkdir(parents=True)
    tar = b"TARBAGBYTES"
    (d / "artifacts.tar.gz").write_bytes(tar)
    (d / "manifest.json").write_text(json.dumps(
        {"status": "succeeded",
         "artifacts_tar_sha256": hashlib.sha256(tar).hexdigest()}))
    return tar


def test_register_streams_verified_tar(staged, monkeypatch):
    seen = {}

    class R:
        returncode = 0
        stderr = b""

    def fake_run(cmd, input=None, capture_output=None, timeout=None):
        seen["cmd"] = cmd
        seen["input"] = input
        return R()
    monkeypatch.setattr(register.subprocess, "run", fake_run)
    res = register.register(RID, name="queer-360m-awq")
    assert res["status"] == "registered"
    assert res["target"] == "me@nano0.lan:/home/me/models/queer-360m-awq"
    assert seen["input"] == staged
    assert "tar -xzf -" in seen["cmd"][-1]
    assert "serve-vllm" in res["serve_hint"]


def test_register_rejects_corrupt(staged, tmp_path):
    (tmp_path / "runs" / RID / "artifacts.tar.gz").write_bytes(b"CORRUPTED")
    res = register.register(RID)
    assert "corrupted" in res["error"]


def test_register_missing_run(tmp_path, monkeypatch):
    monkeypatch.setattr(register, "ROOT", tmp_path)
    res = register.register("nope")
    assert "no such run" in res["error"]


def test_register_dry(staged, monkeypatch):
    monkeypatch.setattr(register.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("dry must not ssh")))
    res = register.register(RID, dry=True)
    assert res.get("dry") and "target" in res
