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


def test_register_large_tar_is_resumable(staged, monkeypatch):
    """Over-threshold tars rsync (--partial --append-verify) to a stage
    path, then unpack + clean on the host — interrupted transfers rerun
    from the partial instead of zero."""
    calls = []

    class R:
        returncode = 0
        stderr = b""

    def fake_run(cmd, input=None, capture_output=None, timeout=None):
        calls.append((list(cmd), input))
        return R()

    monkeypatch.setattr(register.subprocess, "run", fake_run)
    monkeypatch.setattr(register, "RESUME_THRESHOLD", 5)   # tar is 11 B
    res = register.register(RID, name="queer08b-merged")
    assert res["status"] == "registered"
    (cmd1, in1), (cmd2, in2) = calls
    assert cmd1[0] == "rsync" and "--append-verify" in cmd1
    # run-keyed stage (the poisoning fix): path embeds BOTH ids
    stage_path = f"me@nano0.lan:/tmp/qtrain-{RID}-queer08b-merged.tar.gz"
    assert cmd1[-1] == stage_path
    assert in1 is None and cmd2[0] == "ssh"
    # target wiped before unpack (tar -xzf MERGES into a polluted dir)
    assert cmd2[-1].startswith("rm -rf /home/me/models/queer08b-merged")
    assert "tar -xzf /tmp/qtrain-2026-rid-awq001-queer08b-merged.tar.gz" in cmd2[-1]
    assert "rm -f /tmp/qtrain-2026-rid-awq001-queer08b-merged.tar.gz" in cmd2[-1]


def test_register_resume_rerun_survives_partial(staged, monkeypatch,
                                                tmp_path):
    """A failed rsync surfaces a resume hint and does NOT touch the
    target; the rerun uses the SAME stage path (rsync continues)."""
    state = {"n": 0}

    class R:
        def __init__(self, rc):
            self.returncode = rc
            self.stderr = b"link died"

    def fake_run(cmd, input=None, capture_output=None, timeout=None):
        state["n"] += 1
        return R(255 if state["n"] == 1 else 0)

    monkeypatch.setattr(register.subprocess, "run", fake_run)
    monkeypatch.setattr(register, "RESUME_THRESHOLD", 5)
    res = register.register(RID, name="queer08b-merged")
    assert res["status"] == "error" and "resume" in res["error"]
    res2 = register.register(RID, name="queer08b-merged")
    assert res2["status"] == "registered"


def test_register_staging_is_run_keyed():
    # the poisoning bug (observed: 5 consecutive unpack-EOFs): stage
    # keyed by --name alone lets a --partial from one artifact poison
    # every retry of another under the same name. stage path embeds
    # run_id; and a FAILED UNPACK deletes the stage — retrying must
    # re-stream clean, never resume onto junk bytes.
    src = open(register.__file__).read()
    assert 'stage = f"/tmp/qtrain-{run_id}-{name}.tar.gz"' in src
    assert "rm -f {q(stage)}" in src
    # cleanup must be gated to the staged path (stage is undefined in
    # the direct-pipe branch — a NameError here would mask the real
    # unpack error and break every small-tar register)
    assert "if size > RESUME_THRESHOLD" in src
