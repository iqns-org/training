"""Unit tests for the GPU provider layer (src/gpu/, bin/gpu).

All offline: provider CLIs are mocked, sleeps patched out. Run:
  python3 -m pytest src/tests/gpu-providers.py -q
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gpu import available_providers, get_provider  # noqa: E402
from gpu.axolotl import (  # noqa: E402
    TrainingError, run_training, run_with_fallback,
)
from gpu.common import (  # noqa: E402
    GpuError, extract_json_payload, is_active, is_terminal,
    load_job, next_sequential_dir, parse_ssh_connection,
    require_final_artifacts, retry, run, save_job,
)
from gpu.gpuai import GpuAiProvider  # noqa: E402
from gpu.runpod import RunpodProvider, _ssh_from_command  # noqa: E402


# ---------------------------------------------------------------- registry
def test_registry_known_providers():
    assert set(available_providers()) >= {"gpuai", "runpod"}
    assert get_provider("gpuai").name == "gpuai"
    assert get_provider("runpod").name == "runpod"


def test_registry_unknown_provider():
    try:
        get_provider("nope")
    except GpuError as e:
        assert "unknown provider" in str(e)
    else:
        raise AssertionError("expected GpuError")


# ------------------------------------------------------------------ common
def test_status_sets():
    for s in ("running", "creating", "starting", "provisioning", "unknown"):
        assert is_active(s), s
        assert not is_terminal(s), s
    for s in ("terminated", "error", "failed", "exited", "EXITED", "dead"):
        assert is_terminal(s), s
        assert not is_active(s) or s == "unknown", s
    assert not is_terminal("")
    assert not is_terminal("running")


def test_parse_ssh_explicit_fields():
    conn = parse_ssh_connection({"connection": {"user": "u", "host": "h",
                                                "port": 2222}})
    assert (conn["user"], conn["host"], conn["port"]) == ("u", "h", "2222")


def test_parse_ssh_command_forms():
    docs = parse_ssh_connection(
        {"connection": {"ssh_command": "ssh alice@1.2.3.4 -p 2200 -i ~/.ssh/k"}})
    assert (docs["user"], docs["host"], docs["port"]) == ("alice", "1.2.3.4", "2200")
    legacy = parse_ssh_connection(
        {"connection": {"ssh_command": "ssh -p 22 bob@example.com"}})
    assert (legacy["user"], legacy["host"], legacy["port"]) == ("bob", "example.com", "22")


def test_retry_eventual_success():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise GpuError("boom")
        return "ok"

    assert retry(flaky, attempts=3, backoff=0, describe="flaky op") == "ok"


def test_retry_exhaustion_names_operation():
    def doomed():
        raise GpuError("down")
    try:
        retry(doomed, attempts=2, backoff=0, describe="doomed op")
    except GpuError as e:
        assert "doomed op" in str(e)
    else:
        raise AssertionError("expected GpuError")


def test_extract_json_payload():
    assert extract_json_payload('noise\n{"a": 1}\nnoise') == {"a": 1}


def test_job_roundtrip(tmp_path):
    p = tmp_path / "job.json"
    save_job(p, {"instance": {"id": "x"}})
    assert load_job(p)["instance"]["id"] == "x"
    assert load_job(tmp_path / "missing.json") is None


def test_require_final_artifacts(tmp_path):
    try:
        require_final_artifacts(tmp_path)
    except GpuError as e:
        assert "missing final artifacts" in str(e)
    else:
        raise AssertionError("expected GpuError")
    for f in ("adapter_config.json", "adapter_model.safetensors",
              "config.json", "tokenizer.json", "tokenizer_config.json"):
        (tmp_path / f).write_text("{}")
    assert require_final_artifacts(tmp_path) is True


def test_run_timeout():
    try:
        run(["sleep", "5"], timeout=1)
    except GpuError as e:
        assert "timed out" in str(e)
    else:
        raise AssertionError("expected GpuError")


def test_next_sequential_dir(tmp_path):
    a = next_sequential_dir(tmp_path, "m")
    a.mkdir()
    b = next_sequential_dir(tmp_path, "m")
    assert a.name == "001-m" and b.name == "002-m"


# ------------------------------------------------------------------- gpuai
def _gpuai():
    return GpuAiProvider(key_file="~/.ssh/id_ed25519")


def test_gpuai_provision_reuses_live_instance(tmp_path):
    p = _gpuai()
    job = tmp_path / "job.json"
    save_job(job, {"instance": {"id": "i-1"}})
    with mock.patch.object(p, "status",
                           return_value={"id": "i-1", "status": "running"}), \
         mock.patch.object(p, "_gpu") as gpu:
        res = p.provision(app="a", name="n", job_file=job)
    assert res["reused"] is True
    gpu.assert_not_called()


def test_gpuai_terminate_contract():
    p = _gpuai()
    with mock.patch.object(p, "_gpu", return_value=SimpleNamespace(stdout="")):
        assert p.terminate("i-1") == {"id": "i-1", "deleted": True}
    with mock.patch.object(p, "_gpu", side_effect=GpuError("gone")):
        res = p.terminate("i-1")
        assert res["deleted"] is False and "gone" in res["error"]


def test_gpuai_wait_ssh_returns_metadata_with_id():
    p = _gpuai()
    rec = {"id": "i-1", "status": "running",
           "connection": {"user": "u", "host": "h", "port": "22"}}
    with mock.patch.object(p, "status", return_value=rec):
        conn = p.wait_ssh("i-1")
    assert conn["id"] == "i-1" and conn["host"] == "h"


def test_gpuai_wait_ssh_timeout():
    p = _gpuai()
    with mock.patch.object(p, "status",
                           return_value={"id": "i-1", "status": "creating"}):
        try:
            p.wait_ssh("i-1", max_wait=0)
        except GpuError as e:
            assert "no SSH metadata" in str(e)
        else:
            raise AssertionError("expected GpuError")


def test_gpuai_wait_ssh_redeploys_terminal():
    p = _gpuai()
    live = {"id": "i-2", "status": "running",
            "connection": {"user": "u", "host": "h", "port": "22"}}
    with mock.patch.object(p, "status", side_effect=[
            {"id": "i-1", "status": "terminated", "app": "a", "name": "n"},
            live]), \
         mock.patch.object(p, "terminate") as term, \
         mock.patch.object(p, "provision",
                           return_value={"instance": {"id": "i-2"}}) as prov:
        conn = p.wait_ssh("i-1")
    term.assert_called_once_with("i-1")
    assert prov.called and conn["id"] == "i-2"


# ------------------------------------------------------------------ runpod
def _runpod(**kw):
    kw.setdefault("api_key", "k")
    return RunpodProvider(**kw)


def _completed(stdout):
    return SimpleNamespace(stdout=stdout, stderr="", returncode=0)


def test_runpod_uses_runpodctl_binary():
    p = _runpod()
    with mock.patch("subprocess.run",
                    return_value=_completed("[]")) as sr:
        p.list_instances()
    assert sr.call_args[0][0][0] == "runpodctl"


def test_runpod_list_uses_all_and_normalizes():
    p = _runpod()
    payload = json.dumps([{"id": "abc", "name": "n", "status": "RUNNING",
                           "gpu": {"id": "RTX 4090"}, "cost": 0.5}])
    with mock.patch("subprocess.run",
                    return_value=_completed(payload)) as sr:
        recs = p.list_instances()
    argv = sr.call_args[0][0]
    assert "--all" in argv
    assert recs[0]["status"] == "running"
    assert recs[0]["gpu_type"] == "RTX 4090"


def test_runpod_create_flags_are_documented_ones():
    p = _runpod()
    with mock.patch("subprocess.run",
                    return_value=_completed('{"id": "abc123"}')) as sr:
        res = p.provision(app="a", name="n", force=True)
    argv = sr.call_args[0][0]
    assert "--image" in argv and "--gpu-id" in argv and "--gpu-count" in argv
    assert "--imageName" not in argv and "--gpuType" not in argv
    assert res["instance"]["id"] == "abc123"


def test_runpod_terminate_uses_delete():
    p = _runpod()
    with mock.patch("subprocess.run",
                    return_value=_completed("")) as sr:
        assert p.terminate("abc")["deleted"] is True
    assert "delete" in sr.call_args[0][0]


def test_runpod_keys_surface():
    p = _runpod()
    with mock.patch("subprocess.run",
                    return_value=_completed("ok")) as sr:
        p.keys_create("n", "/tmp/k.pub")
    assert sr.call_args[0][0][:3] == ["runpodctl", "ssh", "add-key"]
    assert "--key-file" in sr.call_args[0][0]
    with mock.patch("subprocess.run",
                    return_value=_completed("[]")) as sr:
        assert p.keys_list() == []
    assert sr.call_args[0][0] == ["runpodctl", "ssh", "list-keys"]


def test_runpod_no_api_key_required(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    p = RunpodProvider()  # must not raise; runpodctl owns auth config
    with mock.patch("subprocess.run", return_value=_completed("[]")):
        assert p.list_instances() == []


def test_runpod_missing_binary_is_actionable():
    p = _runpod(binary="runpodctl-missing-xyz")
    try:
        p.list_instances()
    except GpuError as e:
        assert "runpodctl-missing-xyz" in str(e) and "install" in str(e).lower()
    else:
        raise AssertionError("expected GpuError")


def test_runpod_ssh_command_parser_docs_form():
    parsed = _ssh_from_command("ssh alice@1.2.3.4 -p 2200 -i ~/.ssh/k")
    assert (parsed["user"], parsed["host"], parsed["port"]) == (
        "alice", "1.2.3.4", "2200")


def test_runpod_wait_ssh_uses_pod_get_ssh_object():
    p = _runpod()
    pod = {"id": "abc", "name": "n", "status": "RUNNING",
           "ssh": {"ssh_command": "ssh alice@1.2.3.4 -p 2200"}}
    with mock.patch("subprocess.run",
                    return_value=_completed(json.dumps(pod))), \
         mock.patch.object(RunpodProvider, "_ssh_ok", return_value=True):
        conn = p.wait_ssh("abc")
    assert (conn["user"], conn["host"], conn["port"]) == (
        "alice", "1.2.3.4", "2200")


def test_runpod_wait_ssh_terminal_redeploys():
    p = _runpod()
    p._last_provision = {"app": "a", "name": "n"}
    states = [{"id": "old", "name": "n", "status": "EXITED"},
              {"id": "new", "name": "n", "status": "RUNNING",
               "ssh": {"ssh_command": "ssh u@h -p 22"}},
              {"id": "new", "name": "n", "status": "RUNNING",
               "ssh": {"ssh_command": "ssh u@h -p 22"}}]
    with mock.patch.object(RunpodProvider, "_pod_get",
                           side_effect=states), \
         mock.patch.object(p, "terminate") as term, \
         mock.patch.object(p, "provision",
                           return_value={"instance": {"id": "new"}}), \
         mock.patch.object(RunpodProvider, "_ssh_ok", return_value=True):
        conn = p.wait_ssh("old")
    term.assert_called_once_with("old")
    assert conn["host"] == "h"


# ----------------------------------------------------------------- axolotl
class FakeProvider:
    """Minimal provider double for run_training/run_with_fallback."""

    def __init__(self, fail_at=None, artifacts=True):
        self.calls = []
        self.fail_at = fail_at
        self.artifacts = artifacts

    def provision(self, **kw):
        self.calls.append("provision")
        if self.fail_at == "provision":
            raise GpuError("no capacity")
        return {"instance": {"id": "i-9"}}

    def status(self, iid):
        return {"id": iid, "status": "running"}

    def wait_ssh(self, iid):
        self.calls.append("wait_ssh")
        return {"user": "u", "host": "h", "port": "22"}

    def run_remote(self, iid, script, **kw):
        self.calls.append("run_remote")
        return {"rc": 0, "stdout": "", "stderr": ""}

    def upload(self, iid, src, dst):
        self.calls.append("upload")
        return {}

    def download(self, iid, remote, local):
        self.calls.append("download")
        if self.fail_at == "download":
            raise GpuError("rsync broke")
        if self.artifacts:
            d = Path(local)
            d.mkdir(parents=True, exist_ok=True)
            for f in ("adapter_config.json", "adapter_model.safetensors",
                      "config.json", "tokenizer.json", "tokenizer_config.json"):
                (d / f).write_text("{}")
        return {}

    def terminate(self, iid):
        self.calls.append("terminate")
        return {"id": iid, "deleted": True}


def _domain_root(tmp_path):
    d = tmp_path / "axolotl" / "demo"
    d.mkdir(parents=True)
    (d / "SmolLM2-360M-qlora.yaml").write_text("base_model: x\n")
    return tmp_path


def test_run_training_happy_path(tmp_path):
    root = _domain_root(tmp_path)
    fp = FakeProvider()
    res = run_training(fp, root, "demo", quick=True)
    assert res["instance"] == "i-9"
    assert Path(res["model_dir"]).is_dir()
    assert "provision" in fp.calls and "terminate" in fp.calls


def test_run_training_download_failure_keeps_alive(tmp_path):
    root = _domain_root(tmp_path)
    fp = FakeProvider(fail_at="download")
    try:
        run_training(fp, root, "demo", quick=True)
    except TrainingError as e:
        assert e.keep_alive is True and "KEPT ALIVE" in str(e)
        assert "terminate" not in fp.calls  # teardown gated on artifacts
    else:
        raise AssertionError("expected TrainingError")


def test_run_with_fallback_moves_on_safe_failure(tmp_path):
    root = _domain_root(tmp_path)
    bad, good = FakeProvider(fail_at="provision"), FakeProvider()
    res = run_with_fallback([("bad", bad, None), ("good", good, None)],
                            root, "demo", quick=True)
    assert res["provider"] == "good"


def test_run_with_fallback_aborts_on_keep_alive(tmp_path):
    root = _domain_root(tmp_path)
    keep = FakeProvider(fail_at="download")
    try:
        run_with_fallback([("keep", keep, None)], root, "demo", quick=True)
    except TrainingError as e:
        assert e.keep_alive is True
    else:
        raise AssertionError("expected TrainingError")
