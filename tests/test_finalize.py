import json
import os
from pathlib import Path

import pytest

from trainer_agent import finalize as fz
from trainer_agent import joblane, transfer


@pytest.fixture()
def good_outputs(tmp_path):
    out = tmp_path / "outputs" / "run-1"
    out.mkdir(parents=True)
    for f in fz.REQUIRED_FILES:
        (out / f).write_bytes(b"x" * 16)
    (out / "checkpoint-100").mkdir()
    (out / "checkpoint-100" / "adapter_model.safetensors").write_bytes(b"y")
    (out / "train.log").write_text("hello loss=1.0\n")
    return out


def test_gate_ok_excludes_checkpoints(good_outputs):
    ok, problems = fz.gate(good_outputs)
    assert ok, problems


def test_gate_missing_file(tmp_path):
    bad = tmp_path / "outputs"
    bad.mkdir()
    ok, problems = fz.gate(bad)
    assert not ok and any("adapter_config" in p for p in problems)


def test_pack_skips_checkpoints(good_outputs, tmp_path):
    tar = fz.pack(good_outputs, tmp_path / "art.tar.gz")
    import tarfile
    with tarfile.open(tar) as tf:
        names = tf.getnames()
    assert "adapter_config.json" in names
    assert not any("checkpoint-" in n for n in names)


def test_finalize_uploads(monkeypatch, good_outputs):
    calls = []
    monkeypatch.setattr(fz, "upload",
                        lambda path, url: calls.append((path.name, url)))
    monkeypatch.setattr(fz, "gpu_info", lambda: {"gpus": []})
    urls = {"artifacts_put": "u1", "manifest_put": "u2", "log_put": "u3"}
    (log := good_outputs.parent.parent / "run.log").write_text("line\n")
    m = fz.finalize("run-1", good_outputs, {"state": "succeeded"}, log, urls)
    assert m["status"] == "succeeded"
    assert "adapter_config.json" in m["artifacts"]
    uploaded = [u for _, u in calls]
    assert uploaded == ["u1", "u3", "u2"]  # manifest LAST = completion signal
    done = json.loads((good_outputs.parent / "done.json").read_text())
    assert done["status"] == "succeeded"


def test_finalize_volume_writes_manifest_last(monkeypatch, good_outputs, tmp_path):
    """Volume lane: files land under mount/prefix, manifest strictly last,
    and its own dir-listing right before it must not contain manifest.json."""
    mount = tmp_path / "data"
    vol = {"mount": str(mount), "prefix": "runs/r9"}
    seen = []

    real_replace = os.replace

    def spy_replace(src, dst):
        d = Path(dst).parent
        seen.append(sorted(p.name for p in d.iterdir()))
        return real_replace(src, dst)

    monkeypatch.setattr(fz.os, "replace", spy_replace)
    monkeypatch.setattr(fz, "gpu_info", lambda: {"gpus": []})
    (log := good_outputs.parent.parent / "run.log").write_text("vol log\n")
    m = fz.finalize("r9", good_outputs, {"state": "succeeded"}, log, {}, vol)
    assert m["status"] == "succeeded"
    out = mount / "runs" / "r9"
    assert (out / "artifacts.tar.gz").is_file()
    assert (out / "run.log").read_text() == "vol log\n"
    assert seen[-1] and "manifest.json" not in seen[-1]  # only *.part existed
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["artifacts_tar_sha256"].startswith("sha256:") or \
        len(manifest["artifacts_tar_sha256"]) == 64
    assert not list(out.glob("*.part"))


def test_gate_spec_stage_dir(tmp_path, monkeypatch):
    """Stage jobs (ONNX outputs) pass a custom required-files set."""
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "model.onnx").write_bytes(b"\x08\x00")
    ok, problems = fz.gate(out, {"required_files": ["model.onnx"]})
    assert ok and problems == []
    ok, problems = fz.gate(out)          # default adapter gate fails
    assert not ok and any("adapter_config" in p for p in problems)


def test_materialize_stage_inputs_extract(tmp_path):
    import tarfile
    src = tmp_path / "artifacts.tar.gz"
    with tarfile.open(src, "w:gz") as tf:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            p = tmp_path / name
            p.write_bytes(b"x")
            tf.add(p, arcname=name)
    payload = {"run_id": "r9", "command": "true",
               "inputs": [{"name": "model", "path": str(src),
                           "extract": True}],
               "gate": {"required_files": ["model.onnx"]}}
    run = joblane.materialize(payload, work=tmp_path / "work")
    assert (tmp_path / "work/inputs/model/adapter_config.json").is_file()
    assert not (tmp_path / "work/inputs/model.tar.gz").exists()  # cleaned
    assert run["gate"] == {"required_files": ["model.onnx"]}


def test_materialize_no_config_yaml(tmp_path):
    payload = {"run_id": "r10", "command": "true"}
    run = joblane.materialize(payload, work=tmp_path / "w")
    assert not (tmp_path / "w/job/train.yaml").exists()
    assert run["urls"] == {} and run.get("gate") is None


def test_finalize_failed_status(monkeypatch, tmp_path):
    empty = tmp_path / "outputs" / "run-2"
    empty.mkdir(parents=True)
    calls = []
    monkeypatch.setattr(fz, "upload", lambda path, url: calls.append(url))
    monkeypatch.setattr(fz, "gpu_info", lambda: {"gpus": []})
    m = fz.finalize("run-2", empty, {"state": "failed", "exit_code": 3},
                    tmp_path / "nolog.txt", {"manifest_put": "m", "log_put": "l"})
    assert m["status"] == "failed"
    assert m["training"]["exit_code"] == 3
    assert calls == ["m"]          # no artifacts tar on failure


def test_materialize(tmp_path, monkeypatch):
    ds = tmp_path / "data.jsonl"

    def fake_download(url, dest, sha256=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text('{"messages":[]}\n')
        return dest

    monkeypatch.setattr(joblane, "download", fake_download)
    payload = {"run_id": "r", "command": "axolotl train /work/job/train.yaml",
               "config_yaml": "base_model: x\n",
               "dataset": {"name": "queer.jsonl", "url": "presigned",
                           "sha256": None},
               "outputs_dir": str(tmp_path / "out"),
               "urls": {"manifest_put": "m"}}
    run = joblane.materialize(payload, work=tmp_path)
    assert (tmp_path / "job" / "train.yaml").read_text() == "base_model: x\n"
    assert (tmp_path / "datasets" / "queer.jsonl").is_file()
    assert run["urls"]["manifest_put"] == "m"
    assert run["outputs_dir"].is_dir()
