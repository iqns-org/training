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
    (tmp_path / "tmp/queer.jsonl").write_text("\n".join('{"m":1}' * 60))
    monkeypatch.setattr(dispatch, "prepare_dataset",
                        lambda domain, mini=False:
                        (f"datasets/{domain}/abc/queer.jsonl",
                         tmp_path / "tmp/queer.jsonl", "0" * 64))
    res = dispatch.submit("queer", recipe, mini=True, dry=True)
    assert not res.get("error"), res
    assert res.get("dry")
    assert any(k.startswith("datasets/queer/") for k in fake_s3["put"])
    assert any(k.endswith("payload.json") for k in fake_s3["put"])
