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
                        lambda rid, key=None, hours=48, holdout_key=None: {
                            "payload_get": f"g/{rid}", "dataset_get": f"d/{key}",
                            "holdout_get": f"h/{holdout_key}" if holdout_key else "",
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


def test_render_config_preference_lane(tmp_path):
    """R1: pref_datasets recipes mount the domain file into the PREF slot
    and never fabricate an SFT datasets entry (axolotl chat_preference
    reads chosen/rejected)."""
    r = tmp_path / "dpo.yaml"
    r.write_text(yaml.safe_dump({
        "base_model": "Qwen/Qwen3.5-0.8B",
        "beta": 0.1,
        "pref_datasets": [{"path": "./datasets/cognition-dpo.jsonl",
                           "type": "chat_preference"}],
    }))
    out = yaml.safe_load(dispatch.render_config(r, {}, "cognition-dpo.jsonl"))
    assert out["pref_datasets"][0]["path"] == "/work/datasets/cognition-dpo.jsonl"
    assert "datasets" not in out


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


def _vol_puts(tmp_path, monkeypatch):
    """shared plumbing for payload-inspecting dry runs (volume lane)."""
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
    return puts


def test_submit_fft_gate_dry(tmp_path, monkeypatch):
    """A recipe without `adapter:` is FULL FT — the train payload must
    carry the shard-shaped gate, else a perfect round dies at the
    adapter-file default (the whole point of _gate_required)."""
    r = tmp_path / "fft.yaml"
    r.write_text(yaml.safe_dump({"base_model": "Qwen/Qwen3.5-0.8B",
                                 "datasets": [{"path": "x"}]}))
    puts = _vol_puts(tmp_path, monkeypatch)
    res = dispatch.submit("queer", r, mini=True, dry=True)
    payload = json.loads(dict(puts)[f"runs/{res['run_id']}/payload.json"])
    assert "gate" in payload
    assert "model*.safetensors" in payload["gate"]["required_files"]
    # adapter recipes stay gate-less (agent default is already adapter law)
    assert "config_yaml" in payload and "configs" not in payload


def test_submit_pair_dry(tmp_path, monkeypatch):
    """Co-tenant: one run, two configs, chained sh -c command whose per-pid
    waits keep one crash from masking the other; gate requires BOTH trees
    with distinct slugs; ledger records pair provenance."""
    ra = tmp_path / "a.yaml"
    ra.write_text(yaml.safe_dump({"base_model": "Qwen/Qwen3.5-0.8B",
                                  "adapter": "qlora",
                                  "datasets": [{"path": "x"}]}))
    rb = tmp_path / "b.yaml"
    rb.write_text(yaml.safe_dump({"base_model": "Qwen/Qwen3.5-4B",
                                  "adapter": "qlora",
                                  "datasets": [{"path": "x"}]}))
    puts = _vol_puts(tmp_path, monkeypatch)
    res = dispatch.submit("queer", ra, pair=rb, mini=True, dry=True)
    assert not res.get("error"), res
    payload = json.loads(dict(puts)[f"runs/{res['run_id']}/payload.json"])
    names = [c["name"] for c in payload["configs"]]
    assert names == ["qwen3.5-0.8b", "qwen3.5-4b"]
    cmd = payload["command"]
    assert cmd.startswith("sh -c ")
    assert "axolotl train /work/job/qwen3.5-0.8b.yaml" in cmd
    assert "axolotl train /work/job/qwen3.5-4b.yaml" in cmd
    assert "wait $p1 || r=1; wait $p2 || r=1" in cmd
    req = payload["gate"]["required_files"]
    assert any(f.startswith("qwen3.5-0.8b/adapter_model") for f in req)
    assert any(f.startswith("qwen3.5-4b/adapter_model") for f in req)
    # slug isolation reaches the configs themselves
    for c in payload["configs"]:
        cfg = yaml.safe_load(c["config_yaml"])
        assert cfg["output_dir"] == f"/work/outputs/{c['name']}"
        assert cfg["dataset_prepared_path"] == f"/work/prepared/{c['name']}"
    row = dispatch.Ledger(dispatch.ROOT / "state" / "ledger.db") \
        .get(res["run_id"])
    spec = json.loads(row["spec_json"])
    assert spec["pair_recipe"] == str(rb) and len(spec["pair_recipe_sha"]) == 12


def test_submit_pair_slug_collision(tmp_path, monkeypatch):
    """Two recipes with the SAME base_model (e.g. qlora + bf16-lora):
    identical slugs would make two jobs share one output dir — the gate
    would then certify one tree for two runs. Collision gets -2."""
    ra = tmp_path / "a.yaml"
    ra.write_text(yaml.safe_dump({"base_model": "Qwen/Qwen3.5-0.8B",
                                  "adapter": "qlora",
                                  "datasets": [{"path": "x"}]}))
    rb = tmp_path / "b.yaml"
    rb.write_text(yaml.safe_dump({"base_model": "Qwen/Qwen3.5-0.8B",
                                  "adapter": "qlora", "sequence_len": 512,
                                  "datasets": [{"path": "x"}]}))
    puts = _vol_puts(tmp_path, monkeypatch)
    res = dispatch.submit("queer", ra, pair=rb, mini=True, dry=True)
    payload = json.loads(dict(puts)[f"runs/{res['run_id']}/payload.json"])
    names = [c["name"] for c in payload["configs"]]
    assert names[0] == "qwen3.5-0.8b" and names[1] == "qwen3.5-0.8b-2"


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


def test_submit_stage_dry_local(tmp_path, monkeypatch):
    puts = {}
    monkeypatch.setattr(dispatch.storage, "host_put",
                        lambda k, d: puts.__setitem__(k, d))
    monkeypatch.setattr(dispatch, "ROOT", tmp_path)
    pp = tmp_path / "postproc"
    pp.mkdir()
    (pp / "merge_adapter.py").write_text("# stub\n")
    (pp / "holdout_loss.py").write_text("# stub\n")
    (pp / "util.py").write_text("# stub\n")
    res = dispatch.submit_stage("merge", "from-1", base_model="A/B",
                                dry=True, provider="local")
    assert res["lane"] == "host" and res.get("dry")
    payload = json.loads(puts[f"runs/{res['run_id']}/payload.json"])
    assert payload["inputs"][0]["path"] == "/data/runs/from-1/artifacts.tar.gz"
    assert payload["gate"] == {"required_files": ["config.json",
                                "model.safetensors",
                                "merge_stats.json"]}
    assert "merge_adapter.py" in payload["command"] and "--base A/B" in payload["command"]
    assert payload["config_yaml"] == ""
    # scripts ride along, sha-verified, from the repo (not the frozen image)
    assert puts[f"runs/{res['run_id']}/script/merge_adapter.py"] == b"# stub\n"
    assert payload["command"].count("/work/inputs/merge_adapter.py") == 1
    names = [i["name"] for i in payload["inputs"]]
    assert names == ["model", "merge_adapter.py", "util.py"]


def test_submit_stage_dry_runpod(tmp_path, monkeypatch):
    puts = {}
    monkeypatch.setattr(dispatch.storage, "vol_cfg",
                        lambda: {"bucket": "v1", "mount": "/data",
                                 "dc": "EU-RO-1", "key": "k", "secret": "s",
                                 "endpoint": "e", "region": "EU-RO-1"})
    monkeypatch.setattr(dispatch.storage, "vol_put",
                        lambda k, d: puts.__setitem__(k, d))
    monkeypatch.setattr(dispatch, "ROOT", tmp_path)
    pp = tmp_path / "postproc"
    pp.mkdir()
    (pp / "export_onnx.py").write_text("# stub\n")
    (pp / "util.py").write_text("# stub\n")
    res = dispatch.submit_stage("onnx", "from-2", dry=True,
                                provider="runpod")
    assert res["lane"] == "volume" and res.get("dry")
    payload = json.loads(puts[f"runs/{res['run_id']}/payload.json"])
    assert payload["inputs"][0]["path"] == "/data/runs/from-2/artifacts.tar.gz"
    assert payload["gate"] == {"required_files": ["model.onnx"]}


def test_submit_stage_awq_dataset_injection(tmp_path, monkeypatch):
    puts = {}
    monkeypatch.setattr(dispatch.storage, "vol_cfg",
                        lambda: {"bucket": "v1", "mount": "/data",
                                 "dc": "EU-RO-1", "key": "k", "secret": "s",
                                 "endpoint": "e", "region": "EU-RO-1"})
    monkeypatch.setattr(dispatch.storage, "vol_put",
                        lambda k, d: puts.__setitem__(k, d))
    src = {"dataset": {"name": "queer.jsonl",
                       "path": "/data/datasets/queer/abc/queer.jsonl",
                       "sha256": "0" * 64}}

    def vget(k):
        return json.dumps(src).encode() if k.endswith("payload.json") else None
    monkeypatch.setattr(dispatch.storage, "vol_get", vget)
    monkeypatch.setattr(dispatch, "ROOT", tmp_path)
    pp = tmp_path / "postproc"
    pp.mkdir()
    (pp / "quantize_awq.py").write_text("# stub\n")
    (pp / "util.py").write_text("# stub\n")
    res = dispatch.submit_stage("awq", "from-3", dry=True, provider="runpod")
    payload = json.loads(puts[f"runs/{res['run_id']}/payload.json"])
    # model + data + the two riding scripts
    assert len(payload["inputs"]) == 4
    assert payload["inputs"][1]["path"].endswith("queer.jsonl")
    assert payload["inputs"][1]["sha256"] == "0" * 64
    assert "--dataset /work/inputs/data" in payload["command"]
    assert payload["gate"] == {"required_files": ["config.json"]}
    assert puts[f"runs/{res['run_id']}/script/quantize_awq.py"] == b"# stub\n"


def test_submit_stage_awq_missing_source(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch.storage, "vol_cfg",
                        lambda: {"bucket": "v1", "mount": "/data",
                                 "dc": "EU-RO-1", "key": "k", "secret": "s",
                                 "endpoint": "e", "region": "EU-RO-1"})
    monkeypatch.setattr(dispatch.storage, "vol_get", lambda k: None)
    monkeypatch.setattr(dispatch, "ROOT", tmp_path)
    res = dispatch.submit_stage("awq", "ghost", provider="runpod")
    assert "no dataset in lineage" in res["error"]


def test_submit_stage_awq_traces_two_hops(tmp_path, monkeypatch):
    """awq --from-run <merge>: merge payload has no dataset, its model
    input points at the TRAIN run — calibration pointer is inherited."""
    monkeypatch.setattr(dispatch.storage, "vol_cfg",
                        lambda: {"bucket": "v1", "mount": "/data",
                                 "dc": "EU-RO-1", "key": "k", "secret": "s",
                                 "endpoint": "e", "region": "EU-RO-1"})
    puts = {}
    monkeypatch.setattr(dispatch.storage, "vol_put",
                        lambda k, d: puts.__setitem__(k, d))
    train_payload = {"dataset": {"name": "queer.jsonl",
                                 "path": "/data/datasets/queer/x/queer.jsonl",
                                 "sha256": "a" * 64}}
    merge_payload = {"inputs": [{"name": "model",
                                 "path": "/data/runs/train-1/artifacts.tar.gz"}]}

    def vget(k):
        if k == "runs/merge-1/payload.json":
            return json.dumps(merge_payload).encode()
        if k == "runs/train-1/payload.json":
            return json.dumps(train_payload).encode()
        return None
    monkeypatch.setattr(dispatch.storage, "vol_get", vget)
    monkeypatch.setattr(dispatch, "ROOT", tmp_path)
    pp = tmp_path / "postproc"
    pp.mkdir()
    (pp / "quantize_awq.py").write_text("# stub\n")
    (pp / "util.py").write_text("# stub\n")
    res = dispatch.submit_stage("awq", "merge-1", dry=True, provider="runpod")
    payload = json.loads(puts[f"runs/{res['run_id']}/payload.json"])
    assert len(payload["inputs"]) == 4       # model + data + 2 scripts
    assert payload["inputs"][1]["path"] == "/data/datasets/queer/x/queer.jsonl"
    assert payload["inputs"][1]["sha256"] == "a" * 64
    assert "--dataset /work/inputs/data" in payload["command"]


def test_submit_stage_validation(monkeypatch):
    monkeypatch.setattr(dispatch, "ROOT", Path("/nonexistent"))
    assert "unknown stage" in dispatch.submit_stage("bogus", "r")["error"]
    assert "base_model" in dispatch.submit_stage("merge", "r")["error"]
    assert "not wired" in dispatch.submit_stage("onnx", "r",
                                                provider="gpuai")["error"]


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


def test_stage_row_records_image_and_holdout(tmp_path, monkeypatch):
    # 4eed0a law, enforced by test: the STAGE ROW spec must carry the
    # same image the pod boots (_stage_image -> trainer digest for the
    # holdout kind) and the resolved holdout_sha. This is the assertion
    # df3053f promised but never shipped; the dead update_spec call
    # (AttributeError on any real holdout) is exactly what it catches.
    import json
    monkeypatch.setattr(dispatch, "ROOT", tmp_path)
    puts = {}
    monkeypatch.setattr(dispatch.storage, "host_put",
                        lambda k, d: puts.__setitem__(k, d))
    # local lane: mount is LOCAL_MOUNT
    monkeypatch.setenv("LOCAL_MOUNT", "/data")
    pp = tmp_path / "postproc"
    pp.mkdir()
    (pp / "holdout_loss.py").write_text("# stub\n")
    (pp / "util.py").write_text("# stub\n")
    # seed a train-row payload carrying a holdout input on the host lane
    from orchestrator.ledger import Ledger
    led = Ledger(tmp_path / "state" / "ledger.db")
    tr = led.create("train", {"domain": "queer"})
    (tmp_path / "data" / "runs" / tr).mkdir(parents=True)
    hkey = f"runs/{tr}/payload.json"
    # the payload the resolver must walk to find the holdout input
    payload = {"inputs": [{"name": "queer.holdout.jsonl",
                           "path": "/data/datasets/queer/abc/queer.holdout.jsonl",
                           "sha256": "cafe" + "0" * 60}], "from_run": ""}
    puts[hkey] = payload
    monkeypatch.setattr(dispatch.storage, "host_get",
                        lambda k: json.dumps(puts[k]).encode() if k in puts
                        else None)
    res = dispatch.submit_stage("holdout", tr, base_model="Qwen/Qwen3.5-0.8B",
                               provider="local", dry=True)
    assert res.get("dry"), res
    row = led.get(res["run_id"])
    spec = json.loads(row["spec_json"])
    # the books MUST match what the launch expression yields (4eed0a law):
    # holdout declares image:"trainer" -> _stage_image routes to the
    # trainer digest, not the postproc pair the old ternary recorded
    assert spec["image"] == dispatch._stage_image(
        dispatch.STAGE_SPECS["holdout"])
    assert spec["image"] == dispatch._image_ref()   # trainer, not postproc
    assert spec["holdout_sha"] == "cafe" + "0" * 60


def test_mini_estimates_smoke_profile(tmp_path, monkeypatch, fake_s3, recipe):
    """THE ESTIMATE IS A CLAIM ABOUT THE WORK (f44f27, 2026-11-27):
    --mini was gate-estimated as a 60min train ($2.25) while the payload
    declared 48 rows x 1 epoch (~$0.25 real); a $2.50 balance admitted
    neither. 'smoke' (10min) existed in EST_MINUTES but NOTHING selected
    it — dead weight. Pin the seam: same balance $2.50 / reserve $2.00 —
    mini passes the balance gate (0.38), full train is refused (2.25).
    Same law as cb059d opposite direction: wrong estimates either kill
    good runs at the cap or starve cheap validations at the gate."""
    from types import SimpleNamespace
    import subprocess
    monkeypatch.setattr(dispatch, "ROOT", tmp_path)
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp/queer.jsonl").write_text("\n".join(['{"m":1}'] * 60))
    monkeypatch.setattr(dispatch.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0))
    monkeypatch.setattr(dispatch, "get_provider", lambda name: SimpleNamespace(
        balance=lambda: 2.50))
    monkeypatch.setattr(dispatch, "_wave_launch", lambda *a, **k: (
        SimpleNamespace(id="pod-fake", base_url="http://fake"),
        SimpleNamespace(gpu_id="RTX A5000", price_hr=0.90)))
    monkeypatch.setattr(dispatch, "_watch", lambda *a, **k: None)

    res = dispatch.submit("queer", recipe, mini=True, provider="runpod")
    assert not res.get("error"), res
    assert res["estimate"] == 0.38            # 1.50/hr x 10min x 1.5

    res2 = dispatch.submit("queer", recipe, provider="runpod")
    assert "estimate $2.25" in (res2.get("error") or ""), res2


def test_volume_tier_community_fails_fast(tmp_path, monkeypatch, fake_s3,
                                          recipe):
    """CONFIG CONTRADICTION FAILS FAST (120c28): volume lane forces
    SECURE-only in _train_pool, so --tier community + volume is an
    empty-by-construction pool — the wave loop burned 1800s treating an
    impossibility as market flakiness. Now: refused at the top of the
    run, no staging, no launch, no capacity-wait."""
    monkeypatch.setenv("RUNPOD_VOLUME_ID", "vol1")
    monkeypatch.setenv("RUNPOD_S3_ACCESS_KEY", "user_x")
    monkeypatch.setenv("RUNPOD_S3_SECRET_KEY", "rps_y")
    monkeypatch.setattr(dispatch, "ROOT", tmp_path)
    (tmp_path / "tmp").mkdir()
    res = dispatch.submit("queer", recipe, mini=True, provider="runpod",
                          tier="community")
    assert "impossible" in (res.get("error") or "")
    assert res["lane"] == "volume"
    assert not fake_s3["put"]              # nothing staged, nothing waits
