"""SPEC 13ak: provenance — agent attestation + verify_run cross-check."""
import hashlib
import json

import pytest

from orchestrator import provenance
from orchestrator.ledger import Ledger
from trainer_agent import joblane

SHA = "dc9eb6c2" + "0" * 56
IMG = "ghcr.io/iqns-org/trainer@sha256:" + "f" * 64
CFG = "adapter: lora\ndataset: /work/datasets/d.jsonl\n"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------- attestation
def test_materialize_attests_everything_accepted(tmp_path):
    ds = tmp_path / "src.jsonl"
    body = b'{"messages": []}\n' * 3
    ds.write_bytes(body)
    payload = {"run_id": "r1", "command": "true", "config_yaml": CFG,
               "dataset": {"name": "d.jsonl", "path": str(ds),
                           "sha256": _sha(body)}}
    run = joblane.materialize(payload, tmp_path / "work")
    att = run["attestation"]
    assert att["dataset"] == {"name": "d.jsonl", "sha256": _sha(body),
                              "lines": 3}
    assert att["config_sha"] == _sha(CFG.encode())
    # live /v1/sanity contract: the same dict exists on disk
    assert json.loads((tmp_path / "work" / "attest.json").read_text()) == att


def test_materialize_attests_stage_inputs(tmp_path):
    blob = tmp_path / "in.tar"
    blob.write_bytes(b"tarbytes")
    payload = {"run_id": "r2", "command": "true",
               "inputs": [{"name": "data", "path": str(blob),
                           "sha256": _sha(b"tarbytes")}]}
    run = joblane.materialize(payload, tmp_path / "work")
    (att,) = run["attestation"]["inputs"]
    assert att == {"name": "data", "sha256": _sha(b"tarbytes")}


def test_materialize_still_raises_on_corrupt_dataset(tmp_path):
    ds = tmp_path / "src.jsonl"
    ds.write_bytes(b"real")
    payload = {"run_id": "r3", "command": "true",
               "dataset": {"name": "d.jsonl", "path": str(ds),
                           "sha256": "0" * 64}}
    with pytest.raises(ValueError, match="sha mismatch"):
        joblane.materialize(payload, tmp_path / "work")


# ---------------------------------------------------------------- verify_run
@pytest.fixture()
def wired(tmp_path, monkeypatch):
    led = Ledger(tmp_path / "l.db")
    files: dict[str, bytes] = {}
    monkeypatch.setattr(provenance, "lane_get",
                        lambda key, provider="runpod": files.get(key))
    monkeypatch.setattr(provenance, "corpora_git_sha", lambda: "abc123def456")

    def new_run(manifest_att=True, ledger_rec=True):
        spec = {"domain": "queer", "recipe": "x.yaml",
                "corpora": "abc123def456", "recipe_sha": "d9cabd09e5ee"}
        rid = led.create("train", spec)
        if ledger_rec:
            led.merge_spec(rid, dataset_sha=SHA, dataset_lines=42, image=IMG)
        files[f"runs/{rid}/payload.json"] = json.dumps(
            {"run_id": rid, "config_yaml": CFG,
             "dataset": {"name": "d.jsonl", "sha256": SHA}}).encode()
        att = ({"config_sha": _sha(CFG.encode()),
                "dataset": {"sha256": SHA, "lines": 42}}
               if manifest_att else None)
        man = {"agent_version": "0.3.0", "image_tag": IMG}
        if att is not None:
            man["attestation"] = att
        files[f"runs/{rid}/manifest.json"] = json.dumps(man).encode()
        return rid
    return led, files, new_run


def test_all_legs_agree(wired):
    led, _files, new_run = wired
    res = provenance.verify_run(new_run(), led=led)
    assert res["ok"], res["checks"]
    by = {c["check"]: c["status"] for c in res["checks"]}
    assert by["dataset ledger==staged"] == "pass"
    assert by["dataset staged==attested"] == "pass"
    assert by["dataset lines"] == "pass"
    assert by["config staged==attested"] == "pass"
    assert by["image ledger==manifest"] == "pass"
    assert by["corpora drift"] == "pass"


def test_remote_leg_corruption_fails(wired):
    led, files, new_run = wired
    rid = new_run()
    man = json.loads(files[f"runs/{rid}/manifest.json"])
    man["attestation"]["dataset"]["sha256"] = "beef" + "0" * 60
    files[f"runs/{rid}/manifest.json"] = json.dumps(man).encode()
    res = provenance.verify_run(rid, led=led)
    assert not res["ok"]
    assert any(c["check"] == "dataset staged==attested"
               and c["status"] == "fail" for c in res["checks"])


def test_line_truncation_fails(wired):
    led, files, new_run = wired
    rid = new_run()
    man = json.loads(files[f"runs/{rid}/manifest.json"])
    man["attestation"]["dataset"]["lines"] = 41
    files[f"runs/{rid}/manifest.json"] = json.dumps(man).encode()
    res = provenance.verify_run(rid, led=led)
    assert not res["ok"]


def test_legacy_run_warns_not_fails(wired):
    led, _files, new_run = wired
    res = provenance.verify_run(new_run(manifest_att=False,
                                        ledger_rec=False), led=led)
    assert res["ok"]                       # nothing contradicts
    by = {c["check"]: c["status"] for c in res["checks"]}
    assert by["dataset ledger record"] == "warn"
    assert by["agent attestation"] == "warn"
    # ...but the staged leg still answers the forensic question
    staged = [c for c in res["checks"]
              if c["check"] == "dataset ledger record"][0]
    assert SHA[:16] in staged["note"]


def test_missing_run_id_defaults_latest(wired):
    led, _files, new_run = wired
    new_run()
    res = provenance.verify_run(None, led=led)
    assert res["run_id"] and res["ok"]
