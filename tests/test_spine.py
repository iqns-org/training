"""SPEC 13al+13am: family gate, holdout law, eval history, new checks."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from orchestrator import dispatch, families
from orchestrator.bench import score
from orchestrator.ledger import Ledger

ROOT = Path(__file__).resolve().parents[1]

REG = {
    "qwen3.5": {"matches": ["Qwen/Qwen3.5", "qwen3.5"],
                "targets": {"merge": True, "onnx": False, "int8": False,
                            "awq": "untested", "gguf": "deferred"}},
    "smollm2": {"matches": ["HuggingFaceTB/SmolLM2"],
                "targets": {"onnx": "untested"}},
}


@pytest.fixture(autouse=True)
def _registry(monkeypatch):
    monkeypatch.setattr(families, "registry", lambda: REG)


# ------------------------------------------------------------- family gate
def test_family_resolution():
    assert families.family_for("Qwen/Qwen3.5-0.8B")[0] == "qwen3.5"
    assert families.family_for("HuggingFaceTB/SmolLM2-360M")[0] == "smollm2"
    assert families.family_for("nope/model") is None


def test_only_explicit_false_refuses():
    assert "IMPOSSIBLE" in families.refuse_reason("Qwen/Qwen3.5-0.8B", "onnx")
    assert families.refuse_reason("Qwen/Qwen3.5-0.8B", "int8")
    # untested/deferred/unknown-family/unknown-target all proceed
    assert families.refuse_reason("Qwen/Qwen3.5-0.8B", "awq") is None
    assert families.refuse_reason("Qwen/Qwen3.5-0.8B", "gguf") is None
    assert families.refuse_reason("HuggingFaceTB/SmolLM2-360M", "onnx") is None
    assert families.refuse_reason("brand/new", "onnx") is None
    assert families.refuse_reason("", "onnx") is None


def test_lineage_base_walks_stage_chain(tmp_path, monkeypatch):
    led = Ledger(tmp_path / "l.db")
    train = led.create("train", {"recipe": "x.yaml"})
    merged = led.create("stage", {"kind": "merge", "from_run": train,
                                  "base_model": "Qwen/Qwen3.5-0.8B"})
    onnxr = led.create("stage", {"kind": "onnx", "from_run": merged,
                                 "base_model": None})
    # int8's gate must find the family THROUGH onnx -> merge -> base
    assert dispatch._lineage_base(led, onnxr) == "Qwen/Qwen3.5-0.8B"
    assert dispatch._lineage_base(led, train) == ""  # no recipe readable


def test_submit_stage_refuses_before_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch, "ROOT", tmp_path)
    (tmp_path / "state").mkdir()
    led = Ledger(tmp_path / "state" / "ledger.db")
    train = led.create("train", {"recipe": ""})
    merged = led.create("stage", {"kind": "merge", "from_run": train,
                                  "base_model": "Qwen/Qwen3.5-0.8B"})
    rows_before = len(led.list())
    out = dispatch.submit_stage("onnx", merged)
    assert out["status"] == "error"
    assert out["refused"] == "family-registry"
    # NOTHING was created: the refusal costs zero ledger rows and zero pods
    assert len(led.list()) == rows_before


# ------------------------------------------------------------ holdout law
def _pd_module():
    from importlib.machinery import SourceFileLoader
    loader = SourceFileLoader("prepare_domain",
                              str(ROOT / "bin" / "prepare-domain"))
    spec = importlib.util.spec_from_loader("prepare_domain", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def test_holdout_is_stable_and_tiny():
    m = _pd_module()
    lines = [json.dumps({"messages": [{"role": "user", "content": f"q{i}"}]})
             for i in range(20000)]
    held = [l for l in lines if m.is_holdout(l)]
    assert 0.015 < len(held) / len(lines) < 0.035   # ~2%, not 0, not 10%
    # determinism: same content, same verdict, forever
    assert all(m.is_holdout(l) for l in held)
    # append new rows -> old buckets never flip (content-hash, not index)
    more = lines + [json.dumps({"messages": [{"role": "user",
                                              "content": f"z{i}"}]})
                    for i in range(5000)]
    assert all(m.is_holdout(l) for l in more if l in held)


def test_render_config_wires_holdout(tmp_path):
    r = tmp_path / "rec.yaml"
    r.write_text(yaml.safe_dump({
        "base_model": "Qwen/Qwen3.5-0.8B",
        "datasets": [{"path": "./datasets/queer.jsonl", "type": "chat_template"}]}))
    out = yaml.safe_load(dispatch.render_config(r, {}, "queer.jsonl",
                                               holdout_name="queer.holdout.jsonl"))
    (ev,) = out["eval_dataset"]
    assert ev["path"] == "/work/inputs/queer.holdout.jsonl"
    assert ev["type"] == "chat_template"
    # evaluation_strategy MUST ride the wiring: HF defaults to "no" and
    # silently skips evals (daa4d7 proved it — success, zero eval_loss)
    assert out["evaluation_strategy"] == "epoch"
    assert out["per_device_eval_batch_size"] == 1
    assert out["datasets"][0]["path"] == "/work/datasets/queer.jsonl"
    # no holdout -> no eval wiring at all (legacy submissions unaffected)
    out2 = yaml.safe_load(dispatch.render_config(r, {}, "queer.jsonl"))
    assert "eval_dataset" not in out2 and "evaluation_strategy" not in out2


# ------------------------------------------------------------- eval history
def test_record_and_list_evals(tmp_path):
    led = Ledger(tmp_path / "l.db")
    rows = [
        {"host": "nano0", "model": "kandi", "q": "identity", "pass": "PASS",
         "passes": 3, "repeats": 3},
        {"host": "nano0", "model": "kandi", "q": "world", "pass": "FAIL",
         "passes": 0, "repeats": 3},
        {"host": "nano0", "model": "kandi", "q": "none-q", "pass": "—",
         "passes": 0, "repeats": 3},
    ]
    led.record_eval("kandi", rows, run_id="r1", host="nano0")
    (rec,) = led.list_evals(model="kandi")
    assert rec["summary"] == "1/2"          # "—" report-only rows excluded
    assert rec["run_id"] == "r1"
    assert led.list_evals(model="other") == []


# ------------------------------------------------------------- new checks
def test_extended_check_grammar():
    assert score("json_keys:mood,topic", '{"mood":"a","topic":"b"}') == "PASS"
    assert score("json_keys:mood,x", '{"mood":"a"}') == "FAIL"
    assert score("not_contains:qwen", "I'm Kandi.") == "PASS"
    assert score("not_contains:qwen", "I am Qwen") == "FAIL"
    assert score("max_len:30", "short answer") == "PASS"
    assert score("max_len:5", "much too long for five") == "FAIL"
    assert score("nonempty", "hi") == "PASS"
    assert score("nonempty", "   ") == "FAIL"
