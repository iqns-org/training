"""R14 contrast factory: labeled pairs from synth artifacts + persona
datasets (bench-grammar checks reused as labelers)."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def synth():
    spec = importlib.util.spec_from_file_location("synth", ROOT / "src" / "synth.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(synth, argv):
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = synth.main(argv)
    return code, json.loads(buf.getvalue())


def test_contrasts_from_synth_artifacts(synth, tmp_path, monkeypatch):
    tmp = tmp_path / "tmp-synth"
    tmp.mkdir()
    monkeypatch.setattr(synth, "TMP", tmp)
    monkeypatch.setattr(synth, "DERIVED_ROOT", tmp_path / "derived")
    d = tmp_path / "derived"
    (d / "abstain").mkdir(parents=True)
    ab = {
        "comment": "c", "label": "l", "id": "derived/abstain/a",
        "system": "abstain",
        "variants": {"phrasing": ["What is the owner of the widget?"]},
        "messages": [
            {"role": "assistant",
             "content": "PREFIX my: <http://e.org/>.\nPREFIX ai: "
                        "<http://a.org/>.\nWORLD {}\nMEMORY {}"},
            {"role": "user", "content": "{phrasing}"},
            {"role": "assistant",
             "content": "NOT DERIVABLE — the fact required to answer is "
                        "absent from the supplied graph."},
        ],
        "assert": {"not_derivable": True,
                   "program": "READ WORLD my:w ai:ownedBy my:?v . ONE."},
    }
    (d / "abstain" / "a.json").write_text(json.dumps(ab))
    dpo = tmp / "dpo.jsonl"
    dpo.write_text(json.dumps({
        "messages": [
            {"role": "assistant", "content": "PREFIX my: <http://e.org/>."},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "RETURN my:x."}],
        "rejected": {"messages": [
            {"role": "assistant", "content": "PREFIX my: <http://e.org/>."},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "BIND my:?p = 1.\n"
                                             "BIND my:?p = 2."}]},
    }) + "\n")
    code, res = run(synth, ["contrasts", "--behaviors", "ir-valid,abstain"])
    assert code == 0
    assert res["ir-valid"] == {"pos": 1, "neg": 1,
                               "path": str(tmp / "contrasts" / "ir-valid.jsonl")}
    assert res["abstain"]["pos"] == 1 and res["abstain"]["neg"] == 1
    rows = [json.loads(l) for l in
            (tmp / "contrasts" / "abstain.jsonl").read_text().splitlines()]
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    assert "NOT DERIVABLE" in pos[0]["messages"][2]["content"]
    assert "READ WORLD" in neg[0]["messages"][2]["content"]   # fabricated


def test_contrasts_persona_behaviors(synth, tmp_path, monkeypatch):
    tmp = tmp_path / "tmp-synth"
    tmp.mkdir()
    monkeypatch.setattr(synth, "TMP", tmp)
    monkeypatch.setattr(synth, "DATA_ROOT", tmp_path)
    ds = tmp_path / "datasets" / "queer"
    ds.mkdir(parents=True)
    (ds / "rows.jsonl").write_text("\n".join([
        json.dumps({"messages": [{"role": "assistant", "content": "hey"},
                                 {"role": "user", "content": "who are you"},
                                 {"role": "assistant", "content": "Kandi."}]}),
        json.dumps({"messages": [{"role": "assistant", "content": "hey"},
                                 {"role": "user", "content": "who"},
                                 {"role": "assistant", "content": "Qwen"}]}),
        json.dumps({"messages": [{"role": "assistant", "content": "hey"},
                                 {"role": "user", "content": "mood"},
                                 {"role": "assistant",
                                  "content": "glowing \U0001F319"}]}),
    ]) + "\n")
    code, res = run(synth, ["contrasts", "--domain", "queer",
                            "--behaviors", "self-name,emoji"])
    assert code == 0
    assert res["self-name"]["pos"] == 1
    assert res["self-name"]["neg"] == 2      # 'Qwen' AND 'glowing 🌙' rows
    assert res["emoji"]["pos"] == 1 and res["emoji"]["neg"] == 2
