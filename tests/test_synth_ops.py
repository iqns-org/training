"""Close-out family tests (REGIMES R3-R7 + P5/P19): gbnf, grade, gaps,
merge, equiv, conflict, sample (fake endpoint), mix (seeded). Everything
runs on tmp trees — no corpora, no network, no GPU.
"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

NL_TEMPLATE = {
    "comment": "c", "label": "l", "id": "nl-to-ir/t01", "system": "nl-to-ir",
    "variants": {
        "domain": [{"item": "my:doc", "edge1": "ai:origin",
                    "first": "my:depotA", "edge2": "ai:destination",
                    "second": "my:depotB"}],
        "phrasing": ["What is the document's origin and destination?",
                     "Return both roles for the document, in order."],
    },
    "messages": [
        {"role": "assistant", "content":
         "PREFIX my: <http://example.org/entity/>.\n"
         "PREFIX ai: <http://example.org/property/>.\n"
         "WORLD {\n    my:doc ai:origin my:depotA .\n"
         "    my:doc ai:destination my:depotB .\n}\nMEMORY {}"},
        {"role": "user", "content": "{phrasing}"},
        {"role": "assistant", "content":
         "READ WORLD my:doc ai:origin my:?first . ONE.\n"
         "READ WORLD my:doc ai:destination my:?second . ONE.\n"
         "BIND my:?pair = [my:?first, my:?second].\n"
         "RETURN my:?pair."},
    ],
    "assert": {"valid": True, "returned": ["{domain.first}", "{domain.second}"],
               "world_add": [], "world_delete": [], "memory_add": [],
               "memory_delete": [], "emitted": []},
}


@pytest.fixture(scope="module")
def synth():
    spec = importlib.util.spec_from_file_location("synth", ROOT / "src" / "synth.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(synth, argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = synth.main(argv)
    return code, json.loads(buf.getvalue())


def tree(tmp_path):
    d = tmp_path / "derived"
    (d / "nl-to-ir").mkdir(parents=True)
    (d / "nl-to-ir" / "t01.json").write_text(json.dumps(NL_TEMPLATE))
    return d


def test_gbnf_cmd(synth, tmp_path):
    out = tmp_path / "ir.gbnf"
    code, res = run(synth, ["gbnf", "--out", str(out)])
    assert code == 0 and res["undefined_refs"] == [] and res["rules"] > 20
    text = out.read_text()
    assert "::=" in text and "readstmt" in text and "parallelstmt" in text


def test_grade_cmd(synth, tmp_path):
    d = tree(tmp_path)
    msgs, _expected = next(iter(synth.check.expansions(NL_TEMPLATE)))
    rows = tmp_path / "rows.jsonl"
    rows.write_text(
        json.dumps({"messages": msgs}) + "\n" +                       # gold
        json.dumps({"messages": msgs[:2] + [
            {"role": "assistant", "content": "Sure! Orwell."}]}) + "\n")
    reward = tmp_path / "reward.jsonl"
    code, res = run(synth, ["grade", str(rows), "--derived", str(d),
                            "--out", str(reward)])
    assert code == 0 and res["graded"] == 2 and res["kept"] == 1
    scores = [json.loads(l)["s"] for l in reward.read_text().splitlines()]
    assert scores == [1, 0]


def test_gaps_cmd(synth, tmp_path):
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps({"rows": [
        {"model": "kandi", "question": "ir-repair", "passed": False},
        {"model": "kandi", "question": "abstain", "passed": False},
        {"model": "kandi", "question": "ir-translate", "passed": True}]}))
    code, res = run(synth, ["gaps", "--bench", str(bench)])
    assert code == 0 and res["failing"] == 2
    assert all(g["commands"] for g in res["gaps"])
    assert any("E2_UNBOUND" in c for g in res["gaps"] for c in g["commands"])


def test_gaps_real_bench_shape(synth, tmp_path):
    """qtrain bench --json writes a LIST of rows with q/pass/observed —
    gaps must consume it verbatim (the r1 round-1 crash regression)."""
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps([
        {"host": "nano0.lan", "model": "cognition", "q": "ir-translate",
         "pass": "PASS", "passes": 2, "repeats": 2, "observed": None},
        {"host": "nano0.lan", "model": "cognition", "q": "ir-repair",
         "pass": "FAIL", "passes": 0, "repeats": 2, "observed": None},
        {"host": "nano0.lan", "model": "cognition", "q": "feeling",
         "pass": "—", "passes": 0, "repeats": 2, "observed": None}]))
    code, res = run(synth, ["gaps", "--bench", str(bench)])
    assert code == 0 and res["failing"] == 1          # "—" is not a gap
    assert res["gaps"][0]["question"] == "ir-repair"
    assert res["gaps"][0]["models"] == ["cognition"]


def test_merge_cmd(synth, tmp_path):
    out = tmp_path / "merge.json"
    code, res = run(synth, ["merge", "--base", "Qwen/Qwen3.5-0.8B",
                            "--adapters", "/models/a,/models/b",
                            "--method", "dare_ties", "--density", "0.3",
                            "--out", str(out)])
    assert code == 0 and res["adapters"] == 2
    cfg = json.loads(out.read_text())
    assert cfg["merge_method"] == "dare_ties"
    assert cfg["parameters"]["density"] == 0.3
    assert cfg["models"][0]["base_model"] == "Qwen/Qwen3.5-0.8B"


def test_equiv_cmd(synth, tmp_path):
    d = tree(tmp_path)
    code, res = run(synth, ["equiv", "--out", str(d), "--max-pairs", "8"])
    assert code == 0
    assert res["equivalent"] >= 1, res           # rename ⇒ identical delta
    assert res["undecidable"] >= 1, res          # drop ⇒ E2-unbound ⇒ undecidable
    t = d / "equivalence" / "equivalent.json"
    assert t.is_file()
    rec = json.loads(t.read_text())
    e = rec["variants"]["domain"][0]
    assert "my:?r0" in e["b"]                     # uniform rename applied
    assert rec["messages"][2]["content"].startswith("# VERDICT: EQUIVALENT")
    # and it satisfies the fleet equivalence contract on every expansion
    for messages, _expected in synth.check.expansions(rec):
        synth.check.check_record(rec, messages, {"valid": True})


def test_equiv_different_verdict(synth, tmp_path):
    """DIFFERENT requires an OBSERVABLE deletion: a program with EMIT
    (bound, before RETURN); dropping the EMIT changes the delta ->
    DIFFERENT."""
    rec = json.loads(json.dumps(NL_TEMPLATE))
    rec["messages"][0]["content"] = rec["messages"][0]["content"].replace(
        "WORLD {", "EVENT ai:Done.\nWORLD {")
    rec["messages"][2]["content"] = rec["messages"][2]["content"].replace(
        "RETURN my:?pair.", "EMIT ai:Done(my:?first).\nRETURN my:?pair.")
    rec["assert"]["emitted"] = [{"event": "ai:Done",
                                 "args": ["{domain.first}"]}]
    d = tmp_path / "derived"
    (d / "nl-to-ir").mkdir(parents=True)
    (d / "nl-to-ir" / "t02.json").write_text(json.dumps(rec))
    code, res = run(synth, ["equiv", "--out", str(d), "--max-pairs", "8"])
    assert res["different"] >= 1, res
    t = d / "equivalence" / "different.json"
    assert t.is_file()
    r2 = json.loads(t.read_text())
    assert r2["messages"][2]["content"].startswith("# VERDICT: DIFFERENT")
    assert "emitted" in r2["variants"]["domain"][0]["reason"]


def test_conflict_cmd(synth, tmp_path):
    a = tmp_path / "reg.ttl"
    b = tmp_path / "rev.ttl"
    a.write_text("@PREFIX my: <http://example.org/entity/> .\n"
                 "@PREFIX ai: <http://example.org/property/> .\n"
                 "my:site1 ai:riskScore 7 .\n")
    b.write_text("@PREFIX my: <http://example.org/entity/> .\n"
                 "@PREFIX ai: <http://example.org/property/> .\n"
                 "my:site1 ai:riskScore 12 .\n")
    d = tree(tmp_path)
    code, res = run(synth, ["conflict", str(a), str(b), "--out", str(d)])
    assert code == 0 and res["conflicts"] == 1 and res["written"]
    rec = json.loads((d / "conflict" / "sources.json").read_text())
    ctx = rec["variants"]["domain"][0]["ctx"]
    assert "my:site1 ai:riskScore 7 ." in ctx
    assert "my:site1 ai:riskScore 12 ." in ctx
    assert "CONFLICT" in rec["messages"][2]["content"]


def test_sample_cmd_fake_endpoint(synth, tmp_path, monkeypatch):
    d = tree(tmp_path)
    msgs, _ = next(iter(synth.check.expansions(NL_TEMPLATE)))

    def fake_http(url, payload, timeout=120):
        if payload["messages"][1]["content"].startswith("What"):
            return {"choices": [{"message": {"content": msgs[2]["content"]}}]}
        return {"choices": [{"message": {"content": "The depot, probably."}}]}

    monkeypatch.setattr(synth, "http_json", fake_http)
    monkeypatch.setattr(synth, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(synth, "TMP", tmp_path / "tmp-synth")
    code, res = run(synth, ["sample", "--url", "http://fake/v1", "--k", "1",
                            "--limit", "2", "--out", str(d),
                            "--dataset", "cognition-rft"])
    assert code == 0 and res["sampled"] == 2 and res["kept"] == 1
    assert sum(res["rejects"].values()) == 1          # junk answer -> 1 bucket
    assert list(res["rejects"]) == list(res["rejects"])[:1]  # deterministic
    # buckets alone can't aim the data lever; each reject is also
    # attributed to the system it was graded against (r2 lesson)
    rbs = res["rejects_by_system"]
    assert list(rbs) == ["nl-to-ir"]
    assert sum(rbs["nl-to-ir"].values()) == 1
    assert (tmp_path / "tmp-synth" / "rft" / "rft.jsonl").is_file()
    assert (tmp_path / "datasets" / "cognition-rft" / "rft.jsonl").is_file()
    # reject dump: the row BEHIND the bucket — prompt+gold+completion+why
    rej = tmp_path / "tmp-synth" / "rft" / "rejects.jsonl"
    assert rej.is_file() and res["reject_dump"] == str(rej)
    r = json.loads(rej.read_text().splitlines()[0])
    assert r["system"] == "nl-to-ir" and r["id"] == "nl-to-ir/t01"
    assert r["gold"] == msgs[2]["content"]
    assert r["completion"] == "The depot, probably."
    assert r["bucket"] in ("IRError", "AssertionError") and r["why"]


def test_pairs_cmd_fuel_from_rejects(synth, tmp_path):
    src = tmp_path / "rejects.jsonl"
    src.write_text(json.dumps({
        "system": "repair", "id": "repair/t1", "bucket": "AssertionError",
        "why": "AssertionError: boom", "prompt": [
            {"role": "assistant", "content": "A"},
            {"role": "user", "content": "fix"}],
        "gold": "GOLD", "completion": "BAD",
    }) + "\n" + json.dumps({
        "system": "nl-to-ir", "id": "nl-to-ir/t2", "bucket": "IRError",
        "why": "IRError: unparseable", "prompt": [
            {"role": "assistant", "content": "A"},
            {"role": "user", "content": "q"}],
        "gold": "G2", "completion": "junk",
    }) + "\n")
    out = tmp_path / "dpo.jsonl"
    code, res = run(synth, ["pairs", "--in", str(src), "--out", str(out),
                            "--buckets", "AssertionError"])
    assert code == 0 and res["pairs"] == 1          # filter honored
    p = json.loads(out.read_text())
    assert p["messages"][-1]["content"] == "GOLD"   # chosen = gold row
    assert p["rejected"]["messages"][-1]["content"] == "BAD"
    assert p["system"] == "repair"
    code, res = run(synth, ["pairs", "--in", str(src), "--out", str(out)])
    assert res["pairs"] == 2 and res["by_system"] == {"repair": 1,
                                                      "nl-to-ir": 1}


def test_mix_cmd(synth, tmp_path, monkeypatch):
    root = tmp_path / "t"
    (root / "tmp").mkdir(parents=True)
    monkeypatch.setattr(synth, "ROOT", root)
    monkeypatch.setattr(synth, "DATA_ROOT", tmp_path)
    for name in ("a", "b"):
        lines = [json.dumps({"messages": [{"role": "user",
                                           "content": f"{name}{i}"}]})
                 for i in range(10)]
        (root / "tmp" / f"{name}.jsonl").write_text("\n".join(lines) + "\n")
    code, res = run(synth, ["mix", "--sources", "a:1,b:3", "--rows", "12",
                            "--seed", "5", "--name", "demo"])
    assert code == 0 and res["rows"] == 12
    assert res["weights"] == {"a": 0.25, "b": 0.75}
    lines = (tmp_path / "datasets" / "demo" / "mix.jsonl"
             ).read_text().splitlines()
    assert sum(1 for l in lines if '"b' in l) == 9 and len(lines) == 12
    # seeded determinism: same seed -> same content-address
    code2, res2 = run(synth, ["mix", "--sources", "a:1,b:3", "--rows", "12",
                              "--seed", "5", "--name", "demo"])
    assert res2["mix"] == res["mix"]


def test_sample_cmd_interleaves_families(synth, tmp_path, monkeypatch):
    """r3 fix: one big family must not eat the whole --limit —
    expansions round-robin across systems (sorted files let
    r1/r2 measure ONE family while claiming the corpus)."""
    d = tree(tmp_path)
    ab = json.loads(json.dumps(NL_TEMPLATE))
    ab["id"], ab["system"] = "abstain/a01", "abstain"
    ab["variants"]["phrasing"] = [f"Who wrote it? v{i}" for i in range(6)]
    (d / "abstain").mkdir()
    (d / "abstain" / "a01.json").write_text(json.dumps(ab))
    monkeypatch.setattr(
        synth, "http_json",
        lambda url, payload, timeout=120: {
            "choices": [{"message": {"content": "The depot, probably."}}]})
    monkeypatch.setattr(synth, "TMP", tmp_path / "tmp-synth")
    code, res = run(synth, ["sample", "--url", "http://fake/v1",
                            "--k", "1", "--limit", "4", "--out", str(d)])
    assert code == 0 and res["prompts"] == 4
    assert set(res["rejects_by_system"]) == {"abstain", "nl-to-ir"}
