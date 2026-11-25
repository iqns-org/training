"""synth wave-1 + wave-2 engine: mutation/DPO/abstain/audit (FRAMEWORK §4b, §10).

The scenarios tree is rebuilt per test from minimal nl-to-ir templates so
tests never depend on the sibling corpora checkout. t01 exercises the
wave-1 text rules; t02 (MATCH AS handle, BIND literal, CALL with declared
ONE param, EMIT, indented WRITE, TYPE/EVENT/CAPABILITY decls) exercises
every wave-2 AST-dependent family.
"""
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def synth():
    spec = importlib.util.spec_from_file_location("synth", ROOT / "src" / "synth.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NL_TO_IR = {
    "id": "nl-to-ir/t01",
    "label": "nl-to-ir t01",
    "comment": "two-role lookup",
    "system": "nl-to-ir",
    "variants": {
        "domain": [{
            "item": "my:doc", "edge1": "ai:origin", "first": "my:depotA",
            "edge2": "ai:destination", "second": "my:depotB",
            "role1": "origin", "role2": "destination", "text": "document",
        }],
        "phrasing": [
            "Return the document's origin and destination as a list.",
            "Build a two-element result: origin then destination.",
        ],
    },
    "messages": [
        {"role": "assistant", "content":
         "PREFIX my: <http://example.org/entity/>.\n"
         "PREFIX ai: <http://example.org/property/>.\n"
         "WORLD {\n"
         "    my:doc ai:origin my:depotA .\n"
         "    my:doc ai:destination my:depotB .\n"
         "}\n"
         "MEMORY {}"},
        {"role": "user", "content": "{phrasing}"},
        {"role": "assistant", "content":
         "READ WORLD my:doc ai:origin my:?first . ONE.\n"
         "READ WORLD my:doc ai:destination my:?second . ONE.\n"
         "BIND my:?pair = [my:?first, my:?second].\n"
         "RETURN my:?pair."},
    ],
    "assert": {
        "valid": True, "world_add": [], "world_delete": [],
        "memory_add": [], "memory_delete": [], "emitted": [],
        "returned": ["{domain.first}", "{domain.second}"],
    },
}

WAVE2_FIRST = (
    "PREFIX my: <http://example.org/entity/>.\n"
    "PREFIX ai: <http://example.org/property/>.\n"
    "TYPE my:Thing.\n"
    "EVENT ai:Logged.\n"
    "CAPABILITY ai:inspect(my:?t : my:Thing ONE).\n"
    "WORLD {\n    my:box ai:temp my:Warm .\n}\nMEMORY {}")
WAVE2_STMTS = (
    "MATCH WORLD {\n    my:box ai:temp my:?t .\n} AS my:?all.\n"
    "BIND my:?n = 14.\n"
    "CALL ai:inspect(my:?t).\n"
    "BRANCH {\n    WHEN my:?n < 12 THEN {\n"
    "        WRITE WORLD {\n            my:box ai:flag my:Low .\n        }.\n"
    "        EMIT ai:Logged(my:?n).\n        RETURN my:box.\n    }\n"
    "    ELSE {\n        RETURN my:box.\n    }\n}")

WAVE2 = {
    "id": "nl-to-ir/t02",
    "label": "nl-to-ir t02",
    "comment": "threshold + capability + effect shapes for wave 2 rules",
    "system": "nl-to-ir",
    "variants": {
        "domain": [{"x": "1"}],
        "phrasing": [
            "Flag the box when its count is below 12 and return it.",
            "Below 12, mark the box Low and log; return the box.",
        ],
    },
    "messages": [
        {"role": "assistant", "content": WAVE2_FIRST},
        {"role": "user", "content": "{phrasing}"},
        {"role": "assistant", "content": WAVE2_STMTS},
    ],
    "assert": {"valid": True},
}


@pytest.fixture()
def tree(tmp_path, synth):
    scen = tmp_path / "scenarios"
    (scen / "nl-to-ir").mkdir(parents=True)
    (scen / "nl-to-ir" / "t01.json").write_text(json.dumps(NL_TO_IR))
    (scen / "nl-to-ir" / "t02.json").write_text(json.dumps(WAVE2))
    out = tmp_path / "derived"
    tmp = tmp_path / "tmp-synth"
    return scen, out, tmp


def run(synth, argv):
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = synth.main(argv)
    return code, json.loads(buf.getvalue())


def test_source_selection(synth, tree):
    scen, out, tmp = tree
    sources = list(synth.iter_sources(scen, synth.SOURCE_SYSTEMS, 0))
    assert len(sources) == 4                       # 2 phrasings × 2 templates


def test_derive_emits_verified_templates(synth, tree):
    scen, out, tmp = tree
    code, stats = run(synth, ["derive", "--scenarios", str(scen),
                              "--out", str(out), "--tmp", str(tmp)])
    assert code == 0
    assert stats["candidates"] > 0
    assert stats["accepted"], "at least one error class must be reachable"
    assert stats["preflight_fail"] == []
    accepted = set(stats["accepted"])
    # wave 1 — the text-rule families
    assert {"E1_DUP_PREFIX", "E2_UNBOUND", "E3_REDEFINE"} <= accepted
    # wave 2 — every AST-dependent family fires on the t02 shape
    assert {"E4_TYPE_MISMATCH", "E4_TYPE_CYCLE", "E5_CARD_MISMATCH",
            "E6_UNREACHABLE", "E7_IMMUTABLE_SCOPE", "E7_EFFECT_IN_PROJECTION",
            "E1_DUP_TYPE", "E1_DUP_EVENT", "E1_DUP_CAPABILITY"} <= accepted, \
        f"missing: {accepted ^ {'E4_TYPE_MISMATCH', 'E4_TYPE_CYCLE', 'E5_CARD_MISMATCH', 'E6_UNREACHABLE', 'E7_IMMUTABLE_SCOPE', 'E7_EFFECT_IN_PROJECTION', 'E1_DUP_TYPE', 'E1_DUP_EVENT', 'E1_DUP_CAPABILITY'}}"
    # declaration-tier codes are diagnosis-only (no statements-only repair
    # contract exists); statement-tier codes are repairable
    assert {"E4_TYPE_CYCLE", "E7_EFFECT_IN_PROJECTION", "E1_DUP_TYPE"} \
        <= set(stats["no_repair"])
    # wave 3 — inject-then-break families fire on any source; nested E6
    assert {"E6_PARALLEL_ARITY", "E6_BAD_TIMEOUT"} <= accepted, accepted


def test_each_mutant_carries_exactly_the_intended_error(synth, tree):
    scen, out, tmp = tree
    run(synth, ["derive", "--scenarios", str(scen), "--out", str(out),
                "--tmp", str(tmp)])
    checked = 0
    for path in (out / "invalid-program").glob("*.json"):
        rec = json.loads(path.read_text())
        assert rec["system"] == "invalid-program"
        for messages, expected in synth.check.expansions(rec):
            _doc, rep = synth.validate(messages[0]["content"])
            assert rep is not None and not rep["valid"]
            assert [e["code"] for e in rep["errors"]] == \
                [expected["errors"][0]["code"]], path.name
            assert not rep["warnings"]
            assert expected["errors"][0]["code"] in messages[2]["content"]
            checked += 1
    assert checked >= 8
    for path in (out / "repair").glob("*.json"):
        rec = json.loads(path.read_text())
        assert rec["input_error"] in rec["messages"][0]["content"]
        for messages, _ in synth.check.expansions(rec):
            assert messages[2]["content"].strip()
            fixed = synth.check.document(
                synth.check.context_nodes(synth.validate(messages[0]["content"])[0])
                + synth.ir.parse(messages[2]["content"])["body"])
            rep2 = synth.sanity.validate_ast(fixed, strict=True)
            assert rep2["valid"] and not rep2["warnings"], (path.name, rep2["errors"])


def test_inject_families_shape_free(synth, tmp_path):
    """Wave 4: E4/E5/dup-block families fire even on a source WITHOUT the
    fixture shapes — the rules inject them (real corpus had zero of those
    shapes before this)."""
    scen = tmp_path / "scenarios"
    (scen / "nl-to-ir").mkdir(parents=True)
    (scen / "nl-to-ir" / "t01.json").write_text(json.dumps(NL_TO_IR))
    out = tmp_path / "derived"
    tmp = tmp_path / "tmp-synth"
    code, stats = run(synth, ["derive", "--scenarios", str(scen),
                              "--out", str(out), "--tmp", str(tmp)])
    assert code == 0 and stats["preflight_fail"] == []
    accepted = set(stats["accepted"])
    assert {"E4_TYPE_MISMATCH", "E5_CARD_MISMATCH",
            "E1_DUP_PLAN", "E1_DUP_PROJECTION"} <= accepted, accepted
    # DPO pairs only from ctx-identical mutants: every rejected is truly
    # invalid under the ORIGINAL context, every chosen valid
    for line in (tmp / "dpo.jsonl").read_text().splitlines():
        pair = json.loads(line)
        chosen = synth._full(pair["messages"][0]["content"],
                             pair["messages"][2]["content"])
        rejected = synth._full(pair["messages"][0]["content"],
                               pair["rejected"]["messages"][2]["content"])
        assert synth.is_valid_program(chosen)
        assert not synth.is_valid_program(rejected)


def test_p23_dpo_pairs(synth, tree):
    scen, out, tmp = tree
    run(synth, ["derive", "--scenarios", str(scen), "--out", str(out),
                "--tmp", str(tmp)])
    lines = (tmp / "dpo.jsonl").read_text().splitlines()
    assert lines
    for line in lines:
        pair = json.loads(line)
        chosen = synth._full(pair["messages"][0]["content"],
                             pair["messages"][2]["content"])
        rejected = synth._full(pair["messages"][0]["content"],
                               pair["rejected"]["messages"][2]["content"])
        assert synth.is_valid_program(chosen)
        assert not synth.is_valid_program(rejected)


def test_codes_filter(synth, tree):
    scen, out, tmp = tree
    code, stats = run(synth, ["derive", "--scenarios", str(scen),
                              "--out", str(out), "--tmp", str(tmp),
                              "--codes", "E5_CARD_MISMATCH"])
    assert code == 0
    assert set(stats["accepted"]) == {"E5_CARD_MISMATCH"}
    assert list((out / "invalid-program").glob("*e5_card_mismatch*"))
    assert not list((out / "invalid-program").glob("*e7_immutable_scope*"))


def test_abstain_rows_are_not_derivable(synth, tree):
    scen, out, tmp = tree
    code, stats = run(synth, ["abstain", "--scenarios", str(scen),
                              "--out", str(out)])
    assert code == 0
    assert stats["accepted"] >= 1
    for path in (out / "abstain").glob("*.json"):
        rec = json.loads(path.read_text())
        assert rec["messages"][2]["content"].startswith("NOT DERIVABLE")
        assert synth.preflight(rec) == []


def test_rows_expansion(synth, tree):
    scen, out, tmp = tree
    run(synth, ["derive", "--scenarios", str(scen), "--out", str(out),
                "--tmp", str(tmp)])
    code, written = run(synth, ["rows", "--out", str(out),
                                "--rows-out", str(tmp)])
    assert code == 0 and written
    total = 0
    for path in (tmp / "rows").glob("*.jsonl"):
        lines = [json.loads(l) for l in path.read_text().splitlines()]
        assert all(set(l) == {"messages"} and [m["role"] for m in l["messages"]]
                   == ["assistant", "user", "assistant"] for l in lines)
        total += len(lines)
    assert total > 0


def test_audit_runs(synth, tree):
    scen, out, tmp = tree
    code, report = run(synth, ["audit", "--scenarios", str(scen),
                               "--out", str(tmp / "audit.json")])
    assert code == 0
    assert report["templates"] == 2
    assert report["coverage_by_system"]["nl-to-ir"] == 2
    assert (tmp / "audit.json").is_file()


TTL = """\
@PREFIX my: <http://example.org/entity/> .
@PREFIX ai: <http://example.org/property/> .
my:site1 ai:hasStatus my:Active .
my:site1 ai:inspectedBy my:inspector7 .
my:inspector7 ai:holdsCert "Class-A" .
my:inspector7 ai:yearsExp 12 .
my:site1 ai:riskScore 7 .
"""


def test_graph_bridge_templates(synth, tree, tmp_path):
    scen, out, _tmp = tree
    ttl = tmp_path / "site-register.ttl"
    ttl.write_text(TTL)
    code, stats = run(synth, ["graph", str(ttl), "--domain", "construction",
                              "--out", str(out)])
    assert code == 0 and stats["templates"] >= 2 and stats["skipped"] == 0
    assert stats["preflight_fail"] == []
    qa = json.loads((out / "nl-to-ir" / "site-register--qa.json").read_text())
    assert qa["system"] == "nl-to-ir"
    # answers are RECOMPUTED (P3): every expansion executes clean, and the
    # literal object types survive (int stays int, "Class-A" stays str)
    seen = set()
    for messages, expected in synth.check.expansions(qa):
        rep = synth.check.check_record(qa, messages, expected)
        seen.add(str(expected["returned"]))
        assert rep in (True, False)
    assert any("12" in s for s in seen) and any("Class-A" in s for s in seen)
    # hop chain site1 -> inspector7 -> cert exists
    hops = json.loads((out / "nl-to-ir" / "site-register--hops.json").read_text())
    mids = {e["mid"] for e in hops["variants"]["domain"]}
    assert "my:inspector7" in mids
    ab = json.loads((out / "abstain" / "site-register--abstain.json").read_text())
    assert synth.preflight(ab) == []


def test_manifest_bundle(synth, tree, tmp_path, monkeypatch):
    scen, out, tmp = tree
    run(synth, ["derive", "--scenarios", str(scen), "--out", str(out),
                "--tmp", str(tmp)])
    code, man = run(synth, ["manifest", "--out", str(out)])
    assert code == 0
    assert man["corpus_id"].startswith("derived@") and man["count"] >= 8
    assert set(man["oracles"]) == set(synth.ORACLE_FILES)
    monkeypatch.setattr(synth, "DATA_ROOT", tmp_path)
    code, bun = run(synth, ["bundle", "--domain", "cognition",
                            "--out", str(out), "--dpo", str(tmp / "dpo.jsonl")])
    assert code == 0 and bun["rows"] > 0
    dest = tmp_path / "datasets" / "cognition"
    assert sorted(p.name for p in dest.iterdir()) == \
        ["diagnosis.jsonl", "preference.jsonl", "repair.jsonl"]
    first = json.loads((dest / "diagnosis.jsonl").read_text().splitlines()[0])
    assert [m["role"] for m in first["messages"]] == \
        ["assistant", "user", "assistant"]
