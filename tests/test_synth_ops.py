"""Close-out family tests (REGIMES R3-R7 + P5/P19): gbnf, grade, gaps,
merge, equiv, conflict, sample (fake endpoint), mix (seeded). Everything
runs on tmp trees — no corpora, no network, no GPU.
"""
import contextlib
import importlib.util
import io
import json
import os
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


def test_openai_endpoint_forms(synth):
    # the bench-vs-synth trap: synth historically demanded a /v1 suffix;
    # now every shape works, and they all land on the chat endpoint.
    assert (synth.openai_endpoint("http://h:8000")
            == "http://h:8000/v1/chat/completions")
    assert (synth.openai_endpoint("http://h:8000/v1")
            == "http://h:8000/v1/chat/completions")
    assert (synth.openai_endpoint("http://h:8000/v1/")
            == "http://h:8000/v1/chat/completions")
    assert (synth.openai_endpoint("http://h:8000/v1/chat/completions")
            == "http://h:8000/v1/chat/completions")


def _pair(sys_name, user, gold, rejected, shape="dict"):
    rej = ({"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": rejected}]}
           if shape == "dict" else rejected)
    return {"system": sys_name, "why": "test",
            "messages": [{"role": "system", "content": sys_name},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": gold}],
            "rejected": rej}


def test_promote_precedence_quarantine_idempotent(synth, tmp_path,
                                                  monkeypatch):
    # the D7 home: precedence (real student negatives outvote curated),
    # dedup named in receipt, holdout quarantine, malformed refuse,
    # byte-stable rerun. _is_holdout patched to a marker predicate —
    # the law itself is pinned by the mirror tests elsewhere.
    src = tmp_path / "rft"
    src.mkdir()
    (src / "dpo-mining.jsonl").write_text("\n".join(json.dumps(r) for r in [
        _pair("repair", "SAME Q", "GOLD", "real-student-bad"),
        _pair("repair", "uniq q", "GOLD2", "HOLD-marker bad"),
        _pair("repair", "uniq q2", "GOLD3", "GOLD3"),   # malformed: == gold
        _pair("repair", "uniq q3", "GOLD4", "old string shape",
              shape="str"),                              # both shapes work
    ]) + "\n")
    (src / "dpo-craft.jsonl").write_text(json.dumps(
        _pair("repair", "SAME Q", "GOLD", "synthetic-neg")) + "\n")
    monkeypatch.setattr(synth, "_is_holdout", lambda line: "HOLD" in line)
    out = tmp_path / "datasets" / "cognition-dpo"
    code, res = run(synth, ["promote", "--src", str(src), "--out", str(out)])
    assert code == 0
    assert res["kept"] == 2 and res["quarantined_holdout"] == 1
    assert res["dedup_dropped"] == 1 and res["rejects"] == {"malformed": 1}
    assert res["by_src"] == {"mining": 2}
    kept = [json.loads(l) for l in
            (out / "pairs.jsonl").read_text().splitlines()]
    assert kept[0]["_src"] == "mining"       # curated twin was dropped
    assert any("old string shape" in json.dumps(k["rejected"]) for k in kept)
    ho = (out / "pairs.holdout.jsonl").read_text().splitlines()
    assert len(ho) == 1 and "HOLD" in ho[0]
    rec = json.loads((out / "receipt.json").read_text())
    assert rec["sources"]["mining"] and rec["sources"]["craft"]
    assert rec["drops"][0]["reason"] == "real-twin"   # curated twin of a
    # real pair drops, but a DIFFERENT failure mode on an old prompt is
    # kept: add one and it must survive
    assert rec["distinct_prompts"] == 2
    before = (out / "pairs.jsonl").read_bytes()   # idempotent: same input,
    run(synth, ["promote", "--src", str(src), "--out", str(out)])
    assert (out / "pairs.jsonl").read_bytes() == before    # same bytes
    with open(src / "dpo-mining.jsonl", "a") as fh:
        fh.write(json.dumps(_pair("repair", "SAME Q", "GOLD",
                                  "second distinct failure")) + "\n")
    code2, res2 = run(synth, ["promote", "--src", str(src),
                              "--out", str(tmp_path / "d2")])
    assert res2["kept"] == 3 and res2["negatives_per_prompt_max"] == 2


def test_promote_real_law_census_consistent(synth, tmp_path):
    # under the LIVE law, kept + quarantined == valid rows, always;
    # and template-law bucketing is per-system, so 30 rows of ONE
    # system land on one side together (the split-consistency point:
    # a family whose SFT rows are holdout-side cannot train r3.5).
    src = tmp_path / "rft"
    src.mkdir()
    (src / "dpo-mining.jsonl").write_text("\n".join(
        json.dumps(_pair("abstain", f"q{i}", "G", "x"))
        for i in range(30)) + "\n")
    code, res = run(synth, ["promote", "--src", str(src),
                            "--out", str(tmp_path / "d")])
    assert res["kept"] + res["quarantined_holdout"] == 30
    assert res["law"] in ("template", "gold")
    assert res["kept"] in (0, 30)            # one bucket, one side


def test_user_shape_sibling_bucket(synth, monkeypatch):
    # r4 law completion: a row's PACKAGING must never move its holdout
    # bucket. The same ctx shipped 3-message and user-shape must land
    # on the same side under BOTH laws (prepare-domain and the mirror
    # agree — tested via the mirror here, the shared law test pins the
    # pair).
    import json as _json
    ctx = "PREFIX my: <http://e/>.\nWORLD {\n    my:a ai:p my:b .\n}"
    row3 = _json.dumps({"messages": [
        {"role": "assistant", "content": ctx},
        {"role": "user", "content": "equivalent?"},
        {"role": "assistant", "content": "# VERDICT: EQUIVALENT"}]})
    shape = synth.to_user_shape(json.loads(row3)["messages"])
    row2 = _json.dumps({"messages": shape})
    for law in ("template", "gold"):
        monkeypatch.setenv("QTRAIN_HOLDOUT_LAW", law)
        assert synth._is_holdout(row3) == synth._is_holdout(row2), law


def test_user_shape_oracle_roundtrip(synth):
    # the oracle unpacks the fused user turn and applies the task
    # contract unchanged — a gold that passed packed must pass fused
    msgs = [
        {"role": "assistant",
         "content": "PREFIX my: <http://e/>.\nPREFIX ai: <http://p/>.\n"
                    "WORLD {\n    my:s ai:q my:d .\n}"},
        {"role": "user", "content": "Same or not?"},
        {"role": "assistant", "content": "# VERDICT: EQUIVALENT"}]
    rec = {"system": "abstain", "messages": msgs,
           "assert": {"not_derivable": True,
                      "program": "READ WORLD my:s ai:x my:?v . ONE.\n"
                                 "RETURN my:?v."}}
    # abstain preflight-style: fused form keeps the gold verbatim
    fused = synth.to_user_shape(msgs)
    assert fused[0]["content"].endswith("Same or not?")
    assert fused[1]["content"] == "# VERDICT: EQUIVALENT"
    assert len(fused) == 2 and [m["role"] for m in fused] == [
        "user", "assistant"]


def test_abstain_gold_names_the_gap(synth):
    # D3 law in data terms: composed golds QUOTE the missing fact and
    # NEVER the removed value. Build one graph-domain abstain pool the
    # way cmd_graph does and assert both halves of the law.
    dom = {"subj": "my:site1", "pred": "ai:riskScore_27",
           "slabel": "site 1", "plabel": "risk score 27",
           "obj": "my:Depot_95"}
    gvar = synth.pick_variant(0, synth.ABSTAIN_GOLDS).format(
        gap=f"no {dom['plabel']} fact for {dom['slabel']} "
            f"({dom['subj']} {dom['pred']} is what the question needs)")
    assert gvar.startswith("NOT DERIVABLE")
    assert "risk score 27" in gvar and "my:site1" in gvar
    assert "\u2014" not in gvar and "\u2013" not in gvar   # dash-free
    assert "my:Depot_95" not in gvar                        # value hidden


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


def test_audit_cmd_flags_hazards_and_leak(synth, tmp_path, monkeypatch):
    d = tmp_path / "cognition"
    d.mkdir()
    msgs, _ = next(iter(synth.check.expansions(NL_TEMPLATE)))
    prog_gold = msgs[2]["content"]
    clean = {"messages": msgs}
    dashed = {"messages": [msgs[0], msgs[1], {
        "role": "assistant",
        "content": prog_gold.replace("RETURN", "RETURN —", 1)}]}
    rows = [clean] * 4 + [dashed]
    # deterministic bucket: last row of each family is "holdout"
    flags = {"n": 0}

    def fake_hold(line):
        obj = json.loads(line)
        return obj["messages"][-1]["content"] == dashed["messages"][-1]["content"]

    monkeypatch.setattr(synth, "_is_holdout", fake_hold)
    (d / "nl-to-ir.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    code, res = run(synth, ["profile", "--path", str(d)])
    fam = res["nl-to-ir"]
    assert code == 0 and fam["rows"] == 5
    assert fam["kind"] == "program"
    assert fam["hazards_in_program_rows"]["dash"] == 1
    assert fam["uniq_gold_ratio"] == 0.4          # 4 copies + 1 variant
    assert fam["holdout_rows"] == 1
    assert fam["holdout_template_leak"] == 1.0    # sibling trained: 13am law
    assert fam["holdout_gold_leak"] == 0.0        # variant gold unique:
    #   the two leak metrics are ORTHOGONAL — r3 measured template-leak
    #   0.0 with 53% gold-string leak; a metric that moves with the
    #   other is not a second witness


def test_profile_gold_leak_catches_formulaic_family(synth, tmp_path,
                                                   monkeypatch):
    """The 6909eb lesson (holdout NLL 0.0095 on 'unseen' templates):
    two DIFFERENT templates sharing the SAME gold string — template
    law says clean, gold-string law says memorization. Both metrics
    must be reported so the pair is visible."""
    d = tmp_path / "diag"
    d.mkdir()
    gold = "INVALID: E3_REDEFINE - assigned twice."
    train = {"messages": [{"role": "assistant", "content": "CTX-ONE"},
                          {"role": "user", "content": "q?"},
                          {"role": "assistant", "content": gold}]}
    held = {"messages": [{"role": "assistant", "content": "CTX-TWO"},
                         {"role": "user", "content": "q?"},
                         {"role": "assistant", "content": gold}]}
    monkeypatch.setattr(synth, "_is_holdout",
                        lambda line: "CTX-TWO" in line)
    (d / "diagnosis.jsonl").write_text(
        json.dumps(train) + "\n" + json.dumps(held) + "\n")
    code, res = run(synth, ["profile", "--path", str(d)])
    fam = res["diagnosis"]
    assert code == 0
    assert fam["holdout_template_leak"] == 0.0    # v2 law does its job
    assert fam["holdout_gold_leak"] == 1.0        # ...and MISSES this


def test_profile_program_rows_are_structural(synth, tmp_path):
    """D4 closed: program-hood keys on the FIRST LINE's IR syntax, not
    on keyword VOCABULARY anywhere — a prose verdict that QUOTES
    MATCH/TYPE keeps its legal em-dash in hazards_in_prose_rows, and
    a real program gold's dash is caught in the program column."""
    d = tmp_path / "shape"
    d.mkdir()
    em = "\u2014"
    prose = {"messages": [
        {"role": "assistant", "content": "CTX"},
        {"role": "user", "content": "why?"},
        {"role": "assistant",
         "content": f"INVALID: E1_DUP_TYPE {em} TYPE my:Particle "
                    "is declared twice, MATCH never binds."}]}
    program = {"messages": [
        {"role": "assistant", "content": "CTX"},
        {"role": "user", "content": "fix it"},
        {"role": "assistant",
         "content": f"BIND my:?x = 1 {em} .\nRETURN my:?x."}]}
    prose2 = {"messages": [
        {"role": "assistant", "content": "CTX"},
        {"role": "user", "content": "why twice?"},
        {"role": "assistant",
         "content": f"INVALID: E2_SCOPE {em} READ sees no BIND."}]}
    (d / "diagnosis.jsonl").write_text(
        "\n".join(json.dumps(r) for r in (prose, prose2, program)) + "\n")
    code, res = run(synth, ["profile", "--path", str(d)])
    assert code == 0
    fam = res["diagnosis"]
    assert fam["kind"] == "prose"                     # 1 of 3 rows IR
    assert fam["hazards_in_program_rows"]["dash"] == 1   # the real program
    assert fam["hazards_in_prose_rows"]["dash"] == 2     # quoted-keyword prose


def test_holdout_mirror_matches_prepare_domain(synth, monkeypatch):
    """The audit MIRRORS bin/prepare-domain's holdout law; if the law
    moves there and not here, this test fails loud — under EVERY law,
    on rows chosen so the laws DISAGREE (same gold, different
    template)."""
    script = ROOT / "bin" / "prepare-domain"     # extensionless: no loader
    ns = {"__name__": "prepd", "__file__": str(script)}
    exec(compile(script.read_text(), script.name, "exec"), ns)
    same_gold = [
        json.dumps({"messages": [{"role": "assistant",
                                  "content": "CTX-alpha-1"},
                                 {"role": "user", "content": "q"},
                                 {"role": "assistant", "content": "G-77"}]}),
        json.dumps({"messages": [{"role": "assistant",
                                  "content": "CTX-gamma-3j"},
                                 {"role": "user", "content": "q"},
                                 {"role": "assistant", "content": "G-77"}]}),
    ]
    lines = ([json.dumps({"messages": NL_TEMPLATE["messages"]}),
              '{"messages": [{"role":"user","content":"x"},'
              '{"role":"assistant","content":"y"}]}',
              '{"messages": []}', '{"a": 1}'] + same_gold)
    for law_value in ("template", "gold"):
        monkeypatch.setenv("QTRAIN_HOLDOUT_LAW", law_value)
        for line in lines:
            assert synth._is_holdout(line) == ns["is_holdout"](line), \
                f"mirror drift under law={law_value}: {line[:60]}"


def test_holdout_law_gold_cohorts_and_typo_refuses(synth, monkeypatch):
    """law=gold puts rows sharing a gold string on the SAME side even
    across disagreeing templates (the template law splits these two —
    fixture chosen by hash so the contrast is real, not luck); an
    unknown QTRAIN_HOLDOUT_LAW value fails loud on BOTH mirrors; the
    unset default is template (never silently anything else)."""
    a = json.dumps({"messages": [{"role": "assistant",
                                  "content": "CTX-alpha-1"},
                                 {"role": "user", "content": "q"},
                                 {"role": "assistant",
                                  "content": "G-77"}]})
    b = json.dumps({"messages": [{"role": "assistant",
                                  "content": "CTX-gamma-3j"},
                                 {"role": "user", "content": "q"},
                                 {"role": "assistant",
                                  "content": "G-77"}]})
    assert synth._is_holdout(a) is False          # default = template
    assert synth._is_holdout(b) is True           # laws disagree here
    monkeypatch.setenv("QTRAIN_HOLDOUT_LAW", "gold")
    assert synth._is_holdout(a) == synth._is_holdout(b)
    monkeypatch.setenv("QTRAIN_HOLDOUT_LAW", "tamplate")   # typo
    with pytest.raises(SystemExit):
        synth._is_holdout(a)
    with pytest.raises(SystemExit):
        synth._is_holdout(b)


def test_bundle_refuses_unvalidated_system(synth, tmp_path, capsys,
                                           monkeypatch):
    """Fail CLOSED: a family with no declared verification gate ships
    zero rows and names itself in gold_rejects — the 2,562-phantom-
    reject class (verdict prose graded by the concrete contract) can
    only recur as an explicit refusal now."""
    from types import SimpleNamespace
    scen, der, data = tmp_path/"scen", tmp_path/"der", tmp_path/"data"
    scen.mkdir(); der.mkdir()
    unknown = {"system": "quantum-foo", "id": "quantum-foo/x",
               "variants": {},
               "messages": [{"role": "user", "content": "q"},
                            {"role": "assistant", "content": "a"}]}
    (scen / "unknown.json").write_text(json.dumps(unknown))
    (scen / "good.json").write_text(json.dumps(CONCRETE_REC))
    monkeypatch.setattr(synth, "SCEN_ROOT", scen)
    monkeypatch.setattr(synth, "DERIVED_ROOT", der)
    monkeypatch.setattr(synth, "DATA_ROOT", data)
    monkeypatch.setattr(synth, "TMP", tmp_path / "notmp")
    synth.cmd_bundle(SimpleNamespace(domain="probe", out=None, dpo=None))
    out = json.loads(capsys.readouterr().out)
    assert out["gold_rejects"].get("quantum-foo:unvalidated-system") == 1
    assert not (data / "datasets" / "probe" / "quantum-foo.jsonl")\
        .exists()
    assert (data / "datasets" / "probe" / "concrete-examples.jsonl")\
        .read_text().strip()                     # known family ships


CONCRETE_REC = {
    "comment": "c", "label": "l", "id": "concrete-examples/t01",
    "system": "concrete-examples",
    "variants": {
        "domain": [{"subject": {"iri": "my:planet"},
                    "state_predicate": {"iri": "ai:fermState"},
                    "before": {"iri": "my:Fermenting"},
                    "after": {"iri": "my:Bottled"}}],
        "phrasing": ["What state is the planet in now?"],
    },
    "messages": [
        {"role": "assistant", "content":
         "# BEFORE\n\nPREFIX my: <http://example.org/entity/>.\n"
         "PREFIX ai: <http://example.org/property/>.\n\nWORLD {\n"
         "    {domain.subject.iri} {domain.state_predicate.iri} "
         "{domain.before.iri} .\n}\n\n# PROGRAM\n\nDELETE WORLD {\n"
         "    {domain.subject.iri} {domain.state_predicate.iri} "
         "{domain.before.iri} .\n}.\nWRITE WORLD {\n"
         "    {domain.subject.iri} {domain.state_predicate.iri} "
         "{domain.after.iri} .\n}.\n\nRETURN {domain.subject.iri}."},
        {"role": "user", "content": "{phrasing}"},
        {"role": "assistant", "content":
         "WORLD {\n    {domain.subject.iri} {domain.state_predicate.iri} "
         "{domain.after.iri} .\n}\n\n# RETURNED {domain.subject.iri}"},
    ],
    "assert": {"emitted": [], "memory_add": [], "memory_delete": [],
               "returned": "{domain.subject.iri}", "valid": True,
               "world_add": [["{domain.subject.iri}",
                              "{domain.state_predicate.iri}",
                              "{domain.after.iri}"]],
               "world_delete": [["{domain.subject.iri}",
                                 "{domain.state_predicate.iri}",
                                 "{domain.before.iri}"]]},
}


def test_after_return_leak_is_named_assertion(synth):
    """The r1 'Returned' bucket: RETURN executed inside the AFTER
    program escaped check_record as a control-flow EXCEPTION CLASS.
    Now it converts to a named AssertionError; gold still passes; the
    raw block() keeps raising Returned internally (contract unchanged
    there — only the grade boundary is sealed)."""
    msgs, expected = next(iter(synth.check.expansions(CONCRETE_REC)))
    gold = msgs[2]["content"]
    ok, _ = synth.grade_completion(CONCRETE_REC, msgs, expected, gold)
    assert ok, "gold must pass"
    bad = gold + "\nRETURN my:ghost ."
    ok, why = synth.grade_completion(CONCRETE_REC, msgs, expected, bad)
    assert not ok
    assert why.split(":")[0] == "AssertionError", why
    assert "AFTER program" in why


def test_bundle_reads_both_roots_and_verifies_gold(synth, tmp_path, capsys,
                                                   monkeypatch):
    """r3 bundle law: scenarios' craft sources are bundled (never were),
    ids dedupe across roots, and every gold passes check_record before
    it ships — a promoted-but-stale scenario row is counted, not shipped."""
    import copy
    from types import SimpleNamespace
    scen, der, data = tmp_path/"scen", tmp_path/"der", tmp_path/"data"
    scen.mkdir(); der.mkdir()
    (scen / "good.json").write_text(json.dumps(CONCRETE_REC))
    bad = copy.deepcopy(CONCRETE_REC)
    bad["id"] = "concrete-examples/bad01"
    bad["messages"][2]["content"] = bad["messages"][2]["content"].replace(
        "WORLD {", "WORLD \u2014 {", 1)        # parser hazard in gold
    (scen / "bad.json").write_text(json.dumps(bad))
    twin = copy.deepcopy(CONCRETE_REC)          # same id -> must dedupe
    (der / "twin.json").write_text(json.dumps(twin))
    monkeypatch.setattr(synth, "SCEN_ROOT", scen)
    monkeypatch.setattr(synth, "DERIVED_ROOT", der)
    monkeypatch.setattr(synth, "DATA_ROOT", data)
    monkeypatch.setattr(synth, "TMP", tmp_path / "notmp")
    synth.cmd_bundle(SimpleNamespace(domain="probe", out=None, dpo=None))
    out = json.loads(capsys.readouterr().out)
    rows = (data / "datasets" / "probe" / "concrete-examples.jsonl")\
        .read_text().splitlines()
    assert len(rows) == 1, out                  # good once, dedupe held
    assert out["gold_rejects"] == {"concrete-examples:IRError": 1}, out
    assert not (data / "datasets" / "probe" / "bad01.jsonl").exists()
