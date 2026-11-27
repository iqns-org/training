"""GBNF compiler (REGIMES R3): grammar refs + generator<->parser agreement.

The property that matters: every generated sample parses under
src/ir-compile.py (zero E0), for both roots, across seeds — the GBNF and
the mirror generator share one truth.
"""
import importlib.util
import json
import pytest
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gbnf():
    return _load("ir_gbnf", ROOT / "src" / "ir-gbnf.py")


@pytest.fixture(scope="module")
def ir():
    return _load("gbnf_ir", ROOT / "src" / "ir-compile.py")


def test_grammar_rule_refs_resolve(gbnf):
    for root in ("response", "document"):
        assert gbnf.check_refs(gbnf.build(root)) == []


def test_engine_portability_laws(gbnf):
    """Backend-verified 2026-11-27 (r3 decode probe): vLLM v0.34 400s
    structural_tag grammars containing llama-style ~[...] negation and
    needs a root rule named 'root'. These two lines are the difference
    between a grammar the engine ENFORCES and one it rejects."""
    for root in ("response", "document"):
        text = gbnf.build(root)
        assert "~[" not in text, f"{root}: llama-GBNF negation leaked in"
        assert text.splitlines()[0] == f"root ::= {root}"
        assert gbnf.lint(text) == []


def test_lint_catches_dangling_quote(gbnf):
    """The shipped-bug fingerprint: a terminal that consumed its closing
    quote as escaped content left a bare backslash pair — the old rule 4
    literally contained  `"` + \\\\ + `""` . lint must flag that line, and
    leave clean grammars alone."""
    q, b = chr(34), chr(92)
    broken = f"string ::= {q}{b}{b}{q}{q} [x]* {q}{b}{b}{q}{q}"
    assert gbnf.lint(broken) == [1]
    assert gbnf.lint('x ::= "a"\ny ::= "b"\n') == []


def test_response_samples_parse(gbnf, ir):
    for seed in range(40):
        for s in gbnf.generate("response", 2, seed):
            ir.parse(s)                       # raises IRError on syntax fail


def test_document_samples_parse(gbnf, ir):
    for seed in range(25):
        for s in gbnf.generate("document", 1, seed):
            ir.parse(s)


def test_all_statement_families_covered(gbnf):
    seen = set()
    for seed in range(200):
        doc = gbnf.generate("document", 1, seed)[0]
        for kw in ("MATCH", "READ", "WRITE", "DELETE", "BIND", "DERIVE",
                   "CALL", "EMIT", "SPAWN", "WAIT", "BRANCH", "PARALLEL",
                   "EACH", "TRY", "RETURN"):
            if kw in doc:
                seen.add(kw)
    assert len(seen) == 15


def test_cli_sample_validate(gbnf, capsys):
    assert gbnf.main(["--sample", "8", "--seed", "3", "--validate"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["parse_ok"] == 8 and report["parse_fail"] == []


def test_ops_match_lexer_reality(gbnf):
    """The lexer cannot produce '>' or merge '<=' (IRI-branch shadowing);
    the generator must stick to the ops that actually parse."""
    assert gbnf.OPS == ["=", "!=", "<"]


def test_gbnf_is_one_rule_per_line(gbnf):
    text = gbnf.build("document")
    for line in text.splitlines():
        if not line.strip():
            continue
        assert "::=" in line, line
        # continuation lines would break llama.cpp GBNF; none allowed
        assert not line.startswith(" ") or "::=" in line
