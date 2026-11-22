"""Unit tests for PASS 1 RDF-blind AST validation (src/ir-ast-sanity.py).

Run:  python3 -m pytest src/tests/ir-ast-sanity.py -q

Contract under test: the validator answers "is this a structurally coherent
Graph IR program?" with NO RDF / ontology / runtime access.  The CLI emits
exactly one JSONL record on stdout and nothing on stderr.
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sanity = _load("ir_ast_sanity", "src/ir-ast-sanity.py")


def report(src, strict=False):
    rep = sanity.validate_source(src, strict=strict)
    assert "parse_error" not in rep, f"unexpected parse error: {rep['parse_error']}"
    return rep


def codes(rep):
    return {(e["check"], e["code"]) for e in rep["errors"]}


def valid(src, strict=False):
    return report(src, strict=strict)["valid"]


# ------------------------------------------------------------------ fixtures
EXAMPLES = ROOT / "examples" / "ir"


def test_all_canonical_examples_valid():
    for name in ["query.txt", "mutation.txt", "planning.txt",
                 "stateful-reasoning.txt", "full-program.txt"]:
        rep = sanity.validate_source((EXAMPLES / name).read_text(encoding="utf-8"))
        assert "parse_error" not in rep, name
        assert rep["valid"], f"{name}: {rep['errors']}"


# ------------------------------------------------- RDF-blindness guarantees
def test_unknown_class_predicate_individual_capability_are_valid():
    # None of ex:Car / alice / ex:startEngine need to exist anywhere.
    assert valid(
        "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>.\n"
        "PREFIX ex: <http://example.org/>.\n"
        "MATCH WORLD { my:?car rdf:type ex:Car . }.\n"
        "alice owns car1 .\n"
        "CALL ex:startEngine(my:?car).\n"
    )


def test_user_example_valid_and_binding_flows():
    assert valid(
        "MATCH WORLD {\n"
        "    my:?person owns my:?car .\n"
        "} AS my:?match.\n"
        "CALL ex:startEngine(my:?car).\n"
    )


def test_many_into_one_rejected():
    rep = report(
        "MATCH WORLD { my:?p owns my:?c . } AS my:?all.\n"
        "CALL ex:startEngine(my:?all).\n"
    )
    assert not rep["valid"]
    assert (5, "E5_CARD_MISMATCH") in codes(rep)


# ------------------------------------------------------------- check 1: names
def test_duplicate_prefix_rejected():
    rep = report("PREFIX ex: <http://a/>.\nPREFIX ex: <http://b/>.\n")
    assert (1, "E1_DUP_PREFIX") in codes(rep)


def test_undeclared_prefix_is_warning_by_default_error_when_strict():
    rep = report("MATCH WORLD { my:?a rdf:type Car . }.")
    assert rep["valid"]
    assert any(e["code"] == "E1_UNDECLARED_PREFIX" for e in rep["warnings"])
    rep_strict = report("MATCH WORLD { my:?a rdf:type Car . }.", strict=True)
    assert not rep_strict["valid"]


# ------------------------------------------------------------- check 2: scope
def test_unbound_variable_rejected():
    rep = report("CALL ex:f(my:?ghost).\n")
    assert (2, "E2_UNBOUND") in codes(rep)


def test_branch_arm_binding_does_not_leak():
    rep = report(
        "MATCH WORLD { alice owns my:?car . }.\n"
        "BRANCH { WHEN my:?car = car1 THEN { CALL ex:f(my:?car) AS my:?r. } }\n"
        "RETURN my:?r.\n"
    )
    assert not rep["valid"]
    assert (2, "E2_UNBOUND") in codes(rep)


# ---------------------------------------------------------- check 3: dataflow
def test_double_bind_rejected():
    rep = report("BIND my:?x = 1.\nBIND my:?x = 2.\n")
    assert (3, "E3_REDEFINE") in codes(rep)


def test_pattern_join_with_bound_var_allowed():
    assert valid(
        "MATCH WORLD { alice owns my:?car . }.\n"
        "MATCH WORLD { my:?car rdf:type Car . }.\n"
        "RETURN my:?car.\n"
    )


# -------------------------------------------------------------- check 4: types
def test_literal_type_mismatch_rejected():
    rep = report(
        "BIND my:?a = 1.\n"
        "BRANCH { WHEN my:?a = \"s\" THEN { RETURN 1. } }\n"
    )
    assert (4, "E4_TYPE_MISMATCH") in codes(rep)


def test_type_extends_cycle_rejected():
    rep = report("TYPE A EXTENDS B.\nTYPE B EXTENDS A.\n")
    assert (4, "E4_TYPE_CYCLE") in codes(rep)


# ------------------------------------------------------- check 5: cardinality
def test_each_over_match_valid():
    assert valid(
        "EACH my:?w IN MATCH WORLD { my:?c ex:has my:?w . } DO { RETURN my:?w. }\n"
    )


def test_emit_many_rejected():
    rep = report(
        "MATCH WORLD { my:?a ex:p my:?b . } AS my:?all.\n"
        "EMIT ex:E(my:?all).\n"
    )
    assert (5, "E5_CARD_MISMATCH") in codes(rep)


# ------------------------------------------------------ check 6: control flow
def test_unreachable_after_return_rejected():
    rep = report("RETURN 1.\nRETURN 2.\n")
    assert (6, "E6_UNREACHABLE") in codes(rep)


def test_parallel_needs_two_blocks():
    rep = report("PARALLEL { { RETURN 1. } }\n")
    assert (6, "E6_PARALLEL_ARITY") in codes(rep)


# ------------------------------------------------------------- check 7: effects
def test_write_history_rejected():
    rep = report("WRITE HISTORY { car1 engineState Running . }.\n")
    assert (7, "E7_IMMUTABLE_SCOPE") in codes(rep)


def test_effect_in_projection_rejected():
    rep = report(
        "PROJECTION p(my:?x : T) {\n"
        "    WRITE WORLD { my:?x owns car1 . }.\n"
        "    RETURN my:?x.\n"
        "}\n"
    )
    assert (7, "E7_EFFECT_IN_PROJECTION") in codes(rep)


# ------------------------------------------------------------------- cli/json
def test_json_report_shape():
    rep = report("CALL ex:f(my:?ghost).\n")
    assert set(rep) >= {"valid", "errors", "warnings"}
    assert rep["errors"][0]["hint"]  # every error carries a repair hint


def test_parse_errors_reported_not_raised():
    rep = sanity.validate_source("not valid ir (((")
    assert not rep["valid"]
    assert rep["errors"][0]["code"] == "E0_PARSE_ERROR"


def _run_main(src, *argv):
    """Run main() with stdin source; return (rc, record, err)."""
    import io
    import sys
    old_stdin = sys.stdin
    sys.stdin = io.StringIO(src)
    try:
        import contextlib
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            rc = sanity.main(["sanity", "--stdin", *argv])
    finally:
        sys.stdin = old_stdin
    return rc, buf_out.getvalue(), buf_err.getvalue()


def test_cli_valid_is_single_jsonl_record():
    rc, out, err = _run_main("MATCH WORLD { alice owns my:?car . }.\n")
    assert rc == 0
    assert err == ""
    assert out.count("\n") == 1
    record = json.loads(out)
    assert record == {
        "ok": True, "tool": "ir-ast-sanity", "path": "<stdin>",
        "valid": True, "errors": record["errors"],
        "warnings": record["warnings"], "ast": record["ast"],
    }


def test_cli_invalid_is_jsonl_with_hints():
    rc, out, err = _run_main("CALL ex:f(my:?ghost).\n")
    assert rc == 1
    assert err == ""
    record = json.loads(out.strip())
    assert record["ok"] is False
    assert record["valid"] is False
    assert record["errors"][0]["code"] == "E2_UNBOUND"
    assert record["errors"][0]["hint"]


def test_cli_parse_error_is_jsonl_rc2():
    rc, out, err = _run_main("not valid ir (((")
    assert rc == 2
    assert err == ""
    record = json.loads(out.strip())
    assert record["ok"] is False
    assert record["errors"][0]["code"] == "E0_PARSE_ERROR"


def test_cli_missing_file_is_jsonl_rc2(capsys):
    assert sanity.main(["sanity", "no-such-file.txt"]) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    record = json.loads(captured.out.strip())
    assert record["ok"] is False
    assert record["errors"][0]["code"] == "E_IO"
