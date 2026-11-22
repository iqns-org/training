"""Unit tests for the Graph IR parser (src/ir-compile.py, EBNF.md).

Run:  python3 -m pytest src/tests/compile-ir.py -q
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

spec = importlib.util.spec_from_file_location("compile_ir", ROOT / "src" / "ir-compile.py")
ir = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ir)

EXAMPLES = ROOT / "examples" / "ir"


def parse(src):
    return ir.parse(src)


def read_example(name):
    return (EXAMPLES / name).read_text(encoding="utf-8")


# ------------------------------------------------------------------ examples
def test_examples_parse():
    kinds = {
        "query.txt": ["Context", "Context", "Match", "Return"],
        "mutation.txt": ["Write", "Emit", "Return"],
        "stateful-reasoning.txt": ["Match", "Branch"],
        "planning.txt": ["Context", "Context", "Match", "Call", "Write", "Emit", "Return"],
    }
    for name, expected in kinds.items():
        doc = parse(read_example(name))
        assert doc["type"] == "Document", name
        assert [n["type"] for n in doc["body"]] == expected, name


def test_full_program_covers_every_declaration():
    doc = parse(read_example("full-program.txt"))
    top = [n["type"] for n in doc["body"]]
    assert top == [
        "PrefixDecl", "PrefixDecl", "PrefixDecl",
        "Context", "Context", "Context", "Context",
        "TypeDecl", "EventDecl", "CapabilityDecl", "ProjectionDecl",
        "Wait", "PlanDecl",
    ]
    assert [c["kind"] for c in doc["body"][3:7]] == ["WORLD", "MEMORY", "HISTORY", "GOAL"]
    plan = doc["body"][-1]
    assert plan["name"] == "startEngine"
    assert [n["type"] for n in plan["body"]] == [
        "Match", "Bind", "Derive", "Branch", "Each",
        "Parallel", "Try", "Read", "Return",
    ]


# -------------------------------------------------------------- declarations
def test_prefix_decl_absolute_iri():
    (decl,) = parse('PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>.')["body"]
    assert decl == {
        "type": "PrefixDecl",
        "prefix": "rdf",
        "iri": {"type": "Iri", "iri": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"},
    }


def test_type_decl_with_extends_prefixed_iri():
    (decl,) = parse("TYPE Vehicle EXTENDS rdf:Resource.")["body"]
    assert decl["type"] == "TypeDecl"
    assert decl["name"] == "Vehicle"
    assert decl["extends"] == {"type": "Iri", "prefix": "rdf", "local": "Resource"}


def test_capability_decl_params_and_returns():
    (decl,) = parse("CAPABILITY ex:start(my:?e : Vehicle) RETURNS Vehicle.")["body"]
    assert decl["type"] == "CapabilityDecl"
    (param,) = decl["params"]
    assert param["var"] == {"type": "Variable", "name": "e", "namespace": "my"}
    assert param["type"] == {"type": "Name", "value": "Vehicle"}
    assert param["cardinality"] is None
    assert decl["returns"] == {"type": "Name", "value": "Vehicle"}


def test_event_decl_with_schema_graph():
    (decl,) = parse("EVENT ex:E { _:e rdf:type ex:E . }.")["body"]
    assert decl["type"] == "EventDecl"
    assert decl["iri"] == {"type": "Iri", "prefix": "ex", "local": "E"}
    (triple,) = decl["schema"]
    assert triple["subject"] == {"type": "BlankNode", "name": "e"}
    assert triple["predicate"] == {"type": "Iri", "prefix": "rdf", "local": "type"}


def test_plan_decl_params_with_cardinality():
    doc = parse("PLAN p(my:?x : ex:T MANY) { RETURN my:?x. }")
    (decl,) = doc["body"]
    assert decl["type"] == "PlanDecl"
    (param,) = decl["params"]
    assert param["cardinality"] == "MANY"


# ---------------------------------------------------------------- statements
def test_match_with_scope_and_as():
    (stmt,) = parse("MATCH WORLD { alice owns my:?car . } AS my:?r.")["body"]
    assert stmt["type"] == "Match"
    assert stmt["scope"] == "WORLD"
    assert stmt["as"] == {"type": "Variable", "name": "r", "namespace": "my"}
    (triple,) = stmt["pattern"]
    assert triple["object"] == {"type": "Variable", "name": "car", "namespace": "my"}


def test_read_cardinality_before_as():
    (stmt,) = parse("READ WORLD my:?a ex:p my:?b . MANY AS my:?x.")["body"]
    assert stmt["type"] == "Read"
    assert stmt["cardinality"] == "MANY"
    assert stmt["as"]["name"] == "x"


def test_write_and_delete_consume_trailing_dot():
    doc = parse("WRITE WORLD { a b c . }. DELETE MEMORY { a b c . }.")
    assert [n["type"] for n in doc["body"]] == ["Write", "Delete"]
    assert doc["body"][1]["scope"] == "MEMORY"


def test_bind_function_and_list_expressions():
    (b1, b2) = parse("BIND my:?f = ex:f(1, my:?x). BIND my:?l = [1, my:?x].")["body"]
    assert b1["expression"]["type"] == "FunctionCall"
    assert b1["expression"]["ref"] == {"type": "Iri", "prefix": "ex", "local": "f"}
    assert b2["expression"] == {
        "type": "List",
        "items": [
            {"type": "Literal", "value": 1, "datatype": "xsd:integer"},
            {"type": "Variable", "name": "x", "namespace": "my"},
        ],
    }


def test_derive_using_named_projection():
    (stmt,) = parse("DERIVE my:?d = { my:?a ex:p my:?b . } USING fleet.")["body"]
    assert stmt["type"] == "Derive"
    assert stmt["projection"] == {"type": "Name", "value": "fleet"}


def test_emit_bare_and_constructor():
    e1, e2 = parse("EMIT ex:E. EMIT ex:E(my:?x).")["body"]
    assert e1 == {"type": "Emit", "event": {"type": "Iri", "prefix": "ex", "local": "E"}}
    assert e2["args"][0] == {"type": "Variable", "name": "x", "namespace": "my"}


def test_spawn_with_block_and_as():
    (stmt,) = parse("SPAWN ex:A WITH { monitor watches car1 . } AS my:?m.")["body"]
    assert stmt["type"] == "Spawn"
    assert stmt["with"][0]["subject"] == {"type": "Iri", "prefix": "monitor", "local": None, "bare": True}
    assert stmt["as"]["name"] == "m"


def test_wait_event_match_with_timeout():
    (stmt,) = parse("WAIT EVENT ex:Tick { my:?t ex:n 5 . } TIMEOUT 500ms.")["body"]
    assert stmt["type"] == "Wait"
    assert stmt["target"]["type"] == "EventMatch"
    assert stmt["timeout"] == {"type": "Duration", "value": 500, "unit": "ms"}


def test_branch_with_else():
    (stmt,) = parse(
        "BRANCH { WHEN my:?s = ex:Off THEN { RETURN my:?s. } ELSE { RETURN my:?s. } }"
    )["body"]
    assert stmt["type"] == "Branch"
    assert stmt["when"][0]["condition"]["op"] == "="
    assert stmt["else_"] is not None


def test_each_match_do():
    (stmt,) = parse(
        "EACH my:?w IN MATCH WORLD { my:?c ex:has my:?w . } DO { RETURN my:?w. }"
    )["body"]
    assert stmt["type"] == "Each"
    assert stmt["match"] == {
        "type": "MatchExpr",
        "scope": "WORLD",
        "pattern": stmt["match"]["pattern"],
    }
    assert stmt["var"]["name"] == "w"


def test_parallel_blocks():
    (stmt,) = parse("PARALLEL { { RETURN 1. } { RETURN 2. } }")["body"]
    assert stmt["type"] == "Parallel"
    assert len(stmt["blocks"]) == 2


def test_try_catch():
    (stmt,) = parse("TRY { RETURN 1. } CATCH my:?e { RETURN my:?e. }")["body"]
    assert stmt["type"] == "Try"
    assert stmt["catch_var"] == {"type": "Variable", "name": "e", "namespace": "my"}


def test_return_graph_block():
    (stmt,) = parse("RETURN { my:?a ex:p my:?b . } .")["body"]
    assert stmt["type"] == "Return"
    assert stmt["value"][0]["type"] == "Triple"


# --------------------------------------------------------------- conditions
def test_condition_precedence_not_and_or():
    (stmt,) = parse(
        "BRANCH { WHEN NOT my:?a = 1 AND my:?b = 2 OR my:?c = 3 THEN { RETURN 1. } }"
    )["body"]
    cond = stmt["when"][0]["condition"]
    assert cond["type"] == "Or"
    assert cond["left"]["type"] == "And"
    assert cond["left"]["left"]["type"] == "Not"


def test_comparison_operators_and_not_in():
    for src, op in [
        ("BRANCH { WHEN my:?a < 1 THEN { RETURN 1. } }", "<"),
        ("BRANCH { WHEN my:?a >= 1 THEN { RETURN 1. } }", ">="),
        ("BRANCH { WHEN my:?a IN [1] THEN { RETURN 1. } }", "IN"),
        ("BRANCH { WHEN my:?a NOT IN [1] THEN { RETURN 1. } }", "NOT IN"),
    ]:
        (stmt,) = parse(src)["body"]
        assert stmt["when"][0]["condition"]["op"] == op, src


# ------------------------------------------------------------------ terms
def test_variables_default_and_custom_namespace():
    (stmt,) = parse("BIND my:?a = foo:?b.")["body"]
    assert stmt["var"] == {"type": "Variable", "name": "a", "namespace": "my"}
    assert stmt["expression"] == {"type": "Variable", "name": "b", "namespace": "foo"}


def test_keyword_usable_after_colon():
    doc = parse("WORLD { my:?car rdf:type Car . }")
    (triple,) = doc["body"][0]["body"]
    assert triple["predicate"] == {"type": "Iri", "prefix": "rdf", "local": "type"}


def test_literals():
    doc = parse(
        'BIND my:?a = 42. BIND my:?b = -1.5. BIND my:?c = "s\\"q". '
        'BIND my:?d = "2024-01-15"^^xsd:date. BIND my:?e = false.'
    )
    exprs = [n["expression"] for n in doc["body"]]
    assert exprs[0] == {"type": "Literal", "value": 42, "datatype": "xsd:integer"}
    assert exprs[1] == {"type": "Literal", "value": -1.5, "datatype": "xsd:decimal"}
    assert exprs[2] == {"type": "Literal", "value": 's"q'}
    assert exprs[3]["datatype"] == {"type": "Iri", "prefix": "xsd", "local": "date"}
    assert exprs[4] == {"type": "Literal", "value": False, "datatype": "xsd:boolean"}


def test_ast_is_json_serializable():
    doc = parse(read_example("full-program.txt"))
    assert json.loads(json.dumps(doc)) == doc


def test_comments_are_ignored():
    doc = parse("# hello\nMATCH WORLD { a b c . }.")
    assert doc["body"][0]["type"] == "Match"


# ------------------------------------------------------------------- errors
def test_errors_raise_ir_error():
    bad = [
        "MATCH WORLD { a b c .",          # unclosed block
        'BIND my:?a = "oops.',            # unterminated string
        "MATCH WORLD { my:?a ex:p }.",    # missing object term
        "WAIT my:?x TIMEOUT 5x.",         # unknown duration unit
        "TYPE T EXTENDS",                 # truncated
    ]
    for src in bad:
        try:
            parse(src)
        except ir.IRError as exc:
            assert exc.line >= 1
        else:
            raise AssertionError(f"expected IRError for {src!r}")


# ---------------------------------------------------------------------- cli
def test_cli_file_emits_single_jsonl_record(tmp_path, capsys):
    target = tmp_path / "q.txt"
    target.write_text("MATCH WORLD { a b c . }.", encoding="utf-8")
    assert ir.main([ir.__name__ or "ir", str(target)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["ok"] is True
    assert record["tool"] == "ir-compile"
    assert record["path"] == str(target)
    assert record["ast"]["body"][0]["type"] == "Match"


def test_cli_parse_error_is_jsonl(tmp_path, capsys):
    target = tmp_path / "bad.txt"
    target.write_text("MATCH WORLD { a b c .", encoding="utf-8")
    assert ir.main(["ir", str(target)]) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    record = json.loads(captured.out.strip())
    assert record["ok"] is False
    (err,) = record["errors"]
    assert err["code"] == "E0_PARSE_ERROR"
    assert err["line"] >= 1


def test_cli_missing_file_returns_1(capsys):
    assert ir.main(["ir", "no-such-file.txt"]) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    record = json.loads(captured.out.strip())
    assert record["ok"] is False
    assert "no such file" in record["errors"][0]["message"]
