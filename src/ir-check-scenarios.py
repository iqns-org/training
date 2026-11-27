#!/usr/bin/env python3
"""Preflight curriculum templates using stdlib and the local IR tools.

Checks every correlated expansion, all three messages, source verdicts,
declarations/signatures, and semantic assertions for a deliberately small
closed-world fixture subset. This is not the Graph IR runtime. Unsupported
operations with execution assertions fail rather than being guessed.
Run: python3 src/ir-check-scenarios.py
"""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*)\}")
CONTEXTS = {"WORLD", "MEMORY", "WORKING", "HISTORY", "GOAL"}


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ir = load("scenario_ir", "ir-compile.py")
sanity = load("scenario_sanity", "ir-ast-sanity.py")


def expand(value, choices, stack=()):
    if isinstance(value, str):
        def resolve(match):
            key = match[1]
            if key in stack:
                raise ValueError(f"cyclic placeholder {key}")
            result = choices
            for part in key.split("."):
                result = result[part]
            # If result is a dict with iri, return the iri string for string interpolation
            if isinstance(result, dict) and "iri" in result and len(result) == 1:
                return result["iri"]
            if isinstance(result, dict) and "iri" in result:
                # For dicts like {"iri": "my:X", "text": "x"}, return iri for string context
                # But for entire placeholder, return the dict as is for assert handling
                # We need to distinguish: if entire placeholder, return dict; else return iri string
                pass
            return expand(result, choices, (*stack, key))
        entire = PLACEHOLDER.fullmatch(value)
        if entire:
            res = resolve(entire)
            # For entire placeholder, if it's a dict with iri, return iri string (for assert triples)
            if isinstance(res, dict) and "iri" in res:
                return res["iri"]
            return res
        def repl(m):
            res = resolve(m)
            if isinstance(res, dict) and "iri" in res:
                return res["iri"]
            return str(res)
        return PLACEHOLDER.sub(repl, value)
    if isinstance(value, list):
        return [expand(v, choices, stack) for v in value]
    if isinstance(value, dict):
        return {k: expand(v, choices, stack) for k, v in value.items()}
    return value


def expansions(record):
    pools = record["variants"]
    for key, pool in pools.items():
        assert isinstance(pool, list) and pool, f"empty variant pool {key}"
    for selected in itertools.product(*pools.values()):
        choices = dict(zip(pools, selected))
        yield expand(record["messages"], choices), expand(record["assert"], choices)


def walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v)


def name(node):
    if isinstance(node, str):
        return node
    if node.get("type") == "Variable":
        return f"{node.get('namespace', 'my')}:?{node['name']}"
    if node.get("type") == "Name":
        return node["value"]
    return node.get("iri") or f"{node['prefix']}:{node['local']}"


def context_nodes(doc):
    return [n for n in doc["body"] if n["type"] == "Context" or n["type"].endswith("Decl")]


def document(nodes):
    return {"type": "Document", "body": nodes}


def static_check(doc, expected):
    result = sanity.validate_ast(doc, strict=True)
    errors = result["errors"]
    assert result["valid"] == expected.get("valid", True), errors
    if not expected.get("valid", True):
        assert len(errors) == 1, errors
        assert [e["code"] for e in errors] == [e["code"] for e in expected["errors"]], errors
        for entry in expected["errors"]:
            if "variable" in entry:
                assert any(n.get("type") == "Variable" and name(n) == entry["variable"] for n in walk(doc))
    assert not result["warnings"], result["warnings"]
    # Curriculum safeguards are stricter than the RDF-blind checker: a
    # supplied capability must actually have the signature used by CALL.
    caps = {name(n["iri"]): n for n in doc["body"] if n["type"] == "CapabilityDecl"}
    events = {name(n["iri"]) for n in doc["body"] if n["type"] == "EventDecl"}
    types = {name(n["name"]) for n in doc["body"] if n["type"] == "TypeDecl"}
    for n in walk(doc):
        kind = n.get("type")
        if kind == "Iri":
            assert not n.get("bare"), f"bare IRI {n}"
        if kind == "Call":
            key = name(n["capability"])
            assert key in caps, f"undeclared capability {key}"
            assert len(n["args"]) == len(caps[key]["params"]), f"arity mismatch: {key}"
        if kind in ("Emit", "EventMatch"):
            assert name(n["event"]) in events, f"undeclared event {name(n['event'])}"
        if "params" in n:
            for p in n["params"]:
                ref = name(p["type"])
                assert ref in types or ref.startswith("xsd:"), f"undeclared parameter type {ref}"
    return result


class Returned(Exception):
    def __init__(self, value):
        self.value = value


class Fixture:
    """Exact graph sets, no inference; single-row MATCH or explicit EACH."""
    def __init__(self):
        self.state = {k: set() for k in CONTEXTS}
        self.emitted = []
        self.returned = None

    def value(self, n, env):
        kind = n["type"]
        if kind == "Variable":
            return env[name(n)]
        if kind in ("Iri", "Name"):
            return name(n)
        if kind == "Literal":
            return n["value"]
        if kind == "List":
            return [self.value(x, env) for x in n["items"]]
        raise ValueError(f"unsupported fixture expression {kind}")

    def triple(self, n, env):
        return tuple(self.value(n[k], env) for k in ("subject", "predicate", "object"))

    def matches(self, pattern, scope, env):
        rows = [dict(env)]
        for triple in pattern:
            following = []
            for row in rows:
                for fact in sorted(self.state[scope], key=repr):
                    candidate = dict(row)
                    for key, value in zip(("subject", "predicate", "object"), fact):
                        term = triple[key]
                        if term["type"] == "Variable" and name(term) not in candidate:
                            candidate[name(term)] = value
                        elif self.value(term, candidate) != value:
                            break
                    else:
                        following.append(candidate)
            rows = following
        return rows

    def condition(self, n, env):
        kind = n["type"]
        if kind == "Not":
            return not self.condition(n["operand"], env)
        if kind == "And":
            return self.condition(n["left"], env) and self.condition(n["right"], env)
        if kind == "Or":
            return self.condition(n["left"], env) or self.condition(n["right"], env)
        a, b = self.value(n["left"], env), self.value(n["right"], env)
        op = n["op"]
        if op == "=": return a == b
        if op == "!=": return a != b
        if op == "<": return a < b
        if op == "<=": return a <= b
        if op == ">": return a > b
        if op == ">=": return a >= b
        if op == "IN": return a in b
        if op == "NOT IN": return a not in b
        raise ValueError(f"unsupported comparison {op}")

    def block(self, nodes, env):
        for n in nodes:
            kind = n["type"]
            if kind.endswith("Decl"):
                continue
            if kind == "Context":
                self.state[n["kind"].upper()].update(self.triple(t, env) for t in n["body"])
            elif kind in ("Write", "Delete"):
                graph = self.state[n["scope"] or "WORLD"]
                triples = {self.triple(t, env) for t in n["block"]}
                if kind == "Write": graph.update(triples)
                else: graph.difference_update(triples)
            elif kind == "Bind":
                env[name(n["var"])] = self.value(n["expression"], env)
            elif kind in ("Match", "Read"):
                pattern = n["pattern"] if kind == "Match" else [n["triple"]]
                rows = self.matches(pattern, n["scope"] or "WORLD", env)
                # Multiple rows must be consumed explicitly with EACH; do
                # not silently flatten a result set into a scalar answer.
                assert len(rows) == 1, f"fixture needs explicit cardinality handling: {len(rows)} rows"
                env.update(rows[0])
                if n.get("as"):
                    raise ValueError("fixture evaluator requires an explicit result-set contract for AS")
            elif kind == "Each":
                match = n["match"]
                rows = self.matches(match["pattern"], match["scope"] or "WORLD", env)
                for row in rows:
                    assert name(n["var"]) in row
                    self.block(n["body"], row)
            elif kind == "Branch":
                for clause in n["when"]:
                    if self.condition(clause["condition"], env):
                        self.block(clause["body"], dict(env))
                        break
                else:
                    self.block(n.get("else_") or [], dict(env))
            elif kind == "Emit":
                self.emitted.append({"event": name(n["event"]), "args": [self.value(a, env) for a in n.get("args", [])]})
            elif kind == "Return":
                raise Returned(self.value(n["value"], env))
            else:
                raise ValueError(f"unsupported execution assertion for {kind}; supply a deterministic runtime fixture")

    def run(self, doc):
        setup = context_nodes(doc)
        self.block(setup, {})
        before = copy.deepcopy(self.state)
        try:
            self.block([n for n in doc["body"] if n not in setup], {})
        except Returned as result:
            self.returned = result.value
        delta = {"returned": self.returned, "emitted": self.emitted}
        for scope in ("WORLD", "MEMORY", "WORKING", "HISTORY"):
            delta[scope.lower() + "_add"] = [list(t) for t in self.state[scope] - before[scope]]
            delta[scope.lower() + "_delete"] = [list(t) for t in before[scope] - self.state[scope]]
        return delta


def check_record(record, messages, expected):
    assert [m["role"] for m in messages] == ["assistant", "user", "assistant"]
    assert all(m["content"].strip() for m in messages)
    first, user, last = [m["content"] for m in messages]
    assert not re.search(r"\b(?:PREFIX|WORLD|MEMORY|BIND|WRITE|RETURN)\b.*[{:;]", user), "user must be a natural-language request"
    task = record["system"]
    if task == "ir-to-nl":
        initial = ir.parse(first)
        static_check(initial, expected)
        assert last.strip() and not re.search(r"PREFIX|WORLD|MATCH|WRITE|RETURN", last), "ir-to-nl final must be prose"
        return False
    if task == "equivalence":
        assert re.search(r"# VERDICT:\s*(EQUIVALENT|DIFFERENT)", last), "equivalence verdict missing"
        # First contains scenario + Program A + Program B; validate each program separately under shared scenario
        parts = re.split(r"# PROGRAM [AB]", first)
        scenario_text = parts[0] if parts else first
        scenario_doc = ir.parse(scenario_text)
        # scenario alone should be valid (no errors)
        static_check(scenario_doc, {"valid": True})
        for prog_text in parts[1:]:
            if prog_text.strip():
                prog_doc = ir.parse(scenario_text + "\n" + prog_text)
                static_check(prog_doc, {"valid": True})
        return False
    if task in ("program-completion", "program-transform", "semantic-edit"):
        initial = ir.parse(first)
        final = ir.parse(last)
        if task == "program-completion":
            assert first.strip() in last, "completion must preserve supplied source"
        if task in ("program-transform", "semantic-edit"):
            static_check(initial, {"valid": True})
        static_check(final, expected)
        if not any(k in expected for k in ("world_add", "memory_add", "returned", "emitted")):
            return False
        # execution check on final if requested
        fixture = Fixture()
        actual = fixture.run(final)
        for key, value in actual.items():
            if key not in expected:
                assert not value, f"missing assertion {key}: {value}"
            elif key.endswith(("_add", "_delete")):
                assert sorted(value, key=repr) == sorted(expected[key], key=repr), (key, value, expected[key])
            else:
                assert type(value) is type(expected[key]) and value == expected[key], (key, value, expected[key])
        return True
    initial = ir.parse(first)
    if task in ("invalid-program", "concrete-examples"):
        doc = initial
    elif task == "repair":
        broken = sanity.validate_ast(initial, strict=True)
        assert len(broken["errors"]) == 1 and broken["errors"][0]["code"] != "E0_PARSE_ERROR", broken
        if "input_errors" in expected:
            assert [e["code"] for e in broken["errors"]] == expected["input_errors"]
        static_check(initial, {"valid": False, "errors": broken["errors"]})
        doc = document(context_nodes(initial) + ir.parse(last)["body"])
    else:
        assert all(n in context_nodes(initial) for n in initial["body"]), "first message leaks the solution"
        solution = ir.parse(last)
        assert any(not n["type"].endswith("Decl") and n["type"] != "Context" for n in solution["body"]), "empty solution"
        doc = document(initial["body"] + solution["body"])
    static_check(doc, expected)
    if task == "invalid-program":
        assert expected["errors"][0]["code"] in last, "diagnosis/assert disagree"
        assert not re.search(r"# (?:ERROR|BREAKS)", first), "diagnosis leaked in input"
        return False
    if not any(k in expected for k in ("world_add", "memory_add", "returned", "emitted")):
        return False
    fixture = Fixture()
    actual = fixture.run(doc)
    for key, value in actual.items():
        if key not in expected:
            assert not value, f"missing assertion {key}: {value}"
        elif key.endswith(("_add", "_delete")):
            assert sorted(value, key=repr) == sorted(expected[key], key=repr), (key, value, expected[key])
        else:
            assert type(value) is type(expected[key]) and value == expected[key], (key, value, expected[key])
    if task == "concrete-examples":
        after = Fixture()
        try:
            after.block(ir.parse(last)["body"], {})
        except Returned:
            raise AssertionError("RETURN executed inside the AFTER program "
                                 "(the # RETURNED annotation is its channel)")
        assert after.state == fixture.state, "AFTER drops or invents facts"
        returned = re.findall(r"^# RETURNED (.+)$", last, re.M)
        assert len(returned) == 1
        returned_value = fixture.value(ir.parse("RETURN " + returned[0] + ".")["body"][0]["value"], {})
        assert returned_value == fixture.returned, "RETURNED annotation differs from execution"
        emitted = re.findall(r"^# EMITTED (.+)$", last, re.M)
        annotations = Fixture()
        for event in emitted:
            annotations.block(ir.parse("EMIT " + event + ".")["body"], {})
        assert annotations.emitted == fixture.emitted, "EMITTED annotation differs from execution"
    return True


def main():
    failures, count, semantic = [], 0, 0
    files = sorted((ROOT / "generate/scenarios").rglob("*.json"))
    ids = set()
    for path in files:
        try:
            record = json.loads(path.read_text())
            assert set(record) == {"comment", "label", "id", "system", "variants", "messages", "assert"}
            assert record["id"] == str(path.relative_to(ROOT / "generate/scenarios").with_suffix(""))
            assert record["id"] not in ids
            ids.add(record["id"])
            assert record["label"].strip() and record["comment"].strip()
            requests = set()
            for index, (messages, expected) in enumerate(expansions(record)):
                count += 1
                requests.add(messages[1]["content"])
                try:
                    semantic += check_record(record, messages, expected)
                except Exception as exc:
                    failures.append({"id": record["id"], "variant": index, "error": str(exc)})
            assert len(requests) > 1, "request phrasing does not vary"
        except Exception as exc:
            failures.append({"file": str(path), "error": str(exc)})
    print(json.dumps({"ok": not failures, "templates": len(files), "expansions": count, "execution_checks": semantic, "errors": failures}))
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
