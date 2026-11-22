#!/usr/bin/env python3
"""Graph IR — PASS 1: RDF-blind AST sanity validation (ir-ast-sanity).

Pipeline position::

    EBNF Parser  ->  AST  ->  AST Validator (THIS FILE)  ->  Validated AST

This pass answers exactly one question:

    "Is this a structurally coherent Graph IR program?"

It must NEVER ask whether a class, predicate, capability, or individual
exists in the world.  No RDF / ontology / runtime access.  Zero external
dependencies (stdlib only), so it can run in synthetic-training loops
without an RDF database::

    Graph IR -> AST validator -> VALID / INVALID -> error list -> repair

Checks (all syntax-derived, no world knowledge):

  1. Names        — prefixes, declarations, identifiers structurally valid
  2. Scope        — variables visible where referenced
  3. Dataflow     — variables definitely assigned before use (+ single assignment)
  4. Types        — declared/inferred types internally consistent
  5. Cardinality  — ONE / MANY / OPTIONAL usage coherent
  6. Control flow — branches, returns, waits, blocks structurally valid
  7. Effects      — no statically-declared effect rule violations

Explicitly OUT OF SCOPE (later passes):

  * ``my:?car rdf:type ex:Car``      — unknown whether ``ex:Car`` exists
  * ``alice owns car1``               — unknown whether ``alice`` exists
  * ``ex:startEngine(...)``           — unknown whether capability exists
  * predicate / class / actor / projection existence (unless declared
    in the same file, in which case only *internal* consistency applies)

Usage::

    python src/ir-ast-sanity.py program.ir [--strict]
    cat program.ir | python src/ir-ast-sanity.py --stdin

Output contract: exactly one JSONL record on stdout, nothing on stderr.
Valid:   {"ok": true, "tool": "ir-ast-sanity", "path": ..., "valid": true,
           "errors": [], "warnings": [...], "ast": {...}}
Invalid: {"ok": false, "tool": "ir-ast-sanity", "path": ..., "valid": false,
           "errors": [...], "warnings": [...], "ast": {...}}

Exit codes: 0 = VALID, 1 = INVALID, 2 = parse/IO/usage error.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_parser():
    for candidate in ("ir-compile.py", "compile-ir.py"):
        path = ROOT / "src" / candidate
        if path.is_file():
            spec = importlib.util.spec_from_file_location(
                "ir_compile_parser", path
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        "Graph IR parser not found (looked for src/ir-compile.py)"
    )


_PARSER = None


def _parser():
    global _PARSER
    if _PARSER is None:
        _PARSER = _load_parser()
    return _PARSER


IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-]*$")

# Cardinality lattice: OPTIONAL < ONE < MANY  (by "plurality")
CARD_RANK = {"OPTIONAL": 0, "ONE": 1, "MANY": 2}


def _card_compatible(actual, expected):
    """True when a value of `actual` card may flow into `expected` slot.

    ONE satisfies OPTIONAL (a present value is a valid optional); anything
    else must match exactly.  "ANY" accepts everything (RETURN, BIND RHS).
    """
    if expected == "ANY":
        return True
    if actual == expected:
        return True
    return actual == "ONE" and expected == "OPTIONAL"

# Statements with no world effects (safe inside PROJECTION bodies).
PURE_STMTS = {
    "Match", "Read", "Bind", "Derive", "Branch", "Each",
    "Parallel", "Try", "Return", "Triple",
}

# WRITE/DELETE must never target these scopes (HISTORY is append-only).
IMMUTABLE_SCOPES = {"HISTORY"}


def _var_key(var):
    return (var.get("namespace", "my"), var.get("name"))


def _var_label(var):
    ns = var.get("namespace", "my")
    return f"{ns}:?{var.get('name')}"


def _iri_label(node):
    if node is None:
        return "<none>"
    if node.get("type") == "Name":
        return node.get("value", "?")
    if "iri" in node:
        return f"<{node['iri']}>"
    prefix = node.get("prefix", "?")
    local = node.get("local")
    if local is None:
        return f"{prefix}:"
    return f"{prefix}:{local}"


def _type_label(tref):
    if tref is None:
        return "unknown"
    if isinstance(tref, str):
        return tref
    if tref.get("type") == "Name":
        return tref.get("value", "?")
    return _iri_label(tref)


class Issue:
    """A single validation finding.  severity: 'error' or 'warning'."""

    __slots__ = ("check", "code", "message", "hint", "severity", "node")

    def __init__(self, check, code, message, hint="", severity="error", node=None):
        self.check = check
        self.code = code
        self.message = message
        self.hint = hint
        self.severity = severity
        self.node = node

    def to_dict(self):
        d = {
            "check": self.check,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }
        if self.hint:
            d["hint"] = self.hint
        if self.node:
            d["node"] = self.node
        return d


class Binding:
    __slots__ = ("var", "card", "dtype", "origin")

    def __init__(self, var, card="ONE", dtype="unknown", origin=""):
        self.var = dict(var)
        self.card = card
        self.dtype = dtype
        self.origin = origin


class SanityValidator:
    """RDF-blind PASS 1 validator.  No I/O, no imports beyond stdlib."""

    def __init__(self, strict=False):
        self.strict = strict
        self.issues: list[Issue] = []
        # Declaration tables (file-local only; absence proves nothing).
        self.prefixes: dict[str, str] = {}
        self.plans: dict[str, dict] = {}
        self.projections: dict[str, dict] = {}
        self.capabilities: dict[str, dict] = {}
        self.types: dict[str, dict] = {}
        self.events: dict[str, dict] = {}
        # Lexical scope stack: list[dict[key -> Binding]]
        self.scopes: list[dict] = [{}]

    # ------------------------------------------------------------ emit
    def err(self, check, code, message, hint="", node=None):
        self.issues.append(Issue(check, code, message, hint, "error", node))

    def warn(self, check, code, message, hint="", node=None):
        self.issues.append(Issue(check, code, message, hint, "warning", node))

    def undeclared_prefix(self, prefix, node=None):
        msg = f"undeclared prefix '{prefix}:' (no PREFIX {prefix}: ... in file)"
        hint = f"add 'PREFIX {prefix}: <...>' or fix the prefix spelling"
        if self.strict:
            self.err(1, "E1_UNDECLARED_PREFIX", msg, hint, node)
        else:
            self.warn(1, "E1_UNDECLARED_PREFIX", msg, hint, node)

    # ------------------------------------------------------------ entry
    def validate(self, ast):
        self._collect_declarations(ast)
        self._check_type_cycles()
        body = ast.get("body", []) if isinstance(ast, dict) else []
        definite: set = set()
        self._check_statements(body, definite, context="top")
        errors = [i for i in self.issues if i.severity == "error"]
        warnings = [i for i in self.issues if i.severity == "warning"]
        return {
            "valid": not errors,
            "errors": [i.to_dict() for i in errors],
            "warnings": [i.to_dict() for i in warnings],
        }

    # ------------------------------------------------------------ CHECK 1: names
    def _valid_ident(self, name):
        return isinstance(name, str) and bool(IDENT_RE.match(name))

    def _check_ident(self, name, what, node=None):
        if not self._valid_ident(name):
            self.err(
                1, "E1_BAD_IDENTIFIER",
                f"invalid {what} {name!r} (must match [A-Za-z_][A-Za-z0-9_-]*)",
                "rename to a structurally valid identifier",
                node,
            )
            return False
        return True

    def _check_iri_syntax(self, node, what="IRI"):
        """Validate identifier SHAPE only.  Never checks world existence."""
        if node is None:
            return
        if node.get("type") == "Name":
            self._check_ident(node.get("value"), "name", "Name")
            return
        if node.get("type") == "BlankNode":
            self._check_ident(node.get("name"), "blank node name", "BlankNode")
            return
        if "iri" in node:  # absolute <...>: opaque, always structurally fine
            return
        if node.get("bare", False):
            # Bare names (alice, owns, car1 — no colon in source) are a
            # distinct syntactic class: shape only, never a prefix lookup.
            self._check_ident(node.get("prefix"), "name", "Iri")
            return
        prefix = node.get("prefix")
        local = node.get("local")
        bare = node.get("bare", False)
        if prefix is not None:
            self._check_ident(prefix, "prefix", "Iri")
            if prefix not in self.prefixes:
                self.undeclared_prefix(prefix, "Iri")
        if local is not None:
            # Parser accepts keywords after ':' (e.g. rdf:type); those are
            # structurally legal name positions, so only reject empties.
            if not isinstance(local, str) or not local:
                self.err(1, "E1_BAD_IDENTIFIER",
                         f"invalid {what} local name {local!r}",
                         "use a non-empty name after ':'", "Iri")
        elif not bare and prefix is not None:
            self.err(1, "E1_BAD_IDENTIFIER",
                     f"malformed prefixed IRI '{prefix}:'",
                     "write prefix:local or declare the intended form", "Iri")

    def _check_variable_syntax(self, var):
        if not isinstance(var, dict) or var.get("type") != "Variable":
            self.err(1, "E1_BAD_VARIABLE", f"not a variable node: {var!r}",
                     "use the form ns:?name", "Variable")
            return False
        ok = True
        ok &= bool(self._check_ident(var.get("name"), "variable name", "Variable"))
        ns = var.get("namespace", "my")
        if not self._valid_ident(ns):
            self.err(1, "E1_BAD_IDENTIFIER",
                     f"invalid variable namespace {ns!r}",
                     "use 'my' or another valid identifier", "Variable")
            ok = False
        return ok

    def _collect_declarations(self, ast):
        for node in ast.get("body", []):
            t = node.get("type")
            if t == "PrefixDecl":
                prefix = node.get("prefix")
                iri = node.get("iri", {})
                self._check_ident(prefix, "prefix", "PrefixDecl")
                target = iri.get("iri") if isinstance(iri, dict) else str(iri)
                if prefix in self.prefixes and self.prefixes[prefix] != target:
                    self.err(1, "E1_DUP_PREFIX",
                             f"duplicate PREFIX '{prefix}:' with different targets",
                             "keep a single PREFIX declaration per prefix",
                             "PrefixDecl")
                else:
                    self.prefixes[prefix] = target
            elif t == "PlanDecl":
                self._declare_named(node, self.plans, "PLAN", "E1_DUP_PLAN")
            elif t == "ProjectionDecl":
                self._declare_named(node, self.projections, "PROJECTION",
                                    "E1_DUP_PROJECTION")
            elif t == "CapabilityDecl":
                key = _iri_label(node.get("iri"))
                self._check_iri_syntax(node.get("iri"), "capability IRI")
                for p in node.get("params", []):
                    self._check_parameter(p, f"CAPABILITY {key}")
                if key in self.capabilities:
                    self.err(1, "E1_DUP_CAPABILITY",
                             f"duplicate CAPABILITY {key}",
                             "merge into a single CAPABILITY declaration",
                             "CapabilityDecl")
                else:
                    self.capabilities[key] = node
            elif t == "TypeDecl":
                name = node.get("name")
                key = name if isinstance(name, str) else _iri_label(name)
                if isinstance(name, str):
                    self._check_ident(name, "type name", "TypeDecl")
                else:
                    self._check_iri_syntax(name, "type name")
                if node.get("extends") is not None:
                    self._check_type_ref_syntax(node.get("extends"))
                if key in self.types:
                    self.err(1, "E1_DUP_TYPE",
                             f"duplicate TYPE {key}",
                             "keep a single TYPE declaration per name",
                             "TypeDecl")
                else:
                    self.types[key] = node
            elif t == "EventDecl":
                key = _iri_label(node.get("iri"))
                self._check_iri_syntax(node.get("iri"), "event IRI")
                if key in self.events:
                    self.err(1, "E1_DUP_EVENT",
                             f"duplicate EVENT {key}",
                             "keep a single EVENT declaration per IRI",
                             "EventDecl")
                else:
                    self.events[key] = node

    def _declare_named(self, node, table, kind, dup_code):
        name = node.get("name")
        self._check_ident(name, f"{kind.lower()} name", node.get("type"))
        for p in node.get("params", []):
            self._check_parameter(p, f"{kind} {name}")
        if name in table:
            self.err(1, dup_code, f"duplicate {kind} '{name}'",
                     f"rename one of the {kind} declarations", node.get("type"))
        else:
            table[name] = node

    def _check_parameter(self, param, owner):
        var = param.get("var", {})
        self._check_variable_syntax(var)
        self._check_type_ref_syntax(param.get("type"))
        card = param.get("cardinality")
        if card is not None and card not in ("ONE", "OPTIONAL", "MANY"):
            self.err(1, "E1_BAD_CARDINALITY",
                     f"{owner}: bad cardinality {card!r} on {_var_label(var)}",
                     "use ONE, OPTIONAL, or MANY", "Parameter")

    def _check_type_ref_syntax(self, tref):
        if tref is None or isinstance(tref, str):
            if isinstance(tref, str):
                self._check_ident(tref, "type name", "Name")
            return
        if tref.get("type") == "Name":
            self._check_ident(tref.get("value"), "type name", "Name")
        else:
            # A type IRI such as ex:Vehicle: shape only, NEVER existence.
            self._check_iri_syntax(tref, "type reference")

    # ------------------------------------------------------------ scope helpers
    def _lookup(self, key):
        for frame in reversed(self.scopes):
            if key in frame:
                return frame[key]
        return None

    def _define(self, var, card="ONE", dtype="unknown", origin=""):
        """Bind a variable in the current lexical frame + definite set."""
        self._check_variable_syntax(var)
        key = _var_key(var)
        frame = self.scopes[-1]
        if key in frame:
            self.err(3, "E3_REDEFINE",
                     f"variable {_var_label(var)} assigned twice in the same scope "
                     f"(first: {frame[key].origin}; now: {origin})",
                     "use a fresh variable or a nested block for shadowing",
                     origin or None)
        frame[key] = Binding(var, card, dtype, origin)
        return key

    def _use(self, var, definite, context, expect_card="ONE"):
        """Validate a variable *use*: visibility, definite assignment, card."""
        self._check_variable_syntax(var)
        key = _var_key(var)
        binding = self._lookup(key)
        label = _var_label(var)
        if binding is None:
            self.err(2, "E2_UNBOUND",
                     f"variable {label} used in {context} but not visible here",
                     "bind it first with MATCH / READ / BIND / DERIVE / CALL AS, "
                     "or check for a typo in the variable name",
                     context)
            return None
        if key not in definite:
            self.err(3, "E3_USE_BEFORE_DEF",
                     f"variable {label} may be unassigned at {context}",
                     "move the binding statement earlier, or guard the use so it "
                     "only runs on paths where the variable is assigned",
                     context)
        # CHECK 5 (cardinality) rides on uses.  Compatible when the cards
        # match, when a ONE value flows into OPTIONAL, or when the
        # context accepts anything (expect_card == "ANY").
        if expect_card != "ANY" and not _card_compatible(binding.card, expect_card):
            self.err(5, "E5_CARD_MISMATCH",
                     f"variable {label} is {binding.card} (from {binding.origin}) "
                     f"but {context} requires {expect_card}",
                     "iterate with EACH, select/aggregate first, guard an "
                     "OPTIONAL with BRANCH, or change the declaration to "
                     "accept the wider cardinality",
                     context)
        return binding

    # ------------------------------------------------------------ terms
    def _vars_in_term(self, term):
        if isinstance(term, dict) and term.get("type") == "Variable":
            return [term]
        return []

    def _check_term_use(self, term, definite, context, expect_card="ANY"):
        """A ground-position term: variables are uses; IRIs shape-checked."""
        if not isinstance(term, dict):
            return
        t = term.get("type")
        if t == "Variable":
            self._use(term, definite, context, expect_card)
        elif t == "Iri":
            self._check_iri_syntax(term)
        elif t == "BlankNode":
            self._check_ident(term.get("name"), "blank node name", "BlankNode")
        elif t == "Literal":
            pass  # literals are self-contained; nothing RDF-blind to check
        elif t == "List":
            for item in term.get("items", []):
                self._check_term_use(item, definite, context, expect_card)
        elif t == "FunctionCall":
            self._check_function_call(term, definite, context)

    def _check_triple_use(self, triple, definite, context):
        self._check_term_use(triple.get("subject"), definite,
                             f"{context} subject")
        pred = triple.get("predicate")
        if isinstance(pred, dict):
            # Predicates are IRIs, never variables; shape only, no existence.
            self._check_iri_syntax(pred, "predicate")
        self._check_term_use(triple.get("object"), definite,
                             f"{context} object")

    def _bind_pattern_vars(self, triples, definite, origin, card="ONE"):
        """Bind the variables of a MATCH/READ/WAIT pattern.

        SPARQL-style join semantics: the first mention of a variable in a
        pattern *defines* it; repeats in the same pattern are join
        constraints; mentions of an already-visible variable constrain the
        match against the existing binding.  Patterns therefore never
        trigger E3_REDEFINE — only AS-vars and BIND/DERIVE targets do.
        Every mentioned variable is definitely bound afterwards.
        """
        seen: set = set()
        for triple in triples or []:
            if not isinstance(triple, dict):
                continue
            pred = triple.get("predicate")
            if isinstance(pred, dict):
                self._check_iri_syntax(pred, "predicate")
            for pos in ("subject", "object"):
                term = triple.get(pos)
                if not isinstance(term, dict):
                    continue
                if term.get("type") == "Variable":
                    self._check_variable_syntax(term)
                    key = _var_key(term)
                    if key not in seen:
                        seen.add(key)
                        if self._lookup(key) is None:
                            self._define(term, card, "unknown", origin)
                    definite.add(key)
                else:
                    self._check_term_use(term, definite, f"{origin} {pos}")

    # ------------------------------------------------------------ expressions
    def _infer_expr(self, expr, definite, context, expect_card="ONE"):
        """Validate an expression; return (card, dtype)."""
        if not isinstance(expr, dict):
            return ("ONE", "unknown")
        t = expr.get("type")
        if t == "Variable":
            b = self._use(expr, definite, context, expect_card)
            if b is None:
                return ("ONE", "unknown")
            return (b.card, b.dtype)
        if t == "Literal":
            return ("ONE", _literal_dtype(expr))
        if t == "List":
            for item in expr.get("items", []):
                self._infer_expr(item, definite, f"{context} list item")
            return ("MANY", "unknown")
        if t == "FunctionCall":
            return self._check_function_call(expr, definite, context)
        if t == "Iri":
            # Bare IRI in expression position (e.g. ex:Off): a constant.
            self._check_iri_syntax(expr)
            return ("ONE", "unknown")
        if t == "BlankNode":
            self._check_ident(expr.get("name"), "blank node name", "BlankNode")
            return ("ONE", "unknown")
        return ("ONE", "unknown")

    def _check_function_call(self, node, definite, context):
        ref = node.get("ref", {})
        # Function identity is NOT resolved (RDF-blind); shape only.
        if ref.get("type") == "Name":
            self._check_ident(ref.get("value"), "function name", "FunctionCall")
        else:
            self._check_iri_syntax(ref, "function reference")
        for arg in node.get("args", []):
            card, _ = self._infer_expr(arg, definite, f"{context} argument")
            if card == "MANY":
                self.err(5, "E5_CARD_MISMATCH",
                         f"MANY-valued argument passed where ONE is required "
                         f"({context})",
                         "iterate with EACH or aggregate before the call",
                         context)
        return ("ONE", "unknown")

    def _check_condition(self, cond, definite, context):
        if not isinstance(cond, dict):
            return
        t = cond.get("type")
        if t in ("And", "Or"):
            self._check_condition(cond.get("left"), definite, context)
            self._check_condition(cond.get("right"), definite, context)
        elif t == "Not":
            self._check_condition(cond.get("operand"), definite, context)
        elif t == "Comparison":
            lc, lt = self._infer_expr(cond.get("left"), definite,
                                      f"{context} comparison")
            # `x IN [..]` / `x NOT IN [..]`: the right side is *meant* to be
            # a list, so it accepts MANY; every other operator needs ONE.
            right_expect = "ANY" if cond.get("op") in ("IN", "NOT IN") else "ONE"
            rc, rt = self._infer_expr(cond.get("right"), definite,
                                      f"{context} comparison", right_expect)
            for c, side in ((lc, "left"), (rc, "right")):
                if side == "right" and right_expect == "ANY":
                    continue
                if c == "MANY":
                    self.err(5, "E5_CARD_MISMATCH",
                             f"MANY-valued {side} operand in comparison "
                             f"({context}); comparisons require ONE",
                             "iterate with EACH or select a single element first",
                             context)
            self._check_literal_type_pair(lt, rt, context)
        else:
            # Fallback: treat unknown condition shapes as expressions.
            self._infer_expr(cond, definite, context)

    def _check_literal_type_pair(self, lt, rt, context):
        """CHECK 4: only literal-vs-literal mismatches are knowable blind."""
        lit = {"integer", "decimal", "boolean", "string"}
        if lt in lit and rt in lit and lt != rt:
            # integer/decimal are mutually comparable numerics.
            if {lt, rt} <= {"integer", "decimal"}:
                return
            self.err(4, "E4_TYPE_MISMATCH",
                     f"incompatible literal types in {context}: {lt} vs {rt}",
                     "make both sides the same datatype",
                     context)
        if (lt in lit and rt == "nominal") or (lt == "nominal" and rt in lit):
            self.err(4, "E4_TYPE_MISMATCH",
                     f"nominal entity compared against literal in {context}",
                     "compare entities with entities and literals with literals",
                     context)

    # ------------------------------------------------------------ CHECK 4: cycles
    def _check_type_cycles(self):
        edges = {}
        for key, decl in self.types.items():
            ext = decl.get("extends")
            if ext is not None:
                edges[key] = _type_label(ext)
        visited: dict[str, int] = {}

        def dfs(node, stack):
            state = visited.get(node, 0)
            if state == 1:
                cycle = " -> ".join(stack + [node])
                self.err(4, "E4_TYPE_CYCLE",
                         f"cyclic TYPE EXTENDS: {cycle}",
                         "break the inheritance cycle", "TypeDecl")
                return
            if state == 2:
                return
            visited[node] = 1
            nxt = edges.get(node)
            if nxt is not None and nxt in edges:
                dfs(nxt, stack + [node])
            visited[node] = 2

        for node in edges:
            dfs(node, [])

    # ------------------------------------------------------------ statements
    def _check_statements(self, stmts, definite, context):
        for stmt in stmts or []:
            if not isinstance(stmt, dict):
                continue
            t = stmt.get("type")
            handler = getattr(self, f"_stmt_{t.lower()}", None)
            if handler is None:
                self.err(6, "E6_UNKNOWN_STATEMENT",
                         f"unknown statement node '{t}' in {context}",
                         "check the parser output version matches this validator",
                         context)
                continue
            definite = handler(stmt, definite, context)
            if t == "Return":
                # Anything after RETURN in the same block is unreachable.
                idx = (stmts or []).index(stmt)
                if idx < len(stmts) - 1:
                    self.err(6, "E6_UNREACHABLE",
                             f"unreachable statement(s) after RETURN in {context}",
                             "move RETURN to the end of the block or guard it "
                             "with BRANCH",
                             context)
                    break
        return definite

    # -- contexts are data blocks: variables inside are block-local
    # -- (existential), neither requiring nor leaking bindings.
    def _stmt_context(self, stmt, definite, context):
        kind = stmt.get("kind", "?")
        self.scopes.append({})
        for triple in stmt.get("body", []):
            pred = triple.get("predicate")
            if isinstance(pred, dict):
                self._check_iri_syntax(pred, "predicate")
            for pos in ("subject", "object"):
                term = triple.get(pos)
                if isinstance(term, dict) and term.get("type") == "Variable":
                    self._check_variable_syntax(term)
                elif isinstance(term, dict):
                    self._check_term_use(term, definite, f"{kind} block")
        self.scopes.pop()
        return definite

    def _stmt_triple(self, stmt, definite, context):
        # Bare top-level triple: ground fact; variables (if any) are uses.
        self._check_triple_use(stmt, definite, f"{context} triple")
        return definite

    def _stmt_match(self, stmt, definite, context):
        scope = stmt.get("scope")
        if scope is not None and scope not in (
                "WORLD", "MEMORY", "WORKING", "PLAN", "HISTORY"):
            self.err(6, "E6_BAD_SCOPE", f"MATCH with unknown scope {scope!r}",
                     "use WORLD, MEMORY, WORKING, PLAN, or HISTORY", "Match")
        self._bind_pattern_vars(stmt.get("pattern"), definite, "Match pattern")
        for var in self._pattern_vars(stmt.get("pattern")):
            definite.add(_var_key(var))
        as_var = stmt.get("as")
        if as_var is not None:
            # The AS handle reifies the whole match set -> MANY.
            self._define(as_var, "MANY", "unknown", "Match AS")
            definite.add(_var_key(as_var))
        return definite

    def _pattern_vars(self, triples):
        out = []
        for triple in triples or []:
            for pos in ("subject", "object"):
                out.extend(self._vars_in_term(triple.get(pos)))
        return out

    def _stmt_read(self, stmt, definite, context):
        triple = stmt.get("triple") or {}
        # Single-triple pattern: same join semantics as MATCH (a mention of
        # an already-visible variable constrains the read; fresh variables
        # are bound ONE each).
        pred = triple.get("predicate")
        if isinstance(pred, dict):
            self._check_iri_syntax(pred, "predicate")
        for pos in ("subject", "object"):
            term = triple.get(pos)
            if isinstance(term, dict) and term.get("type") == "Variable":
                self._check_variable_syntax(term)
                key = _var_key(term)
                if self._lookup(key) is None:
                    self._define(term, "ONE", "unknown", "Read pattern")
                definite.add(key)
            elif isinstance(term, dict):
                self._check_term_use(term, definite, f"Read pattern {pos}")
        card = stmt.get("cardinality") or "ONE"
        if card not in ("ONE", "OPTIONAL", "MANY"):
            self.err(1, "E1_BAD_CARDINALITY", f"READ with bad cardinality {card!r}",
                     "use ONE, OPTIONAL, or MANY", "Read")
            card = "ONE"
        as_var = stmt.get("as")
        if as_var is not None:
            self._define(as_var, card, "unknown", "Read AS")
            definite.add(_var_key(as_var))
        return definite

    def _stmt_write(self, stmt, definite, context):
        return self._stmt_write_delete(stmt, definite, context, "WRITE")

    def _stmt_delete(self, stmt, definite, context):
        return self._stmt_write_delete(stmt, definite, context, "DELETE")

    def _stmt_write_delete(self, stmt, definite, context, kind):
        scope = stmt.get("scope")
        if scope is None:
            self.warn(7, "E7_MISSING_SCOPE",
                      f"{kind} without explicit scope in {context}",
                      f"write '{kind} WORLD {{ ... }}' (or MEMORY/WORKING)",
                      kind)
        elif scope in IMMUTABLE_SCOPES:
            self.err(7, "E7_IMMUTABLE_SCOPE",
                     f"{kind} targets immutable scope {scope} in {context}",
                     "HISTORY is append-only; record history with EMIT instead",
                     kind)
        for triple in stmt.get("block", []):
            self._check_triple_use(triple, definite, f"{kind} block")
        return definite

    def _stmt_bind(self, stmt, definite, context):
        var = stmt.get("var", {})
        card, dtype = self._infer_expr(stmt.get("expression"), definite,
                                       "BIND right-hand side", "ANY")
        # CHECK 4: rebinding a nominally-typed plan param to a literal.
        key = _var_key(var) if isinstance(var, dict) else None
        prior = self._lookup(key) if key else None
        if prior is not None and prior.dtype == "nominal" and dtype in (
                "integer", "decimal", "boolean", "string"):
            self.err(4, "E4_TYPE_MISMATCH",
                     f"BIND {_var_label(var)}: literal assigned to "
                     f"nominally-typed {prior.dtype} variable from {prior.origin}",
                     "bind a compatible entity or introduce a fresh variable",
                     "Bind")
        self._define(var, card, dtype, "Bind")
        definite.add(_var_key(var))
        return definite

    def _stmt_derive(self, stmt, definite, context):
        var = stmt.get("var", {})
        # Pattern vars are local derivations: bind in a child scope.
        self.scopes.append({})
        self._bind_pattern_vars(stmt.get("pattern"), definite, "Derive pattern")
        self.scopes.pop()
        proj = stmt.get("projection", {})
        if proj.get("type") == "Name":
            self._check_ident(proj.get("value"), "projection name", "Derive")
            # In-file arity/cardinality cross-checks would go here; the
            # *existence* of the projection is NOT required (Pass 2).
        else:
            self._check_iri_syntax(proj, "projection reference")
        self._define(var, "ONE", "unknown", "Derive")
        definite.add(_var_key(var))
        return definite

    def _stmt_call(self, stmt, definite, context):
        cap = stmt.get("capability", {})
        if cap.get("type") == "Name":
            self._check_ident(cap.get("value"), "capability name", "Call")
        else:
            # Shape only — existence is a Pass 2 (semantic binding) question.
            self._check_iri_syntax(cap, "capability reference")
        # Expected cardinalities: from the in-file CAPABILITY decl when
        # present, else conservative ONE for every argument.
        key = _iri_label(cap) if cap.get("type") != "Name" else cap.get("value")
        decl = self.capabilities.get(key)
        params = decl.get("params", []) if isinstance(decl, dict) else []
        args = stmt.get("args", [])
        for i, arg in enumerate(args):
            expect = "ONE"
            if i < len(params):
                expect = params[i].get("cardinality") or "ONE"
            card, _ = self._infer_expr(arg, definite, f"CALL {_iri_label(cap)}",
                                       "ANY")
            if not _card_compatible(card, expect):
                self.err(5, "E5_CARD_MISMATCH",
                         f"CALL {_iri_label(cap)} argument {i + 1} is {card} but "
                         f"parameter {i + 1} requires {expect}",
                         "iterate with EACH, select one element, guard an "
                         "OPTIONAL with BRANCH, or declare the parameter "
                         f"{card}",
                         "Call")
        as_var = stmt.get("as")
        if as_var is not None:
            self._define(as_var, "ONE", "unknown", "Call AS")
            definite.add(_var_key(as_var))
        return definite

    def _stmt_emit(self, stmt, definite, context):
        self._check_iri_syntax(stmt.get("event"), "emitted event")
        for arg in stmt.get("args", []) or []:
            card, _ = self._infer_expr(arg, definite, "EMIT argument", "ANY")
            if card == "MANY":
                self.err(5, "E5_CARD_MISMATCH",
                         "EMIT with MANY-valued argument; events carry single values",
                         "iterate with EACH and emit once per element",
                         "Emit")
        return definite

    def _stmt_spawn(self, stmt, definite, context):
        actor = stmt.get("actor", {})
        if actor.get("type") == "Name":
            self._check_ident(actor.get("value"), "actor name", "Spawn")
        else:
            self._check_iri_syntax(actor, "actor reference")
        for triple in stmt.get("with") or []:
            self._check_triple_use(triple, definite, "SPAWN WITH block")
        as_var = stmt.get("as")
        if as_var is not None:
            self._define(as_var, "ONE", "unknown", "Spawn AS")
            definite.add(_var_key(as_var))
        return definite

    def _stmt_wait(self, stmt, definite, context):
        target = stmt.get("target", {})
        if target.get("type") == "EventMatch":
            self._check_iri_syntax(target.get("event"), "waited event")
            # Event pattern vars become bound after the WAIT.
            self._bind_pattern_vars(target.get("pattern"), definite, "Wait EVENT")
            for var in self._pattern_vars(target.get("pattern")):
                definite.add(_var_key(var))
        else:
            self._check_condition(target, definite, f"{context} WAIT")
        timeout = stmt.get("timeout")
        if timeout is not None and timeout.get("value", 1) <= 0:
            self.err(6, "E6_BAD_TIMEOUT", "WAIT TIMEOUT must be positive",
                     "use e.g. TIMEOUT 30s", "Wait")
        return definite

    def _stmt_branch(self, stmt, definite, context):
        outs = []
        for i, clause in enumerate(stmt.get("when") or []):
            self._check_condition(clause.get("condition"), definite,
                                  f"{context} WHEN {i + 1}")
            self.scopes.append({})
            out = self._check_statements(list(clause.get("body") or []),
                                         set(definite), f"{context} WHEN {i + 1}")
            if not clause.get("body"):
                self.warn(6, "E6_EMPTY_BLOCK",
                          f"empty WHEN {i + 1} body in {context}",
                          "add statements or remove the clause", "Branch")
            self.scopes.pop()
            outs.append(out)
        else_block = stmt.get("else_")
        if else_block is not None:
            self.scopes.append({})
            out = self._check_statements(list(else_block), set(definite),
                                         f"{context} ELSE")
            if not else_block:
                self.warn(6, "E6_EMPTY_BLOCK", f"empty ELSE body in {context}",
                          "add statements or remove the ELSE", "Branch")
            self.scopes.pop()
            outs.append(out)
        if not outs:
            return definite
        if else_block is None:
            # Without ELSE the branch may not run: only incoming vars are sure.
            sometimes = set()
            for out in outs:
                sometimes |= out
            for key in sometimes - definite:
                pass  # maybe-bound: visible in-arm only, not definite after
            return definite
        joined = set(outs[0])
        for out in outs[1:]:
            joined &= out
        joined |= definite
        return joined

    def _stmt_parallel(self, stmt, definite, context):
        blocks = stmt.get("blocks") or []
        if len(blocks) < 2:
            self.err(6, "E6_PARALLEL_ARITY",
                     f"PARALLEL with {len(blocks)} block(s) in {context}; need >= 2",
                     "add another block or drop PARALLEL", "Parallel")
        # Arms share the enclosing scope and are threaded in author order, so
        # a handle spawned in one arm (e.g. SPAWN ... AS ?m) is visible in
        # later arms.  True data-race analysis is a runtime concern, not
        # Pass 1: structurally, PARALLEL behaves like a block here.
        for i, block in enumerate(blocks):
            definite = self._check_statements(list(block), definite,
                                              f"{context} PARALLEL arm {i + 1}")
        return definite

    def _stmt_each(self, stmt, definite, context):
        var = stmt.get("var", {})
        match = stmt.get("match") or {}
        # One child frame covers the MATCH pattern and the loop body: the
        # element handle and pattern variables live only for the iteration.
        # Outer variables mentioned in the pattern are join constraints.
        self.scopes.append({})
        self._check_variable_syntax(var)
        self._define(var, "ONE", "unknown", "Each var")
        inner_definite = set(definite) | {_var_key(var)}
        self._bind_pattern_vars(match.get("pattern"), inner_definite,
                                "Each MATCH")
        for v in self._pattern_vars(match.get("pattern")):
            inner_definite.add(_var_key(v))
        self._check_statements(list(stmt.get("body") or []), inner_definite,
                               f"{context} EACH body")
        if not stmt.get("body"):
            self.warn(6, "E6_EMPTY_BLOCK", f"empty EACH body in {context}",
                      "add statements or remove the EACH", "Each")
        self.scopes.pop()
        return definite  # body bindings do not leak

    def _stmt_try(self, stmt, definite, context):
        self.scopes.append({})
        try_out = self._check_statements(list(stmt.get("body") or []),
                                         set(definite), f"{context} TRY")
        self.scopes.pop()
        catch_var = stmt.get("catch_var")
        catch_body = stmt.get("catch_body")
        if catch_var is None:
            # No handler: success-or-abort, try bindings stand on success.
            return try_out
        self.scopes.append({})
        self._define(catch_var, "ONE", "unknown", "Try CATCH var")
        catch_definite = set(definite) | {_var_key(catch_var)}
        catch_out = self._check_statements(list(catch_body or []),
                                           catch_definite, f"{context} CATCH")
        self.scopes.pop()
        # Either path may have run: definite = intersection.
        return (try_out & catch_out) | definite

    def _stmt_return(self, stmt, definite, context):
        value = stmt.get("value")
        if isinstance(value, list):  # graph block form
            for triple in value:
                if isinstance(triple, dict) and triple.get("type") == "Triple":
                    self._check_triple_use(triple, definite, "RETURN block")
        elif isinstance(value, dict):
            # RETURN accepts any cardinality (including a MANY match handle).
            self._infer_expr(value, definite, "RETURN", "ANY")
        return definite

    # -- declarations with bodies: fresh lexical scope, params pre-bound
    def _stmt_plandecl(self, stmt, definite, context):
        return self._check_decl_body(stmt, definite, "PLAN", pure=False)

    def _stmt_projectiondecl(self, stmt, definite, context):
        return self._check_decl_body(stmt, definite, "PROJECTION", pure=True)

    def _check_decl_body(self, stmt, definite, kind, pure):
        name = stmt.get("name", "?")
        self.scopes.append({})
        local_definite: set = set()
        for p in stmt.get("params", []):
            var = p.get("var", {})
            card = p.get("cardinality") or "ONE"
            dtype = ("nominal" if _type_label(p.get("type")) not in (
                "unknown", "?", None) else "unknown")
            # Types are nominal refs here; exact kind rarely matters blind.
            self._define(var, card, "unknown", f"{kind} {name} param")
            local_definite.add(_var_key(var))
        body = list(stmt.get("body") or [])
        if not body:
            self.warn(6, "E6_EMPTY_BLOCK", f"empty {kind} {name} body",
                      "add statements or remove the declaration", kind)
        if pure:
            for sub in body:
                if isinstance(sub, dict) and sub.get("type") not in PURE_STMTS:
                    self.err(7, "E7_EFFECT_IN_PROJECTION",
                             f"{sub.get('type')} inside PROJECTION {name}; "
                             f"projections must be pure queries",
                             "move the effectful statement into a PLAN",
                             "ProjectionDecl")
        out = self._check_statements(body, local_definite, f"{kind} {name}")
        self.scopes.pop()
        return definite  # decl bodies never leak into the enclosing flow

    # -- leaf declarations: already handled in _collect_declarations
    def _stmt_prefixdecl(self, stmt, definite, context):
        return definite

    def _stmt_typedecl(self, stmt, definite, context):
        return definite

    def _stmt_eventdecl(self, stmt, definite, context):
        schema = stmt.get("schema")
        if schema:
            self.scopes.append({})
            for triple in schema:
                pred = triple.get("predicate")
                if isinstance(pred, dict):
                    self._check_iri_syntax(pred, "predicate")
                for pos in ("subject", "object"):
                    term = triple.get(pos)
                    if isinstance(term, dict) and term.get("type") == "Variable":
                        self._check_variable_syntax(term)
            self.scopes.pop()
        return definite

    def _stmt_capabilitydecl(self, stmt, definite, context):
        return definite


def _literal_dtype(lit):
    dt = lit.get("datatype")
    if dt in ("xsd:integer", None) and isinstance(lit.get("value"), bool):
        return "boolean"
    if dt == "xsd:integer" or isinstance(lit.get("value"), int) and not isinstance(
            lit.get("value"), bool):
        if dt is None or dt == "xsd:integer":
            return "integer"
    if dt == "xsd:decimal" or isinstance(lit.get("value"), float):
        return "decimal"
    if dt == "xsd:boolean" or isinstance(lit.get("value"), bool):
        return "boolean"
    if isinstance(dt, dict):  # typed literal "x"^^<...>: opaque here
        return "unknown"
    return "string"


def validate_ast(ast, strict=False):
    """Validate a parsed AST dict.  Returns the report dict."""
    return SanityValidator(strict=strict).validate(ast)


def validate_source(src, strict=False):
    """Parse Graph IR source, then RDF-blind validate.  Returns report."""
    mod = _parser()
    try:
        ast = mod.parse(src)
    except Exception as exc:  # IRError from the EBNF parser
        return {
            "valid": False,
            "parse_error": str(exc),
            "errors": [{
                "check": 0,
                "code": "E0_PARSE_ERROR",
                "message": str(exc),
                "severity": "error",
                "hint": "fix the syntax error before structural validation",
            }],
            "warnings": [],
        }
    report = validate_ast(ast, strict=strict)
    report["ast"] = ast
    return report


def emit(record):
    """Emit one JSONL record to stdout.  This is the tool's only output."""
    print(json.dumps(record))


def main(argv=None):
    """CLI: validate one file (or stdin), emit exactly one JSONL record.

    The record always carries ``ok`` / ``tool`` / ``path`` / ``valid`` /
    ``errors`` / ``warnings`` (plus ``ast`` whenever parsing succeeded),
    so pipelines can consume stdout without ever parsing freeform text.

    Exit codes: 0 = VALID, 1 = INVALID, 2 = parse/IO/usage error.
    Nothing human-readable is ever printed; stderr stays silent.
    """
    argv = list(sys.argv if argv is None else argv)
    prog = Path(argv[0]).name if argv else "ir-ast-sanity"
    args = argv[1:]
    # Accepted for backward compatibility; output is always one compact
    # JSONL line regardless.
    strict = "--strict" in args
    args = [a for a in args if a not in ("--json", "--strict", "--compact")]
    if "--help" in args or "-h" in args:
        emit({"ok": True, "tool": "ir-ast-sanity", "path": None,
              "usage": f"{prog} <file|-|--stdin> [--strict]"})
        return 0

    use_stdin = "--stdin" in args
    args = [a for a in args if a != "--stdin"]
    if use_stdin or args == ["-"]:
        path, src = "<stdin>", sys.stdin.read()
    elif len(args) == 1:
        path = args[0]
        try:
            src = Path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            emit({"ok": False, "tool": "ir-ast-sanity", "path": path,
                  "valid": False,
                  "errors": [{"check": None, "code": "E_IO", "severity": "error",
                              "message": f"{path}: no such file or directory",
                              "hint": "check the file path"}],
                  "warnings": []})
            return 2
        except OSError as exc:
            emit({"ok": False, "tool": "ir-ast-sanity", "path": path,
                  "valid": False,
                  "errors": [{"check": None, "code": "E_IO", "severity": "error",
                              "message": f"{path}: {exc}",
                              "hint": "check file permissions and encoding"}],
                  "warnings": []})
            return 2
    else:
        emit({"ok": False, "tool": "ir-ast-sanity", "path": None,
              "valid": False,
              "errors": [{"check": None, "code": "E_USAGE", "severity": "error",
                          "message": f"usage: {prog} <file|-|--stdin> [--strict]",
                          "hint": "pass exactly one input file"}],
              "warnings": []})
        return 2

    report = validate_source(src, strict=strict)
    record = {"ok": bool(report.get("valid")),
              "tool": "ir-ast-sanity",
              "path": path,
              "valid": bool(report.get("valid")),
              "errors": report.get("errors", []),
              "warnings": report.get("warnings", [])}
    if "parse_error" in report:
        record["parse_error"] = report["parse_error"]
    if "ast" in report:
        record["ast"] = report["ast"]
    emit(record)
    if report.get("parse_error"):
        return 2
    return 0 if report.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
