#!/usr/bin/env python3
"""Graph IR parser — implements the EBNF in EBNF.md.

Parses Graph IR source into a canonical AST (plain dicts, JSON-serializable)
and reduces nothing else: PARALLEL / EACH / TRY are preserved as explicit
compiler-level nodes, all twelve VM primitives appear as statement nodes.

CLI: ir-compile.py <file>   (file may be "-" or --stdin for stdin)

Output contract: exactly one JSONL record on stdout, nothing on stderr.
Success: {"ok": true, "tool": "ir-compile", "path": ..., "ast": {...}}
Failure: {"ok": false, "tool": "ir-compile", "path": ..., "errors": [...]}
"""
import json
import re
import sys
from pathlib import Path

KEYWORDS = {
    "PREFIX", "PLAN", "PROJECTION", "EVENT", "TYPE", "CAPABILITY", "RETURNS",
    "EXTENDS", "WORLD", "MEMORY", "GOAL", "WORKING", "HISTORY", "MATCH",
    "READ", "WRITE", "DELETE", "BIND", "DERIVE", "AS", "CALL", "EMIT",
    "SPAWN", "WAIT", "WITH", "TIMEOUT", "BRANCH", "WHEN", "THEN", "ELSE",
    "PARALLEL", "EACH", "IN", "DO", "TRY", "CATCH", "RETURN", "USING",
    "ONE", "OPTIONAL", "MANY", "NOT", "AND", "OR", "TRUE", "FALSE", "WHEN",
}

DURATION_UNITS = {"ms", "s", "m", "h", "d"}

IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]*")
IRI_RE = re.compile(r'<[^<>"{}|^`\x00-\x20]*>')
ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "'": "'"}


class IRError(Exception):
    def __init__(self, message, line, column):
        super().__init__(f"{line}:{column}: {message}")
        self.line = line
        self.column = column


class Token:
    __slots__ = ("kind", "value", "line", "column")

    def __init__(self, kind, value, line, column):
        self.kind = kind
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self):
        return f"Token({self.kind}, {self.value!r}, {self.line}:{self.column})"


def tokenize(src):
    tokens = []
    i = 0
    line = 1
    col = 1
    n = len(src)

    def emit(kind, value, s_line, s_col, end):
        nonlocal i, line, col
        col = s_col + (end - i)
        tokens.append(Token(kind, value, s_line, s_col))
        i = end

    while i < n:
        c = src[i]
        if c == "\n":
            i += 1
            line += 1
            col = 1
            continue
        if c in " \t\r":
            i += 1
            col += 1
            continue
        if c == "#":
            while i < n and src[i] != "\n":
                i += 1
            continue
        s_line, s_col = line, col
        if c == '"':
            j = i + 1
            buf = []
            while j < n:
                ch = src[j]
                if ch == "\\" and j + 1 < n:
                    esc = src[j + 1]
                    if esc == "u" and j + 5 < n:
                        buf.append(chr(int(src[j + 2 : j + 6], 16)))
                        j += 6
                        continue
                    buf.append(ESCAPES.get(esc, esc))
                    j += 2
                    continue
                if ch == '"':
                    break
                if ch == "\n":
                    raise IRError("unterminated string literal", s_line, s_col)
                buf.append(ch)
                j += 1
            if j >= n:
                raise IRError("unterminated string literal", s_line, s_col)
            emit("STRING", "".join(buf), s_line, s_col, j + 1)
            continue
        if c == "<":
            m = IRI_RE.match(src, i)
            if m:
                emit("IRI", src[i + 1 : m.end() - 1], s_line, s_col, m.end())
                continue
            emit("<", "<", s_line, s_col, i + 1)
            continue
        if c.isdigit() or (c == "-" and i + 1 < n and src[i + 1].isdigit()):
            neg = c == "-"
            start = i + 1 if neg else i
            j = start
            while j < n and src[j].isdigit():
                j += 1
            if j < n and src[j] == "." and j + 1 < n and src[j + 1].isdigit():
                j += 1
                while j < n and src[j].isdigit():
                    j += 1
                text = ("-" if neg else "") + src[start:j]
                emit("DECIMAL", text, s_line, s_col, j)
            else:
                text = ("-" if neg else "") + src[start:j]
                emit("INTEGER", text, s_line, s_col, j)
            continue
        m = IDENT_RE.match(src, i)
        if m:
            word = m.group(0)
            upper = word.upper()
            kind = upper if upper in KEYWORDS else "IDENT"
            emit(kind, word, s_line, s_col, m.end())
            continue
        two = src[i : i + 2]
        if two in ("!=", "<=", ">=", "^^"):
            emit(two, two, s_line, s_col, i + 2)
            continue
        if c in "=:,.?!(){}[]":
            kind = "DOT" if c == "." else c
            emit(kind, c, s_line, s_col, i + 1)
            continue
        raise IRError(f"unexpected character {c!r}", s_line, s_col)

    tokens.append(Token("EOF", None, line, col))
    return tokens


class Parser:
    def __init__(self, src):
        self.tokens = tokenize(src)
        self.pos = 0

    @property
    def tok(self):
        return self.tokens[self.pos]

    def advance(self):
        t = self.tok
        self.pos += 1
        return t

    def at(self, *kinds):
        return self.tok.kind in kinds

    def expect(self, kind):
        if not self.at(kind):
            raise IRError(
                f"expected {kind!r} but found {self.tok.value or 'end of file'!r}",
                self.tok.line,
                self.tok.column,
            )
        return self.advance()

    def expect_ident(self):
        t = self.tok
        if t.kind != "IDENT":
            raise IRError(f"expected identifier but found {t.value or 'end of file'!r}", t.line, t.column)
        return self.advance()

    def expect_name(self):
        """Accept identifiers and grammar keywords used as name positions
        after ':' (e.g. rdf:type, my:?type)."""
        t = self.tok
        if t.kind == "EOF" or not t.kind.isalpha() or t.kind in ("IRI", "STRING"):
            raise IRError(f"expected name but found {t.value or 'end of file'!r}", t.line, t.column)
        return self.advance()

    def node(self, type_, **fields):
        return {"type": type_, **fields}

    # --------------------------------------------------------------- document
    def parse_document(self):
        body = []
        while self.tok.kind != "EOF":
            kind = self.tok.kind
            if kind == "PREFIX":
                body.append(self.parse_prefix())
            elif kind in ("PLAN", "PROJECTION"):
                body.append(self.parse_plan(kind))
            elif kind == "EVENT":
                body.append(self.parse_event())
            elif kind == "TYPE":
                body.append(self.parse_type())
            elif kind == "CAPABILITY":
                body.append(self.parse_capability())
            elif kind in ("WORLD", "MEMORY", "GOAL", "WORKING", "HISTORY") and (
                self.tokens[self.pos + 1].kind == "{"
            ):
                body.append(self.parse_context())
            else:
                body.append(self.parse_statement())
        return self.node("Document", body=body)

    def parse_prefix(self):
        self.advance()
        prefix = self.expect_ident().value
        self.expect(":")
        iri = self.parse_iri()
        self.expect("DOT")
        return self.node("PrefixDecl", prefix=prefix, iri=iri)

    def parse_plan(self, kind):
        self.advance()
        name = self.expect_ident().value
        params = []
        if self.at("("):
            self.advance()
            params = self.parse_parameter_list()
            self.expect(")")
        block = self.parse_block()
        return self.node(kind.capitalize() + "Decl" if kind == "PLAN" else "ProjectionDecl",
                         name=name, params=params, body=block)

    def parse_event(self):
        self.advance()
        iri = self.parse_iri()
        schema = None
        if self.at("{"):
            schema = self.parse_graph_block()
        self.expect("DOT")
        return self.node("EventDecl", iri=iri, schema=schema)

    def parse_type(self):
        self.advance()
        name_ref = self.parse_type_ref()
        # Keep bare identifiers as plain strings for backward compat
        # (`TYPE Vehicle ...` -> "Vehicle"), prefixed IRIs as Iri nodes
        # (`TYPE ex:Vehicle ...` -> {"type": "Iri", ...}).
        if name_ref.get("type") == "Name":
            name = name_ref["value"]
        else:
            name = name_ref
        extends = None
        if self.at("EXTENDS"):
            self.advance()
            extends = self.parse_type_ref()
        self.expect("DOT")
        return self.node("TypeDecl", name=name, extends=extends)

    def parse_capability(self):
        self.advance()
        iri = self.parse_iri()
        self.expect("(")
        params = []
        if not self.at(")"):
            params = self.parse_parameter_list()
        self.expect(")")
        returns = None
        if self.at("RETURNS"):
            self.advance()
            returns = self.parse_type_ref()
        self.expect("DOT")
        return self.node("CapabilityDecl", iri=iri, params=params, returns=returns)

    def parse_parameter_list(self):
        params = [self.parse_parameter()]
        while self.at(","):
            self.advance()
            params.append(self.parse_parameter())
        return params

    def parse_parameter(self):
        var = self.parse_variable()
        self.expect(":")
        type_ref = self.parse_type_ref()
        cardinality = self.parse_cardinality()
        return self.node("Parameter", var=var, type=type_ref, cardinality=cardinality)

    def parse_type_ref(self):
        t = self.tok
        if t.kind == "IRI" or (t.kind == "IDENT" and self.tokens[self.pos + 1].kind == ":"):
            return self.parse_iri()
        return self.node("Name", value=self.expect_ident().value)

    def parse_cardinality(self):
        if self.at("ONE"):
            self.advance()
            return "ONE"
        if self.at("OPTIONAL"):
            self.advance()
            return "OPTIONAL"
        if self.at("MANY"):
            self.advance()
            return "MANY"
        return None

    # ---------------------------------------------------------------- graphs
    def parse_context(self):
        kind = self.advance().value
        return self.node("Context", kind=kind, body=self.parse_graph_block())

    def parse_graph_block(self):
        self.expect("{")
        triples = []
        while not self.at("}"):
            triples.append(self.parse_triple())
        self.expect("}")
        return triples

    def parse_triple(self):
        subject = self.parse_term()
        predicate = self.parse_iri(lenient=True)
        obj = self.parse_term()
        self.expect("DOT")
        return self.node("Triple", subject=subject, predicate=predicate, object=obj)

    def parse_term(self):
        t = self.tok
        if t.kind == "IRI":
            return self.parse_iri()
        if t.kind == "?":
            return self.parse_variable()
        if t.kind == "IDENT" and self.tokens[self.pos + 1].kind == ":":
            nxt = self.tokens[self.pos + 2].kind
            if nxt == "?":
                return self.parse_variable()
            return self.parse_iri()
        if t.kind == "IDENT" and t.value == "_" and self.tokens[self.pos + 1].kind == ":":
            self.advance()
            self.advance()
            return self.node("BlankNode", name=self.expect_ident().value)
        if t.kind == "STRING":
            self.advance()
            if self.at("^^"):
                self.advance()
                return self.node("Literal", value=t.value, datatype=self.parse_iri())
            return self.node("Literal", value=t.value)
        if t.kind == "INTEGER":
            self.advance()
            return self.node("Literal", value=int(t.value), datatype="xsd:integer")
        if t.kind == "DECIMAL":
            self.advance()
            return self.node("Literal", value=float(t.value), datatype="xsd:decimal")
        if t.kind == "TRUE":
            self.advance()
            return self.node("Literal", value=True, datatype="xsd:boolean")
        if t.kind == "FALSE":
            self.advance()
            return self.node("Literal", value=False, datatype="xsd:boolean")
        if t.kind == "IDENT":
            self.advance()
            return self.node("Iri", prefix=t.value, local=None, bare=True)
        raise IRError(f"expected term but found {t.value or 'end of file'!r}", t.line, t.column)

    def parse_iri(self, lenient=False):
        t = self.tok
        if t.kind == "IRI":
            self.advance()
            return self.node("Iri", iri=t.value)
        if t.kind == "IDENT" and t.value == "_" and self.tokens[self.pos + 1].kind == ":":
            self.advance()
            self.expect(":")
            return self.node("BlankNode", name=self.expect_name().value)
        if t.kind == "IDENT" and self.tokens[self.pos + 1].kind == ":" and self.tokens[self.pos + 2].kind.isalpha() and self.tokens[self.pos + 2].kind != "EOF":
            prefix = self.advance().value
            self.expect(":")
            local = self.expect_name().value
            return self.node("Iri", prefix=prefix, local=local)
        if lenient and t.kind != "EOF" and t.kind.isalpha() and t.kind not in ("IRI", "STRING"):
            self.advance()
            return self.node("Iri", prefix=t.value, local=None, bare=True)
        raise IRError(f"expected IRI but found {t.value or 'end of file'!r}", t.line, t.column)

    def parse_variable(self):
        namespace = "my"
        t = self.tok
        if t.kind == "IDENT" and self.tokens[self.pos + 1].kind == ":" and self.tokens[self.pos + 2].kind == "?":
            namespace = self.advance().value
            self.advance()
        self.expect("?")
        name = self.expect_name().value
        return self.node("Variable", name=name, namespace=namespace)

    # ------------------------------------------------------------- statements
    def parse_statement(self):
        kind = self.tok.kind
        if kind == "MATCH":
            return self.parse_match()
        if kind == "READ":
            return self.parse_read()
        if kind in ("WRITE", "DELETE"):
            return self.parse_write_delete(kind)
        if kind == "BIND":
            return self.parse_bind()
        if kind == "DERIVE":
            return self.parse_derive()
        if kind == "CALL":
            return self.parse_call()
        if kind == "EMIT":
            return self.parse_emit()
        if kind == "SPAWN":
            return self.parse_spawn()
        if kind == "WAIT":
            return self.parse_wait()
        if kind == "BRANCH":
            return self.parse_branch()
        if kind == "PARALLEL":
            return self.parse_parallel()
        if kind == "EACH":
            return self.parse_each()
        if kind == "TRY":
            return self.parse_try()
        if kind == "RETURN":
            return self.parse_return()
        return self.parse_triple()

    def parse_scope(self):
        return self.advance().value if self.tok.kind in ("WORLD", "MEMORY", "WORKING", "HISTORY", "PLAN") else None

    def parse_match(self):
        self.advance()
        scope = self.parse_scope()
        pattern = self.parse_graph_pattern()
        as_var = None
        if self.at("AS"):
            self.advance()
            as_var = self.parse_variable()
        self.expect("DOT")
        return self.node("Match", **{"scope": scope, "pattern": pattern, "as": as_var})

    def parse_graph_pattern(self):
        return self.parse_graph_block()

    def parse_read(self):
        self.advance()
        scope = self.parse_scope()
        triple = self.parse_triple()
        cardinality = self.parse_cardinality()
        as_var = None
        if self.at("AS"):
            self.advance()
            as_var = self.parse_variable()
        self.expect("DOT")
        return self.node("Read", **{"scope": scope, "triple": triple, "cardinality": cardinality, "as": as_var})

    def parse_write_delete(self, kind):
        self.advance()
        scope = self.parse_scope()
        block = self.parse_graph_block()
        self.expect("DOT")
        return self.node(kind.capitalize(), scope=scope, block=block)

    def parse_bind(self):
        self.advance()
        var = self.parse_variable()
        self.expect("=")
        expr = self.parse_expression()
        self.expect("DOT")
        return self.node("Bind", var=var, expression=expr)

    def parse_derive(self):
        self.advance()
        var = self.parse_variable()
        self.expect("=")
        pattern = self.parse_graph_pattern()
        self.expect("USING")
        ref = self.parse_ref()
        self.expect("DOT")
        return self.node("Derive", var=var, pattern=pattern, projection=ref)

    def parse_ref(self):
        if self.at("IRI"):
            return self.parse_iri()
        t = self.expect_ident()
        if self.tokens[self.pos].kind == ":":
            self.advance()
            local = self.advance().value
            return self.node("Iri", prefix=t.value, local=local)
        return self.node("Name", value=t.value)

    def parse_call(self):
        self.advance()
        ref = self.parse_ref()
        self.expect("(")
        args = self.parse_argument_list()
        self.expect(")")
        as_var = None
        if self.at("AS"):
            self.advance()
            as_var = self.parse_variable()
        self.expect("DOT")
        return self.node("Call", **{"capability": ref, "args": args, "as": as_var})

    def parse_argument_list(self):
        args = []
        if not self.at(")"):
            args.append(self.parse_expression())
            while self.at(","):
                self.advance()
                args.append(self.parse_expression())
        return args

    def parse_emit(self):
        self.advance()
        iri = self.parse_iri()
        args = None
        if self.at("("):
            self.advance()
            args = self.parse_argument_list()
            self.expect(")")
        if args is None:
            emit = self.node("Emit", event=iri)
            self.expect("DOT")
            return emit
        emit = self.node("Emit", event=iri, args=args)
        self.expect("DOT")
        return emit

    def parse_spawn(self):
        self.advance()
        ref = self.parse_ref()
        with_block = None
        if self.at("WITH"):
            self.advance()
            with_block = self.parse_graph_block()
        as_var = None
        if self.at("AS"):
            self.advance()
            as_var = self.parse_variable()
        self.expect("DOT")
        return self.node("Spawn", **{"actor": ref, "with": with_block, "as": as_var})

    def parse_wait(self):
        self.advance()
        target = self.parse_wait_target()
        timeout = None
        if self.at("TIMEOUT"):
            self.advance()
            timeout = self.parse_duration()
        self.expect("DOT")
        return self.node("Wait", target=target, timeout=timeout)

    def parse_wait_target(self):
        if self.at("EVENT"):
            self.advance()
            iri = self.parse_iri()
            pattern = None
            if self.at("{"):
                pattern = self.parse_graph_pattern()
            return self.node("EventMatch", event=iri, pattern=pattern)
        return self.parse_condition()

    def parse_duration(self):
        t = self.expect("INTEGER")
        unit = self.expect_name().value
        if unit not in DURATION_UNITS:
            raise IRError(
                f"unknown duration unit {unit!r} (expected one of {sorted(DURATION_UNITS)})",
                t.line,
                t.column,
            )
        return self.node("Duration", value=int(t.value), unit=unit)

    def parse_block(self):
        self.expect("{")
        statements = []
        while not self.at("}"):
            statements.append(self.parse_statement())
        self.expect("}")
        return statements

    def parse_branch(self):
        self.advance()
        self.expect("{")
        clauses = [self.parse_when_clause()]
        while self.at("WHEN"):
            clauses.append(self.parse_when_clause())
        else_block = None
        if self.at("ELSE"):
            self.advance()
            else_block = self.parse_block()
        self.expect("}")
        return self.node("Branch", when=clauses, else_=else_block)

    def parse_when_clause(self):
        self.expect("WHEN")
        condition = self.parse_condition()
        self.expect("THEN")
        return self.node("When", condition=condition, body=self.parse_block())

    def parse_parallel(self):
        self.advance()
        self.expect("{")
        blocks = [self.parse_block()]
        while not self.at("}"):
            blocks.append(self.parse_block())
        self.expect("}")
        return self.node("Parallel", blocks=blocks)

    def parse_each(self):
        self.advance()
        var = self.parse_variable()
        self.expect("IN")
        match = self.parse_match_expression()
        self.expect("DO")
        return self.node("Each", var=var, match=match, body=self.parse_block())

    def parse_match_expression(self):
        self.expect("MATCH")
        scope = self.parse_scope()
        return self.node("MatchExpr", scope=scope, pattern=self.parse_graph_pattern())

    def parse_try(self):
        self.advance()
        body = self.parse_block()
        catch_var = None
        catch_block = None
        if self.at("CATCH"):
            self.advance()
            catch_var = self.parse_variable()
            catch_block = self.parse_block()
        return self.node("Try", body=body, catch_var=catch_var, catch_body=catch_block)

    def parse_return(self):
        self.advance()
        if self.at("{"):
            value = self.parse_graph_block()
            self.expect("DOT")
            return self.node("Return", value=value)
        value = self.parse_expression()
        self.expect("DOT")
        return self.node("Return", value=value)

    # ------------------------------------------------------------- conditions
    def parse_condition(self):
        return self.parse_or()

    def parse_or(self):
        left = self.parse_and()
        while self.at("OR"):
            self.advance()
            left = self.node("Or", left=left, right=self.parse_and())
        return left

    def parse_and(self):
        left = self.parse_not()
        while self.at("AND"):
            self.advance()
            left = self.node("And", left=left, right=self.parse_not())
        return left

    def parse_not(self):
        if self.at("NOT") and self.tokens[self.pos + 1].kind != "IN":
            self.advance()
            return self.node("Not", operand=self.parse_not())
        return self.parse_primary_condition()

    def parse_primary_condition(self):
        if self.at("("):
            self.advance()
            inner = self.parse_condition()
            self.expect(")")
            return inner
        left = self.parse_expression()
        if self.at("NOT") and self.tokens[self.pos + 1].kind == "IN":
            self.advance()
            self.advance()
            return self.node("Comparison", op="NOT IN", left=left, right=self.parse_expression())
        op = self.parse_comparison_operator()
        right = self.parse_expression()
        return self.node("Comparison", op=op, left=left, right=right)

    def parse_comparison_operator(self):
        t = self.tok
        if t.kind in ("=", "!=", "<", "<=", ">", ">="):
            self.advance()
            return t.kind
        if t.kind == "IN":
            self.advance()
            return "IN"
        raise IRError(f"expected comparison operator but found {t.value!r}", t.line, t.column)

    # ------------------------------------------------------------- expressions
    def _is_function_head(self):
        """True when current tokens start a function_expression: ref '('."""
        t1 = self.tok
        t2 = self.tokens[self.pos + 1]
        if t1.kind == "IDENT" and t2.kind == "(":
            return True
        if t1.kind == "IDENT" and t2.kind == ":":
            t3 = self.tokens[self.pos + 2]
            t4 = self.tokens[self.pos + 3]
            return t3.kind != "EOF" and t3.kind.isalpha() and t4.kind == "("
        return False

    def parse_expression(self):
        return self.parse_primary()

    def parse_primary(self):
        t = self.tok
        if t.kind == "[":
            self.advance()
            items = []
            if not self.at("]"):
                items.append(self.parse_expression())
                while self.at(","):
                    self.advance()
                    items.append(self.parse_expression())
            self.expect("]")
            return self.node("List", items=items)
        if t.kind == "IRI" or self._is_function_head():
            if t.kind == "IRI":
                ref = self.parse_iri()
            elif self.tokens[self.pos + 1].kind == "(":
                ref = self.parse_ref()
            else:
                ref = self.parse_iri()
            self.expect("(")
            args = self.parse_argument_list()
            self.expect(")")
            return self.node("FunctionCall", ref=ref, args=args)
        if t.kind == "-":
            nxt = self.tokens[self.pos + 1]
            if nxt.kind == "INTEGER":
                self.advance()
                self.advance()
                return self.node("Literal", value=-int(nxt.value), datatype="xsd:integer")
            if nxt.kind == "DECIMAL":
                self.advance()
                self.advance()
                return self.node("Literal", value=-float(nxt.value), datatype="xsd:decimal")
            raise IRError("expected number after '-'", t.line, t.column)
        return self.parse_term()


def parse(src):
    return Parser(src).parse_document()


def to_json(ast, compact=False):
    return json.dumps(ast, indent=None if compact else 2, sort_keys=False)


def emit(record):
    """Emit one JSONL record to stdout.  This is the tool's only output."""
    print(json.dumps(record))


def main(argv=None):
    """CLI: parse one file (or stdin) and emit exactly one JSONL record.

    Success: ``{"ok": true, "tool": "ir-compile", "path": ..., "ast": {...}}``
    Failure: ``{"ok": false, "tool": "ir-compile", "path": ...|null,
    "errors": [{code, message, ...}]}``

    Exit codes: 0 = parsed, 1 = parse/IO error, 2 = usage error.
    Nothing human-readable is ever printed; stderr stays silent.
    """
    argv = sys.argv if argv is None else argv
    prog = Path(argv[0]).name if argv else "ir-compile"
    args = list(argv)
    # Accepted for backward compatibility; output is always one compact
    # JSONL line regardless.
    args = [a for a in args[1:] if a not in ("--compact", "--json")]
    if "--help" in args or "-h" in args:
        emit({"ok": True, "tool": "ir-compile", "path": None,
              "usage": f"{prog} <file|-|--stdin> [--stdin]"})
        return 0
    use_stdin = "--stdin" in args
    args = [a for a in args if a != "--stdin"]
    if use_stdin or args == ["-"]:
        path, src = "<stdin>", sys.stdin.read()
    elif len(args) == 1:
        path = args[0]
        try:
            src = sys.stdin.read() if path == "-" else Path(path).read_text(
                encoding="utf-8")
        except FileNotFoundError:
            emit({"ok": False, "tool": "ir-compile", "path": path,
                  "errors": [{"code": "E_IO",
                              "message": f"{path}: no such file or directory",
                              "hint": "check the file path"}]})
            return 1
        except OSError as exc:
            emit({"ok": False, "tool": "ir-compile", "path": path,
                  "errors": [{"code": "E_IO", "message": f"{path}: {exc}",
                              "hint": "check file permissions and encoding"}]})
            return 1
    else:
        emit({"ok": False, "tool": "ir-compile", "path": None,
              "errors": [{"code": "E_USAGE",
                          "message": f"usage: {prog} <file|-|--stdin>",
                          "hint": "pass exactly one input file"}]})
        return 2
    try:
        ast = parse(src)
    except IRError as exc:
        emit({"ok": False, "tool": "ir-compile", "path": path,
              "errors": [{"code": "E0_PARSE_ERROR",
                          "message": f"{path}: {exc}",
                          "line": exc.line, "column": exc.column,
                          "hint": "fix the syntax error and re-run"}]})
        return 1
    emit({"ok": True, "tool": "ir-compile", "path": path, "ast": ast})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
