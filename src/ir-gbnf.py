#!/usr/bin/env python3
"""ir-gbnf — Graph IR grammar as GBNF (guided decoding) + mirror generator.

REGIMES R3: emit GBNF (one rule per line, llama.cpp/vLLM compatible) so IR
*syntax* becomes a decode-time guarantee. The same productions are mirrored
in a seeded generator, which is what makes the grammar trustworthy: every
generated sample must parse under src/ir-compile.py (property-tested).

Grammar facts pinned from the tokenizer/parser (NOT guessed, all verified
against ir-compile behaviour):

  READ   ::= 'READ' scope? triple card? ('AS' var)? '.'      # two dots
  MATCH  ::= 'MATCH' scope? block ('AS' var)? '.'
  WRITE/DELETE ::= ('WRITE'|'DELETE') scope? block '.'
  EACH   ::= 'EACH' var 'IN' 'MATCH' scope? block 'DO' block
  PARALLEL arms  ::= 'PARALLEL' '{' block block+ '}'
  BRANCH/TRY/PLAN/PROJECTION blocks carry NO trailing dot
  comparison ops: '=' '!=' '<' ONLY — the lexer cannot produce '>' (not in
      its punctuation set) and '<=' / '>=' are shadowed by the IRI branch
      of '<'; those are parser limitations, not omissions here.
  numeric literals must NOT end a statement (the lexer reads '1.' as a
      DECIMAL); the rules below guard with GBNF negative lookahead (~).

Usage:
  bin/ir-gbnf [--root response|document] [--out FILE]     # print GBNF
  bin/ir-gbnf --sample N [--seed S] [--validate]          # generate programs
"""
import argparse
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── GBNF rules (one rule = one line) ───────────────────────────────────

CORE = {
    "name": r'[a-zA-Z_] [a-zA-Z0-9_-]*',
    "iri": r'"<" [^<>{}" ]* ">" | name ":" name',
    "var": r'"?" name | name ":" "?" name',
    "string": r'"\\"" [^"\\]* "\\""',
    "unsigned": r'[0-9]+ | [0-9]+ "." [0-9]+',
    "integer": r'"-"? [0-9]+ ~[ \n\t]',
    "decimal": r'"-"? [0-9]+ "." [0-9]+ ~[ \n\t]',
    "boolean": r'("true" | "false") ~[ \n\t]',
    "term": r'iri | var | string | decimal | integer | boolean',
    "card": r'"ONE" | "OPTIONAL" | "MANY"',
    "scope": r'"WORLD" | "MEMORY" | "WORKING" | "HISTORY" | "PLAN"',
    "WS": r'[ \n\t]+',
    "wsp": r'[ \n\t]*',
    "args": r'expr (wsp "," wsp expr)*',
    "pattern": r'"{" (wsp term wsp+ term wsp+ term wsp ".")* wsp "}"',
    "triple": r'term wsp+ term wsp+ term wsp "."',
    "exprlist": r'"[" wsp (expr (wsp "," wsp expr)*)? wsp "]"',
    "ref": r'name ":" name | name ~[ \n\t(]',
    "funccall": r'ref wsp* "(" wsp* args? wsp* ")"',
    "expr": r'exprlist | funccall | term',
    "cmpop": r'"=" | "!=" | "<"',
    "cmp": r'expr wsp* cmpop wsp expr | expr wsp "IN" wsp expr '
           r'| expr wsp "NOT" wsp "IN" wsp expr',
    "cond": r'("NOT" wsp cond | "(" wsp* cond wsp* ")" | cmp) '
            r'( wsp ("AND" | "OR") wsp ("NOT" wsp cond | "(" wsp* cond wsp* ")" | cmp))*',
    "block": r'"{" (wsp stmt)* wsp "}"',
    "matchstmt": r'"MATCH" wsp scope? wsp* pattern wsp* ("AS" wsp var wsp*)? "."',
    "readstmt": r'"READ" wsp scope? wsp* triple wsp card? wsp* '
                r'("AS" wsp var wsp*)? "."',
    "writestmt": r'("WRITE" | "DELETE") wsp scope? wsp* pattern wsp "."',
    "bindstmt": r'"BIND" wsp var wsp "=" wsp expr wsp "."',
    "derivestmt": r'"DERIVE" wsp var wsp "=" wsp pattern wsp "USING" wsp ref wsp "."',
    "callstmt": r'"CALL" wsp funccall wsp* ("AS" wsp var wsp*)? "."',
    "emitstmt": r'"EMIT" wsp iri wsp* ("(" wsp* args wsp* ")")? wsp "."',
    "spawnstmt": r'"SPAWN" wsp ref wsp* ("WITH" wsp* pattern wsp*)? '
                 r'("AS" wsp var wsp*)? "."',
    "waitstmt": r'"WAIT" wsp (("EVENT" wsp iri wsp* pattern?) | cond) wsp* '
                r'("TIMEOUT" wsp integer wsp* name wsp*)? "."',
    "whencase": r'"WHEN" wsp cond wsp "THEN" wsp* block',
    "branchstmt": r'"BRANCH" wsp* "{" wsp* whencase (wsp whencase)* '
                  r'(wsp "ELSE" wsp* block)? wsp* "}"',
    "eachstmt": r'"EACH" wsp var wsp "IN" wsp "MATCH" wsp scope? wsp* pattern '
                r'wsp "DO" wsp* block',
    "trystmt": r'"TRY" wsp* block wsp* ("CATCH" wsp var wsp* block)?',
    "parallelstmt": r'"PARALLEL" wsp* "{" wsp block wsp block (wsp block)* wsp "}"',
    "returnstmt": r'"RETURN" wsp* (expr | pattern) wsp "."',
    "stmt": r'matchstmt | readstmt | writestmt | bindstmt | derivestmt '
            r'| callstmt | emitstmt | spawnstmt | waitstmt | branchstmt '
            r'| eachstmt | trystmt | parallelstmt | returnstmt | triple',
}

DOC = {
    "prefixdecl": r'"PREFIX" wsp+ name ":" wsp* iri wsp "."',
    "typedecl": r'"TYPE" wsp+ (name ":" name) wsp* '
                r'("EXTENDS" wsp+ (name ":" name) wsp*)? "."',
    "eventdecl": r'"EVENT" wsp+ iri wsp* (pattern wsp*)? "."',
    "param": r'var wsp* ":" wsp* (name ":" name) wsp* card? wsp*',
    "params": r'param (wsp* "," wsp* param)*',
    "capdecl": r'"CAPABILITY" wsp+ name ":" name wsp* "(" wsp* params? wsp* ")" '
               r'wsp* ("RETURNS" wsp+ (name ":" name) wsp*)? "."',
    "plandecl": r'"PLAN" wsp+ name wsp* ("(" wsp* params wsp* ")")? wsp* block',
    "projectiondecl": r'"PROJECTION" wsp+ name wsp* ("(" wsp* params wsp* ")")? wsp* block',
    "context": r'("WORLD" | "MEMORY" | "GOAL" | "WORKING" | "HISTORY") wsp* pattern',
    "document": r'(wsp (prefixdecl | typedecl | eventdecl | capdecl | plandecl '
                r'| projectiondecl | context | stmt))*',
}

RESP = {"response": r'(wsp stmt)*'}


def build(root="response"):
    rules = dict(CORE)
    rules.update(RESP if root == "response" else DOC)
    return "\n".join(f"{k} ::= {v}" for k, v in rules.items()) + "\n"


def check_refs(gbnf_text):
    """Every rule-like token referenced must be defined (grammar lint)."""
    defined = set(re.findall(r"^(\w+) ::=", gbnf_text, re.M))
    body = re.sub(r'"(?:[^"\\]|\\.)*"', " ", gbnf_text)   # drop literals
    body = re.sub(r"\[[^\]]*\]", " ", body)               # drop char classes
    tokens = set(re.findall(r"\b[a-z][a-z0-9_]*\b", body))
    reserved = {"and", "or", "not", "in"}                  # keyword words
    return sorted(tokens - defined - reserved)


# ── mirror generator (seeded; parse-tested against ir-compile) ────────

SCOPES = ["WORLD", "MEMORY", "WORKING", "HISTORY", "PLAN"]
CARD = ["ONE", "OPTIONAL", "MANY"]
OPS = ["=", "!=", "<"]          # lexer reality — see module docstring


class Gen:
    def __init__(self, rng):
        self.rng = rng
        self.n = 0

    def fresh(self, p="v"):
        self.n += 1
        return f"my:?{p}{self.n}"

    def ent(self):
        return f"my:e{self.rng.randint(1, 9)}"

    def pred(self):
        return f"ai:p{self.rng.randint(1, 9)}"

    def obj(self, depth=0):
        r = self.rng.random()
        if r < 0.4:
            return self.ent()
        if r < 0.6:
            return self.fresh("o")
        if r < 0.75:
            return f'"{self.rng.choice(["alpha", "beta", "gamma"])}"'
        if r < 0.9:
            return f"{self.rng.randint(0, 99)}  "     # lexer: keep dot off numbers
        return self.rng.choice(["true  ", "false  "])

    def triple(self, bindable=False):
        s, p = self.ent(), self.pred()
        o = self.fresh("x") if bindable else self.obj()
        return f"{s} {p} {o} ."

    def pattern(self, n=None):
        n = n or self.rng.randint(1, 2)
        rows = []
        for _ in range(n):
            o = self.obj() if self.rng.random() < .5 else self.fresh("m")
            rows.append(f"  {self.ent()} {self.pred()} {o} .")
        return "{\n" + "\n".join(rows) + "\n}"

    def block(self, depth=0):
        n = 1 if depth >= 2 else self.rng.randint(1, 2)
        pad = "  " * (depth + 1)
        body = "\n".join(pad + self.stmt(depth + 1) for _ in range(n))
        return "{\n" + body + "\n" + "  " * depth + "}"

    def cond(self):
        a = self.ent() if self.rng.random() < .5 else self.fresh("c")
        op = self.rng.choice(OPS)
        b = self.rng.choice([self.ent(), f"{self.rng.randint(1, 50)}  ",
                             '"x"  ', "true  "])
        return f"{a} {op} {b}"

    def expr(self, depth=0):
        r = self.rng.random()
        if r < 0.12 and depth < 2:
            return "[" + ", ".join(self.ent() for _ in range(2)) + "]"
        if r < 0.22 and depth < 2:
            return f"ai:f{self.rng.randint(1, 3)}({self.ent()})"
        return self.obj(depth)

    def stmt(self, depth=0):
        kind = self.rng.choice([
            "read", "match", "write", "delete", "bind", "call", "emit",
            "branch", "each", "return", "triple", "spawn", "wait", "try",
            "parallel", "derive"])
        scope = self.rng.choice(SCOPES[:2])
        if kind == "read":
            return (f"READ {scope} {self.ent()} {self.pred()} {self.fresh('r')} "
                    f". {self.rng.choice(CARD)}.")
        if kind == "match":
            tail = f" AS {self.fresh('h')}" if self.rng.random() < .4 else ""
            return f"MATCH {scope} {self.pattern()}{tail}."
        if kind in ("write", "delete"):
            return f"{kind.upper()} {scope} {self.pattern()}."
        if kind == "bind":
            return f"BIND {self.fresh('b')} = {self.expr(depth)}."
        if kind == "derive":
            return (f"DERIVE {self.fresh('d')} = {self.pattern(1)} "
                    f"USING ai:proj{self.rng.randint(1, 2)}.")
        if kind == "call":
            tail = f" AS {self.fresh('c')}" if self.rng.random() < .4 else ""
            return f"CALL ai:cap{self.rng.randint(1, 3)}({self.expr(depth)}){tail}."
        if kind == "emit":
            args = f"({self.ent()})" if self.rng.random() < .7 else ""
            return f"EMIT ai:ev{self.rng.randint(1, 3)}{args}."
        if kind == "spawn":
            withb = (f"WITH {{ {self.ent()} ai:role {self.ent()} . }} "
                     if self.rng.random() < .5 else "")
            return f"SPAWN ai:worker{self.rng.randint(1, 2)} {withb}AS {self.fresh('w')}."
        if kind == "wait":
            if self.rng.random() < .5:
                return (f"WAIT EVENT ai:ev{self.rng.randint(1,3)} "
                        f"{{ my:x ai:by {self.fresh('w')} . }} "
                        f"TIMEOUT {self.rng.randint(1, 60)}s.")
            return f"WAIT {self.cond()} TIMEOUT {self.rng.randint(1, 60)}s."
        if kind == "branch":
            whens = "\n  ".join(
                f"WHEN {self.cond()} THEN {self.block(depth + 1)}"
                for _ in range(self.rng.randint(1, 2)))
            els = f"\n  ELSE {self.block(depth + 1)}" if self.rng.random() < .6 else ""
            return "BRANCH {\n  " + whens + els + "\n}"
        if kind == "each":
            v = self.fresh("i")
            return (f"EACH {v} IN MATCH {scope} {{ {self.ent()} {self.pred()} {v} . }} "
                    f"DO {self.block(depth + 1)}")
        if kind == "try":
            catch = f" CATCH {self.fresh('e')} {self.block(depth + 1)}" \
                if self.rng.random() < .6 else ""
            return f"TRY {self.block(depth + 1)}{catch}"
        if kind == "parallel":
            return "PARALLEL {\n " + "\n ".join(self.block(depth + 1)
                                                for _ in range(2)) + "\n}"
        if kind == "return":
            return f"RETURN {self.expr(depth)}."
        return self.triple()

    def decl(self):
        kind = self.rng.choice(["type", "event", "capability", "plan",
                                "projection"])
        if kind == "type":
            return (f"TYPE my:T{self.rng.randint(1, 9)}."
                    if self.rng.random() < .7 else
                    f"TYPE my:T{self.rng.randint(1, 9)} "
                    f"EXTENDS my:T{self.rng.randint(1, 9)}.")
        if kind == "event":
            return (f"EVENT ai:ev{self.rng.randint(1, 3)}."
                    if self.rng.random() < .6 else
                    f"EVENT ai:ev{self.rng.randint(1, 3)} "
                    f"{{ my:x ai:by my:?who . }}.")
        if kind == "capability":
            card = self.rng.choice(CARD)
            ret = " RETURNS my:T1." if self.rng.random() < .5 else "."
            return (f"CAPABILITY ai:cap{self.rng.randint(1, 3)}"
                    f"(my:?a : my:T{self.rng.randint(1, 9)} {card}){ret}")
        body = self.block(1) if self.rng.random() < .4 else \
            "{\n  READ WORLD my:x ai:p my:?z . ONE.\n}"
        return f"{kind.upper()} step{self.rng.randint(1, 3)}(my:?p : my:T1) {body}"

    def document(self, n_stmts=3):
        parts = ["PREFIX my: <http://example.org/entity/>.",
                 "PREFIX ai: <http://example.org/property/>."]
        parts += [self.decl() for _ in range(self.rng.randint(0, 2))]
        parts.append(f"{self.rng.choice(['WORLD', 'MEMORY'])} {self.pattern(2)}")
        parts += [self.stmt() for _ in range(n_stmts)]
        return "\n".join(parts)


CTX = ("PREFIX my: <http://example.org/entity/>.\n"
       "PREFIX ai: <http://example.org/property/>.\n")


def generate(root="response", n=1, seed=0):
    rng = random.Random(seed)
    g = Gen(rng)
    if root == "document":
        return [g.document() for _ in range(n)]
    return [CTX + "\n".join(g.stmt() for _ in range(rng.randint(1, 3)))
            for _ in range(n)]


def main(argv=None):
    p = argparse.ArgumentParser(prog="ir-gbnf",
                                description="IR grammar as GBNF + generator")
    p.add_argument("--root", choices=("response", "document"), default="response")
    p.add_argument("--out", help="write GBNF here (default stdout)")
    p.add_argument("--sample", type=int, default=0, help="generate N programs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--validate", action="store_true",
                   help="parse-check samples against src/ir-compile.py")
    args = p.parse_args(argv)

    if args.sample:
        samples = generate(args.root, args.sample, args.seed)
        if args.validate:
            ir = _load("gbnf_ir", "ir-compile.py")
            ok, bad = 0, []
            for s in samples:
                try:
                    ir.parse(s)
                    ok += 1
                except Exception as e:
                    bad.append({"error": str(e), "sample": s[:200]})
            print(json.dumps({"root": args.root, "samples": len(samples),
                              "parse_ok": ok, "parse_fail": bad[:3]}))
            return 0 if not bad else 1
        print(json.dumps(samples, indent=1))
        return 0

    text = build(args.root)
    missing = check_refs(text)
    if args.out:
        Path(args.out).write_text(text)
        print(json.dumps({"out": args.out, "root": args.root,
                          "undefined_refs": missing}))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
