#!/usr/bin/env python3
"""synth — deterministic synthesis engine over Graph IR scenario templates.

FRAMEWORK.md §4b wave 1 (patterns P2, P3-verify, P12, P23, P28, P29, P31).
Zero LLM, zero new dependencies, stdlib only — the engine is a *proposal
machine whose every proposal passes the oracle*:

    select source    templates whose final IR validates under its context
    mutate  (P2)     rulebook of text-level single-violation mutations
    verify           Pass-1 checker: EXACTLY one error, the intended code,
                     zero warnings — SAFEGUARDS law by construction
    emit  (P12/P23)  derived invalid-program / repair / abstain templates
                     + DPO preference pairs; NOTHING lands unverified
    audit (P28/P29)  near-duplicate scan, phrasing entropy, coverage matrix

Oracles are REUSED, never reimplemented: ir-compile.py (parser),
ir-ast-sanity.py (Pass-1 validator), ir-check-scenarios.py (variant
expansion + closed-world Fixture executor for mechanical asserts — P3).

Derived output is regenerable, not curated (FRAMEWORK §10): templates land
under $CORPORA_DIR/generate/derived/; rows/DPO/audit under training tmp.

Usage:
  bin/synth derive [--systems a,b] [--per-code N] [--limit N] [--no-dpo]
  bin/synth abstain [--per-code N]
  bin/synth rows [--task NAME]
  bin/synth audit
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ir = _load("synth_ir", "ir-compile.py")
sanity = _load("synth_sanity", "ir-ast-sanity.py")
check = _load("synth_check", "ir-check-scenarios.py")

# CORPORA_DIR law (expand.py / SPEC §4b): env wins, relative resolves against
# ROOT; else the sibling checkout; else legacy in-repo layout.
_d = os.environ.get("CORPORA_DIR", "").strip()
if _d:
    _p = Path(_d)
    DATA_ROOT = _p if _p.is_absolute() else (ROOT / _p).resolve()
else:
    DATA_ROOT = ROOT.parent / "corpora" if (
        ROOT.parent / "corpora").is_dir() else ROOT
SCEN_ROOT = DATA_ROOT / "generate" / "scenarios"
DERIVED_ROOT = DATA_ROOT / "generate" / "derived"
TMP = ROOT / "tmp" / "synth"

# Source tasks whose learned target is an IR program under its own context.
SOURCE_SYSTEMS = ("nl-to-ir", "valid-program", "program-transform", "semantic-edit")

VARIANT = "qzx"  # renaming pool, never collides with template vars by audit
DIAG_PHRASINGS = [
    "Identify the single Pass-1 error in this program.",
    "What is wrong with this program? Name the error.",
    "Diagnose exactly one structural violation in the program above.",
]
REPAIR_PHRASINGS = [
    "Repair this program with the smallest change that restores validity.",
    "Fix the error in this program. Change as little as possible.",
    "This program is invalid. Return the minimal corrected version.",
    "Correct this program while preserving its original intent.",
]
ABSTAIN_PHRASINGS = [
    "{request}",
    "Answer from the supplied graph only — if the facts are absent, say so.",
    "Resolve the request against the current state; refuse to invent.",
]

ERROR_HINTS = {
    "E1_DUP_PREFIX": "the prefix is declared twice — keep exactly one PREFIX per prefix.",
    "E1_DUP_TYPE": "the type is declared twice — keep a single TYPE declaration per name.",
    "E1_DUP_CAPABILITY": "the capability is declared twice — merge into a single declaration.",
    "E1_DUP_EVENT": "the event is declared twice — keep a single EVENT declaration per IRI.",
    "E1_DUP_PLAN": "the plan is declared twice — keep a single PLAN declaration per name.",
    "E1_DUP_PROJECTION": "the projection is declared twice — keep a single declaration.",
    "E2_UNBOUND": "the variable has no visible binding at its use site.",
    "E3_REDEFINE": "the variable is assigned twice in the same scope.",
    "E3_USE_BEFORE_DEF": "the variable is used before its first binding.",
    "E4_TYPE_CYCLE": "the TYPE EXTENDS chain is cyclic — break the inheritance cycle.",
    "E4_TYPE_MISMATCH": "the comparison mixes incompatible literal datatypes — make both sides the same datatype.",
    "E5_CARD_MISMATCH": "a MANY value is passed where ONE is required — iterate with EACH or select one element first.",
    "E6_UNREACHABLE": "statements follow a RETURN in the same block — move RETURN to the end of the block.",
    "E6_PARALLEL_ARITY": "PARALLEL has fewer than two arms — add another block or drop PARALLEL.",
    "E6_BAD_TIMEOUT": "WAIT TIMEOUT must be positive — use e.g. TIMEOUT 30s.",
    "E7_EFFECT_IN_PROJECTION": "a PROJECTION body performs an effect — projections must be pure queries.",
    "E7_IMMUTABLE_SCOPE": "a WRITE/DELETE targets the immutable HISTORY scope — record history with EMIT instead.",
}


# ── oracle helpers ──────────────────────────────────────────────────────

def validate(text):
    """(ast, report), or (None, None) when the source does not parse."""
    try:
        doc = ir.parse(text)
    except Exception:
        return None, None
    return doc, sanity.validate_ast(doc, strict=True)


def is_valid_program(text):
    doc, rep = validate(text)
    return rep is not None and rep["valid"] and not rep["warnings"]


def single_error(text, code):
    """The error dict iff text fails with EXACTLY one error of `code`,
    zero warnings — else None (the mutation is discarded, never patched)."""
    doc, rep = validate(text)
    if rep is None or rep["valid"] or rep["warnings"]:
        return None
    errs = rep["errors"]
    if len(errs) == 1 and errs[0]["code"] == code:
        return errs[0]
    return None


# ── mutation rulebook (P2) ──────────────────────────────────────────────
#
# Uniform rule contract. Each candidate:
#   (code, mutant_full, mutated_stmts_or_None, var_or_None)
# mutated_stmts is None when the violation lives in the DECL/context tier
# (repair not expressible under the statements-only repair contract →
# diagnosis rows only; never a broken repair template).

def _full(ctx, stmts):
    return ctx.rstrip() + "\n\n" + stmts.strip() + "\n"


def mut_dup_prefix(ctx, stmts):
    """E1_DUP_PREFIX: re-declare the first PREFIX with a different IRI."""
    m = re.search(r"^PREFIX\s+([A-Za-z0-9_\-]+):\s*<([^<>]+)>\s*\.", ctx, re.M)
    if not m:
        return []
    dup = f"PREFIX {m.group(1)}: <{m.group(2)}-dup>."
    mutated = ctx[:m.end()] + "\n" + dup + ctx[m.end():]
    return [("E1_DUP_PREFIX", _full(mutated, stmts), None, None)]


def mut_redefine(ctx, stmts):
    """E3_REDEFINE: duplicate a BIND of the same variable right after itself."""
    out = []
    lines = stmts.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"^\s*BIND\s+my:\?([A-Za-z0-9_\-]+)\s*=", line)
        if not m:
            continue
        mutated = "\n".join(lines[:i + 1] + [line] + lines[i + 1:])
        out.append(("E3_REDEFINE", _full(ctx, mutated), mutated, m.group(1)))
    return out


def mut_unbound(ctx, stmts):
    """E2_UNBOUND: rename ONE true use-site occurrence of a bound variable."""
    out = []
    if f"my:?{VARIANT}" in stmts:
        return out
    for m in re.finditer(r"my:\?([A-Za-z0-9_\-]+)", stmts):
        var = m.group(1)
        line_start = stmts.rfind("\n", 0, m.start()) + 1
        prefix_on_line = stmts[line_start:m.start()]
        # skip binding positions: BIND LHS, AS, EACH — renaming there creates
        # a fresh bind, not an unbound use
        if re.search(r"(BIND|AS|EACH)\s+(my:\?\w+\s*=\s*)?$", prefix_on_line) \
                or re.search(r"\bAS\s+$", prefix_on_line) \
                or re.search(r"BIND\s+$", prefix_on_line):
            continue
        mutated = stmts[:m.start()] + f"my:?{VARIANT}" + stmts[m.end():]
        out.append(("E2_UNBOUND", _full(ctx, mutated), mutated, var))
    return out


def mut_use_before_def(ctx, stmts):
    """E3_USE_BEFORE_DEF: move a BIND below a later SAME-BLOCK line that
    uses its var. Depth-aware (wave 3): the swap stays inside one block so
    it never crosses a closing brace and breaks the structure."""
    out = []
    lines = stmts.split("\n")
    depths = line_depths(lines)
    for i, line in enumerate(lines):
        m = re.match(r"^\s*BIND\s+my:\?([A-Za-z0-9_\-]+)\s*=", line)
        if not m:
            continue
        var = m.group(1)
        for j in range(i + 1, len(lines)):
            # same depth = same block; the closing `}` line has shallower
            # depth and the line that OPENED the block is shallower too
            if depths[j] == depths[i] and lines[j].strip() \
                    and re.search(rf"my:\?{re.escape(var)}\b", lines[j]):
                mutated = "\n".join(lines[:i] + lines[i + 1:j + 1]
                                    + [line] + lines[j + 1:])
                out.append(("E3_USE_BEFORE_DEF", _full(ctx, mutated),
                            mutated, var))
                break
    return out


def line_depths(lines):
    """Brace depth BEFORE each line (line i's own braces change the depth
    for line i+1). A line inside `BRANCH { ... }` has depth 1."""
    depths, d = [], 0
    for line in lines:
        depths.append(d)
        stripped = re.sub(r'"[^"]*"', "", line)        # braces in literals
        for ch in stripped:
            if ch == "{":
                d += 1
            elif ch == "}":
                d -= 1
        d = max(0, d)
    return depths


# ── wave 2 rules: the AST-dependent families ───────────────────────────
#
# Checker mechanics these exploit (ir-ast-sanity): MATCH `AS` binds MANY;
# IMMUTABLE_SCOPES = {HISTORY}; statements after a RETURN in the same block
# are unreachable; PROJECTION bodies must be pure (PURE_STMTS); TYPE EXTENDS
# edges are dfs-checked; duplicate declarations collide by name.

def _has_many_handle(stmts):
    """The `AS my:?x` many-cardinality handle, if the program has one."""
    m = re.search(r"\bAS\s+my:\?([A-Za-z0-9_\-]+)\b", stmts)
    return m.group(1) if m else None


def mut_many_call(ctx, stmts):
    """E5_CARD_MISMATCH: pass the MATCH-AS (MANY) handle to a CALL argument.
    (The comparison route double-fires — _check_term_use AND the comparison
    side check — so only arg/emit positions yield single-error mutants.)"""
    many = _has_many_handle(stmts)
    if not many:
        return []
    out = []
    for m in re.finditer(r"CALL\s+([A-Za-z0-9_\-:]+)\(([^()]*)\)\.", stmts):
        args = [a.strip() for a in m.group(2).split(",") if a.strip()]
        if not args or f"my:?{many}" in m.group(0):
            continue
        swapped = (f"CALL {m.group(1)}(my:?{many}"
                   + "".join(f", {a}" for a in args[1:]) + ").")
        mutated = stmts[:m.start()] + swapped + stmts[m.end():]
        out.append(("E5_CARD_MISMATCH", _full(ctx, mutated), mutated, many))
    return out


def mut_many_emit(ctx, stmts):
    """E5_CARD_MISMATCH: EMIT with the MANY handle (events carry singles)."""
    many = _has_many_handle(stmts)
    if not many:
        return []
    out = []
    for m in re.finditer(r"EMIT\s+([A-Za-z0-9_\-:]+)\(([^()]*)\)\.", stmts):
        if f"my:?{many}" == m.group(2).strip():
            continue
        mutated = stmts[:m.start()] + f"EMIT {m.group(1)}(my:?{many})." + stmts[m.end():]
        out.append(("E5_CARD_MISMATCH", _full(ctx, mutated), mutated, many))
    return out


LIT_NUM = re.compile(r"-?\d+(?:\.\d+)?")
LIT_STR = re.compile(r'"[^"]*"')


def mut_lit_flip(ctx, stmts):
    """E4_TYPE_MISMATCH: flip the datatype of a literal bound into exactly
    one comparison whose other operand is itself a literal."""
    out = []
    lines = stmts.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"^\s*BIND\s+my:\?([A-Za-z0-9_\-]+)\s*=\s*([^.;]+?)\s*\.\s*$",
                     line)
        if not m:
            continue
        var, expr = m.group(1), m.group(2)
        hits = []
        for j, other in enumerate(lines):
            if j == i or not re.search(rf"my:\?{re.escape(var)}\b", other):
                continue
            if not re.search(r"[=!<>]=?", other):
                continue
            op_side = re.search(rf"(?:[=!<>]=?)\s*([^\s,;()]+)", other)
            if not op_side:
                continue
            tok = op_side.group(1)
            if tok.startswith("my:?") and var in tok:
                continue
            hits.append((j, tok))
        if len(hits) != 1:
            continue
        other_tok = hits[0][1]
        if not (LIT_NUM.fullmatch(other_tok) or LIT_STR.fullmatch(other_tok)):
            continue
        if LIT_NUM.fullmatch(expr):
            flipped = f"    BIND my:?{var} = \"1\"."
        elif LIT_STR.fullmatch(expr):
            flipped = f"    BIND my:?{var} = 1."
        else:
            continue
        mutated = "\n".join(lines[:i] + [flipped] + lines[i + 1:])
        out.append(("E4_TYPE_MISMATCH", _full(ctx, mutated), mutated, var))
    return out


def mut_type_cycle(ctx, stmts):
    """E4_TYPE_CYCLE: make a plain TYPE self-extending."""
    out = []
    for m in re.finditer(r"^TYPE\s+(my:[A-Za-z0-9_\-]+)\s*\.\s*$", ctx, re.M):
        mutated = (ctx[:m.start()] + f"TYPE {m.group(1)} EXTENDS {m.group(1)}."
                   + ctx[m.end():])
        out.append(("E4_TYPE_CYCLE", _full(mutated, stmts), None, None))
    return out


def mut_immutable_write(ctx, stmts):
    """E7_IMMUTABLE_SCOPE: redirect a WRITE/DELETE block at HISTORY."""
    out = []
    for m in re.finditer(
            r"^(\s*)(WRITE|DELETE)\s+(WORLD|MEMORY|GOAL|WORKING)\s*\{", stmts, re.M):
        mutated = (stmts[:m.start()] + m.group(1) + m.group(2)
                   + " HISTORY {" + stmts[m.end():])
        out.append(("E7_IMMUTABLE_SCOPE", _full(ctx, mutated), mutated, None))
    return out


DEAD_PROJ = ("\nPROJECTION synthprobe {\n"
             "    WRITE WORLD {\n        my:sink ai:probe my:sunk .\n    }.\n}")


def mut_effect_in_projection(ctx, stmts):
    """E7_EFFECT_IN_PROJECTION: append a PROJECTION whose body is an effect.
    Universal — needs nothing from the source program. Declaration tier."""
    if "synthprobe" in ctx or "synthprobe" in stmts:
        return []
    return [("E7_EFFECT_IN_PROJECTION", _full(ctx + DEAD_PROJ, stmts), None, None)]


def mut_dead_after_return(ctx, stmts):
    """E6_UNREACHABLE: live statement after a RETURN in the same block."""
    if re.search(r"my:\?dead\b", stmts):
        return []
    out = []
    body = stmts.strip()
    if re.search(r"RETURN[^{}\n]*\.?\s*$", body):
        # ends with a RETURN (possibly after a closing brace line)
        mutated = body + "\nBIND my:?dead = 1."
        out.append(("E6_UNREACHABLE", _full(ctx, mutated), mutated, None))
    mutated = body + "\nRETURN 1.\nBIND my:?dead = 1."
    out.append(("E6_UNREACHABLE", _full(ctx, mutated), mutated, None))
    return out


DUP_CODES = {"TYPE": "E1_DUP_TYPE", "CAPABILITY": "E1_DUP_CAPABILITY",
             "EVENT": "E1_DUP_EVENT", "PLAN": "E1_DUP_PLAN",
             "PROJECTION": "E1_DUP_PROJECTION"}


def mut_dup_decl(ctx, stmts):
    """E1_DUP_*: restate a single-line declaration verbatim. PLAN/PROJECTION
    carry blocks and rarely match the one-liner shape; oracle filters."""
    out = []
    for kind, code in DUP_CODES.items():
        for m in re.finditer(rf"^{kind}\s+[^\n{{}}]*\.\s*$", ctx, re.M):
            line = m.group(0).rstrip()
            mutated = ctx[:m.end()] + "\n" + line + ctx[m.end():]
            out.append((code, _full(mutated, stmts), None, None))
    return out


# ── wave 4 rules: shape-free inject families ───────────────────────────
#
# E4/E5 and block-declaration dups depend on fixture shapes (BIND-literal
# comparisons, MATCH-AS handles, PLAN/PROJECTION declarations) the curated
# tree lacks. These rules INJECT the shape as part of the proposal — the
# oracle still disposes every row. Grammar facts (verified against
# ir-compile): PROJECTION/PLAN are `NAME(params) { body }` with NO trailing
# dot; MATCH ... AS binds MANY; comparisons reject mixed literal dtypes.

CARD_CTX = "\nCAPABILITY ai:synkcap(my:?a : my:synkt ONE)."
CARD_STMTS = ("MATCH WORLD {\n    my:synke9 ai:p9 my:?x9 .\n} AS my:?m9.\n"
              "CALL ai:synkcap(my:?m9).")
TYPE_STMTS = ('BIND my:?e4 = 1.\nBRANCH {\n    WHEN my:?e4 < "synth" THEN {\n'
              '        RETURN my:synke4.\n    }\n}')
# TYPE is injected ONCE; only the block declaration is restated, else the
# mutant carries E1_DUP_TYPE alongside the intended error.
DUP_TYPE = "TYPE synkt9."
DUP_PROJ = "PROJECTION synq(my:?a : synkt9) {\n    RETURN my:?a.\n}"
DUP_PLAN = "PLAN synp(my:?a : synkt9) {\n    RETURN my:?a.\n}"


def mut_inject_card(ctx, stmts):
    """E5_CARD_MISMATCH: injected CAPABILITY (ONE param) called with the
    injected MATCH-AS handle (MANY)."""
    if "synkcap" in ctx or "my:?m9" in stmts:
        return []
    mutated = CARD_STMTS + "\n" + stmts.strip()
    return [("E5_CARD_MISMATCH", _full(ctx + CARD_CTX, mutated), mutated, "m9")]


def mut_inject_type(ctx, stmts):
    """E4_TYPE_MISMATCH: injected integer BIND compared against a string
    literal inside an injected guard."""
    if "my:?e4" in stmts:
        return []
    mutated = TYPE_STMTS + "\n" + stmts.strip()
    return [("E4_TYPE_MISMATCH", _full(ctx, mutated), mutated, "e4")]


def mut_dup_blockdecl(ctx, stmts):
    """E1_DUP_{PROJECTION,PLAN}: inject a block-carrying declaration and
    restate it verbatim — the dup family line-oriented rules cannot see."""
    out = []
    if "synq" not in ctx:
        double = (ctx + "\n" + DUP_TYPE + "\n" + DUP_PROJ + "\n" + DUP_PROJ)
        out.append(("E1_DUP_PROJECTION", _full(double, stmts), None, None))
    if "synp" not in ctx:
        double = (ctx + "\n" + DUP_TYPE + "\n" + DUP_PLAN + "\n" + DUP_PLAN)
        out.append(("E1_DUP_PLAN", _full(double, stmts), None, None))
    return out


def mut_dead_in_block(ctx, stmts):
    """E6_UNREACHABLE inside a nested block: a live statement inserted
    right after a RETURN that ends a WHEN/ELSE arm (depth >= 1)."""
    if re.search(r"my:\?dead\b", stmts):
        return []
    out = []
    lines = stmts.split("\n")
    depths = line_depths(lines)
    for i, line in enumerate(lines):
        if depths[i] >= 1 and re.match(r"^\s*RETURN\s", line):
            ind = re.match(r"^\s*", line).group(0)
            mutated = "\n".join(lines[:i + 1] + [f"{ind}BIND my:?dead = 1."]
                                + lines[i + 1:])
            out.append(("E6_UNREACHABLE", _full(ctx, mutated), mutated, None))
    return out


PARALLEL_INJECT = ("PARALLEL {\n"
                   "    { READ WORLD my:sink ai:probeA my:?pp . ONE. }\n"
                   "    { READ WORLD my:sink ai:probeB my:?qq . ONE. }\n}")
PARALLEL_BROKEN = ("PARALLEL {\n"
                   "    { READ WORLD my:sink ai:probeA my:?pp . ONE. }\n}")
WAIT_INJECT = "WAIT EVENT ai:syntheticAlarm TIMEOUT 30s."
WAIT_BROKEN = "WAIT EVENT ai:syntheticAlarm TIMEOUT 0s."


def mut_parallel_arity(ctx, stmts):
    """E6_PARALLEL_ARITY: prepend a VALID two-arm PARALLEL, then remove an
    arm — inject-then-break, so the base shape never has to exist in the
    corpus. Prepend (not append): after a top-level RETURN anything is
    unreachable and would double-fire."""
    if "my:?pp" in stmts or "my:?qq" in stmts:
        return []
    return [("E6_PARALLEL_ARITY",
             _full(ctx, PARALLEL_BROKEN + "\n" + stmts.strip()),
             PARALLEL_BROKEN + "\n" + stmts.strip(), None)]


def mut_bad_timeout(ctx, stmts):
    """E6_BAD_TIMEOUT: WAIT ... TIMEOUT must be positive; inject a valid
    WAIT, then zero the timeout."""
    if "syntheticAlarm" in stmts or "syntheticAlarm" in ctx:
        return []
    return [("E6_BAD_TIMEOUT",
             _full(ctx, WAIT_BROKEN + "\n" + stmts.strip()),
             WAIT_BROKEN + "\n" + stmts.strip(), None)]


RULEBOOK = (mut_dup_prefix, mut_redefine, mut_unbound, mut_use_before_def,
            # wave 2
            mut_many_call, mut_many_emit, mut_lit_flip, mut_type_cycle,
            mut_immutable_write, mut_effect_in_projection,
            mut_dead_after_return, mut_dup_decl,
            # wave 3
            mut_dead_in_block, mut_parallel_arity, mut_bad_timeout,
            # wave 4
            mut_inject_card, mut_inject_type, mut_dup_blockdecl)


# ── template construction ───────────────────────────────────────────────

def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48]


def make_template(task, sid, code, messages, assertion, extra=None):
    name = f"{slug(sid)}--{code.lower()}"
    rec = {
        "comment": f"derived by synth from {sid} (FRAMEWORK P2/P12)",
        "label": f"{task} derived {code} from {sid}",
        "id": f"derived/{task}/{name}",
        "system": task,
        "variants": {"phrasing": (DIAG_PHRASINGS if task == "invalid-program"
                                  else REPAIR_PHRASINGS if task == "repair"
                                  else ABSTAIN_PHRASINGS)},
        "messages": messages,
        "assert": assertion,
    }
    if extra:
        rec.update(extra)
    return rec


def execution_assert(doc):
    """P3 law: computes the semantic assert from the program — never types it."""
    fixture = check.Fixture()
    try:
        delta = fixture.run(doc)
    except Exception:
        return None
    assertion = {"valid": True}
    for key, value in delta.items():
        if key == "returned" and value is None:
            continue
        if key.endswith(("_add", "_delete")) and not value:
            continue
        if key == "emitted":
            assertion[key] = value
            continue
        assertion[key] = value
    return assertion


# ── derived preflight (same laws as ir-check-scenarios, derived-aware) ──

def preflight(rec):
    """Every expansion must satisfy its task contract. Returns error list."""
    failures = []
    for idx, (messages, expected) in enumerate(check.expansions(rec)):
        try:
            roles = [m["role"] for m in messages]
            assert roles == ["assistant", "user", "assistant"], roles
            assert all(m["content"].strip() for m in messages)
            first, user, last = [m["content"] for m in messages]
            task = rec["system"]
            if task == "invalid-program":
                doc, rep = validate(first)
                assert rep is not None, "mutant must parse"
                assert rep["valid"] is False and len(rep["errors"]) == 1, rep["errors"]
                assert rep["errors"][0]["code"] == expected["errors"][0]["code"]
                assert not rep["warnings"], rep["warnings"]
                assert expected["errors"][0]["code"] in last, "diagnosis/assert disagree"
                assert not re.search(r"# (?:ERROR|BREAKS)", first), "diagnosis leaked"
            elif task == "repair":
                doc, rep = validate(first)
                assert rep is not None
                errs = rep["errors"]
                assert len(errs) == 1 and errs[0]["code"] != "E0_PARSE_ERROR", errs
                assert errs[0]["code"] == rec["input_error"]
                assert "ERROR " + errs[0]["code"] in first, "error label missing"
                fixed = check.document(check.context_nodes(doc)
                                       + ir.parse(last)["body"])
                rep2 = sanity.validate_ast(fixed, strict=True)
                assert rep2["valid"] and not rep2["warnings"], rep2["errors"]
            elif task == "abstain":
                doc, rep = validate(first)
                assert rep is not None and rep["valid"] and not rep["warnings"], rep
                assert last.strip().startswith("NOT DERIVABLE")
                # determinism: the recorded program must fail on the reduced
                # context via the fixture (exactly one MATCH/READ with 0 rows)
                stmts = expected["program"]
                combined = check.document(check.context_nodes(doc)
                                          + ir.parse(stmts)["body"])
                try:
                    check.Fixture().run(combined)
                    raise AssertionError("expected non-derivability, ran fine")
                except AssertionError as exc:
                    assert "0 rows" in str(exc) or "expected non-derivability" in str(exc), exc
                    if "expected non-derivability" in str(exc):
                        raise
            else:
                raise AssertionError(f"unknown derived task {task}")
        except Exception as exc:
            failures.append({"variant": idx, "error": str(exc)})
    return failures


def write_template(rec, out_root):
    path = out_root / rec["system"] / (Path(rec["id"]).name + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    return path


# ── sources ─────────────────────────────────────────────────────────────

def iter_sources(scen_root, systems, limit):
    """Yield (record, messages, stmts, full_text) per template-expansion
    whose final IR validates under its own context."""
    count = 0
    for path in sorted(scen_root.rglob("*.json")):
        if "derived" in path.parts:
            continue
        try:
            rec = json.loads(path.read_text())
        except Exception:
            continue
        if rec.get("system") not in systems:
            continue
        for messages, _expected in check.expansions(rec):
            first, stmts = messages[0]["content"], messages[2]["content"]
            full = _full(first, stmts)
            if not is_valid_program(full):
                continue
            count += 1
            if limit and count > limit:
                return
            yield rec, messages, first, stmts, full


# ── derive (P2 + P23) ───────────────────────────────────────────────────

def cmd_derive(args):
    systems = tuple(args.systems.split(",")) if args.systems else SOURCE_SYSTEMS
    scen_root = Path(args.scenarios) if args.scenarios else SCEN_ROOT
    out_root = Path(args.out) if args.out else DERIVED_ROOT
    tmp = Path(args.tmp) if args.tmp else TMP
    tmp.mkdir(parents=True, exist_ok=True)
    stats = {"sources": 0, "candidates": 0, "accepted": {},
             "no_repair": {}, "preflight_fail": [], "by_template": {}}
    codes = set(args.codes.split(",")) if args.codes else None
    # accumulate accepted mutants per (template, code): ONE file per cell,
    # variants as correlated fixture pools (VARIANTS.md law — mutant, var
    # and diagnosis travel together; the phrasing pool multiplies on top)
    accepted_ip = {}   # (sid, code) -> [ {"mutant":..., "var":...} ]
    accepted_rp = {}   # (sid, code) -> [ {"mutant":..., "stmts":...} ]
    dpo_path = tmp / "dpo.jsonl"
    dpo = dpo_path.open("w") if not args.no_dpo else None
    try:
        for rec, messages, first, stmts, full in iter_sources(
                scen_root, systems, args.limit):
            stats["sources"] += 1
            sid = rec["id"]
            for rule in RULEBOOK:
                for code, mutant_full, mutated_stmts, var in rule(first, stmts):
                    if codes and code not in codes:
                        continue
                    stats["candidates"] += 1
                    err = single_error(mutant_full, code)
                    if err is None:
                        continue
                    # dedupe structurally identical mutants (same code+var)
                    pool = accepted_ip.setdefault((sid, code), [])
                    if any(e["var"] == var and e["mutant"] == mutant_full
                           for e in pool):
                        continue
                    if len(pool) >= args.per_code:
                        continue
                    pool.append({"mutant": mutant_full, "var": var})
                    stats["accepted"][code] = stats["accepted"].get(code, 0) + 1
                    if mutated_stmts is not None:
                        accepted_rp.setdefault((sid, code), []).append(
                            {"mutant": mutant_full, "stmts": stmts.strip(),
                             "msg": err["message"]})
                        # DPO requires the shared context to be IDENTICAL
                        # on both sides: inject rules that edit the ctx
                        # (CAPABILITY decl etc.) would make the rejected
                        # side valid against the ORIGINAL ctx — skip those.
                        if dpo and mutant_full == _full(first, mutated_stmts):
                            dpo.write(json.dumps({
                                "messages": messages[:2] + [
                                    {"role": "assistant", "content": stmts}],
                                "rejected": {"messages": messages[:2] + [
                                    {"role": "assistant",
                                     "content": mutated_stmts}]},
                            }, ensure_ascii=False) + "\n")
                    else:
                        stats["no_repair"][code] = stats["no_repair"].get(code, 0) + 1
        # ── emit one verified template per (template, code) cell ──
        for (sid, code), pool in sorted(accepted_ip.items()):
            stats["by_template"][sid] = stats["by_template"].get(sid, 0) + 1
            var = pool[0]["var"]
            diag = (f"INVALID: {code} — variable my:?{var} is illegal "
                    f"here. {ERROR_HINTS[code]}" if var
                    else f"INVALID: {code} — {ERROR_HINTS[code]}")
            assertion = {"valid": False, "errors": [
                {"code": code, "variable": f"my:?{var}"} if var
                else {"code": code}]}
            ip = make_template("invalid-program", sid, code, [
                {"role": "assistant", "content": "{domain.mutant}"},
                {"role": "user", "content": "{phrasing}"},
                {"role": "assistant", "content": diag},
            ], assertion)
            ip["variants"]["domain"] = pool
            if preflight(ip):
                stats["preflight_fail"].append({"id": ip["id"],
                                                "task": "invalid-program"})
                continue
            write_template(ip, out_root)
        for (sid, code), pool in sorted(accepted_rp.items()):
            rp = make_template("repair", sid, code, [
                {"role": "assistant",
                 "content": "{domain.mutant}\n\n# ERROR " + code
                            + ": {domain.msg}"},
                {"role": "user", "content": "{phrasing}"},
                {"role": "assistant", "content": "{domain.stmts}"},
            ], {"valid": True}, extra={"input_error": code})
            rp["variants"]["domain"] = pool
            if preflight(rp):
                stats["preflight_fail"].append({"id": rp["id"],
                                                "task": "repair"})
                continue
            write_template(rp, out_root)
    finally:
        if dpo:
            dpo.close()
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    return 0


# ── abstain (P12) ───────────────────────────────────────────────────────

TRIPLE_RE = re.compile(r"^\s+\S+\s+\S+\s+\S+\s*\.\s*$")


def cmd_abstain(args):
    scen_root = Path(args.scenarios) if args.scenarios else SCEN_ROOT
    out_root = Path(args.out) if args.out else DERIVED_ROOT
    stats = {"templates": 0, "accepted": 0, "preflight_fail": []}
    seen = set()
    for path in sorted(scen_root.rglob("*.json")):
        if "derived" in path.parts:
            continue
        try:
            rec = json.loads(path.read_text())
        except Exception:
            continue
        if rec.get("system") != "nl-to-ir" or rec["id"] in seen:
            continue
        seen.add(rec["id"])
        stats["templates"] += 1
        accepted = 0
        for messages, _exp in check.expansions(rec):
            if accepted >= args.per_code:
                break
            first, user, stmts = (m["content"] for m in messages)
            doc, rep = validate(first)
            if rep is None or not rep["valid"]:
                continue
            lines = first.split("\n")
            for i, line in enumerate(lines):
                if not TRIPLE_RE.match(line):
                    continue
                reduced = "\n".join(lines[:i] + lines[i + 1:])
                rdoc, rrep = validate(reduced)
                if rrep is None or not rrep["valid"] or rrep["warnings"]:
                    continue
                combined = check.document(check.context_nodes(rdoc)
                                          + ir.parse(stmts)["body"])
                try:
                    check.Fixture().run(combined)
                    continue                       # still derivable → wrong row
                except AssertionError as exc:
                    if "0 rows" not in str(exc):
                        continue
                except Exception:
                    continue
                ab = make_template("abstain", rec["id"], "ABSTAIN", [
                    {"role": "assistant", "content": reduced},
                    {"role": "user", "content": "{phrasing}"},
                    {"role": "assistant", "content":
                     "NOT DERIVABLE — the fact required to answer is absent "
                     "from the supplied graph. Name what is missing; do not "
                     "invent facts."},
                ], {"not_derivable": True, "program": stmts.strip(),
                    "removed": line.strip()})
                ab["variants"] = {"phrasing": [
                    ABSTAIN_PHRASINGS[0].format(request=user),
                    ABSTAIN_PHRASINGS[1],
                    ABSTAIN_PHRASINGS[2],
                ]}
                fails = preflight(ab)
                if fails:
                    stats["preflight_fail"].append({"id": ab["id"], "errors": fails})
                    continue
                write_template(ab, out_root)
                stats["accepted"] += 1
                accepted += 1
                break
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    return 0


# ── rows: derived templates → expanded chat-envelope jsonl ─────────────

def cmd_rows(args):
    out_root = Path(args.out) if args.out else DERIVED_ROOT
    tmp = TMP if not args.rows_out else Path(args.rows_out)
    per_system = {}
    for path in sorted(out_root.rglob("*.json")):
        rec = json.loads(path.read_text())
        if args.task and rec["system"] != args.task:
            continue
        rows = per_system.setdefault(rec["system"], [])
        for messages, _expected in check.expansions(rec):
            rows.append(messages)
    written = {}
    for system, rows in sorted(per_system.items()):
        dest = tmp / "rows" / f"{system}.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w") as fh:
            for messages in rows:
                fh.write(json.dumps({"messages": messages},
                                    ensure_ascii=False) + "\n")
        written[system] = len(rows)
    print(json.dumps(written, indent=1))
    return 0


# ── graph bridge (P4 round-trip QA, P6 hop-ladder, P12 abstention) ─────
#
# Real ingested TTL graphs become Graph IR scenario templates: every fact
# yields a lookup task whose answer is RECOMPUTED by the fixture executor
# (never typed), every entity chain yields a 2-hop task, and dropping a
# fact yields a NOT-DERIVABLE task. Templates land under
# derived/graph/<system>/ and pass the same preflight as mutations.

RDF_PREFIX = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>."

QA_PHRASINGS = [
    "What is the {domain.plabel} of the {domain.slabel}?",
    "Find the {domain.slabel}'s {domain.plabel}.",
    "Read the {domain.plabel} for {domain.slabel}.",
]
HOP_PHRASINGS = [
    "Follow the {domain.p1label} of the {domain.slabel}, then its "
    "{domain.p2label}; return both.",
    "From {domain.slabel}: give its {domain.p1label} and that value's "
    "{domain.p2label}, in that order.",
]


def parse_ttl(text):
    """Minimal Turtle subset: PREFIX lines + plain triple lines whose terms
    are qnames or literals (predicate/object lists, @base, full IRIs are
    skipped and counted). Returns (prefix_lines, facts, skipped, rdf_used)
    where facts are (subj, pred, obj_ir, obj_native) — obj_ir is the IR
    rendering (quoted strings), obj_native the typed Python value."""
    prefixes, facts, skipped, rdf_used = {}, [], 0, False

    def qname(term):
        return re.match(r"^[A-Za-z_][\w\-]*:[\w\-]+$", term)

    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^@?PREFIX\s+(\S+):\s*(\S+?)\s*\.?$", line, re.I)
        if m:
            prefixes[m.group(1)] = m.group(2).strip("<>")
            lines.append(f"PREFIX {m.group(1)}: <{prefixes[m.group(1)]}>.")
            continue
        if line.startswith("@") or ";" in line or "," in line:
            skipped += 1
            continue
        m = re.match(r"^(\S+)\s+(\S+)\s+(.+?)\s*\.\s*$", line)
        if not m:
            skipped += 1
            continue
        s, p, o = m.groups()
        if p == "a":
            p, rdf_used = "rdf:type", True
        if not (qname(s) and qname(p) and qname(o)):
            # literal object?
            lm = re.match(r'^"(.*)"(?:@\w+|\^\^\S+)?$', o)
            nm = re.match(r"^-?\d+(\.\d+)?$", o)
            bm = re.match(r"^(true|false)$", o)
            if not (qname(s) and qname(p) and (lm or nm or bm)):
                skipped += 1
                continue
            if lm:
                val = lm.group(1)
                obj_ir = f'"{val}"'
                if re.fullmatch(r"-?\d+", val):
                    native = int(val)
                elif re.fullmatch(r"-?\d+\.\d+", val):
                    native = float(val)
                elif val in ("true", "false"):
                    native = val == "true"
                else:
                    native = val
            elif nm:
                obj_ir, native = o, float(o) if "." in o else int(o)
            else:
                obj_ir, native = o, o == "true"
            facts.append((s, p, obj_ir, native))
            continue
        if qname(o):
            facts.append((s, p, o, None))

    if rdf_used and "rdf" not in prefixes:
        lines.insert(0, RDF_PREFIX)
    return lines, facts, skipped, rdf_used


def ttl_context(prefix_lines, triples):
    """prefix_lines + a WORLD block over (s, p, o_ir, _native) facts."""
    return ("\n".join(prefix_lines) + "\nWORLD {\n"
            + "".join(f"    {s} {p} {o} .\n" for s, p, o, _ in triples)
            + "}\nMEMORY {}")


def label(term):
    return term.split(":", 1)[1].replace("_", " ").lower()


def cmd_graph(args):
    src = Path(args.ttl)
    files = [src] if src.is_file() else sorted(src.rglob("*.ttl"))
    out_root = Path(args.out) if args.out else DERIVED_ROOT
    stats = {"files": 0, "templates": 0, "facts": 0, "skipped": 0,
             "preflight_fail": []}
    for f in files:
        stats["files"] += 1
        domain = args.domain or slug(f.stem)
        prefix_lines, facts, skipped, _rdf = parse_ttl(f.read_text())
        stats["skipped"] += skipped
        if not facts:
            continue
        gsha = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
        stats["facts"] += len(facts)
        ctx = ttl_context(prefix_lines, facts)
        stem = slug(f.stem)
        entries = []
        for s, p, o, native in facts[:args.max_facts]:
            entries.append({
                "subj": s, "pred": p, "obj": o if native is None else native,
                "slabel": label(s), "plabel": label(p),
            })
        if entries:
            stmts = ("READ WORLD {domain.subj} {domain.pred} my:?v . ONE.\n"
                     "RETURN my:?v.")
            rec = make_template("nl-to-ir", f"graph.{domain}.{stem}--qa",
                                "QA", [
                {"role": "assistant", "content": ctx},
                {"role": "user", "content": "{phrasing}"},
                {"role": "assistant", "content": stmts},
            ], {"valid": True, "returned": "{domain.obj}",
                "world_add": [], "world_delete": [],
                "memory_add": [], "memory_delete": [], "emitted": []})
            rec["id"] = f"graph/{domain}/{stem}--qa"
            rec["comment"] = f"graph QA over {f.name}@{gsha} (FRAMEWORK P4)"
            rec["variants"] = {"domain": entries, "phrasing": QA_PHRASINGS}
            if _verify_scenario(rec):
                write_template(rec, out_root)
                stats["templates"] += 1
            else:
                stats["preflight_fail"].append(rec["id"])
        # hop-2: s -> m -> o chains through shared entity nodes
        by_subj = {}
        for s, p, o, native in facts:
            by_subj.setdefault(s, []).append((p, o, native))
        hops = []
        for s, p1, m, _n1 in facts:
            if _n1 is not None or m not in by_subj or m == s:
                continue
            for p2, o, native2 in by_subj[m]:
                if o != m and len(hops) < args.max_facts:
                    hops.append({"subj": s, "p1": p1, "mid": m, "p2": p2,
                                 "obj": o if native2 is None else native2,
                                 "slabel": label(s),
                                 "p1label": label(p1), "p2label": label(p2)})
        if hops:
            stmts = ("READ WORLD {domain.subj} {domain.p1} my:?m . ONE.\n"
                     "READ WORLD my:?m {domain.p2} my:?o . ONE.\n"
                     "BIND my:?pair = [my:?m, my:?o].\n"
                     "RETURN my:?pair.")
            rec = make_template("nl-to-ir", f"graph.{domain}.{stem}--hops",
                                "HOPS", [
                {"role": "assistant", "content": ctx},
                {"role": "user", "content": "{phrasing}"},
                {"role": "assistant", "content": stmts},
            ], {"valid": True, "returned": ["{domain.mid}", "{domain.obj}"],
                "world_add": [], "world_delete": [],
                "memory_add": [], "memory_delete": [], "emitted": []})
            rec["id"] = f"graph/{domain}/{stem}--hops"
            rec["comment"] = f"hop-ladder over {f.name}@{gsha} (FRAMEWORK P6)"
            rec["variants"] = {"domain": hops, "phrasing": HOP_PHRASINGS}
            if _verify_scenario(rec):
                write_template(rec, out_root)
                stats["templates"] += 1
            else:
                stats["preflight_fail"].append(rec["id"])
        # CFKG (REGIMES R2): counterfactual edits — rewrite ONE fact, the
        # right answer is what the EDITED graph says, never what memory
        # says. Trains graph-reading over parametric priors — the skill
        # that makes retrieval-grounded serving actually hold (legal/
        # construction facts CHANGE). Alt values preserve the literal type
        # so the fixture's typed assert comparison holds.
        qnames = sorted({o for _s, _p, o, native in facts if native is None})
        cf_entries = []
        for i, (s, p, o_ir, native) in enumerate(facts[:args.max_facts]):
            if native is None:                       # qname object
                pool = [q for q in qnames if q != o_ir]
                alt = alt_ir = pool[i % len(pool)] if pool else f"my:cf{i}"
            elif isinstance(native, bool):
                alt = not native
                alt_ir = "true" if alt else "false"
            elif isinstance(native, (int, float)):
                alt = native + 100
                alt_ir = str(alt)
            else:                                    # string literal
                alt = f"{native}-cf"
                alt_ir = f'"{alt}"'
            edited = [t if k != i else (s, p, alt_ir,
                                        None if native is None else alt)
                      for k, t in enumerate(facts)]
            cf_entries.append({"ctx": ttl_context(prefix_lines, edited),
                               "subj": s, "pred": p, "obj": alt,
                               "orig": o_ir, "slabel": label(s),
                               "plabel": label(p)})
        if cf_entries:
            rec = make_template("nl-to-ir", f"graph.{domain}.{stem}--cfkg",
                                "CFKG", [
                {"role": "assistant", "content": "{domain.ctx}"},
                {"role": "user", "content": "{phrasing}"},
                {"role": "assistant", "content":
                 "READ WORLD {domain.subj} {domain.pred} my:?v . ONE.\n"
                 "RETURN my:?v."},
            ], {"valid": True, "returned": "{domain.obj}",
                "world_add": [], "world_delete": [],
                "memory_add": [], "memory_delete": [], "emitted": []})
            rec["id"] = f"graph/{domain}/{stem}--cfkg"
            rec["comment"] = (f"counterfactual graph edits over {f.name}@"
                              f"{gsha} (REGIMES R2): answer follows the "
                              f"EDITED graph, never memory")
            rec["variants"] = {"domain": cf_entries, "phrasing": QA_PHRASINGS}
            if _verify_scenario(rec):
                write_template(rec, out_root)
                stats["templates"] += 1
                stats["counterfactuals"] = stats.get("counterfactuals", 0) \
                    + len(cf_entries)
            else:
                stats["preflight_fail"].append(rec["id"])
        # abstain: remove each fact, the lookup must stop being derivable
        ab_entries = []
        for i, (s, p, o, native) in enumerate(facts[:args.max_facts]):
            reduced = ttl_context(prefix_lines,
                                  [t for k, t in enumerate(facts) if k != i])
            ab_entries.append({"ctx": reduced, "subj": s, "pred": p,
                               "obj": o if native is None else native,
                               "slabel": label(s), "plabel": label(p)})
        if ab_entries:
            ab = make_template("abstain", f"graph.{domain}.{stem}",
                               "ABSTAIN", [
                {"role": "assistant", "content": "{domain.ctx}"},
                {"role": "user", "content":
                 "What is the {domain.plabel} of the {domain.slabel}?"},
                {"role": "assistant", "content":
                 "NOT DERIVABLE — the fact required to answer is absent "
                 "from the supplied graph. Name what is missing; do not "
                 "invent facts."},
            ], {"not_derivable": True,
                "program": ("READ WORLD {domain.subj} {domain.pred} my:?v "
                            ". ONE.\nRETURN my:?v.")})
            ab["id"] = f"graph/{domain}/{stem}--abstain"
            ab["comment"] = f"abstention over {f.name}@{gsha} (FRAMEWORK P12)"
            ab["variants"] = {"domain": ab_entries}
            if preflight(ab) == []:
                write_template(ab, out_root)
                stats["templates"] += 1
            else:
                stats["preflight_fail"].append(ab["id"])
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    return 0


def _verify_scenario(rec):
    """check_record over every expansion — the fleet's own preflight law."""
    try:
        for messages, expected in check.expansions(rec):
            check.check_record(rec, messages, expected)
        return True
    except Exception:
        return False


# ── audit (P28 / P29 / P3-verify) ──────────────────────────────────────

def shingles(text, k=8):
    words = re.findall(r"\w+", text)
    if len(words) <= k:
        return {" ".join(words)}
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def cmd_audit(args):
    scen_root = Path(args.scenarios) if args.scenarios else SCEN_ROOT
    out = Path(args.out) if args.out else (TMP / "audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    by_system, entropy, shapes = {}, [], []
    for path in sorted(scen_root.rglob("*.json")):
        if "derived" in path.parts:
            continue
        try:
            rec = json.loads(path.read_text())
        except Exception:
            continue
        by_system.setdefault(rec["system"], []).append(rec)
        total = 1
        for pool in rec.get("variants", {}).values():
            total *= max(1, len(pool))
        phrases = {messages[1]["content"]
                   for messages, _ in check.expansions(rec)}
        entropy.append({"id": rec["id"], "expansions": total,
                        "distinct_requests": len(phrases),
                        "ratio": round(len(phrases) / max(1, total), 3)})
        shapes.append({"id": rec["id"], "system": rec["system"],
                       "sh": shingles(rec["messages"][-1]["content"])})
    # P28: cross-template near-duplicate scan on final-IR shape
    near = []
    for a, b in itertools.combinations(shapes, 2):
        if a["system"] != b["system"]:
            continue
        score = jaccard(a["sh"], b["sh"])
        if score >= 0.8:
            near.append({"a": a["id"], "b": b["id"], "similarity": round(score, 3)})
    coverage = {k: len(v) for k, v in sorted(by_system.items())}
    derived = {}
    # audit a scenario tree's OWN sibling derived dir when overridden, so
    # test/alternate roots never see the live corpora manifest
    droot = scen_root.parent / "derived" if args.scenarios else DERIVED_ROOT
    if droot.is_dir():
        for path in sorted(droot.rglob("*.json")):
            if path.name == "manifest.json":
                continue
            try:
                rec = json.loads(path.read_text())
            except Exception:
                continue
            derived[rec["system"]] = derived.get(rec["system"], 0) + 1
    report = {
        "templates": sum(coverage.values()),
        "coverage_by_system": coverage,
        "derived_by_system": derived,
        "near_duplicate_pairs": sorted(near, key=lambda n: -n["similarity"])[:20],
        "lowest_phrasing_entropy": sorted(entropy, key=lambda e: e["ratio"])[:10],
        "expansion_totals": sum(e["expansions"] for e in entropy),
    }
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("templates", "coverage_by_system", "derived_by_system",
                       "expansion_totals")}, indent=1))
    print(f"full report: {out}", file=sys.stderr)
    return 0


# ── manifest & bundle (Phase 0 kernel: content-addressed lineage) ──────

ORACLE_FILES = ("synth.py", "ir-compile.py", "ir-ast-sanity.py",
                "ir-check-scenarios.py")


def cmd_manifest(args):
    out_root = Path(args.out) if args.out else DERIVED_ROOT
    files = []
    for path in sorted(out_root.rglob("*.json")):
        if path.name == "manifest.json":
            continue
        blob = path.read_bytes()
        files.append({"id": json.loads(blob)["id"],
                      "path": path.relative_to(out_root).as_posix(),
                      "sha256": hashlib.sha256(blob).hexdigest()[:16]})
    oracles = {}
    for name in ORACLE_FILES:
        p = ROOT / "src" / name
        oracles[name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    corpus = hashlib.sha256(
        "".join(f["sha256"] for f in files).encode()).hexdigest()[:16]
    manifest = {"corpus_id": f"derived@{corpus}", "files": files,
                "oracles": oracles, "count": len(files)}
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"corpus_id": manifest["corpus_id"],
                      "count": len(files), "oracles": oracles}))
    return 0


BUNDLE_MAP = {"invalid-program": "diagnosis", "repair": "repair",
              "abstain": "abstain"}


def cmd_bundle(args):
    """derived + graph templates → $CORPORA_DIR/datasets/<domain>/*.jsonl
    consumable by the existing lane: bin/prepare-domain <domain> → S3 →
    qtrain. Regenerable output — corpora/.gitignore keeps it out of git."""
    out_root = Path(args.out) if args.out else DERIVED_ROOT
    domain = args.domain
    dest = DATA_ROOT / "datasets" / domain
    dest.mkdir(parents=True, exist_ok=True)
    buckets = {}
    for path in sorted(out_root.rglob("*.json")):
        if path.name == "manifest.json":
            continue
        rec = json.loads(path.read_text())
        sysname = rec["system"]
        gid = rec["id"]
        if "graph" in gid:
            # split graph-bridge families by id suffix (qa/hops/cfkg)
            tag = ("graph-qa" if gid.endswith("--qa")
                   else "graph-hops" if gid.endswith("--hops")
                   else "counterfactual" if gid.endswith("--cfkg")
                   else sysname)
        else:
            tag = BUNDLE_MAP.get(sysname, sysname)
        rows = buckets.setdefault(tag, [])
        for messages, _expected in check.expansions(rec):
            rows.append({"messages": messages})
    written = {}
    for tag, rows in sorted(buckets.items()):
        out = dest / f"{tag}.jsonl"
        with out.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        written[tag] = {"rows": len(rows),
                        "sha256": hashlib.sha256(out.read_bytes()).hexdigest()[:16]}
    # P23 preference pairs: axolotl chat_preference format ({chosen,
    # rejected}) written into a SEPARATE dataset domain — prepare-domain
    # merges every jsonl under datasets/<domain>/, so preference rows
    # must never live beside the SFT envelopes (REGIMES R1; qtrain
    # submit --domain cognition-dpo + a pref_datasets recipe).
    dpo = Path(args.dpo) if getattr(args, "dpo", None) else TMP / "dpo.jsonl"
    if dpo.is_file():
        pdest = DATA_ROOT / "datasets" / f"{domain}-dpo"
        pdest.mkdir(parents=True, exist_ok=True)
        with (pdest / "preference.jsonl").open("w") as fh:
            n = 0
            for line in dpo.read_text().splitlines():
                if not line.strip():
                    continue
                pair = json.loads(line)
                fh.write(json.dumps({
                    "chosen": pair["messages"],
                    "rejected": pair["rejected"]["messages"],
                }, ensure_ascii=False) + "\n")
                n += 1
        written["preference->" + f"{domain}-dpo"] = {
            "rows": n,
            "sha256": hashlib.sha256(
                (pdest / "preference.jsonl").read_bytes()).hexdigest()[:16]}
    total = sum(w["rows"] for w in written.values())
    print(json.dumps({"dataset": f"{domain}@{hashlib.sha256(str(sorted(written.items())).encode()).hexdigest()[:16]}",
                      "path": str(dest), "files": written, "rows": total},
                     indent=1))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="synth",
                                description="deterministic IR corpus synthesis")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("derive", help="P2/P23: mutation + DPO synthesis")
    d.add_argument("--systems",
                   help="comma list, default: " + ",".join(SOURCE_SYSTEMS))
    d.add_argument("--codes",
                   help="comma list of error classes to attempt (default all)")
    d.add_argument("--per-code", type=int, default=2,
                   help="max variants per (template, code) cell")
    d.add_argument("--limit", type=int, default=0, help="cap expansions scanned")
    d.add_argument("--scenarios", help="override scenarios root")
    d.add_argument("--out", help="override derived root")
    d.add_argument("--tmp", help="override tmp dir (dpo.jsonl)")
    d.add_argument("--no-dpo", action="store_true")
    d.set_defaults(func=cmd_derive)
    a = sub.add_parser("abstain", help="P12: not-derivable rows from graphs")
    a.add_argument("--per-code", type=int, default=1,
                   help="max accepted rows per source template")
    a.add_argument("--scenarios")
    a.add_argument("--out")
    a.set_defaults(func=cmd_abstain)
    g = sub.add_parser("graph", help="P4/P6/P12: scenarios over real TTL")
    g.add_argument("ttl", help=".ttl file or directory")
    g.add_argument("--domain", help="domain name (default: file stem)")
    g.add_argument("--max-facts", type=int, default=50)
    g.add_argument("--out")
    g.set_defaults(func=cmd_graph)
    r = sub.add_parser("rows", help="expand derived templates → jsonl")
    r.add_argument("--task")
    r.add_argument("--out", help="derived root to read")
    r.add_argument("--rows-out", help="dir to write rows")
    r.set_defaults(func=cmd_rows)
    u = sub.add_parser("audit", help="P28/P29: coverage, entropy, near-dup")
    u.add_argument("--scenarios")
    u.add_argument("--out")
    u.set_defaults(func=cmd_audit)
    m = sub.add_parser("manifest", help="content-address derived tree + "
                                       "oracle digests (L2/L3 lineage)")
    m.add_argument("--out", help="derived root to walk")
    m.set_defaults(func=cmd_manifest)
    b = sub.add_parser("bundle", help="derived rows → "
                                      "$CORPORA_DIR/datasets/<domain>/")
    b.add_argument("--domain", default="cognition")
    b.add_argument("--out", help="derived root to read")
    b.add_argument("--dpo", help="override dpo.jsonl path (default tmp)")
    b.set_defaults(func=cmd_bundle)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
