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


# ── R4: student-side RFT sampling + grading (the recursion half) ──────
#
# Prompts come from derived templates (first+user, gold hidden); the
# served model answers; a completion is KEPT only if the same fleet
# contract that gates every corpus artifact accepts it (check_record /
# NOT-DERIVABLE verdict). Teacher-independent; content-addressed when
# bundled.

import random
import urllib.request


def http_json(url, payload, timeout=120):
    """Minimal OpenAI-compatible POST (bench.py's seam, no streaming)."""
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def grade_completion(rec, messages, expected, completion):
    """Student-side oracle: build [a,u,a] with the completion as the final
    assistant turn and run the fleet's own record contract over it.
    prose-verdict families (abstain/conflict) are graded on the verdict
    phrase; everything else must satisfy check_record exactly."""
    try:
        trial = messages[:2] + [{"role": "assistant", "content": completion}]
        if rec["system"] == "abstain":
            return ("NOT DERIVABLE" in completion.upper(), "abstain verdict")
        if rec["system"] == "conflict":
            return ("CONFLICT" in completion.upper(), "conflict verdict")
        check.check_record(rec, trial, expected)
        return True, "oracle pass"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:140]}"


def cmd_sample(args):
    if not args.url:
        print(json.dumps({"error": "sample needs --url <openai-endpoint>"}))
        return 1
    roots = ([Path(args.out)] if args.out else [SCEN_ROOT, DERIVED_ROOT])
    tmp = TMP / "rft"
    tmp.mkdir(parents=True, exist_ok=True)
    grammar = None
    if getattr(args, "guided", False):
        # R3 serve-flag, PROVEN 2026-11-27: this engine (vLLM v0.34)
        # silently IGNORES top-level guided_* fields — enforcement lives
        # in response_format structural_tag, grammar must be
        # xgrammar-portable (ir-gbnf build(): root alias + [^...] sets).
        # LOAD WARNING (same day): the first full guided pass wedged
        # EngineCore on Orin (guided costs MINUTES per generation and
        # client timeouts leave orphan server work) — single probes or
        # small --max-tokens only; AS-BUILT §7.
        gmod = _load("ir_gbnf_cli", "ir-gbnf.py")
        grammar = gmod.build()
    # rejects: WHY the oracle disposed no (r1 lesson — yield=0 with an
    # empty by_family taught us nothing; the exception class from
    # grade_completion is the actionable bucket: CompileError/AssertionError/
    # parse-*/verdict-fail/empty-completion)
    stats = {"prompts": 0, "sampled": 0, "kept": 0, "by_family": {},
             "rejects": {}, "rejects_by_system": {}, "errors": []}
    rows_path = tmp / "rft.jsonl"
    rejects_path = tmp / "rejects.jsonl"
    paths = [p for r in roots if r.is_dir()
             for p in sorted(r.rglob("*.json")) if p.name != "manifest.json"]
    # r3 fix: interleave EXPANSIONS round-robin across systems — sorted
    # file order let one alphabetically-first family eat the whole
    # --limit (r1/r2's "yield 0.0" measured ONE family; rejects_by_system
    # exposed it, 315331c). Coverage is structural now, not discovered.
    streams: dict[str, list] = {}
    for path in paths:
        rec = json.loads(path.read_text())
        if args.task and rec["system"] != args.task:
            continue
        for pair in check.expansions(rec):
            streams.setdefault(rec["system"], []).append((rec,) + pair)
    ordered = []
    for i in range(max((len(v) for v in streams.values()), default=0)):
        for sysname in sorted(streams):
            if i < len(streams[sysname]):
                ordered.append(streams[sysname][i])
        if len(ordered) >= args.limit > 0:
            break
    with rows_path.open("w") as fh, rejects_path.open("w") as rj:
        for rec, messages, expected in ordered:
            if stats["prompts"] >= args.limit:
                break
            prompt = messages[:2]
            stats["prompts"] += 1
            for _ in range(args.k):
                try:
                    body = {"model": args.model, "messages": prompt,
                            "temperature": args.temperature,
                            "max_tokens": args.max_tokens}
                    if grammar:
                        body["response_format"] = {
                            "type": "structural_tag",
                            "format": {"type": "grammar",
                                       "grammar": grammar}}
                    resp = http_json(args.url.rstrip("/")
                                     + "/chat/completions", body)
                    completion = (resp["choices"][0]["message"]
                                  ["content"] or "").strip()
                except Exception as exc:
                    stats["errors"].append(str(exc)[:120])
                    continue
                if not completion:
                    stats["rejects"]["empty-completion"] = \
                        stats["rejects"].get("empty-completion", 0) + 1
                    continue
                stats["sampled"] += 1
                ok, why = grade_completion(rec, messages, expected,
                                           completion)
                if not ok:
                    key = why.split(":", 1)[0] if ":" in why else why
                    stats["rejects"][key] = \
                        stats["rejects"].get(key, 0) + 1
                    # r2 lesson: the global buckets said "shape ok,
                    # contract dead" but not WHICH contract — per-system
                    # is what aims r3's data lever.
                    bysys = stats["rejects_by_system"].setdefault(
                        rec["system"], {})
                    bysys[key] = bysys.get(key, 0) + 1
                    # the rows BEHIND the counts: (prompt, gold, the
                    # model's own completion, the assert that disposed
                    # it) — consumed by `synth pairs` as DPO fuel and
                    # by human eyes as the family->cause map.
                    rj.write(json.dumps({
                        "system": rec["system"], "id": rec["id"],
                        "bucket": key, "why": why,
                        "prompt": prompt,
                        "gold": messages[2]["content"],
                        "completion": completion[:1500],
                    }, ensure_ascii=False) + "\n")
                    stats["reject_dump"] = str(rejects_path)
                if ok:
                    stats["kept"] += 1
                    fam = ("graph-" if "graph" in rec["id"]
                           else "") + rec["system"]
                    stats["by_family"][fam] = \
                        stats["by_family"].get(fam, 0) + 1
                    fh.write(json.dumps({"messages": messages[:2] + [
                        {"role": "assistant", "content": completion}]},
                        ensure_ascii=False) + "\n")
    stats["yield"] = round(stats["kept"] / max(1, stats["sampled"]), 3)
    if args.dataset:
        dest = DATA_ROOT / "datasets" / args.dataset
        dest.mkdir(parents=True, exist_ok=True)
        out = dest / "rft.jsonl"
        out.write_text(rows_path.read_text())
        stats["dataset"] = (f"{args.dataset}@" +
                            hashlib.sha256(out.read_bytes()).hexdigest()[:16])
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    return 0

def cmd_pairs(args):
    """R2-lesson tool: reject dump -> DPO pairs. RFT can't bootstrap at
    zero yield, but a parseable near-miss completion under the SAME
    prompt is exactly a preference negative. Rows land in the expand-
    lane pair shape ({messages: gold, rejected: {messages: bad}}) so
    bin/prepare-domain's preference.jsonl writer consumes them verbatim."""
    src = Path(args.infile) if args.infile else TMP / "rft" / "rejects.jsonl"
    out = Path(args.out) if args.out else TMP / "rft" / "dpo.jsonl"
    keep = set(args.buckets.split(",")) if args.buckets else None
    n, by_sys = 0, {}
    with src.open() as fh, out.open("w") as oh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if keep and r["bucket"] not in keep:
                continue
            oh.write(json.dumps({
                "messages": r["prompt"] + [
                    {"role": "assistant", "content": r["gold"]}],
                "rejected": {"messages": r["prompt"] + [
                    {"role": "assistant", "content": r["completion"]}]},
                "system": r["system"], "why": r["why"],
            }, ensure_ascii=False) + "\n")
            n += 1
            by_sys[r["system"]] = by_sys.get(r["system"], 0) + 1
    print(json.dumps({"pairs": n, "by_system": by_sys, "out": str(out)}))
    return 0


# ── P1 (regime plan): distributional corpus QA ─────────────────────────
#
# Row oracles gate CORRECTNESS; audit measures what the corpus TEACHES:
# per-family shape, style classes the IR parser rejects (the em-dash
# bleed was found via model failure — this finds it at build), near-dup
# rate, and holdout->train TEMPLATE leakage (SPEC 13am splits by row
# content; templated families leak siblings, so holdout NLL partially
# measures memorization — this quantifies it for the split-law v2 call).

STYLE_CLASSES = {          # prose-bleed fingerprints: chars NO legal IR
    "dash": "\u2014\u2013",   # program contains. (paren/square were
    "curly": "\u2018\u2019\u201c\u201d",  # dropped: IR has legal list
}                          # literals — presence-based = false positives
PROGRAM_MARK = re.compile(
    r"\b(PREFIX|WORLD|MEMORY|READ|BIND|WRITE|MATCH|RETURN|EMIT)\b")


def _norm(text: str) -> str:
    return " ".join(text.split())


def _is_holdout(line: str) -> bool:
    """Mirror of bin/prepare-domain's holdout law v2 (TEMPLATE key:
    normalized first message, whole line as fallback; test_holdout_
    mirror pins agreement — if the law moves there and not here, it
    fails loud)."""
    try:
        msgs = json.loads(line).get("messages") or []
        first = " ".join((msgs[0].get("content") or "").split())
    except Exception:
        first = ""
    key = first or line.strip()
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 50 == 0


def _fam(fams: dict, name: str) -> dict:
    return fams.setdefault(name, {
        "rows": 0, "gold_lens": [], "uniq": set(), "prog": 0,
        "style": {c: 0 for c in STYLE_CLASSES},
        "style_prose": {c: 0 for c in STYLE_CLASSES},
        "train_templates": set(), "hold_templates": []})


def _observe(st: dict, gold: str, tmpl: str, held: bool) -> None:
    st["rows"] += 1
    st["gold_lens"].append(len(gold))
    st["uniq"].add(_norm(gold))
    is_prog = bool(PROGRAM_MARK.search(gold))
    st["prog"] += int(is_prog)
    for cls, chars in STYLE_CLASSES.items():
        if any(c in gold for c in chars):
            (st["style"] if is_prog else st["style_prose"])[cls] += 1
    if held:
        st["hold_templates"].append(tmpl)
    else:
        st["train_templates"].add(tmpl)


def cmd_profile(args):
    fams: dict[str, dict] = {}
    if args.path:                        # jsonl datasets (prepared rows)
        p = Path(args.path)
        files = sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]
        for f in files:
            famname = (args.family or f.stem if f.parent == p
                       else args.family or f.parent.name)
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                msgs = row.get("messages") or row.get("chosen") or []
                gold = next((m["content"] for m in reversed(msgs)
                             if m["role"] == "assistant"), "")
                tmpl = (hashlib.sha256(_norm(msgs[0]["content"]).encode())
                        .hexdigest()[:12] if msgs else "?")
                _observe(_fam(fams, famname), gold, tmpl, _is_holdout(line))
    else:                                # scenario trees (rec + gold turn)
        roots = [Path(args.scenarios)] if args.scenarios \
            else [SCEN_ROOT, DERIVED_ROOT]
        for r in roots:
            if not r.is_dir():
                continue
            for path in sorted(r.rglob("*.json")):
                if path.name == "manifest.json":
                    continue
                rec = json.loads(path.read_text())
                if args.family and rec["system"] != args.family:
                    continue
                _observe(_fam(fams, rec["system"]),
                         rec["messages"][2]["content"],
                         rec["id"].split("/")[0], False)
    report = {}
    for name, st in fams.items():
        lens = sorted(st["gold_lens"]) or [0]
        out = {
            "rows": st["rows"],
            "kind": "program" if st["prog"] * 2 >= st["rows"] else "prose",
            "gold_len_p50": lens[len(lens) // 2],
            "uniq_gold_ratio": round(len(st["uniq"]) / max(1, st["rows"]), 3),
            "hazards_in_program_rows": st["style"],
            "hazards_in_prose_rows": st["style_prose"],
        }
        if st["hold_templates"]:
            leaked = sum(1 for t in st["hold_templates"]
                         if t in st["train_templates"])
            out["holdout_rows"] = len(st["hold_templates"])
            out["holdout_template_leak"] = round(
                leaked / len(st["hold_templates"]), 3)
        report[name] = out
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0


# ── R5: data-mixing regression (grid primitive) ───────────────────────

def cmd_mix(args):
    """Weighted merge of PREPARED domain files (run bin/prepare-domain on
    each source first) into a reproducible, seeded, content-addressed
    dataset — the mixing-law experiment primitive (REGIMES R5)."""
    spec = []
    for part in args.sources.split(","):
        name, _, w = part.partition(":")
        spec.append((name.strip(), float(w) if w else 1.0))
    total = sum(w for _n, w in spec) or 1
    rng = random.Random(args.seed)
    pools = {}
    for name, _w in spec:
        src = ROOT / "tmp" / f"{name}.jsonl"
        if not src.is_file():
            print(json.dumps({"error": f"no prepared file {src} — run "
                                       f"bin/prepare-domain {name} first"}))
            return 1
        pools[name] = [l for l in src.read_text().splitlines() if l.strip()]
    rows = []
    for name, w in spec:
        take = int(round(args.rows * w / total))
        pool = pools[name]
        if not pool:
            continue
        if take <= len(pool):
            idx = rng.sample(range(len(pool)), take)
            rows.extend(pool[i] for i in idx)
        else:
            rows.extend(pool[rng.randrange(len(pool))] for _ in range(take))
    rng.shuffle(rows)
    rows = rows[:args.rows]
    dest = DATA_ROOT / "datasets" / args.name
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "mix.jsonl"
    out.write_text("".join(l + "\n" for l in rows))
    manifest = {"mix": f"{args.name}@{hashlib.sha256(out.read_bytes()).hexdigest()[:16]}",
                "seed": args.seed, "rows": len(rows),
                "weights": {n: round(w / total, 4) for n, w in spec},
                "sources": {n: hashlib.sha256(
                    (ROOT / "tmp" / f"{n}.jsonl").read_bytes()).hexdigest()[:16]
                    for n, _ in spec}}
    (dest / "mix.manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))
    return 0


# ── R6/R7/P5/P19 close-out: gbnf, grade, gaps, merge, equiv, conflict ─
#
# All deterministic, all oracle-decided: gbnf emits the R3 grammar (via
# src/ir-gbnf.py, property-tested there); grade is the fleet contract as
# an offline RLVR reward function (R7 harness); gaps turns bench failures
# into targeted synthesis proposals (P7/R10); merge wraps
# postproc/merge_loras.py (R6); equiv and conflict mine two new families
# (P5 equivalence pairs, P19 dual-source contradictions) whose verdicts
# are COMPUTED by the fixture executor, never typed.

def cmd_gbnf(args):
    g = _load("ir_gbnf_cli", "ir-gbnf.py")
    text = g.build(args.root)
    refs = g.check_refs(text)
    res = {"root": args.root, "rules": text.count("::="),
           "undefined_refs": refs, "grammar": text}
    if args.out:
        Path(args.out).write_text(text)
        res["out"] = args.out
        res.pop("grammar", None)
    print(json.dumps(res, ensure_ascii=False))
    return 0


def _grade_index(roots):
    """(first, user) -> (record, expanded messages, expected): the prompt
    space students are asked in — the curated scenario tree (gold
    programs) AND the derived tree (mutants/verdicts), so any completion
    anywhere in the loop grades against the fleet's own record contract."""
    index = {}
    for root in roots:
        if not Path(root).is_dir():
            continue
        for path in sorted(Path(root).rglob("*.json")):
            if path.name == "manifest.json":
                continue
            try:
                rec = json.loads(path.read_text())
            except Exception:
                continue
            if rec.get("system") not in ("nl-to-ir", "valid-program",
                                         "program-transform", "semantic-edit",
                                         "program-completion", "repair",
                                         "invalid-program", "abstain",
                                         "conflict", "equivalence"):
                continue
            for messages, expected in check.expansions(rec):
                index.setdefault((messages[0]["content"],
                                  messages[1]["content"]),
                                 (rec, messages, expected))
    return index


def cmd_grade(args):
    """R7 harness: offline grade of {messages} jsonl rows — the exact
    (prompt -> s in {0,1}) reward pairs a GRPO loop reuses, without any
    model weights."""
    roots = [Path(args.derived)] if args.derived else [SCEN_ROOT, DERIVED_ROOT]
    index = _grade_index(roots)
    rows = [l for l in Path(args.path).read_text().splitlines() if l.strip()]
    graded = kept = unmatched = 0
    out_rows = []
    for line in rows:
        row = json.loads(line)
        msgs = row["messages"]
        key = (msgs[0]["content"], msgs[1]["content"])
        hit = index.get(key)
        if hit is None:
            unmatched += 1
            out_rows.append({**row, "s": 0, "why": "unmatched prompt"})
            continue
        rec, exp_msgs, expected = hit
        ok, why = grade_completion(rec, exp_msgs, expected, msgs[-1]["content"])
        graded += 1
        kept += ok
        out_rows.append({**row, "s": 1 if ok else 0, "why": why})
    if args.out:
        Path(args.out).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows))
    print(json.dumps({"graded": graded, "kept": kept, "unmatched": unmatched,
                      "yield": round(kept / max(1, graded), 3),
                      "out": args.out}, indent=1))
    return 0


GAP_COMMANDS = {
    "ir-translate": ["synth derive --systems nl-to-ir",
                     "synth graph <ttl> --domain <d>"],
    "ir-diagnose": ["synth derive --codes E1_DUP_PREFIX,E1_DUP_TYPE,"
                    "E1_DUP_CAPABILITY,E1_DUP_EVENT,E1_DUP_PLAN,"
                    "E1_DUP_PROJECTION,E7_EFFECT_IN_PROJECTION"],
    "ir-repair": ["synth derive --codes E2_UNBOUND,E3_REDEFINE,"
                  "E3_USE_BEFORE_DEF,E5_CARD_MISMATCH,E6_UNREACHABLE,"
                  "E6_PARALLEL_ARITY,E6_BAD_TIMEOUT,E7_IMMUTABLE_SCOPE"],
    "abstain": ["synth abstain --per-code 3", "synth graph <ttl> --domain <d>"],
    "json-guard": ["bin/expand <domain>  # format-fixture floors (P8)"],
    "ir-emit": ["synth derive --codes E7_EFFECT_IN_PROJECTION,E1_DUP_EVENT"],
}


def cmd_gaps(args):
    """P7/R10: failed bench questions -> targeted synthesis proposals
    (the loop's propose step). Input: `qtrain bench --json` output —
    the REAL row shape (a list of {q, pass, model, ...}) or an older
    {"rows": [...]} wrapper; "—" (none-check) rows are report-only,
    never gaps."""
    data = json.loads(Path(args.bench).read_text())
    rows = data if isinstance(data, list) else data.get("rows", [])

    def failed(r):
        v = r.get("pass", r.get("passed"))
        if isinstance(v, bool):
            return not v
        return v in ("FAIL", "?")          # "—" excluded by design

    failing = [r for r in rows if failed(r)]
    gaps = {}
    for r in failing:
        q = r.get("q") or r.get("question", "?")
        g = gaps.setdefault(q, {"question": q, "models": [],
                                "commands": GAP_COMMANDS.get(q, [
                                    "synth derive",
                                    "bin/ir-scenarios  # manual curriculum"])})
        g["models"].append(r.get("model", "?"))
    print(json.dumps({"failing": len(failing), "rows": len(rows),
                      "gaps": list(gaps.values())}, indent=1,
                     ensure_ascii=False))
    return 0


def cmd_merge(args):
    """R6: mergekit config generation (postproc/merge_loras.py)."""
    spec = importlib.util.spec_from_file_location(
        "merge_loras", ROOT / "postproc" / "merge_loras.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    adapters = [a.strip() for a in args.adapters.split(",") if a.strip()]
    cfg = m.gen_merge_config(args.base, adapters, args.method, args.density,
                             out_path=(args.out if args.out != "-"
                                       else "./merged"))
    if args.out and args.out != "-":
        Path(args.out).write_text(json.dumps(cfg, indent=1) + "\n")
        print(json.dumps({"out": args.out, "adapters": len(adapters),
                          "method": args.method, "density": args.density}))
    else:
        print(json.dumps(cfg, indent=1))
    return 0


def _run_program(ctx_text, program):
    """Execution delta of a program under a context, or None when the
    fixture cannot decide deterministically (same honesty law)."""
    doc, rep = validate(ctx_text)
    if rep is None or not rep["valid"] or rep["warnings"]:
        return None
    try:
        combined = check.document(check.context_nodes(doc)
                                  + ir.parse(program)["body"])
        delta = check.Fixture().run(combined)
    except Exception:
        return None
    return {k: (sorted(v, key=repr) if isinstance(v, list) else v)
            for k, v in delta.items()}


def _preflight_equivalence(rec):
    try:
        for messages, _expected in check.expansions(rec):
            check.check_record(rec, messages, {"valid": True})
        return True
    except Exception:
        return False


EQUIV_PHRASINGS = [
    "Are these programs equivalent? Answer with the verdict.",
    "Compare A and B on this scenario: EQUIVALENT or DIFFERENT?",
    "Do A and B behave identically? Give the verdict.",
]


def cmd_equiv(args):
    """P5 equivalence mining: uniform variable renames (must be
    EQUIVALENT) and targeted BIND/EMIT deletions (must be DIFFERENT) over
    derived programs; the fixture executor computes the verdict by
    comparing execution deltas. Undecidable pairs are dropped."""
    root = Path(args.out) if args.out else DERIVED_ROOT
    index = {}
    for path in sorted(root.rglob("*.json")):
        if path.name == "manifest.json":
            continue
        try:
            rec = json.loads(path.read_text())
        except Exception:
            continue
        if rec["system"] != "nl-to-ir":
            continue
        for messages, expected in check.expansions(rec):
            ctx, _user, program = (m["content"] for m in messages)
            if not expected.get("valid", True) or "returned" not in expected:
                continue
            index.setdefault((rec["id"], ctx, program), True)
    stats = {"sources": len(index), "pairs": 0, "equivalent": 0,
             "different": 0, "undecidable": 0, "templates": 0}
    by_v = {"EQUIVALENT": [], "DIFFERENT": []}
    for sid, ctx, a in sorted(index):
        if len(by_v["EQUIVALENT"]) + len(by_v["DIFFERENT"]) >= args.max_pairs:
            break
        da = _run_program(ctx, a)
        if da is None:
            stats["undecidable"] += 1
            continue
        # EQUIVALENT candidate: uniform rename of every variable
        vars_ = sorted({m.group(1) for m in re.finditer(r"my:\?(\w+)", a)},
                       key=len, reverse=True)
        if vars_:
            b = a
            for i, v in enumerate(vars_):
                b = re.sub(rf"my:\?{re.escape(v)}\b", f"my:?r{i}", b)
            db = _run_program(ctx, b)
            stats["pairs"] += 1
            if db is not None and da == db:
                stats["equivalent"] += 1
                by_v["EQUIVALENT"].append({
                    "ctx": ctx, "a": a, "b": b,
                    "reason": "identical execution delta (uniform rename)"})
        # DIFFERENT candidate: delete a BIND/EMIT/READ line that is
        # OBSERVABLE in the delta — each drop is tried until one changes
        # the outcome (drops that break binding are undecidable, skipped)
        lines = a.split("\n")
        for k, l in enumerate(lines):
            if not re.match(r"\s*(BIND|EMIT|READ)\b", l):
                continue
            c = "\n".join(x for i, x in enumerate(lines) if i != k)
            dc = _run_program(ctx, c)
            stats["pairs"] += 1
            if dc is None:
                stats["undecidable"] += 1
                continue
            if da != dc:
                stats["different"] += 1
                by_v["DIFFERENT"].append({
                    "ctx": ctx, "a": a, "b": c,
                    "reason": "deltas differ: " + "; ".join(
                        key for key in da if da.get(key) != dc.get(key))[:80]})
                break
    for verdict, pool in sorted(by_v.items()):
        if not pool:
            continue
        rec = make_template("equivalence", f"equiv.{verdict.lower()}",
                            verdict, [
            {"role": "assistant",
             "content": "{domain.ctx}\n\n# PROGRAM A\n{domain.a}\n\n"
                        "# PROGRAM B\n{domain.b}"},
            {"role": "user", "content": "{phrasing}"},
            {"role": "assistant",
             "content": "# VERDICT: " + verdict + "\n# BECAUSE: "
                        "{domain.reason}"},
        ], {"verdict": verdict, "valid": True})
        rec["id"] = f"equiv/{verdict.lower()}"
        rec["variants"] = {"domain": pool, "phrasing": EQUIV_PHRASINGS}
        if _preflight_equivalence(rec):
            write_template(rec, root)
            stats["templates"] += 1
    print(json.dumps(stats, indent=1))
    return 0


def cmd_conflict(args):
    """P19 contradiction mining: same (subject, predicate) asserted with
    different objects across two sources -> CONFLICT rows naming both
    values and citing the files. Trains refuse-to-pick-silently — the
    retrieval-serving failure mode where two sources disagree."""
    base = Path(args.ttl).read_text()
    alt = Path(args.other).read_text()
    bp, bf, _bs, _br = parse_ttl(base)
    _ap, af, _as_, _ar = parse_ttl(alt)
    apairs = {}
    for s, p, o, _n in af:
        apairs.setdefault((s, p), []).append(o)
    entries = []
    for s, p, o, _n in bf:
        for o2 in apairs.get((s, p), []):
            if o2 != o:
                merged = [t for t in bf if (t[0], t[1]) != (s, p)]
                entries.append({
                    "s": s, "p": p, "v1": o, "v2": o2,
                    "label": label(s), "plabel": label(p),
                    "src1": Path(args.ttl).name,
                    "src2": Path(args.other).name,
                    "ctx": ttl_context(bp, merged + [(s, p, o, None),
                                                     (s, p, o2, None)])})
    root = Path(args.out) if args.out else DERIVED_ROOT
    written = False
    if entries:
        rec = make_template("conflict", "conflict.sources", "CONFLICT", [
            {"role": "assistant", "content": "{domain.ctx}"},
            {"role": "user", "content": "{phrasing}"},
            {"role": "assistant",
             "content": "CONFLICT — {domain.label} has two asserted "
                        "{domain.plabel} values across sources "
                        "({domain.src1}: {domain.v1} | {domain.src2}: "
                        "{domain.v2}). Report both, citing sources; do not "
                        "pick silently."},
        ], {"conflict": True, "valid": True})
        rec["id"] = "conflict/sources"
        rec["variants"] = {"domain": entries, "phrasing": [
            "What is the {domain.plabel} of the {domain.label}?",
            "Read the {domain.label}'s {domain.plabel} from the graph."]}
        ok = True
        for messages, _expected in check.expansions(rec):
            _d, rep = validate(messages[0]["content"])
            if rep is None or not rep["valid"] or rep["warnings"]:
                ok = False
        if ok:
            write_template(rec, root)
            written = True
    print(json.dumps({"conflicts": len(entries), "written": written},
                     indent=1))
    return 0


import re


def _emoji(text):
    return "".join(c for c in text if ord(c) > 0x2190)


def _sample_rows(path: Path, limit: int):
    rows = []
    for f in sorted(path.rglob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(line)
    if len(rows) > limit:
        rows = rows[::max(1, len(rows) // limit)][:limit]   # deterministic
    return [json.loads(l) for l in rows]


def cmd_contrasts(args):
    """R14 contrast-set factory (ORACLES.md §2): labeled (doing X, not
    doing X) pairs — the fuel for steering/ablation/interference probes.
    Sources: bench-grmamar checks over any dataset, plus synth's own
    artifacts (ir-valid from dpo pairs, abstain from refusal rows)."""
    from orchestrator import bench as _b
    out_dir = TMP / "contrasts"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}

    def emit(behavior, pairs):
        path = out_dir / f"{behavior}.jsonl"
        with path.open("w") as fh:
            for label, msgs in pairs:
                fh.write(json.dumps({"behavior": behavior, "label": label,
                                     "messages": msgs},
                                    ensure_ascii=False) + "\n")
        stats[behavior] = {"pos": sum(1 for l, _ in pairs if l == 1),
                           "neg": sum(1 for l, _ in pairs if l == 0),
                           "path": str(path)}

    want = set((args.behaviors or "").split(",")) - {""}
    if not want or {"ir-valid"} & want:
        dpo = TMP / "dpo.jsonl"
        if dpo.is_file():
            pairs = []
            for line in dpo.read_text().splitlines():
                if not line.strip():
                    continue
                p = json.loads(line)
                pairs.append((1, p["messages"]))
                pairs.append((0, p["rejected"]["messages"]))
            emit("ir-valid", pairs[:2 * args.limit])

    if not want or {"abstain"} & want:
        pairs = []
        for path in sorted(DERIVED_ROOT.rglob("*.json")) if DERIVED_ROOT.is_dir() else []:
            if path.name == "manifest.json":
                continue
            rec = json.loads(path.read_text())
            if rec["system"] != "abstain":
                continue
            for messages, expected in check.expansions(rec):
                fabricated = ("READ WORLD my:whatever ai:whatever my:?v . "
                              "ONE.\nRETURN my:?v.")
                pairs.append((1, messages))
                pairs.append((0, messages[:2] + [
                    {"role": "assistant", "content": fabricated}]))
        emit("abstain", pairs[:2 * args.limit])

    if not want or {"self-name", "emoji", "json"} & want:
        domain = args.domain or "queer"
        ds = DATA_ROOT / "datasets" / domain
        if not ds.is_dir():
            stats["_error"] = f"no dataset {ds}"
        else:
            checks = {"self-name": "contains:kandi",
                      "emoji": "regex:.*[\\U0001F300-\\U0001FAFF\\u2600-\\u27BF]",
                      "json": "json"}
            for behavior, checkspec in checks.items():
                if behavior not in (want or checks):
                    continue
                pairs = []
                for row in _sample_rows(ds, args.limit):
                    msgs = row.get("messages")
                    if not msgs:
                        continue
                    final = msgs[-1]["content"]
                    verdict = _b._primitive(checkspec, final)
                    if verdict in ("PASS", "FAIL"):
                        pairs.append((1 if verdict == "PASS" else 0, msgs))
                emit(behavior, pairs)
    print(json.dumps(stats, indent=1, ensure_ascii=False))
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
              "abstain": "abstain", "equivalence": "equivalence",
              "conflict": "conflict"}


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
    s = sub.add_parser("sample", help="R4: student-side RFT (serve -> k -> "
                                      "oracle keep -> rft.jsonl)")
    s.add_argument("--url", help="OpenAI-compatible endpoint, e.g. "
                                 "http://nano0.lan:8000/v1")
    s.add_argument("--model", default="kandi")
    s.add_argument("--k", type=int, default=4)
    s.add_argument("--limit", type=int, default=50, help="max prompts")
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--max-tokens", type=int, default=512)
    s.add_argument("--task", help="filter by system (e.g. nl-to-ir)")
    s.add_argument("--out", help="derived root to read prompts from")
    s.add_argument("--dataset", help="also write "
                                     "$CORPORA_DIR/datasets/<name>/rft.jsonl")
    s.add_argument("--guided", action="store_true",
                   help="force IR grammar via response_format "
                        "structural_tag (v0.34+ portable grammar; ORIN "
                        "WARNING: minutes per generation, wedged the "
                        "first full pass — probe small, AS-BUILT 7)")
    s.set_defaults(func=cmd_sample)
    pj = sub.add_parser("pairs", help="reject dump -> DPO pairs (r3 fuel)")
    pj.add_argument("--in", dest="infile", help="rejects jsonl (default tmp/rft/rejects.jsonl)")
    pj.add_argument("--out", help="dpo jsonl (default tmp/rft/dpo.jsonl)")
    pj.add_argument("--buckets", help="comma list to keep (e.g. AssertionError,Returned)")
    pj.set_defaults(func=cmd_pairs)
    ad = sub.add_parser("profile", help="P1: distributional corpus QA "
                        "(shape, parser hazards, dup + holdout leak)")
    ad.add_argument("--path", help="jsonl file or dir of dataset rows; "
                    "default: scenario trees")
    ad.add_argument("--family", help="restrict to one system/dataset")
    ad.add_argument("--scenarios", help="override scenarios root")
    ad.set_defaults(func=cmd_profile)
    x = sub.add_parser("mix", help="R5: weighted mix of prepared datasets")
    x.add_argument("--sources", required=True,
                   help="comma list name:weight, e.g. queer:0.3,cognition:0.6")
    x.add_argument("--rows", type=int, default=4000)
    x.add_argument("--seed", type=int, default=7)
    x.add_argument("--name", required=True, help="output dataset name")
    x.set_defaults(func=cmd_mix)
    gb = sub.add_parser("gbnf", help="R3: emit the Graph IR GBNF grammar")
    gb.add_argument("--root", choices=("response", "document"),
                    default="response")
    gb.add_argument("--out")
    gb.set_defaults(func=cmd_gbnf)
    g2 = sub.add_parser("grade", help="R7 harness: offline RLVR grading of "
                                      "{messages} jsonl vs derived prompts")
    g2.add_argument("path")
    g2.add_argument("--out", help="write graded rows (with reward s) here")
    g2.add_argument("--derived", help="derived root to grade against")
    g2.set_defaults(func=cmd_grade)
    gp = sub.add_parser("gaps", help="P7/R10: bench failures -> synthesis "
                                     "proposals")
    gp.add_argument("--bench", required=True)
    gp.set_defaults(func=cmd_gaps)
    mg = sub.add_parser("merge", help="R6: LoRA merge-algebra config "
                                      "(mergekit)")
    mg.add_argument("--base", required=True)
    mg.add_argument("--adapters", required=True)
    mg.add_argument("--method", default="ties",
                    choices=("ties", "dare_ties", "dare_linear", "slerp"))
    mg.add_argument("--density", type=float, default=0.5)
    mg.add_argument("--out", default="-")
    mg.set_defaults(func=cmd_merge)
    eq = sub.add_parser("equiv", help="P5: equivalence pair mining "
                                      "(verdicts computed, never typed)")
    eq.add_argument("--max-pairs", type=int, default=40)
    eq.add_argument("--out", help="derived root")
    eq.set_defaults(func=cmd_equiv)
    co = sub.add_parser("conflict", help="P19: dual-source contradiction rows")
    co.add_argument("ttl")
    co.add_argument("other")
    co.add_argument("--out", help="derived root")
    co.set_defaults(func=cmd_conflict)
    ct = sub.add_parser("contrasts", help="R14: labeled contrast-set factory "
                                          "(fuel for steering/ablation)")
    ct.add_argument("--domain", help="persona dataset (default queer)")
    ct.add_argument("--behaviors", help="comma list: ir-valid,abstain,"
                                        "self-name,emoji,json")
    ct.add_argument("--limit", type=int, default=200)
    ct.set_defaults(func=cmd_contrasts)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
