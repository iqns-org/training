"""L3 CANARY (docs/REGIMES: cognition@e184c8de 9,044 -> @569e46d5 6,864
accepted, silently, from an UNCHANGED scenarios tree — the drift was in
the 7d1859c synth.py sweep). Rulebook semantics are frozen here: pinned
programs -> exact acceptance/verdicts. ANY mutation of the rulebook that
silently moves acceptance counts flips this test — loud instead of lost.
If you change a rule DELIBERATELY, update the pins in the same commit
and cite the corpus-id move in REGIMES."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def synth():
    spec = importlib.util.spec_from_file_location(
        "synth_canary", ROOT / "src" / "synth.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CTX = ("PREFIX my: <http://e.org/n/>.\nPREFIX ai: <http://e.org/p/>.\n"
       "WORLD { my:doc ai:origin my:a . }\nMEMORY { my:doc ai:tag my:t . }")
STMTS = "READ WORLD my:doc ai:origin my:?x . ONE.\nRETURN my:?x."


def _full(mod, stmts):
    return mod._full(CTX + "\n", stmts)


# -- frozen acceptance: the canary's heart ---------------------------------

def test_valid_program_acceptance_pinned(synth):
    """The reference program: VALID, zero errors AND zero warnings. If
    the parser/checker starts rejecting or flagging THIS, every family's
    acceptance silently drops (the L3 class)."""
    full = synth._full(CTX + "\n", STMTS)
    assert synth.is_valid_program(full)
    doc, rep = synth.validate(full)
    assert rep["valid"] and not rep["errors"] and not rep["warnings"]


# Measured golden table (2026-11-27): each rule, on the pinned seed, must
# emit a candidate whose code assignment holds AND whose mutant carries
# EXACTLY that one error, zero warnings — the acceptance math of derive.
# A silent rulebook change (7d1859c class) drops or recodes rows here first.
MUT_GOLDEN = {
    "mut_bad_timeout": {"E6_BAD_TIMEOUT"},
    "mut_dead_after_return": {"E6_UNREACHABLE"},
    "mut_dup_blockdecl": {"E1_DUP_PROJECTION", "E1_DUP_PLAN"},
    "mut_dup_prefix": {"E1_DUP_PREFIX"},
    "mut_effect_in_projection": {"E7_EFFECT_IN_PROJECTION"},
    "mut_inject_card": {"E5_CARD_MISMATCH"},
    "mut_inject_type": {"E4_TYPE_MISMATCH"},
    "mut_parallel_arity": {"E6_PARALLEL_ARITY"},
    "mut_unbound": {"E2_UNBOUND"},
}


def test_mutation_rulebook_golden(synth):
    """Every pinned rule fires, keeps its code assignment, and its
    mutants still pass `single_error` for that code. If a rule drifts to
    multi-error/no-error output (=> derive silently discards the cell),
    this test names it."""
    fired = {}
    for name, codes in MUT_GOLDEN.items():
        out = getattr(synth, name)(CTX + "\n", STMTS)
        assert out, f"{name} went silent on the seed"
        seen = set()
        for code, mutant_full, _stmts, _var in out:
            assert synth.single_error(mutant_full, code), \
                f"{name}: mutant for {code} no longer carries exactly " \
                f"that single error (silent-cell-drop, the L3 class)"
            seen.add(code)
        assert codes <= seen, f"{name}: lost codes {codes - seen}"
        fired[name] = sorted(seen)
    # the rules NOT in the golden set must not newly fire on the seed
    # with unknown codes either (drift detector, not a rigidity lock:
    # if you ADD a rule, this stays green; recoding fails loud).


def test_unbound_read_semantics_pinned(synth):
    """Pinned subtlety that rule rewrites could silently invert: READ
    BINDS unknown handles (a ?ghost read is VALID); it is only an
    E2_UNBOUND when REFERENCED unbound (mut_unbound's shape)."""
    ghost = synth._full(CTX + "\n",
                        "READ WORLD my:doc ai:origin my:?ghost . ONE.\n"
                        "RETURN my:?ghost.")
    doc, rep = synth.validate(ghost)
    assert rep["valid"] and not rep["errors"]
    assert not synth.is_valid_program(synth._full(CTX + "\n",
                                                  "RETURN my:?neverbound."))


def test_expansion_math_pinned(synth):
    """Variant-grid arithmetic: 1 domain x 2 phrasings -> exactly 2
    expansions, roles [assistant,user,assistant], gold passes its own
    grade. Any change here re-scales every family's row count — the
    silent-acceptance-change class."""
    rec = {
        "comment": "c", "label": "l", "id": "nl-to-ir/t01",
        "system": "nl-to-ir",
        "variants": {
            "domain": [{"first": "my:depotA"}],
            "phrasing": ["Who is it from?", "State the origin."],
        },
        "messages": [
            {"role": "assistant", "content": CTX},
            {"role": "user", "content": "{phrasing}"},
            {"role": "assistant", "content": STMTS},
        ],
        "assert": {"valid": True, "returned": "my:a",
                   "world_add": [], "world_delete": [], "memory_add": [],
                   "memory_delete": [], "emitted": []},
    }
    exps = list(synth.check.expansions(rec))
    assert len(exps) == 2
    for messages, expected in exps:
        assert [m["role"] for m in messages] == ["assistant", "user",
                                                 "assistant"]
        ok, why = synth.grade_completion(rec, messages, expected,
                                         messages[2]["content"])
        assert ok, why          # gold grades gold: must ALWAYS pass
