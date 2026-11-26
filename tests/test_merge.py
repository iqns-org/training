"""Merge no-op forensics (c7cde2): the namespace surgery is the fix, so
the pure functions get their own tests (torch-free via the script's
import guard)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    ldr = importlib.machinery.SourceFileLoader(
        "ma", str(ROOT / "postproc" / "merge_adapter.py"))
    spec = importlib.util.spec_from_loader("ma", ldr)
    m = importlib.util.module_from_spec(spec)
    ldr.exec_module(m)
    return m


ma = _mod()


def test_key_re_handles_both_spellings():
    for k in ("base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.weight",
              "base_model.model.model.layers.3.self_attn.q_proj.lora_B.default.weight"):
        m = ma.KEY_RE.match(k)
        assert m, k
    assert not ma.KEY_RE.match("base_model.model.model.layers.0.weight")


def test_pair_ckpt_groups_halves():
    ckpt = {
        "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.weight": 1,
        "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_B.weight": 2,
        "base_model.model.model.layers.3.self_attn.q_proj.lora_A.default.weight": 3,
        "register_that_is_not_lora": 4,
    }
    pairs = ma.pair_ckpt(ckpt)
    assert pairs == {"model.language_model.layers.0.mlp.up_proj":
                     {"A": 1, "B": 2},
                     "model.layers.3.self_attn.q_proj": {"A": 3}}


def test_candidate_paths_both_directions():
    got = ma.candidate_paths("model.language_model.layers.3.self_attn.q_proj")
    assert got[0] == "model.language_model.layers.3.self_attn.q_proj"
    assert "model.layers.3.self_attn.q_proj" in got
    got = ma.candidate_paths("model.layers.0.mlp.gate_proj")
    assert "model.language_model.layers.0.mlp.gate_proj" in got


def test_lora_scaling_ranks_and_rslora():
    assert ma.lora_scaling({"r": 16, "lora_alpha": 32}) == 2.0
    assert ma.lora_scaling({"r": 16, "lora_alpha": 32,
                            "use_rslora": True}) == 8.0   # 32/sqrt(16)
    assert ma.lora_scaling({"r": 8, "lora_alpha": 8}) == 1.0


def test_merge_laws_all_refuse_a_base_clone():
    # the whole point of the 13ao rewrite: every silent-noop shape dies
    assert "ZERO" in ma.merge_law_error(0, 0, [], 0)
    # partial merge (namespace miss / half pair / shape clash)
    err = ma.merge_law_error(4, 3, ["model.layers.9.mlp.up_proj (half "
                                   "pair)"], 3)
    assert "incomplete" in err and "3/4" in err
    # every pair landed but NOTHING moved numerically = the base clone
    err = ma.merge_law_error(4, 4, [], 0)
    assert "numerically unchanged" in err and "base clone" in err


def test_merge_law_passes_only_on_real_delta():
    assert ma.merge_law_error(186, 186, [], 186) is None
    # a couple of fp16-rounded-to-zero tensors is normal; SILENCE is not
    assert ma.merge_law_error(186, 186, [], 184) is None
