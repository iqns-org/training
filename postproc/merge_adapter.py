#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model (CPU-friendly, fp16 out).

REWRITTEN after c7cde2/5c698d forensics (the merge no-op): the adapter
checkpoint is saved in the TRAINING-time namespace — composite keys
`model.language_model.layers.N....lora_A.weight`, no `.default` segment —
while this image's AutoModelForCausalLM builds the TEXT class
(`model.layers.N...`) and this peft generation expects `.lora_A.default.`
param names. peft found ZERO overlap, logged "missing AND unexpected
adapter keys", and merge_and_unload returned the UNMODIFIED base: every
qwen3.5 merge this pipeline ever shipped (v2 served for a bench, v3,
minis) was the base model in fp16 — byte-identical model.safetensors
across different adapters (sha 2cdf190a everywhere). The engine was never
serving a failed persona; it was serving the base wearing a persona's
name. peft's own conversion can't be trusted across these image
generations, so the merge is manual: string-normalized namespace
resolution, delta = (B @ A) * scaling, per-module accumulation. A merge
that moves zero (or not exactly all) tensors dies with a nonzero exit
instead of publishing a base clone. The applied-count dict line is
printed for finalize to scrape into the manifest.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

try:  # heavy stack only exists on stage pods; helpers below are pure
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:                                    # noqa: F811
    torch = None                                       # noqa: F811

KEY_RE = re.compile(
    r"^base_model\.model\.(?P<mod>.+?)\.lora_(?P<side>[AB])(?:\.default)?\.weight$")


def lora_scaling(cfg: dict) -> float:
    rank = int(cfg["r"])
    alpha = float(cfg.get("lora_alpha", rank))
    return alpha / math.sqrt(rank) if cfg.get("use_rslora") else alpha / rank


def candidate_paths(mod: str):
    """Namespace spellings of a target module, as-saved first, then the
    text<->composite swap. This family has shipped in exactly two
    spellings: 'model.layers...' (text-class loads) and
    'model.language_model.layers...' (composite/training-time saves)."""
    swapped = mod.replace("model.language_model.", "model.", 1)
    raised = re.sub(r"^model\.", "model.language_model.", mod, count=1)
    out, seen = [], set()
    for p in (mod, swapped, raised):
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def pair_ckpt(ckpt: dict):
    """ckpt tensors -> {module_path: {A: t, B: t}} (lora halves grouped)."""
    pairs: dict[str, dict] = {}
    for k, v in ckpt.items():
        m = KEY_RE.match(k)
        if m:
            pairs.setdefault(m.group("mod"), {})[m.group("side")] = v
    return pairs


def merge_law_error(n_pairs: int, applied: int, unresolved: list,
                    changed: int) -> str | None:
    """The FAIL-FAST LAWS as one pure check the suite can pin (silence
    about the delta is how the no-op shipped for a month):

      n_pairs == 0        adapter keys never matched the namespace —
                          publishing would clone the base.
      unresolved or
      applied != n_pairs  a partial merge is a lie of a different shape.
      changed == 0        every pair landed but NO weight moved
                          numerically (all-zero deltas): the output bytes
                          ARE the base — refuse to publish a clone."""
    if n_pairs == 0:
        return ("merge: ZERO lora pairs parsed — refusing to publish a "
                "base clone")
    if unresolved or applied != n_pairs:
        return (f"merge incomplete: applied {applied}/{n_pairs}; "
                f"unresolved: {unresolved[:8]}")
    if changed == 0:
        return (f"merge: all {applied} applied tensors left weights "
                "numerically unchanged — adapter contributed nothing, "
                "refusing to publish a base clone")
    return None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if torch is None:
        raise SystemExit("merge_adapter runs on a stage pod (torch absent)")

    acfg = json.loads((Path(a.input) / "adapter_config.json").read_text())
    scaling = lora_scaling(acfg)
    ckpt = load_file(str(Path(a.input) / "adapter_model.safetensors"))
    pairs = pair_ckpt(ckpt)
    if not pairs:
        # cheap pre-check BEFORE the base download (same law as below)
        raise SystemExit(merge_law_error(0, 0, [], 0))

    model = AutoModelForCausalLM.from_pretrained(a.base,
                                                 torch_dtype=torch.float16)
    params = dict(model.named_parameters())
    applied, unresolved, changed = 0, [], 0
    for mod, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            unresolved.append(f"{mod} (half pair)")
            continue
        tgt = None
        for cand in candidate_paths(mod):
            if (p := params.get(cand + ".weight")) is not None:
                tgt = p
                break
        if tgt is None:
            unresolved.append(f"{mod} (no module in either namespace)")
            continue
        A = ab["A"].to(torch.float32)              # [r, in]
        B = ab["B"].to(torch.float32)              # [out, r]
        delta = (B @ A) * scaling
        if tuple(delta.shape) != tuple(tgt.shape):
            unresolved.append(f"{mod} shape {tuple(delta.shape)} != "
                              f"{tuple(tgt.shape)}")
            continue
        before = tgt.data.clone()
        with torch.no_grad():
            tgt.data.copy_((tgt.data.to(torch.float32) + delta).to(tgt.dtype))
        applied += 1
        if not torch.equal(tgt.data, before):
            changed += 1

    # FAIL-FAST LAW (one pure function, suite-pinned): zero pairs, a lost
    # pair, or a numerically silent delta all die instead of publishing
    # a base clone.
    if (err := merge_law_error(len(pairs), applied, unresolved, changed)):
        raise SystemExit(err)

    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.base).save_pretrained(a.out)
    sys.path.insert(0, str(Path(__file__).parent))
    from util import sanitize_tokenizer_config  # noqa: E402
    sanitize_tokenizer_config(a.out)
    # STATS RIDE A GATED ARTIFACT, not the in-image manifest scrape:
    # pods run frozen agent images (the 5ce31b lesson — v0.3.0 finalize
    # lacked the wanted keys, so the dict line never reached the summary).
    stats = {"merge_applied_pairs": applied, "merge_expected_pairs": len(pairs),
             "merge_changed_tensors": changed,
             "merge_scaling": round(scaling, 6)}
    (Path(a.out) / "merge_stats.json").write_text(json.dumps(stats, indent=1))
    print(stats, flush=True)
    print(f"merged -> {a.out} ({applied} tensors moved)", flush=True)


if __name__ == "__main__":
    main()
