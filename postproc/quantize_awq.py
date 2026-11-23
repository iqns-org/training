#!/usr/bin/env python3
"""AWQ-style int4 (W4A16) quantization of a merged model via
llm-compressor, CALIBRATED ON THE DOMAIN CORPUS — the model's specialty
survives quantization because calibration sees its own distribution.

Output is compressed-tensors format: vLLM (nano0/Orin included) loads it
natively (auto-detected from config.json).

Version-tolerant ON PURPOSE — llmcompressor has drifted under us five
times (530b30 tokenizer shape; 88a662 divisibility; 805ff2 swallowed
variant errors; c38f58/54c2f6: my dict args used type='float', which is
FP8-only — the REAL W4A16 preset is type=INT/4bit/group/128/symmetric/
dynamic=False, so the version-agnostic move is to CLONE the installed
preset via compressed_tensors preset_name_to_scheme and override ONLY
group_size). Two hard-won facts steer the ladder:

  * THIS model (llama-type, hidden 960, heads 15x64) has 960 % 128 != 0
    on every linear except down_proj -> group_size=128 is impossible
    for most of the model, and an `ignore` list there would preserve
    almost nothing. group_size 64 divides 960 and 2560 exactly.
  * group_size is NOT a constructor kwarg and dict-schemes under
    `scheme` never validate in 0.11 — custom configs live in
    config_groups (built from the cloned preset, above).

Variants are ordered by a safetensors-header scan (no tensor load): if
the non-divisible-by-128 modules hold >50% of weight bytes, group-size
variants go first; otherwise ignore-fit (proven end-to-end in c38f58:
valid compressed-tensors, just mostly-fp16) leads. Every construction
rejection is LOGGED — silent swallowing was 805ff2's blind spot.
"""
import argparse
import inspect
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from util import sanitize_tokenizer_config  # noqa: BLE001, E402

DIVISOR = 128


def scan_shapes(model_dir: Path, divisor: int = DIVISOR):
    """(offenders, off_bytes, tot_bytes) from safetensors headers only.
    Linear weight matrices are [out, in]; group quantization requires
    in % group_size == 0. data_offsets are byte ranges -> cheap share."""
    files = []
    idx = model_dir / "model.safetensors.index.json"
    if idx.is_file():
        wm = json.loads(idx.read_text()).get("weight_map", {})
        files = sorted({model_dir / f for f in wm.values()})
    elif (model_dir / "model.safetensors").is_file():
        files = [model_dir / "model.safetensors"]
    offenders, off_b, tot_b = [], 0, 0
    for f in files:
        try:
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
        except (OSError, ValueError) as e:
            print(f"header scan skip {f.name}: {e}", flush=True)
            continue
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            s_off, e_off = (meta.get("data_offsets") or [0, 0])[:2]
            tot_b += e_off - s_off
            shp = meta.get("shape") or []
            if len(shp) == 2 and shp[1] % divisor:
                off_b += e_off - s_off
                offenders.append(name[:-7] if name.endswith(".weight")
                                 else name)
    return sorted(set(offenders)), off_b, tot_b


def calib_texts(tok, dataset, samples):
    """Chat rows -> templated text; the samples cap applies to EVERY
    appended row (88a662: an early `continue` bypassed it)."""
    texts = []
    for line in Path(dataset).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        t = ""
        if "messages" in row:
            try:
                t = tok.apply_chat_template(row["messages"], tokenize=False,
                                            add_generation_prompt=False)
            except Exception:                          # noqa: BLE001
                t = row.get("text") or ""
        else:
            t = row.get("text") or row.get("prompt") or ""
        if t:
            texts.append(t)
        if len(texts) >= samples:
            break
    return texts


def variant_builds(QM, AWQ, ignore, group_first):
    """Priority-ordered (label, recipe) ladder; construction rejections
    LOGGED, identical recipes deduped.

    0.11.0 forensics (c38f58): the modifier's REAL fields are
    scheme (preset name only) / config_groups / ignore / targets / ...
    -- dict-schemes NEVER validate; custom group size goes through
    config_groups built with compressed_tensors model objects (the
    package the installed version actually ships). 'group' is recipe
    bookkeeping, not group_size. AWQ lives in modifiers.transform.awq
    now (import-path deprecation warning in the same log)."""
    def attempt_build(label, fn):
        try:
            return (label, fn())
        except Exception as e:                         # noqa: BLE001
            print(f"variant unavailable {label}: "
                  f"{type(e).__name__}: {str(e)[:260]}", flush=True)
            return None

    def cfg_groups(gs):
        """THE version-agnostic path, settled from installed source:
        W4A16's real preset is type=INT num_bits=4 strategy=GROUP
        group_size=128 symmetric=True dynamic=False, activations fp16
        (input_activations omitted). My earlier variants used
        type='float' — which is FP8-only and raised
        'Only num_bits in (8) are supported' (54c2f6 attempt 1). Clone
        the INSTALLED preset and change only group_size, so the answer
        is never a guess about a future version."""
        try:
            from compressed_tensors.quantization.quant_scheme import (
                preset_name_to_scheme)
            base = preset_name_to_scheme("W4A16", ["Linear"])
            w = base.weights.model_copy(update={"group_size": gs})
            return {"group_0": base.model_copy(update={"weights": w})}
        except Exception:                              # noqa: BLE001
            pass
        from compressed_tensors.quantization import (QuantizationArgs,
                                                     QuantizationScheme)
        w = QuantizationArgs(num_bits=4, type="int", strategy="group",
                             group_size=gs, symmetric=True, dynamic=False)
        return {"group_0": QuantizationScheme(targets=["Linear"],
                                              weights=w,
                                              input_activations=None)}

    cands = []
    if QM:
        first, second = [], []
        if group_first:
            first = [("g64/config-groups",
                      lambda: [QM(config_groups=cfg_groups(64),
                                  ignore=["lm_head"])])]
            second = [("g32/config-groups",
                       lambda: [QM(config_groups=cfg_groups(32),
                                   ignore=["lm_head"])])]
        else:
            first = [("W4A16-ignore-fit",
                      lambda: [QM(targets="Linear", scheme="W4A16",
                                  ignore=ignore)])]
            second = [("g64/config-groups",
                       lambda: [QM(config_groups=cfg_groups(64),
                                   ignore=["lm_head"])])]
        cands += first + second + [
            ("W4A16-ignore-fit",
             lambda: [QM(targets="Linear", scheme="W4A16",
                         ignore=ignore)]),
            ("W4A16-preset",
             lambda: [QM(targets="Linear", scheme="W4A16")]),
        ]
    if AWQ and QM:
        cands.append(("AWQ+QM[g64]",
                      lambda: [AWQ(), QM(config_groups=cfg_groups(64),
                                         ignore=["lm_head"])]))

    ordered, seen = [], set()
    for label, fn in cands:
        built = attempt_build(label, fn)
        if not built:
            continue
        key = str(built[1])
        if key not in seen:
            seen.add(key)
            ordered.append(built)
            print(f"variant ready: {label}", flush=True)
    return ordered


def build_kwargs(oneshot, ds, recipe, out, texts):
    """Only kwargs the installed signature actually accepts."""
    sig = inspect.signature(oneshot).parameters
    kw = {"dataset": ds, "recipe": recipe}
    if "output_dir" in sig:
        kw["output_dir"] = out
    elif "save_dir" in sig:
        kw["save_dir"] = out
    for key, val in (("max_seq_length", 1024),
                     ("num_calibration_samples", len(texts))):
        if key in sig:
            kw[key] = val
    return kw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input", required=True,
                    help="merged model dir (from stage input)")
    ap.add_argument("--dataset", required=True,
                    help="calibration jsonl (chat rows)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=256)
    a = ap.parse_args()

    import llmcompressor
    from transformers import AutoTokenizer
    try:
        from llmcompressor import oneshot
    except ImportError:                                # pre-0.4 layout
        from llmcompressor.transformers import oneshot
    QM = AWQ = None
    try:
        from llmcompressor.modifiers.quantization import QuantizationModifier
        QM = QuantizationModifier
    except Exception:                                  # noqa: BLE001
        pass
    try:
        from llmcompressor.modifiers.transform.awq import AWQModifier
        AWQ = AWQModifier
    except Exception:                                  # noqa: BLE001
        try:
            from llmcompressor.modifiers.awq import AWQModifier
            AWQ = AWQModifier
        except Exception:                              # noqa: BLE001
            pass

    print(f"llmcompressor {getattr(llmcompressor, '__version__', '?')} | "
          f"oneshot params: "
          f"{list(inspect.signature(oneshot).parameters)[:12]}", flush=True)
    if QM:
        try:
            print(f"QM fields: {sorted(QM.model_fields)}", flush=True)
        except Exception:                              # noqa: BLE001
            pass

    sanitize_tokenizer_config(a.input)
    model_dir = Path(a.input).resolve()
    offenders, off_b, tot_b = scan_shapes(model_dir)
    frac = off_b / tot_b if tot_b else 0.0
    group_first = frac > 0.5
    print(f"scan: {len(offenders)} modules non-divisible by {DIVISOR} "
          f"({frac:.0%} of weight bytes) -> "
          f"{'group-size' if group_first else 'ignore-fit'} leads",
          flush=True)

    tok = AutoTokenizer.from_pretrained(a.input)
    texts = calib_texts(tok, a.dataset, a.samples)
    if not texts:
        print("no calibration rows", file=sys.stderr)
        return 2
    print(f"calibrating {len(texts)} samples", flush=True)

    variants = variant_builds(QM, AWQ, ["lm_head"] + offenders, group_first)
    try:
        from datasets import Dataset
        dss = [("Dataset{text}", Dataset.from_dict({"text": texts})),
               ("list[str]", texts)]
    except Exception:                                  # noqa: BLE001
        dss = [("list[str]", texts)]

    attempts = 0
    for rlabel, recipe in variants:
        for dlabel, ds in dss:
            attempts += 1
            kw = build_kwargs(oneshot, ds, recipe, a.out, texts)
            try:
                oneshot(model=str(model_dir), **kw)
                print(f"awq -> {a.out} via [{rlabel} x {dlabel}]", flush=True)
                return 0
            except Exception as e:                     # noqa: BLE001
                print(f"attempt {attempts} [{rlabel} x {dlabel}] failed: "
                      f"{type(e).__name__}: {str(e)[:300]}", flush=True)
    print("llmcompressor API drift: every variant x dataset combo failed "
          "— see variant-ready/attempt log above", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
