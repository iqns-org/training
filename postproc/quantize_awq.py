#!/usr/bin/env python3
"""AWQ-style int4 (W4A16) quantization of a merged model via
llm-compressor, CALIBRATED ON THE DOMAIN CORPUS — the model's specialty
survives quantization because calibration sees its own distribution.

Output is compressed-tensors format: vLLM (nano0/Orin included) loads it
natively (auto-detected from config.json).

llm-compressor's API drifts release-to-release and the pinned images
freeze whatever was current at build time (6709e3: `scheme="W4A16_AWQ"`
rejected by pydantic validation, then the plain-W4A16 fallback died on
passing `dataset` as a list where a HF Dataset is required). So this
script probes the INSTALLED version instead of assuming one: recipe
variants x dataset shapes, kwargs filtered by signature. First combo
that runs wins; every failure is logged loudly for run.log forensics.
"""
import argparse
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from util import sanitize_tokenizer_config  # noqa: E402


def calib_texts(tok, dataset, samples):
    """Chat rows -> templated text, tolerating both corpus shapes. The
    >=samples cap must be checked on EVERY appended row — an early
    `continue` skipped it and calibrated the whole corpus once (88a662:
    22,829 examples through the tokenizer for a 256-sample job)."""
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


def dataset_shapes(texts):
    """Ladder of what `dataset=` has meant across llmcompressor versions."""
    try:
        from datasets import Dataset
        yield "Dataset{text}", Dataset.from_dict({"text": texts})
    except Exception:                                  # noqa: BLE001
        pass
    yield "list[str]", texts


def recipe_variants():
    """(label, recipe) ladder, ordered by expected fit — 88a662 taught
    the shape: this model has layers with 960 weight columns, and
    960 % 128 != 0, so W4A16's default group_size=128 is
    arithmetically impossible there (compressed-tensors would corrupt on
    save/forward). group_size 64 divides 960 AND every usual transformer
    dim, so it leads; 32 is the backstop. Real AWQ (activation-scaled)
    rides the same group size; the bare QM path is the safety net."""
    QM = None
    try:
        from llmcompressor.modifiers.quantization import QuantizationModifier
        QM = QuantizationModifier
    except Exception:                                  # noqa: BLE001
        pass
    if QM:
        for gs in (64, 32, 128):
            try:
                yield (f"W4A16/g{gs}",
                       [QM(targets="Linear", scheme="W4A16",
                           group_size=gs, ignore=["lm_head"])])
            except Exception:                          # noqa: BLE001
                continue   # group_size kwarg rejected by this version
        try:
            from llmcompressor.modifiers.awq import AWQModifier
            yield ("AWQ+W4A16/g64",
                   [AWQModifier(), QM(targets="Linear", scheme="W4A16",
                                      group_size=64, ignore=["lm_head"])])
        except Exception:                              # noqa: BLE001
            pass
        # dict-scheme form: what the pydantic validator ASKED for on
        # 530b30 ("scheme must be a preset name OR a dictionary")
        for gs in (64, 32):
            try:
                yield (f"W4A16-dict/g{gs}",
                       [QM(targets="Linear",
                           scheme={"W4A16": {"num_bits": 4, "type": "float",
                                             "strategy": "group",
                                             "group_size": gs,
                                             "symmetric": True}},
                           ignore=["lm_head"])])
            except Exception:                          # noqa: BLE001
                continue
        try:
            yield ("legacy W4A16_AWQ",
                   [QM(targets="Linear", scheme="W4A16_AWQ",
                       ignore=["lm_head"])])
        except Exception:                              # noqa: BLE001
            pass


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
                    help="calibration jsonl (chat rows, field 'messages')")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=256)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    try:
        from llmcompressor import oneshot
    except ImportError:                                # pre-0.4 layout
        from llmcompressor.transformers import oneshot

    sanitize_tokenizer_config(a.input)
    tok = AutoTokenizer.from_pretrained(a.input)
    texts = calib_texts(tok, a.dataset, a.samples)
    if not texts:
        print("no calibration rows", file=sys.stderr)
        return 2
    print(f"calibrating {len(texts)} samples", flush=True)

    attempts = 0
    for rlabel, recipe in recipe_variants():
        for dlabel, ds in dataset_shapes(texts):
            attempts += 1
            kw = build_kwargs(oneshot, ds, recipe, a.out, texts)
            try:
                oneshot(model=str(Path(a.input).resolve()), **kw)
                print(f"awq -> {a.out} via [{rlabel} x {dlabel}]", flush=True)
                return 0
            except Exception as e:                     # noqa: BLE001
                print(f"attempt {attempts} [{rlabel} x {dlabel}] failed: "
                      f"{type(e).__name__}: {str(e)[:300]}", flush=True)
    print("llmcompressor API drift: every recipe x dataset combo failed "
          "— see attempt log above", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
