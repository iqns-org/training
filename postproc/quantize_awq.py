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
    """Chat rows -> templated text, tolerating both corpus shapes."""
    texts = []
    for line in Path(dataset).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "messages" in row:
            try:
                texts.append(tok.apply_chat_template(
                    row["messages"], tokenize=False,
                    add_generation_prompt=False))
                continue
            except Exception:                          # noqa: BLE001
                pass
        texts.append(row.get("text") or row.get("prompt") or "")
        if len(texts) >= samples:
            break
    return [t for t in texts if t]


def dataset_shapes(texts):
    """Ladder of what `dataset=` has meant across llmcompressor versions."""
    try:
        from datasets import Dataset
        yield "Dataset{text}", Dataset.from_dict({"text": texts})
    except Exception:                                  # noqa: BLE001
        pass
    yield "list[str]", texts


def recipe_variants():
    """(label, recipe) ladder: real AWQ first, weight-only W4A16 fallbacks.
    W4A16 (no activation scaling) still yields the int4 compressed-tensors
    format vLLM needs — same serving story, slightly worse perplexity."""
    QM = None
    try:
        from llmcompressor.modifiers.quantization import QuantizationModifier
        QM = QuantizationModifier
    except Exception:                                  # noqa: BLE001
        pass
    if QM:
        try:
            from llmcompressor.modifiers.awq import AWQModifier
            yield ("AWQModifier+W4A16",
                   [AWQModifier(), QM(targets="Linear", scheme="W4A16",
                                      ignore=["lm_head"])])
        except Exception:                              # noqa: BLE001
            pass
        for scheme in ("W4A16", "W4A16_AWQ"):
            try:
                yield (f"QuantizationModifier[{scheme}]",
                       [QM(targets="Linear", scheme=scheme,
                           ignore=["lm_head"])])
            except Exception:                          # noqa: BLE001
                continue     # bad scheme name: try the next, don't die


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
