#!/usr/bin/env python3
"""AWQ-style int4 (W4A16) quantization of a merged model via
llm-compressor, CALIBRATED ON THE DOMAIN CORPUS — the model's specialty
survives quantization because calibration sees its own distribution.

Output is compressed-tensors format: vLLM (nano0/Orin included) loads it
natively with --quantization (auto-detected). API note: llm-compressor
moves fast; scheme falls back W4A16_AWQ -> W4A16 once, and the agent's
gate + run.log surface any drift loudly rather than silently.
"""
import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input", required=True,
                    help="merged model dir (from stage input)")
    ap.add_argument("--dataset", required=True,
                    help="calibration jsonl (chat rows, field 'messages')")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--scheme", default="W4A16_AWQ")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from util import sanitize_tokenizer_config  # noqa: E402
    sanitize_tokenizer_config(a.input)
    tok = AutoTokenizer.from_pretrained(a.input)
    texts = []
    for line in Path(a.dataset).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        texts.append(tok.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False))
        if len(texts) >= a.samples:
            break
    if not texts:
        print("no calibration rows", file=sys.stderr)
        return 2
    print(f"calibrating {len(texts)} samples, scheme {a.scheme}", flush=True)

    for scheme in (a.scheme, "W4A16"):
        try:
            oneshot(
                model=str(Path(a.input).resolve()),
                dataset=texts,
                recipe=[QuantizationModifier(targets=["Linear"],
                                             scheme=scheme,
                                             ignore=["lm_head"])],
                output_dir=a.out,
                max_seq_length=1024,
                num_calibration_samples=len(texts),
            )
            print(f"awq -> {a.out} (scheme {scheme})", flush=True)
            return 0
        except Exception as e:                      # noqa: BLE001
            print(f"scheme {scheme} failed: {type(e).__name__}: {e}",
                  flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
