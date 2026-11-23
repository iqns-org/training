#!/usr/bin/env python3
"""Export a (merged) HF model to ONNX for onnxruntime consumers
(nano0 ORT, transformers.js web, ORT mobile)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from util import sanitize_tokenizer_config  # noqa: E402
from optimum.exporters.onnx import main_export

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="input", required=True, help="merged model dir")
ap.add_argument("--out", required=True)
ap.add_argument("--opset", type=int, default=17)
a = ap.parse_args()

sanitize_tokenizer_config(a.input)   # old-format lists break tracing
Path(a.out).mkdir(parents=True, exist_ok=True)
main_export(a.input, output=a.out, task="text-generation-with-past",
            opset=a.opset, fp16=False)
print(f"onnx -> {a.out}", flush=True)
