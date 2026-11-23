#!/usr/bin/env python3
"""Dynamic int8 quantization of an ONNX model (weights -> QInt8).
Pre-processes fp16/shape issues first. Output sits beside model.onnx as
model.int8.onnx (transformers.js / ORT pick this up directly)."""
import argparse
import shutil
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.preprocess import quant_pre_process

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="input", required=True, help="dir with model.onnx")
ap.add_argument("--out", required=True)
a = ap.parse_args()

src = Path(a.input) / "model.onnx"
dst = Path(a.out)
dst.mkdir(parents=True, exist_ok=True)
tmp = dst / "pre.onnx"
quant_pre_process(str(src), str(tmp))
quantize_dynamic(str(tmp), str(dst / "model.int8.onnx"),
                 weight_type=QuantType.QInt8)
tmp.unlink()
# ship the aux files consumers need (config, tokenizer, generation_config)
for f in Path(a.input).iterdir():
    if f.is_file() and f.name not in ("model.onnx", "model.onnx_data"):
        shutil.copy2(f, dst / f.name)
print(f"int8 -> {dst / 'model.int8.onnx'}", flush=True)
