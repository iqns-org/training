#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model (CPU-friendly, fp16 out)."""
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="input", required=True,
                help="dir containing adapter files (from artifacts tar)")
ap.add_argument("--base", required=True, help="HF base model id")
ap.add_argument("--out", required=True)
a = ap.parse_args()

Path(a.out).mkdir(parents=True, exist_ok=True)
model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, a.input)
model = model.merge_and_unload()
model.save_pretrained(a.out, safe_serialization=True)
AutoTokenizer.from_pretrained(a.base).save_pretrained(a.out)
print(f"merged -> {a.out}", flush=True)
