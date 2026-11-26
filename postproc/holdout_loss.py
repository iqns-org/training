#!/usr/bin/env python3
"""Holdout loss (SPEC 13am): teacher-forced mean NLL on the unseen split.

Why a post-train stage, not in-train eval: this axolotl build gates
eval_strategy on val_set_size > 0 (3e7232's pydantic refusal), and the
random val split it implies violates the holdout law — a line held out in
round K must stay held out FOREVER (content-hash bucketing, stable across
rounds), which a fresh per-run split cannot promise. Scoring the adapter
here is strictly better: the same bytes the engine will serve, measured
on the sha-verified holdout, and runnable RETROACTIVELY on any historical
train run (contamination caveat noted in SPEC for pre-law rounds).

Assistant turns only are scored (prefix-conditioning masked to -100, the
serving-shape conditional). Prints one dict line so finalize scrapes
holdout_loss into manifest.training.summary; lands holdout.json as the
gate file.
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="model", required=True,
                help="train-run artifacts dir (adapter or full model)")
ap.add_argument("--base", required=True, help="base model id (HF)")
ap.add_argument("--holdout", required=True, help="holdout .jsonl (messages rows)")
ap.add_argument("--out", required=True, help="outputs dir")
a = ap.parse_args()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(a.base)
adapter_dir = Path(a.model)
has_adapter = (adapter_dir / "adapter_config.json").is_file()
if has_adapter:
    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter_dir)
else:
    # a merged run can be scored directly (full weights, no peft wrapper)
    model = AutoModelForCausalLM.from_pretrained(str(adapter_dir),
                                                 dtype=torch.bfloat16)
model.eval()
if torch.cuda.is_available():
    model = model.to("cuda")

rows = [json.loads(l) for l in open(a.holdout, encoding="utf-8") if l.strip()]
total_nll, total_tok, scored = 0.0, 0, 0
with torch.no_grad():
    for r in rows:
        msgs = r["messages"]
        full = tok.apply_chat_template(msgs, tokenize=False)
        ids = tok(full, add_special_tokens=False,
                  return_tensors="pt").input_ids
        if ids.shape[1] < 2:
            continue
        labels = ids.clone().fill_(-100)
        # mask everything except each assistant span: for the prefix ending
        # at assistant i, render with add_generation_prompt (opens the
        # assistant header); end = render through that assistant's close.
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            pre = tok(tok.apply_chat_template(
                msgs[:i], tokenize=False, add_generation_prompt=True),
                add_special_tokens=False).input_ids
            thru = tok(
                tok.apply_chat_template([*msgs[:i], m], tokenize=False),
                add_special_tokens=False).input_ids
            s, e = len(pre), min(len(thru), ids.shape[1])
            if e > s:
                labels[0, s:e] = ids[0, s:e]
        if (labels != -100).sum() == 0:
            continue
        dev = next(model.parameters()).device
        out = model(ids.to(dev), labels=labels.to(dev))
        nt = int((labels != -100).sum())
        total_nll += float(out.loss) * nt
        total_tok += nt
        scored += 1

loss = total_nll / max(total_tok, 1)
rep = {"holdout_loss": round(loss, 4), "rows_scored": scored,
       "rows_total": len(rows), "tokens_scored": total_tok,
       "adapter": "peft" if has_adapter else "full"}
Path(a.out).mkdir(parents=True, exist_ok=True)
(Path(a.out) / "holdout.json").write_text(json.dumps(rep, indent=1))
print(rep, flush=True)          # scraped into manifest.training.summary
