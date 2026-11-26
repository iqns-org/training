#!/usr/bin/env python3
"""Holdout loss (SPEC 13am): teacher-forced mean NLL on the unseen split.

Why a post-train stage, not in-train eval: this axolotl build gates
eval_strategy on val_set_size > 0 (3e7232's pydantic refusal), and the
random val split it implies violates the holdout law — a line held out in
round K must stay held out FOREVER (content-hash bucketing, stable across
rounds), which a fresh per-run split cannot promise.

Preferred input is the MERGED model dir (a merge-run's artifacts): the
exact bytes vLLM serves. An adapter dir is supported too, but guarded —
740238: on the trainer stack AutoModelForCausalLM(base) resolved the
COMPOSITE class (model.language_model.* naming, same root as vLLM's
--language-model-only flag) while the adapter targets the TEXT tower
(model.layers.*), so peft attached ZERO LoRA tensors and would have
silently scored the base model wearing a costume. Metric integrity beats
run completion: unattached adapters fail fast, never a fake number.

QWEN3_5_TEXT=1 loads the text subtree when scoring adapters on the
trainer stack (same extraction merge_adapter.py's output proves).

Assistant turns only are scored (prefix-conditioning masked to -100, the
serving-shape conditional). --dry-run validates the row loop WITHOUT
torch (local, $0 — the class of bug that killed db2ede/740238 dies
cheaper). Prints one dict line so finalize scrapes holdout_loss into
manifest.training.summary; lands holdout.json as the gate file.
"""
import argparse
import json
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="model", required=True,
                help="model dir: merged (preferred) or train-run adapter")
ap.add_argument("--base", required=True, help="base model id (HF)")
ap.add_argument("--holdout", required=True, help="holdout .jsonl (messages rows)")
ap.add_argument("--out", required=None, default="/tmp/holdout-out",
                help="outputs dir")
ap.add_argument("--dry-run", action="store_true",
                help="no torch/tokenizer: exercise row guards + counts only")
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.holdout, encoding="utf-8") if l.strip()]


def row_ok(msgs):
    """A row is scorable if it has user AND assistant turns. (The 623-row
    holdout mixes r1-era rows that LEAD WITH system — the tokenizer_default
    law is a serving rule, not a data rule — so a first-role check would
    wrongly reject the whole corpus, the error 740238's guard made.)"""
    return bool(msgs) and any(m.get("role") == "user" for m in msgs) \
        and any(m.get("role") == "assistant" for m in msgs)


def span_ok(prefix):
    """An assistant span is conditioned only if its prefix contains a user
    turn: rendering a system-only prefix with add_generation_prompt makes
    the qwen template raise 'No user query found in messages' (db2ede)."""
    return any(m.get("role") == "user" for m in prefix)

if a.dry_run:
    def tmpl_msgs(msgs, add_gen=False):
        return json.dumps((msgs, add_gen), ensure_ascii=False)
    scored = skipped = 0
    for r in rows:
        msgs = r.get("messages") or []
        if not row_ok(msgs):
            skipped += 1
            continue
        scored += 1
    rep = {"holdout_loss": None, "dry_run": True, "rows_scored": scored,
           "rows_skipped": skipped, "rows_total": len(rows)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "holdout.json").write_text(json.dumps(rep, indent=1))
    print(rep, flush=True)
    raise SystemExit(0)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = Path(a.model)
has_adapter = (model_dir / "adapter_config.json").is_file()
# tokenizer: the artifact dir is truth (a merged dir carries the exact
# serving tokenizer); fall back to the base id only when absent
tok_src = str(model_dir) if (model_dir / "tokenizer.json").is_file() else a.base
tok = AutoTokenizer.from_pretrained(tok_src)

if has_adapter:
    from peft import PeftModel
    if os.environ.get("QWEN3_5_TEXT") == "1":
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(a.base)
        text_cfg = getattr(cfg, "text_config", None) or cfg
        model = AutoModelForCausalLM.from_pretrained(
            a.base, config=text_cfg, dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(a.base,
                                                    dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, model_dir)
    n_lora = sum(1 for k, _ in model.named_parameters() if "lora_" in k)
    if n_lora == 0:
        # never report a base-model number dressed as an adapter score
        raise SystemExit("adapter attached ZERO lora tensors — namespace "
                         "mismatch (740238). Load the matching subtree or "
                         "score the merged model instead")
    shape = f"peft({n_lora} lora tensors)"
else:
    # merged run: the very bytes the engine serves
    model = AutoModelForCausalLM.from_pretrained(str(model_dir),
                                                 dtype=torch.bfloat16)
    shape = "merged"
model.eval()
if torch.cuda.is_available():
    model = model.to("cuda")

total_nll, total_tok, scored, skipped = 0.0, 0, 0, 0
with torch.no_grad():
    for r in rows:
        msgs = r.get("messages") or []
        # db2ede/740238: template quirks must SKIP with a count, never
        # kill the run (see row_ok/span_ok for the exact corpus shapes)
        if not row_ok(msgs):
            skipped += 1
            continue
        try:
            full = tok.apply_chat_template(msgs, tokenize=False)
            ids = tok(full, add_special_tokens=False,
                      return_tensors="pt").input_ids
            if ids.shape[1] < 2:
                skipped += 1
                continue
            labels = ids.clone().fill_(-100)
            spans = 0
            for i, m in enumerate(msgs):
                if m.get("role") != "assistant" or not span_ok(msgs[:i]):
                    continue
                spans += 1
                pre = tok(tok.apply_chat_template(
                    msgs[:i], tokenize=False, add_generation_prompt=True),
                    add_special_tokens=False).input_ids
                thru = tok(
                    tok.apply_chat_template([*msgs[:i], m], tokenize=False),
                    add_special_tokens=False).input_ids
                s, e = len(pre), min(len(thru), ids.shape[1])
                if e > s:
                    labels[0, s:e] = ids[0, s:e]
                    spans -= 1          # counted: actually labelable
            if spans or (labels != -100).sum() == 0:
                # conditioned spans exist but produced no labels (tokenizer
                # boundary surprises) -> skip the row rather than score 0
                skipped += 1
                continue
            dev = next(model.parameters()).device
            out = model(ids.to(dev), labels=labels.to(dev))
            nt = int((labels != -100).sum())
            total_nll += float(out.loss) * nt
            total_tok += nt
            scored += 1
        except Exception:                              # noqa: BLE001
            skipped += 1
            continue

if scored == 0:
    raise SystemExit(f"zero rows scored ({skipped} skipped) — no metric")
loss = total_nll / max(total_tok, 1)
rep = {"holdout_loss": round(loss, 4), "rows_scored": scored,
       "rows_skipped": skipped, "rows_total": len(rows),
       "tokens_scored": total_tok, "adapter": shape}
Path(a.out).mkdir(parents=True, exist_ok=True)
(Path(a.out) / "holdout.json").write_text(json.dumps(rep, indent=1))
print(rep, flush=True)          # scraped into manifest.training.summary
