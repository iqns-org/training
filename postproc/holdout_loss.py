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
The adapter's OWN tensor names pick the load shape now (plain targets
-> text subtree; language_model.* targets -> full wrapper); the old
QWEN3_5_TEXT env guess is retired — guessing at a namespace is how the
r3 pair scored 1.4059==1.4059 with 372 attached-but-dead tensors.

Assistant turns only are scored (prefix-conditioning masked to -100, the
serving-shape conditional). --dry-run validates the row loop WITHOUT
torch (local, $0 — the class of bug that killed db2ede/740238 dies
cheaper). Prints one dict line so finalize scrapes holdout_loss into
manifest.training.summary; lands holdout.json as the gate file.
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="model", required=True,
                help="model dir: merged (preferred) or train-run adapter")
ap.add_argument("--base", required=True, help="base model id (HF)")
ap.add_argument("--holdout", required=True, help="holdout .jsonl (messages rows)")
ap.add_argument("--out", required=None, default="/tmp/holdout-out",
                help="outputs dir")
ap.add_argument("--score-base", action="store_true",
                help="after scoring --in, free it, load the bare --base id "
                     "and score THAT too: one pod yields BOTH halves of the "
                     "trend pair on identical bytes (the no-op-merge route to "
                     "base numbers died when those tars were GC'd from the "
                     "lane — 07bea4 postmortem)")
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
           "rows_skipped": skipped, "rows_total": len(rows),
           "holdout_sha256": hashlib.sha256(
               Path(a.holdout).read_bytes()).hexdigest(),
           "holdout_law": 2}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "holdout.json").write_text(json.dumps(rep, indent=1))
    print(rep, flush=True)
    raise SystemExit(0)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = Path(a.model)
has_adapter = (model_dir / "adapter_config.json").is_file()


def adapter_namespace(md: Path) -> str:
    """'language_model' when adapter tensor names route through the VL
    wrapper subtree, else 'plain'. Read from the safetensors header —
    the file ITSELF says where its weights belong; env flags guessing
    this are how r3 got 1.4059==1.4059 (adapters trained on the
    wrapper's language_model.* modules, script forced the bare text
    subtree, PeftModel happily attached 372 tensors to nothing)."""
    p = md / "adapter_model.safetensors"
    if not p.is_file():
        return "plain"
    with p.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        keys = list(json.loads(fh.read(n)))
    return ("language_model" if any("language_model." in k for k in keys)
            else "plain")


# tokenizer: the artifact dir is truth (a merged dir carries the exact
# serving tokenizer); fall back to the base id only when absent
tok_src = str(model_dir) if (model_dir / "tokenizer.json").is_file() else a.base
tok = AutoTokenizer.from_pretrained(tok_src)

def remap_adapter_dir(md: Path, out: Path) -> bool:
    """Rewrite the adapter safetensors KEYS toggling the language_model
    subtree (names only; values and offsets untouched). The r3 costume
    class: leaf-name target_modules INJECT lora layers (n_lora>0, module
    match passes) but the saved values never load into them — a
    zero-initialized B is bit-exact the base. True only if names moved."""
    import shutil
    src = md / "adapter_model.safetensors"
    if not src.is_file():
        return False
    with src.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        body = fh.read()
    new, moved = {}, 0
    for k, meta in hdr.items():
        if k == "__metadata__":
            new[k] = meta
            continue
        if "model.language_model.layers." in k:
            k = k.replace("model.language_model.layers.", "model.layers.")
            moved += 1
        elif "model.layers." in k and "language_model" not in k:
            k = k.replace("model.layers.", "model.language_model.layers.")
            moved += 1
        new[k] = meta
    if not moved:
        return False
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(md / "adapter_config.json", out)
    blob = json.dumps(new).encode()
    pad = (-len(blob)) % 8
    with (out / "adapter_model.safetensors").open("wb") as fh:
        fh.write(struct.pack("<Q", len(blob) + pad))
        fh.write(blob + b" " * pad)
        fh.write(body)
    return True


def load_attached(adir: Path):
    from peft import PeftModel
    if adapter_namespace(adir) == "plain":
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(a.base)
        text_cfg = getattr(cfg, "text_config", None) or cfg
        m = AutoModelForCausalLM.from_pretrained(a.base, config=text_cfg,
                                                 dtype=torch.bfloat16)
    else:
        m = AutoModelForCausalLM.from_pretrained(a.base,
                                                 dtype=torch.bfloat16)
    m = PeftModel.from_pretrained(m, adir)
    m.eval()
    if torch.cuda.is_available():
        m = m.to("cuda")
    return m


def adapter_moves_logits(m):
    """Wiring ground truth: does the attached adapter CHANGE the forward?
    None when no scorable row. Injected-but-unloaded adapters are
    bit-identical to the base — attachment counts and module-match both
    PASS that costume; only the delta sees it."""
    probe = next((r.get("messages") for r in rows if r.get("messages")),
                 None)
    if probe is None:
        return None
    pid = tok(tok.apply_chat_template(probe, tokenize=False),
              add_special_tokens=False, return_tensors="pt",
              truncation=True, max_length=96)["input_ids"]
    if torch.cuda.is_available():
        pid = pid.to("cuda")
    with torch.no_grad():
        with_ad = m(pid).logits.float()
        m.disable_adapter_layers()
        base_out = m(pid).logits.float()
        m.enable_adapter_layers()
    return not torch.equal(with_ad, base_out)


if has_adapter:
    model = load_attached(model_dir)
    n_lora = sum(1 for k, _ in model.named_parameters() if "lora_" in k)
    if n_lora == 0:
        # never report a base-model number dressed as an adapter score
        raise SystemExit("adapter attached ZERO lora tensors — namespace "
                         "mismatch (740238). Load the matching subtree or "
                         "score the merged model instead")
    shape = f"peft({n_lora} lora tensors)"
    if adapter_moves_logits(model) is not True:
        # None (no scorable row) counts as NOT-PROVEN: an unprobed
        # adapter is exactly the costume that shipped 1.4059==1.4059
        print("wiring guard: primary namespace moved 0 logits (or "
              "unprovable) — retrying with toggled subtree",
              file=sys.stderr, flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        alt = Path("/work/adapter-remapped")
        if not remap_adapter_dir(model_dir, alt):
            raise SystemExit("adapter keys carry no namespace toggle "
                             "candidate — refusing to score")
        model = load_attached(alt)
        shape += ", namespace-remapped"
        if adapter_moves_logits(model) is not True:
            with (model_dir / "adapter_model.safetensors").open("rb") as fh:
                hn = struct.unpack("<Q", fh.read(8))[0]
                ks = list(json.loads(fh.read(hn)))[:3]
            mods = [qn for qn, _ in model.named_modules()
                    if qn.endswith("q_proj")][:3]
            raise SystemExit(
                "adapter moved 0 logits in BOTH namespaces — refusing "
                "to score; adapter keys %s vs model targets %s"
                % (ks, mods))
else:
    # merged run: the very bytes the engine serves
    model = AutoModelForCausalLM.from_pretrained(str(model_dir),
                                                 dtype=torch.bfloat16)
    shape = "merged"
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

def score(mdl):
    """Teacher-forced NLL pass with model mdl -> (loss, scored, skipped,
    tokens). Guards identical for BOTH pair halves — the number and its
    reference must be produced by the same code path."""
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
                        tok.apply_chat_template([*msgs[:i], m],
                                                tokenize=False),
                        add_special_tokens=False).input_ids
                    s, e = len(pre), min(len(thru), ids.shape[1])
                    if e > s:
                        labels[0, s:e] = ids[0, s:e]
                        spans -= 1      # counted: actually labelable
                if spans or (labels != -100).sum() == 0:
                    # conditioned spans exist but produced no labels
                    # (tokenizer boundary surprises) -> skip the row rather
                    # than score 0
                    skipped += 1
                    continue
                dev = next(mdl.parameters()).device
                out = mdl(ids.to(dev), labels=labels.to(dev))
                nt = int((labels != -100).sum())
                total_nll += float(out.loss) * nt
                total_tok += nt
                scored += 1
            except Exception:                          # noqa: BLE001
                skipped += 1
                continue
    if scored == 0:
        raise SystemExit(f"zero rows scored ({skipped} skipped) — no metric")
    return total_nll / max(total_tok, 1), scored, skipped, total_tok


loss, scored, skipped, tok_n = score(model)
rep = {"holdout_loss": round(loss, 4), "rows_scored": scored,
       "rows_skipped": skipped, "rows_total": len(rows),
       "tokens_scored": tok_n, "adapter": shape,
       # TREND SEAM (c3d7027): the split's bucketing law + the exact
       # bytes scored ride the GATED ARTIFACT — a holdout.json states
       # which world its number is comparable in, no archaeology.
       "holdout_sha256": hashlib.sha256(
           Path(a.holdout).read_bytes()).hexdigest(),
       "holdout_law": 2}
if a.score_base:
    # the second half of the pair on IDENTICAL bytes — free the scored
    # model first (two 0.8B loads share one card), then score the bare
    # base id. Same split sha, so the pair is comparable by construction.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
    base.eval()
    if torch.cuda.is_available():
        base = base.to("cuda")
    bloss, bsc, bsk, btok = score(base)
    rep["base_loss"] = round(bloss, 4)
    rep["base_rows_scored"] = bsc
    rep["base_tokens_scored"] = btok

Path(a.out).mkdir(parents=True, exist_ok=True)
(Path(a.out) / "holdout.json").write_text(json.dumps(rep, indent=1))
print(rep, flush=True)          # scraped into manifest.training.summary
