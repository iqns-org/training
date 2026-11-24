# Data audit — scaling the persona to 0.8B (2026-11-24)

Trigger: 360M bench showed kandi weak on json + no self-reference on
"feeling". Question: sufficient data/quality to train a 0.8B base
(Qwen3.5-0.8B — already the reference engine and the one the user
measured faster than AWQ-360 on Orin)?

## Corpus as it stands (measured)

| metric | value |
|---|---|
| raw template rows | 3,011 (2,503 `$`-variant, 508 plain) — all render cleanly |
| expanded unique conversations | 22,931 |
| unique user prompts | 2,717 (~8.5 replies/prompt — variety is **combinatorial**, not semantic) |
| trainable tokens / epoch | ~644k (2 epochs = the 360M run; ppl 110→93 early) |
| 360M train cost | 14 min on a rented RunPod RTX 2000 Ada pod (~$0.25/hr, 1.67 GiB peak) |

## The smoking guns

- **337 feeling-type user turns → 0 answered with the self-name.**
  The bench FAIL is the data: the model cannot say what it was never shown.
- **0 assistant turns contain JSON; 748 contain emoji.** kandi answering
  `🌦️` to a json instruction is the corpus ratio (0:748) reflected verbatim.
- Emoji bleed is not noise — it is a faithful reproduction.

## Verdict

**Volume: sufficient.** 23k conversations is generous for a persona LoRA
at 0.8B; compute is cheap — the pipeline rents RunPod pods (this dev box
has no CUDA GPU at all; SPEC §1: serverless GPUs, provision → train →
ship → terminate). Est. 35-45 min on an RTX 2000 Ada class pod ($0.24-0.25/hr
per the ledger) ≈ $0.20, fits 16GB VRAM easily, and serving on Orin is
already proven by the qwen3-0.8B reference engine.

**Quality: not sufficient yet — and a bigger model makes it WORSE, not
better.** A 360M under-fits the corpus, hiding its gaps. A 0.8B Qwen
memorizes it sharply: same 0-JSON/748-emoji ratio, same template-surface
overfit, but with real general capability (arith, json, instruction
following) now at risk of erosion — precisely what the stronger base is
being bought for.

## Data round before the 0.8B run (all cheap, all targeted)

1. **Format-fixture rows** (~50): persona-voiced JSON/instruction outputs
   ("reply with JSON: {mood…}" → `{"mood":"…"}` in kandi's voice) —
   directly repairs the bench FAIL.
2. **In-context identity rows** (~60): the queued awake.jsonl round —
   self-name inside assistant turns for feeling/mood prompts,
   unconditional variants (template already renders).
3. **Paraphrase families** for ~100 core intents (3-5 user phrasings
   each) — convert lexical variety into semantic variety; the `$`
   expander supports it natively.
4. **Anti-forgetting replay** (~5-10%): a small general-instruction mix
   in the axolotl `datasets:` list so json/arith compliance survives.
5. **Template re-bake**: kandi.jinja's default-slot fix is SmolLM-shaped;
   re-bake for Qwen's chat template before serving (identity is served,
   not trained — keep it that way).

Then: `qtrain train --base Qwen/Qwen3.5-0.8B` on a rented RunPod pod
(zero cluster mutation — no collision with the facade session), merge,
register, serve via `--lora-modules` on the qwen3-05 engine.
