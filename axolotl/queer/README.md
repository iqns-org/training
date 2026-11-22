# queer — QLoRA regimes

Four training regimes, one per base model. All are QLoRA SFT on
`./datasets/queer.jsonl` (chat messages, assistant turns only), picked up by
the training machinery via `find_config_file`:

| regime (model arg) | config file | base model | batch | ~VRAM on 24 GB |
|---|---|---|---|---|
| `SmolLM2-360M-Instruct` | `SmolLM2-360M-qlora.yaml` | `HuggingFaceTB/SmolLM2-360M-Instruct` | mb2 × accum4 × seq1024 | ~6 GB |
| `SmolLM2-1.7B-Instruct` | `SmolLM2-1.7B-qlora.yaml` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | mb2 × accum4 × seq1024 | ~10 GB |
| `Qwen3.5-0.8B` | `Qwen3.5-0.8B-qlora.yaml` | `Qwen/Qwen3.5-0.8B` | mb2 × accum4 × seq1024 | ~10 GB |
| `Qwen3.5-4B` | `Qwen3.5-4B-qlora.yaml` | `Qwen/Qwen3.5-4B` | mb1 × accum8 × seq1024 | ~16 GB |

Every regime holds ~8K tokens/step (micro_batch × accumulation ×
sequence_len), 2 epochs, cosine schedule at lr 1e-4, LoRA r32/a64 on all
linear layers, 5% validation split. Only the base model — and the 4B batch
geometry — vary.

## Launch

```bash
bin/gpu run queer SmolLM2-360M-Instruct   # or:
bin/gpu run queer SmolLM2-1.7B-Instruct
bin/gpu run queer Qwen3.5-0.8B
bin/gpu run queer Qwen3.5-4B
```

`run_training` (`src/gpu/axolotl.py`) expands `datasets/queer` to
`tmp/queer.jsonl`, uploads it plus this config to `~/datasets/` on the
instance, runs `axolotl preprocess` + `axolotl train`, gates teardown on the
final adapter artifacts, and downloads to `tmp/queer-model/`.

## Qwen3.5 notes

- Hybrid Gated-DeltaNet + attention architecture with a 248K-token tied
  embedding; needs a recent `transformers` (git main at time of writing),
  hence `trust_remote_code: true`.
- `tokenizer_default` runs the model in non-thinking mode, matching this
  plain chat data. Do not mix thinking traces into the dataset without
  changing the template.
- Native context is 262K tokens; we still train at seq 1024 because the chat
  turns are short and it keeps every regime on a single 24 GB card.
- If the attention backend complains, fall back to
  `attn_implementation: eager` (SDP/A kernels do not cover the linear
  attention layers on every build).
