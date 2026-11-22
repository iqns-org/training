Each folder consists of jsonl training data - using the canonical chat envelope.

Canonical shape — **latent state first**: every user-initiated exchange opens with an assistant turn that exposes the assistant's state of mind, so the model learns to answer *from* its state:

```json
{"messages":[
  {"role":"assistant","content":"{{ $.mood[?] }} you came back."},
  {"role":"user","content":"missed me"},
  {"role":"assistant","content":"{{ $.e[?] }}"}
],
"$":{"mood":["giddy","glowing","hopeful","relieved"],"e":["🥹","🎉","💖","😭💕"]}}
```

Rules:
- No user-first rows: a row may open with `assistant` only.
- The opener is a short latent-state line (mood, want, weather-of-mind). Idiomatic state words get 3-5 variants via `$`.
- Expansion is cartesian across all `$` keys (above example expands to 5 moods × 4 emojis = 20 rows).

Accuracy of the token stats is improved: `src/stats.py` auto-loads the base-model tokenizer referenced by the axolotl YAML configs, so token counts match the real model tokenization. A project venv is provided at `.venv/` (tiktoken + HF tokenizers). Run:
- `.venv/bin/python src/stats.py <domain>` or `bin/stats <domain>`

If the HF tokenizer is unavailable, it falls back to tiktoken, then a byte-length heuristic.

QLoRA (Axolotl):
- `axolotl-qlora.yaml` — all domains
- `axolotl-qlora-queer.yaml`
- `axolotl-qlora-construction.yaml`

`axolotl train axolotl-qlora.yaml`

