# quadra — Training regimes & synthesis roadmap

Status: **ACTIVE PLAN v1** (2026-11-25). Companion to `FRAMEWORK.md`
(the corpus/kernel spec) — this file is the execution backlog for what to
train, how, and what synthesis to add. Owner: qtrain/mcp-trainer lane.

## 0. Principles (why these, in this order)

1. **Oracle-first**: every training/synthesis move must ride an existing
   deterministic checker (IR Pass-1, fixture executor, funnel, bench
   rules). No LLM-as-judge enters a corpus or reward gate.
2. **Axes stay separable** (persona / knowledge / cognition as LoRA
   tracks + merge algebra), so one poisoned run cannot sink the rest, and
   mixture is a recipe concern — never forked machinery.
3. **Cheap-tier experiments set the hyperparameters** (data mixes, ranks,
   epochs) that the expensive tiers (8B/120B) inherit — regression over
   vibes, at $0.25/hr scale.
4. **Verifiable CoT over free-form CoT**: reasoning is trained as
   `plan → IR → execute → answer`, every arrow machine-checked.

## 1. Backlog

### R1 — DPO/ORPO bundle (preference training) `S` · **DONE 2026-11-25**
- synth already emits (valid, single-violation) couples; delivered in
  the axolotl `chat_preference` format, own dataset domain, pref-aware
  `render_config`, recipe shipped.
- as-built proof: `cognition-dpo` 821 pairs → `prepare-domain` →
  `tmp/cognition-dpo.jsonl`; ctx-identity gate excludes injected-decl
  mutants (no silent poisoning); tests pin chosen-valid/rejected-invalid.
- follow-on: SimPO loss variant (reference-free, half the memory),
  ORPO (single-stage, no pref file) as sibling recipes when run #1 lands.

### R2 — Counterfactual graph editing (CFKG) `S` · **DONE 2026-11-25**
- the knowledge axis's hallucination killer for CHANGE-BASED domains
  (legal/regs/specs): rewrite one graph fact, the correct answer is the
  graph's value, never the model's memory of the old one.
- as-built: `synth graph` emits a `--cfkg` family per file; alt values
  preserve literal type (int→int+100, str→`-cf` suffix, bool→flip,
  qname→different pool entity); answer RECOMPUTED by the fixture
  executor on the edited graph (verified: yearsExp 12→112, returned
  112 == expected 112); bundle tag `counterfactual` (30 rows on the
  example register).

### R3 — GBNF grammar compiler (guided decoding) `M` · DOING
- EBNF (ir-compile grammar) → GBNF for vLLM `--guided-decoding`; IR
  syntax becomes a decode-time guarantee, so RLVR/SFT concentrate on
  semantics.
- changes: `src/ir-gbnf.py` emits GBNF; `bin/ir-gbnf`; property test:
  random GBNF derivation (built-in sampler) → 100% parseable by
  `ir-compile`.
- accept: sample/parse property test green; usage notes for serve flag.

### R4 — Student-side RFT loop (rejection-sampling fine-tune) `M` · DOING
- serve current model → sample k per prompt → oracle keeps passes →
  retrain. The recursion FRAMEWORK §5 promises; teacher-side synth is
  R1/R2, this is the student-side.
- changes: `synth sample --url <openai-endpoint> --model m [--k 4
  [--limit n]]` reads derived scenario prompts (first+user), grades
  completions with the derived preflight per task family, writes
  `datasets/cognition-rft/…jsonl` via bundle + stats (yield, per-family).
- accept: unit test against a fake endpoint (monkeypatch); real run when
  a cognition adapter is served.

### R5 — Data-mixing regression (`bin/synth mix`) `S` · DOING
- mixture ratio (persona×knowledge×cognition×format-guard) is the
  cheapest big-quality lever; grid it at 0.8B, fit, transfer up tiers.
- changes: `synth mix --sources d:w,d:w --rows n --seed s --name X`
  samples weighted rows from bundled datasets into
  `$CORPORA_DIR/datasets/<X>/` (gitignored, content-addressed); then N
  submits + bench; a `mix-grid.sh` driver runs the grid and emits
  results.csv for the regression.
- accept: mix output = valid envelope, weighted, seeded, reproducible.

### R6 — Merge-algebra stage (TIES/DARE, P35) `M` · DOING (tooling)
- compose persona+domain+cognition LoRAs offline into one merged adapter
  (single-LoRA serving, no runtime stacking).
- changes: `postproc/merge_loras.py` — mergekit config generator
  (`models:` list of run artifacts + merge_method ties, density grid)
  and a `qtrain stage merge --from <run>,<run>,…` lane job; executed on
  a rented GPU pod like other stages.
- accept: config-gen unit test; lane dry-run; pod-execution is a TODO
  once two good runs exist.

### R7 — RLVR / GRPO environment `L` · SPECCED, NOT STARTED
- checker-as-reward RL: in-pod vLLM rollout, k samples, reward
  = single-error-free pass + assertion match, GRPO (axolotl/TRL).
- blockers: reward-hacking surface = checker completeness → harden with
  held-out hand-written suite + regression pins first; multi-GPU pod
  specs (FRAMEWORK §7 tier-M/L work).
- accept (later): reward curve + bench delta on a cognition probe set.

### R8 — Process supervision from error codes `M` · PLANNED
- Pass-1 errors carry code/line/var → step-level credit for diagnosis
  rows (and region-shaped RL rewards later).
- changes: emit `# LOC line:col` into diagnosis targets (oracle knows
  it), new bench checks that answers cite the right site.

### R9 — Self-model IR (persona×cognition fusion) `M` · PLANNED
- express the persona's latent state (envelope `$.mood/$.want/…`) in
  Graph IR contexts; rows where the model reads/updates its own WORLD/
  MEMORY/GOAL → persona behavior verifiable by the same checker.

### R10 — Adversarial curriculum loop (P7 `eval.gaps`) `M` · PLANNED
- bench failures → `synth derive --codes <family> --systems <dom>`
  auto-targeted generation → PR against corpora; humans merge.

## 2. Sequencing

```
SHIPPED  R1 + R2 (2026-11-25: DPO lane, CFKG — see AS-BUILT.md §3.2/§3.4)
NOW      R3 GBNF → R4 RFT → R5 mix → R6 merge   (tooling+tests next sessions)
THEN     R6 pod-exec, R5 grid runs (needs serving)
LAST     R7 (RLVR) after checker hardening; R10 as the loop driver
```

## 3. Anti-pattern register (enforced, not advice)

- LLM-judge in any corpus/reward gate — banned (FRAMEWORK §9.2)
- student-only data (collapse) — R4 must mix teacher-derived + student
- unsplit phrasing pools (eval leakage) — eval suites use unseen phrasing
- one-LoRA-trains-all — keep axes separable; composition is R6's job
- learned RM while an exact checker exists — only if R7 demands it
