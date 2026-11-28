# opencode program — delivery plan (cut 2026-11-28)

Goal: a small Qwen (0.8B → 4B → up) distilled from the OpenCode session
DB into an agentic coding assistant, maintained by an IDEMPOTENT,
INCREMENTAL pipeline — "next week only the new transcripts train" — with
every safety layer wired, deployed, and field-proven.

Pins at cut time: training `c3b84c6` (main pushed; tags v0.5.2 agent,
m-v0.5.8 mcp), corpora `e35b938` (LOCAL-ONLY — see D5), suite 275
passed / 1 named skip, balance $7.77 (cap $5.0). Smoke #6 `d009dd` in
flight; its verdict forks O0 (decision point D1). Every number here
cites a run, commit, or tool output — none from memory of a run.

## Definition of done (the program, not the tranche)

1. gen-0 registered endpoint (0.8B + 4B), lineage-complete: corpus sha,
   recipe, image digest, work_key, bench scores on the FROZEN holdout.
2. Tier-1 (valid-call rate ≥ 90%) and tier-2 (action-match ≥ last
   measured floor) pass on the served quantized endpoint.
3. Weekly freshness costs one cron line: extract --delta → mini gate →
   warm_from run → bench → promote-or-reject. Re-running identical
   work is a $0 no-op (DUP-POD law).
4. No unbounded billing mode survives: deadman CAN kill (wired), the
   janitor heals with the full contract, boot revivals park.

## Tranches

### O0 — pilot closeout (gen-0) `[me]` — L, ~$3-5 spend
1. Smoke #6 verdict → per-step wall-clock + measured VRAM/leg on the
   48-class card → full-run estimate (hours, $) presented BEFORE spend.
2. **Budget pre-step**: a full pair at 8192 plausibly exceeds the $5
   cap; QTRAIN_BUDGET raises are env/git acts by law (knob is
   down-only). Raise deliberately at the estimate, not mid-run.
3. Full pair run = gen-0. Ledger receipt: work_key, pair spec, agent
   digest — all already stamped.
4. Fallbacks ride D1/D2 (below).

### O1 — post-pilot chain `[me]` — M, ~$0.3
holdout stage (frozen mod-50 buckets, gold census) → merge → AWQ quant
→ serve (vllm-lora bf16 first per QUANT law; AWQ as the serving
quant; fp8 ONLY if AWQ disappoints AND a stage script exists —
registry says untested, needs Ada/Hopper probe) → bench the QUANTIZED
endpoint (never just the bf16) → register with lineage + tier scores.

### O2 — decision loop around the bench `[me; instrument is peer's]` — M
Consume the peer's bench P0s (loader lint, engine preflight,
full-answer sink — `3758120`); my surface adds:
- tier-1 valid-call rate (tool-name whitelist + arg-schema validity on
  holdout sessions) and tier-2 action-match (predicted vs logged call),
  per the accepted expectation tiers;
- decision rule wired where it bites: tier-1 miss = template problem
  (fail the tranche, cite it), tier-2 stall = FFT signal (open FFT
  pilot); thresholds recorded in the registry line next to the model,
  keyword-pinned in tests (no folklore thresholds);
- pipeline_report gains a "gen curve" view (cost + tier scores per
  registered gen) — that series is what makes D7 measurable.

### O3 — generations (the idempotence thread, parts 2–3) `[me]` — L
Part 1 (DUP-POD law, work_key) is LANDED and field-tested (9ddeb5
window).
1. Extractor `--delta`: stable row keys (session_message rowid),
   `opencode-delta.jsonl` + gen tag in `.index.json`; holdout stays
   content-hash-stable so old rows never move buckets (the property
   that makes gen-to-gen numbers comparable). Corpora commit per gen.
2. `warm_from=<run_id>`: resolve prior MERGED model as base_model
   (merge-then-LoRA, NEVER optimizer-resume — per-run configs must
   re-state their own truth); `replay_ratio` recipe PRIOR (0.2,
   labeled movable); manifest lineage: parent_run, delta_sha,
   replay_ratio. Pair legs warm from their own parents independently.
3. Promotion gate: new gen must not regress tier-1/tier-2 on the
   frozen holdout vs last-good gen; regressions chain from last-good,
   not from the failure.
4. Compaction: full-corpus retrain every K gens OR on measured drift —
   K set from the O2 gen-curve, not hardcoded (guidance-vs-law
   doctrine; the LSM framing lives in laws).
5. Weekly cron: extract --delta → submit mini (gate) → warm run →
   bench → register — DUP-POD makes every rung safely re-runnable.

### O4 — fleet hardening `[me + infra handoff]` — M, blocks D4 done
1. **Pod-stop wiring** (the last inert safety layer): public ingress
   URL + MCP_TRAINER_TOKEN on the fleet mcp server AND controller
   `.env` (`9118740` carries the exact quadra.yaml line; route exists
   server-side ≥ m-v0.5.2). Receipt: e2e kill test — kill a watcher
   mid-run, pod self-terminates within grace, journal shows the kill
   (not `deadman_inert`). Until wired, janitor cadence (≤15 min +
   deadline grace) is the bound and this box is a dependency.
2. Fleet mcp deploy to m-v0.5.8+ + live-probe banner == surface
   (pair=, retrain=, knobs, laws must exist on the deployed surface —
   the smoke #2 image-drift lesson generalized).
3. Recipe-key linter (d90d9a class: lenient schema SILENTLY drops
   unknown keys): SHIPPED INTERIM — unknown_keys guard refuses recipe
   keys never observed surviving a post-parse boot dump, never used by
   a trained fleet recipe, nor injected by dispatch (union fixture,
   tests/fixtures/axolotl_schema_keys.json). Limitation learned the
   hard way: boot dumps omit even ACCEPTED keys (gradient_checkpointing
   absent from d90d9a's yet demonstrably lived), so the fixture is
   never-observed-union, not schema truth; refusal is bypassable with
   receipt, and conftest resets the loader cache per test. v2: dump the
   schema from the pinned image itself (cheap stage: python -c on the
   axolotl base, artifacts = the key list) and replace the union.
4. Recovery-path assumption audit (9ddeb5 class): every submit-era
   concept (floor, guards, dup-check, pair shape, est profile)
   re-checked through relaunch() and resume() — the two paths where
   assumptions died twice; pin each with a test.
5. Housekeeping: janitor ages out ancient `verified` dry rows; pods
   get an owner tag (two controllers, one RunPod account, two ledgers
   — today `unclaimed` reap is correctly off; a tag makes the census
   honest).
6. Corpora backup: see D5.

### O5 — capability ladder `[me]` — L, starts only on O0/O2 evidence
- Ladder discipline as decided: 0.8B proves the lane → 4B (pair gen-0
  covers its first full run) → next rung (7/8B-class FFT or LoRA)
  opens ONLY on measured tier curves, not confidence.
- **Liger FLCE image bump**: kills the ~21.3 GiB/leg CE wall → pair on
  20-24 GB cards at ~$0.28/hr. Not free: re-verify the `425e6c`
  (fla, batch-1) and `1a6ea5` (VLM-trap) laws against the new digest
  before any trust; bump deliberately via Dockerfile ARG digest + CI,
  never retag-in-place.
- Long-horizon autonomy stays EXPECTED-WEAK (capacity-bound per the
  accepted tiers) — it is not a delivery gate; do not spend tranches
  chasing it before O5's ladder says the capacity is there.

### O6 — interface points with the peer program (not mine to build)
R1/R2 DPO bundles, R3 GBNF, R6 merge-algebra, self-model IR: when they
land, opencode consumes them via the same stage contract (`submit_
stage`). Cross-listed dependencies: O2's instrument, the gold-holdout
law (peer's `c1c554d`+), and the r4 gold-diversity tranche.

## Decision points (explicit forks, not vibes)

| # | Fork | Trigger | Branches |
|---|------|---------|----------|
| D1 | smoke #6 | result | RESOLVED GREEN: sequential on PRO 4500 Blackwell 32GB ($0.72) trained 44/44 steps both legs (658ab2, done, 18 artifacts, train_loss 0.90, 18.1 min pod). Floors now: concurrent 48 (unsourceable in this volume DC — DC-alignment, not market: preflight shows A40/A6000/L40S in catalog), sequential 28 (quiet-card 25.2/leg + headroom, c4f1ec) |

## Measured economics (658ab2 — the numbers that gate D2)

Mini sequential, PRO 4500 Blackwell: 0.8B ~3.7 s/it, 4B ~5.2 s/it
(5.15 avg incl. warmup), 0.917 packed steps per row at 8192.
Full corpus (22,947 rows) per EPOCH:
- 0.8B: ~21,000 steps x 5.15s ~= 30 h -> $21.6
- 4B:   ~21,000 steps x 5.2s+  ~= 31-41 h -> $22-30
- sequential pair: 1 epoch ~= $44-52; recipes' num_epochs=2 ~= $88-104
Balance at cut: $7.41. The cap was set BEFORE the pilot measured this —
D2 is now a real fork:

| Path | Full-run cost | What it buys |
|------|--------------|--------------|
| F1: 5-6K stratified gen0, 2 epochs | ~$16-22 | real tier scores now; weekly deltas (~582 rows, ~$2/wk) fill in distribution anyway |
| F2: full corpus, 1 epoch | ~$44-52 | maximal data pass; no epoch-2 generalization-repair |
| F3: full corpus, 2 epochs | ~$88-104 | the textbook gen0; pays ~2x F2 for pass 2 |
| F4: faster class (A100-80/H100) | ~$50-90 full-2ep | 3-4x s/it cut at 2-3x rate; net ~= F3, days -> hours of wall |

Recommendation: F1 now (fits current balance + cap raise ~$25), F2/F3 as
a deliberate compaction event later — the GENERATIONS design makes
gen0-size a strategy choice, not a one-way door: deltas keep any gen
current. Weekly delta run at 582 rows: ~$2-3, and DUP-POD makes the
whole cron safely re-runnable.
| D2 | budget raise | before full run | raise QTRAIN_BUDGET in .env with the measured estimate as the receipt (down-only knobs make this the honest path) |
| D3 | FFT pilot | tier-2 stall at gen-0 | run `Qwen3.5-0.8B-fft.yaml` single-leg (gate/tooling already built) |
| D4 | "all safety wired" claim | until O4.1 lands | janitor-cadence-bound is NOT wired; the plan's DoD #4 fails while pod-stop is inert — do not call the loop production-grade before the kill-test receipt |
| D5 | corpora remote | disk is a single point | push corpora to Forgejo (recipes/tools are precious; rows are regenerable) — needs explicit user go, like every push |
| D6 | image bump timing | after O1 registers | liger bump goes AFTER gen-0 registers, so a kernel-law regression never contaminates the baseline lineage |
| D7 | compaction K | after ≥3 gens | from the O2 gen-curve; until then K is unset (a movable PRIOR, not a law) |

## Risks & kill criteria

- **Capacity ceiling (accept, don't fight)**: self-distillation from
  same-family qwen3.8-flash proves the LANE, not a ceiling; if tier-1
  passes but tier-2 sits flat across ladder rungs, the program's
  verdict flips from "scale up" to "stronger teacher or RL tranche" —
  that is a finding, logged, not a failure mode to hide.
- **Stock fragility EU-RO SECURE**: 48-class offers exist ($0.59 A40)
  but volume-DC affinity is unproven under load; D1 sequential mode is
  the built-in hedge.
- **Shared worktree, two agents**: interface drift risk — banner==
  surface discipline + tests pin NO version strings; peer files
  (bench.py, synth, infra) stay untouched, commits boundary-clean
  (proven today: peer synth failures excluded from my commits, my
  pin/recipe sweeps never touched their files).
- **Spend guardrail**: program total ≈ $15-25 through O5 first rung;
  aggregate gate + balance gate + rate ceiling are the machines;
  every tranche's cost lands in the ledger, so `qtrain budget` census
  IS the burn chart.

## Today's queue (after this doc)

1. Smoke #6 verdict → D1 branch, estimate table for D2.
2. Build O3.1 (`--delta` extractor mode) WHILE waiting — zero fleet
   cost, unblocks the weekly cron.
3. Draft O4.3 linter (recipe keys vs boot-dump schema fixture).
4. Send O4.1 handoff (quadra.yaml line + .env pair) to the infra-
   holding session — it is the only tranche I cannot land alone.
