# quadra — Oracle program & weight-space engineering

Status: **ROADMAP v1** (2026-11-26). Companion to `FRAMEWORK.md` (corpus
kernel), `REGIMES.md` (execution backlog — this file feeds it R13+), and
`AS-BUILT.md`. Owner: qtrain/mcp-trainer lane.

## 0. The doctrine

**An oracle is any mechanism that disposes of a model's proposal.** The
one idea — propose → oracle → keep/reject — only generalizes as far as
our oracles do. Three laws:

1. **Use the strongest tier available per axis.** Deterministic gates
   (parse/execute/decide) outrank statistical checks, which outrank
   judges. LLM-as-judge is permitted ONLY where no ground truth exists
   (persona voice, taste), and it must be *pinned*: fixed model+prompt+
   temperature, a human-labeled calibration set, periodic spot-checks,
   and never inside a loop that grades a model like itself.
2. **Every oracle is versioned** (FRAMEWORK L3): its digest lands in
   `manifest.json`; oracle edits invalidate downstream rows and gates.
3. **Nothing permanent ships unmeasured.** Inference-time (reversible) →
   adapter-space (composable) → weight-space (permanent), in that order;
   every weight edit produces a before/after table from bench + grade.

## 1. The oracle catalogue (tier → ours → to build)

| Tier | Mechanism | We have | To build |
|---|---|---|---|
| **Syntax** | EBNF/GBNF, parsers, turtle/JSON parse | `ir-compile`, `ir-gbnf`, funnel parse gate | GBNF per dialect (Solidity, SQL…) |
| **Structure** | SHACL, schemas, type/cardinality | funnel SHACL+ASK gates; Pass-1 checker | schema registry per domain |
| **Query** | SPARQL ASK/SELECT, Datalog | fixture executor (closed-world) | **R16**: Datalog oracle for legal rules; SPARQL over ttl@h |
| **Execution** | pytest, tsc, solc, EVM, SQL engines | — | **R17**: language-oracle adapter (coding corpus w/ tests-as-oracle) |
| **Model check** | explicit-state checker, LTL | — | **R18**: concurrent IR (PARALLEL/SPAWN/WAIT) — beyond fixture subset |
| **Differential** | round-trips, two impls, two teachers | transpiler plan; teacher API lane | equiv mining done (R11); solc/EVM round-trip |
| **Retrieval** | citation-grounding, span match, provenance | `qtrain verify`, manifest | cite-the-proof rows already (P13) — bench probe |
| **Statistical** | embedding distance, entropy, calibration | `synth audit` entropy/near-dup | **R13**: `embed:` + `rate:` bench checks; ECE vs verifier labels |
| **Judge** | LLM-as-judge | (banned at corpus gates) | persona grading only, pinned+calibrated, human-gold sampled |
| **Human** | merge gates, release tags | git + tags | contrast-review queue |

The key realization: **the bench is already an oracle DSL**
(`contains:/regex:/json`) — upgrading its grammar (R13) turns it into the
universal soft-oracle harness, the gate every weight-space experiment
below needs.

## 2. Weight-space program

**The unlock: we are a contrast-set factory.** Every representation
technique needs labeled pairs ("model doing X" vs "model not doing X").
Our verified corpus generates them for free — DPO couples (valid/invalid),
abstain rows (refuse/fabricate), self-name rows (in-voice/out), emoji
compliance, refusal boundaries. R14 emits them as a first-class artifact.

### R15 — Abliteration / steering, the measured way

Abliteration = isolation of a behavior direction (mean-diff or top SVD
component of contrast activations), then either:

- **SOFT (default): the direction becomes a LoRA adapter.** Activation
  steering at inference is mathematically a low-rank delta — so it rides
  the existing vLLM multi-LoRA lane, composes with persona/cognition
  adapters (R6 merge algebra), is per-persona, and is removable.
- **HARD (gated): weight orthogonalization** (`W ← W (I − vvᵀ)`) →
  merged artifact. Permanent; ships only with the before/after table.

Gate: `bench --config probes.json` + `synth sample` yield delta. Known
risks (documented, not hidden): capability regression adjacent to the
removed direction; brittleness across prompts; refusal-direction removal
changes safety posture — acceptable in a private single-operator stack,
never on shared endpoints without explicit operator decision.

Generalizes to **concept sliders**: any verifiable behavior → contrast
set → direction → dial (emoji density, self-naming eagerness, hedging,
"cockiness" — persona as a control surface instead of a prompt plea).

### R19 — Broader weight-space backlog

| Idea | What | First-class when |
|---|---|---|
| Task-vector algebra | persona+knowledge+cognition as additive deltas | R6 pod-exec proves merge fidelity |
| Spectral bleed-editing | SVD the persona delta, prune components correlated with unwanted bleed (emoji-as-direction!) | R13 gives the rate probe |
| Concept sliders | R15 machinery per behavior | R15 lands |
| Soups/seed-stability | weight-average across seeds at 0.8B | trivial, needs run volume |
| Mode-connectivity probe | α-interpolate two runs, does persona survive? | decides merge safety |
| QA-LoRA for edge | quant-aware training so int4 doesn't eat the persona | nano0 int4 lane exists |
| Parametric knowledge editing (ROME/MEMIT) | fact surgery in weights | **deliberately NOT chosen** — CFKG (R2) trains graph-reading instead; facts change, edits rot |
| Adversarial curriculum | proposer generates checker-passing-but-wrong prompts; oracle grades both sides | R10 loop driver |
| Self-consistency (weak) | sample-k agreement as signal, never gate | cheap addition to grade |
| Oracle red-team (mutation testing) | fuzz for checker gaps — the checker IS the ceiling | R17/R18 give the substrate |

## 3. The enabler ladder (build order)

```
R13 bench grammar upgrade      (day)  — turns bench into the oracle harness
R14 contrast factory           (day)  — fuel for everything below
R15 steering/ablation tooling  (week) — soft mode first, hard mode gated
R16 Datalog/legal oracle       (week) — legal axis joins the closed loop
R17 language-oracle adapter    (week) — pytest/solc; coding axis; proves
                                         "any EBNF'd language" generalizes
R18 concurrent-IR model checker (research) — validates what fixtures can't
R20 calibration probes          (days)  — ECE vs verifier labels
```

Dependency: R15/R19 all consume R13 (gate) + R14 (fuel). R13/R14 are the
"close out" of this document — they make the rest executable, not
aspirational.

## 4. Scope discipline

Each oracle added must state: what it cannot see (its ceiling), what the
held-out complement is (hand suites, human spot-check), and which regime
consumes it. An oracle nobody gates on is a toy; a gate nobody audits is
a hazard.
