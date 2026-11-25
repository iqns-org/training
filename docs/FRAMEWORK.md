# quadra — Cognition Loop: unified corpus→graph→model framework

Status: **DRAFT v1** (2026-11-25). Companion to `SPEC.md` (§4b corpus boundary
holds) and to `rmcp/docs/RMCP_SPEC.md`. Focus repo: `quadra/corpora`;
machinery: `quadra/training` + `rmcp-extract` + `mcp_trainer`.

## 0. The one idea

Everything in this stack that works, works because of a single pattern:

> **A model proposes. A deterministic oracle disposes. Only verified artifacts
> enter the corpus.**

- `rmcp-extract`: teacher LLM emits TTL → **funnel**: Turtle parse → SHACL
  shapes → ASK constraints → quarantine violating foci. (SPEC §5 gates 5–7.)
- Graph IR (`training/src/ir-compile.py`, `ir-ast-sanity.py`,
  `ir-check-scenarios.py`): generator LLM emits programs → parser → Pass-1
  validator (E0–E7) → sandboxed fixture executor vs `assert` deltas.
- Bench (`orchestrator/bench.py`): served model answers → rule-checked
  pass/fail per question.

Because every artifact is **mechanically checkable**, corpus quality can
compound without human review at every step — this is what makes a recursive
loop safe. Personas, domain knowledge, and cognition are not three systems;
they are three **corpus axes** flowing through the same pipeline with
different oracles.

## 1. Corpus axes (what we train)

| Axis | Substrate | Oracle | Example domains |
|---|---|---|---|
| **Persona** | chat envelope (`$`-variant jsonl), jinja template slot | template-vs-weights bench; ratio audits (emoji/json lesson) | kandi |
| **Knowledge** | **TTL graphs** in cluster S3 (content-addressed) | funnel: parse/SHACL/ASK; round-trip NL↔graph QA | construction (SWMS, specs), legal (FWC curricula), coding (repos/docs) |
| **Cognition** | **Graph IR** programs executed over those graphs | parser + Pass-1 checker + deterministic executor | 10 task families; thinking strategies = IR-verified reasoning |

The key upgrade over the status quo: cognition scenarios stop being only
hand-written toy fixtures and start being **generated from real ingested
TTL graphs** — legal/regulatory facts become the WORLD of nl-to-ir / repair /
equivalence tasks. Knowledge and cognition train together and teach the same
thing: reasoning over structured state.

## 2. The kernel (where everything lives)

Content-addressed artifacts with a provenance DAG — reuse what exists
(`orchestrator/storage.py`, `ledger.py`, `provenance.py`), generalize it.

```
source@h   raw pdf/docx/xlsx/md          (vault → mcp-files/rmcp-extract)
md@h       layout markdown                (docling / native engines)
ttl@h      verified RDF graph             (funnel-gated; S3 quadra-ml/graphs/, NOT git)
rule@h     extraction rule (md+frontmatter, vocab-restricted, SHACL+ASK)  → corpora git
vocab@h    ontology registry subset (DiCon/DICE + domain packs)           → corpora git
scenario@h IR task template (variants+messages+assert)                    → corpora git
dataset@h  expanded jsonl (sha256 of expansion — existing convention)     → corpora git + S3
recipe@h   axolotl overlay (domain×model)                                 → corpora git
run@id     training/stage run (ledger row)
model@digest adapter/merged/quant artifact                                → S3
eval@h     bench result suite                                              → corpora/docs + ledger
```

**Laws**
- L1 — Git holds only human-curatable text (rules, vocab packs, scenarios,
  recipes, datasets). Binary sources and TTL graphs live in S3
  (`quadra-ml/graphs/<domain>/…`), referenced by hash. No LFS.
- L2 — Every dataset row carries the ids of what produced it
  (`{scenario@h, ttl@h?, rule@h?, teacher@model, oracle@digest}`). A dataset
  with unverifiable lineage cannot be trained on. (This generalizes the
  extract `.info` envelope law.)
- L3 — Oracles are versioned artifacts too. Changing `ir-compile.py`'s
  semantics or a SHACL shapes file invalidates downstream rows → gap report.

## 3. Ingestion lane (raw → TTL), per format

Extending `rmcp-extract` — the pipeline stages already exist; what's needed
is **rule packs per domain** and two new engines:

```
drop ──► mcp-files vault ──► notify ──► rmcp-extract
  PDF         docling-rs → md ─┐
  DOCX        docx.rs → md ────┼──► markdown ──► rule (llm|cmd) ──► funnel ──► vault.meta/*.ttl
  XLSX   [NEW] xlsx.rs: sheets→ canonical tables ──► mostly cmd-driver rules (deterministic rows→TTL,
         LLM only for headers/units semantics)
  MD     ─────────────────────┘
```

- **xlsx is the easy win**: structured tables map to RDF predicate/subject
  rows almost without LLM help; the funnel's `cmd` driver can be a sheet→ttl
  compiler. Budget burns on prose docs, not spreadsheets.
- **Rule packs** in `corpora/rules/<domain>/*.md` (frontmatter = config:
  `vocabulary.use/classes/properties`, shapes, ASK; body = teacher prompt),
  matched by store+pattern+media_type exactly like the fleet's existing law.
- **Domain packs**: `corpora/vocab/legal.fwc.*` (acts, instruments,
  obligations, entitlements, deadlines), `corpora/vocab/construction.*`
  (SWMS: plant, hazard, control, duty-holder; specs: element, material,
  standard, test), `corpora/vocab/coding.*` (symbol, call graph, module dep,
  API schema — fed from repo ASTs via `cmd` rules, no LLM needed at all).

## 4. Synthesis patterns (create/test/synthesize from raw + structured)

A small algebra of corpus operations; every pattern is a `cmd`-driven tool
with deterministic verification, and every generator LLM call is a
`llm`-driver rule. This is the API surface the MCP exposes (§6).

| # | Pattern | Input | Rows produced | Oracle |
|---|---|---|---|---|
| P1 | **propose–verify–keep** | scenario template + vocab | task-family jsonl | parser/AST/executor (existing ir-generator loop) |
| P2 | **perturb-and-repair** | any valid IR/graph | invalid-program, repair, diagnosis | mutation ledger records the single intended error (E-code); checker must agree |
| P3 | **execute-and-assert** | program + graph | concrete-examples (BEFORE/AFTER, events, returns) | sandbox executor — `assert` deltas computed, never typed |
| P4 | **round-trip QA** | ttl@h + vocab | knowledge QA, nl-to-ir, ir-to-nl | answer recomputed by SPARQL/READ over the graph |
| P5 | **back-translate** | IR program | NL spec → fresh IR → compare ASTs | structural equivalence verdicts (equivalence family) |
| P6 | **hop-ladder** | graph | multi-hop reasoning at depth k=1..n | executor enumerates exact hop paths |
| P7 | **failure mining** | eval@h + run logs | targeted rows for each failed check | the `data-audit-2026-11-24.md` workflow, mechanized: gap → rule/scenario diff → new rows |
| P8 | **format-guard** | — | instruct/json/format fixtures mix | ratio floor enforced at build (`$`-grid discipline) |
| P9 | **distill** | teacher model + rows | student jsonl (persona voice via teacher under template+seed; rejection-sampled by oracles) | same oracle as source pattern; teacher identity recorded per L2 |

### 4b. Missing patterns (inventory beyond P1–P9)

Grouped by what they teach. These were absent from the first pass — each
still obeys the one idea (LLM proposes, deterministic oracle disposes).

**Self-correction & reasoning discipline (cognition):**
| # | Pattern | What | Oracle |
|---|---|---|---|
| P10 | **critic pass** | model's own program + checker errors → corrected program; teaches repair of *its own* output, not hand-planted bugs | Pass-1 checker on both versions |
| P11 | **plan-then-code** | GOAL → PLAN/PROJECTION declarations → executable program (chain-of-*action*, verifiable at every stage) | plan↔program consistency check |
| P12 | **abstention** | query whose facts are absent from WORLD → "not derivable"; the anti-hallucination row type | graph query: derivable(y,n) is computable |
| P13 | **cite-the-proof** | answer + the exact READ/MATCH pattern that yields it; groundability as output format | re-execute the cited pattern |
| P14 | **isomorphism mining** | same IR shape rendered across unrelated domains (book→author ≡ sonata→composer ≡ call→callee) → abstraction not memorization | AST equivalence up to renaming |
| P15 | **scaffold fading** | worked example → hinted completion → bare request for the same task | same task oracle |
| P16 | **deontic reading** | legal/construction statutes → obligation/permission/prohibition + conditional trigger + deadline rows | rule fires deterministically on fixture facts |
| P17 | **number/unit/time** | takeoffs, conversions, relative dates (xsd datatypes the funnel already validates) | arithmetic recomputed |

**Memory & state governance (knowledge axis, already in the fleet but not
yet a corpus pattern — mcp-facts is literally the reference implementation):**
| # | Pattern | What | Oracle |
|---|---|---|---|
| P18 | **assert/retract/correct** | temporal ledger rows: supersede, asOf ("what was true then"), retracted-now queries | ledger semantics |
| P19 | **contradiction detection** | two sources asserting conflicting cardinality/type facts → conflict report, not silent overwrite | SHACL cardinality + dual-source diff |
| P20 | **entity resolution** | same real thing across docs (aliases, typos, case) → `sameAs` graph + merge/no-merge rows | transitive-closure check + golden pairs |
| P21 | **provenance QA** | "where did this come from" → cite doc@sha/rule@h path | DAG walk (provenance.py) |

**Preference & RL (the cheapest high-value ones — P2 gives them for free):**
| # | Pattern | What | Oracle |
|---|---|---|---|
| P22 | **RFT / rejection sampling** | sample-k teacher/student outputs, keep only oracle-passing rows | existing checkers = reward model |
| P23 | **DPO pairs** | valid vs single-violation mutated program = free preference pair | P2 mutation ledger |
| P24 | **RLVR curriculum** | online: generate→verify→reward→train loop against the checker (needs P27 first) | the runtime |

**Inference-time patterns (training-data analogues of serving behavior):**
| # | Pattern | What | Oracle |
|---|---|---|---|
| P25 | **grammar-constrained decoding** | compile `ebnf.md` to a GBNF/grammar sampler on vLLM — structural validity *at decode time*, not learned hope | the grammar itself |
| P26 | **self-verify loop** | generate → execute → repair at serve time, with corpus rows mirroring exactly this trajectory shape | P27 runtime |
| P27 | **IR runtime** | **critical gap**: scenarios/README says the fixture evaluator "is not a runtime implementation"; until a real Graph IR engine exists, there is no online verifier and no agentic environment — P24/P26 stay blocked | — |

**Data hygiene & scale (meta-patterns over the corpus itself):**
| # | Pattern | What | Oracle |
|---|---|---|---|
| P28 | **near-dup / entropy guard** | minhash/embedding dedup across scenarios; the data-audit lesson — "variety is combinatorial, not semantic" — enforced as a build gate | distinct-n / embedding spread floors |
| P29 | **coverage matrix** | cells = domain × construct × task-family (cases.jsonl sections make this real); empty cells = generation targets | build-time quota check |
| P30 | **contamination & held-out domains** | train on literature/music, eval on legal/construction *shapes*; synthetic rows can leak eval answers | set-disjointness gate |
| P31 | **DAG-driven incremental rebuild** | template/rule/oracle version change invalidates exactly downstream rows (L3, mechanized) | kernel manifest |
| P32 | **teacher-call cache** | content-address LLM responses (prompt@hash → rows); reproducibility + cost control | hash identity |
| P33 | **rows-per-dollar** | verified-row efficiency per pattern/domain/teacher; drives which patterns get budget | ledger + usage stats |
| P34 | **guard-mix & interference** | persona×domain×format interference tests (emoji-bleed generalized: any axis poisoning another); auto-tuned mixture floors per recipe tier | bench matrix deltas |
| P35 | **adapter consolidation** | many narrow LoRAs → merged/composed models (TIES/DARE) or router policy; the serve-side twin of corpus mixing | bench parity vs stacked LoRAs |

**Multimodal (sources have drawings, you have mcp-vision):**
| # | Pattern | What | Oracle |
|---|---|---|---|
| P36 | **figure→graph** | spec tables, plans, circuit/schematic crops → TTL (cross-checked against text lane of same doc) | two-lane agreement gate |

**Implementation wave 1 (no LLM required):** P2, P3(verify), P12, P23, P28,
P29, P31 ship now as `training/src/synth.py` + `bin/synth`. The engine works
purely on existing scenario templates: it (1) selects every template whose
final message is a valid IR program under its own context, (2) applies a
**rulebook of text-level single-violation mutations** (E1_DUP_PREFIX,
E2_UNBOUND, E3_REDEFINE, E3_USE_BEFORE_DEF, …), (3) keeps only mutants the
oracle accepts as *exactly one intended error* — the SAFEGUARDS law enforced
by construction, not by hope, and (4) emits derived `invalid-program`,
`repair`, and `abstain` templates under `corpora/generate/derived/` (the
curation tree stays pristine; the derived tree carries its own preflight),
`dpo.jsonl` preference pairs from the same (valid, mutant) coupling, and
`audit.json` (P28/P29/P31: coverage matrix, phrasing entropy, template-shape
near-duplicates, mutant-candidate yield per error class). LLM-dependent
patterns (P1, P4–P11, P13–P27, P32–P36) attach to the same engine as new
generators — the oracle gate is the extension point.

Curriculum tiers: each task family × hop depth × domain complexity gets a
difficulty grade; generation for tier N+1 is gated on model pass-rate at
tier N (bench/scenario eval results feed the gate). That is the recursion's
ratchet.

## 5. Recursive loop (end-to-end automation)

```
        ┌───────────────────────────────────────────────────────────────┐
        │                          OUTER LOOP                           │
  ingest│ source→md→ttl │ synth P1..P9 │ build+hash │ train │ stage │   │
  ────► │  (rmcp-extract)│  (corpus-gen)│ (expand.py)│(qtrain)│(merge/ │  │
        │                │              │            │       │ quant) │  │
        │   eval/bench ──┴── gap analysis (P7) ──────┴───────┴─────────┘
        │        ▲                          │
        │        └── register→serve ── next-round plan (curriculum gate)
        └── human gates: budget caps (ledger/budget.py), release tags, vocab/rule merges
```

- Every edge is deterministic or oracle-checked, so an **agent may drive the
  loop**; humans gate only: money (budget caps already exist), taste
  (persona/rule merges via git), and release (`domain-YYYY.NN` tags already
  exist).
- The loop never auto-publishes: `register` (serve) and `tag` (release) are
  the two intentional checkpoints, exactly like today — promoted from
  convention to law.
- Self-improvement ladder: bigger student ⇒ better generator ⇒ harder
  curriculum ⇒ bigger gap mining. The oracle does not improve with model
  size (it's deterministic), which is what keeps the loop honest.
- Teacher lane: Groq/OpenRouter APIs (already in `ir-generator.py`) + big
  self-hosted open models (§7) for rows small models can't draft; teacher
  outputs are treated **exactly like any proposal** — funnel/parser/executor
  gates apply identically.

## 6. Control plane: one MCP

Extend `mcp_trainer` (port 3022) into the corpus+training control plane; do
not build a new service. Tool surface (agents + humans, same API):

```
ingest.put(path|url, domain)        ingest.status            graphs.query(ttl@h|domain, ASK/SPARQL-ish)   [planned: needs rmcp-extract wiring]
rules.list/get/propose              vocab.get/propose                                                          [planned]
synth.derive(codes,systems,…)       synth.abstain(…)      synth.graph(ttl_path,domain,…)     [SHIPPED]
synth.equiv() — P5 mining           synth.conflict(a,b) — P19 mining                          [SHIPPED]
synth.gbnf() — R3 grammar export    synth.sample(url,k,…) — R4 student RFT                    [SHIPPED]
synth.grade(jsonl) — R7 reward fct   synth.gaps(bench) — P7 propose step                       [SHIPPED]
synth.merge(base,adapters) — R6 cfg  synth.mix(sources,rows,seed,name) — R5 grid prim          [SHIPPED]
synth.manifest() → derived@h        synth.audit()         synth.bundle(domain) → <domain>@h  [SHIPPED]
synth.status(job) — background jobs [SHIPPED]
dataset.build(domain) → h           dataset.audit(h)       # token stats, ratio floors, lineage (L2)
train.submit(domain@h, recipe)      stage.submit(kind)     job.status/logs                    [existing]
eval.bench(url, suite) → eval@h     eval.gaps(eval@h)      # P7 entry point                   [bench shipped, gaps planned]
loop.plan() / loop.step() / loop.history()                                                           [planned]
provenance.get(artifact@h)          # DAG walk (extends provenance.py)                        [partial: verify_run]
```

`qtrain` CLI mirrors the tools (already the convention). The infra manifests
deploy `mcp-trainer` unchanged in shape — it just grows the new tools.

## 7. Model tiers (no ceiling)

Same kernel, same datasets — recipes scale:

| Tier | Models | Train lane | Serve lane |
|---|---|---|---|
| S | 0.8B–4B (SmolLM2, Qwen3.5) | rented 16–24GB pod, QLoRA (today) | nano0 vLLM multi-LoRA |
| M | 8B–32B | single 48–80GB (L40S/A100/H100 spot), QLoRA; `resources.gpus` already supports it | cloud serve or cluster node |
| L | 70–120B+ OSS (Qwen, GLM, Llama-class) | 8×80GB pod, QLoRA + ZeRO-3/FSDP (axolotl native); new `images/Dockerfile` lane unchanged | cloud vLLM (multi-worker) — serving is an infra concern, not training's |
| T | teachers (API + any tier-L model) | never trained here | OpenRouter/Groq or self-hosted |

Needed work for L, concretely: (1) multi-GPU pod launch specs in
`orchestrator/providers/runpod.py` (`gpu_type_ids` array + worker count —
the API supports it; our pod spec pins one); (2) multipart S3 upload +
checkpoint-shard assembly for 60 GB artifacts (agent `storage.py` lane);
(3) `max_cost_per_hour` ~10× but same budget law; (4) eval matrix comparing
tiers per axis — a tier-L model is most valuable as a **teacher (P9)** and
as the reference for gap analysis, not as the deployed persona.

## 8. Phasing

- **Phase 0 — kernel freeze** *(partial)*: `synth manifest` gives
  `derived@h` + oracle source digests (L2/L3); `synth bundle` gives
  `cognition@h` dataset ids. Remaining: per-row lineage stamps in
  `expand.py`/`register.py`; ledger `artifact(kind,id,parents,meta)` table.
- **Phase 1 — knowledge lanes** *(bridge shipped, rule packs open)*:
  `synth graph` turns funnel-shaped TTL into graph-qa/hop/abstain
  scenarios (proven on `generate/examples/site-register.ttl` → bundled as
  `cognition@4006d94f`, 6,409 rows, trained-lane compatible via
  `bin/prepare-domain`). Remaining: xlsx engine + rule packs for real
  construction/legal sources; rmcp-extract → `ttl@h` → `synth graph`
  webhook wiring (`ingest.*`, `graphs.*` MCP tools).
- **Phase 2 — IR↔graph bridge** *(started)*: mutation rulebook complete
  over curated templates (waves 1–3); hop-ladder E4/E5 families activate
  as Phase-1 graphs feed scenario finals. Curriculum tiers + bench-gated
  ramp (P7 gap-analysis) still open.
- **Phase 3 — big lane**: ZeRO-3/FSDP recipes + multi-GPU pod specs +
  multipart artifacts; tier-L teacher self-hosted; P9 distillation runs.
- **Phase 4 — loop automation**: `loop.step` as cluster cron/agent;
  gap-analysis proposes corpora diffs (PRs into `quadra/corpora`), humans
  merge, loop continues. Dashboard = ledger views (progress/notify exist).

## 9. Decisions log (why this shape)

1. **RDF/TTL as the interchange** — the only format that is simultaneously
   machine-validated (SHACL/ASK), model-irrelevant, queryable as ground
   truth, and already what rmcp-extract emits and Graph IR consumes.
2. **Oracles over judges** — no LLM-as-judge anywhere in the corpus gate;
   LLMs propose, code disposes. (An LLM judge in the loop makes the
   recursion self-confirming; a deterministic oracle makes it convergent.)
3. **Graphs in S3, not git** — derived from sources, regenerable, content-
   addressed; git is for what humans curate (rules, vocab, scenarios).
4. **Persona/knowledge/cognition are one pipeline, three axes** — dataset
   mixing is a recipe concern (per `domains`/`format` rows already
   practiced); the machinery must never fork per axis.
5. **One MCP control plane** — fleet law already says every capability is an
   MCP tool; the outer loop is therefore an agent-drivable API, not a
   bespoke scheduler.
6. **Deterministic wave first** — the engine ships LLM-free patterns
   (mutation/abstention/audit) first: they exercise every kernel law
   (L1–L3), produce corpus value immediately, and become the verifiable
   substrate the LLM patterns (P1, P9, P22, P24) later plug into.
7. **Mutants are proposals, not data** — every generated violation is kept
   only if the Pass-1 checker reports *exactly the intended single error*
   with zero warnings; a mutation the oracle disagrees with is discarded
   and counted as yield loss, never hand-patched.

## 10. Synthesis engine: `bin/synth` / mcp-trainer `synth.*`

```
bin/synth derive [--codes a,b] [--systems a,b] [--per-code N] [--limit N]
    # P2/P23: sources = templates whose final IR validates under its
    # context; oracle filter = EXACTLY one intended error, zero warnings —
    # multi-error mutants are yield loss, never hand-patched.
    # Rulebook (15 rules):
    #   wave 1 (text): E1_DUP_PREFIX, E2_UNBOUND, E3_REDEFINE,
    #                  E3_USE_BEFORE_DEF (depth-aware)
    #   wave 2 (AST): E4_TYPE_CYCLE, E4_TYPE_MISMATCH (literal flip),
    #                  E5_CARD_MISMATCH (CALL/EMIT arg from MATCH-AS many
    #                  handle), E6_UNREACHABLE, E7_IMMUTABLE_SCOPE,
    #                  E7_EFFECT_IN_PROJECTION, E1_DUP_{TYPE,CAPABILITY,
    #                  EVENT,PLAN,PROJECTION}
    #   wave 3 (inject-then-break + brace depths): nested E6 inside
    #                  blocks, PARALLEL-arity, WAIT timeout, dup-decl
    #                  family — shapes need not exist in the source
    #   wave 4 (shape-free inject): E5_CARD + E4_TYPE + E1_DUP_{PLAN,
    #                  PROJECTION} via injected fixtures — the dormant
    #                  wave-2 families now fire on EVERY source template
    # statement-tier mutants → repairable templates; declaration-tier →
    #   diagnosis-only (stats.no_repair records the split honestly);
    #   DPO pairs only from ctx-identical mutants (undeclared-capability
    #   traps would make a rejected side valid)
    # outputs: corpora/generate/derived/{invalid-program,repair,abstain}/*.json
    #          training/tmp/synth/dpo.jsonl
    # every template is preflighted (expansion × role contract × checker ×
    # fixture execution) before it is written — nothing lands unverified.
bin/synth abstain                # P12 NOT-DERIVABLE rows (anti-hallucination)
bin/synth graph <ttl|dir> [--domain d]
    # P4/P6/P12 knowledge↔cognition bridge: turtle facts → graph-qa /
    # hop-ladder / counterfactual (R2 CFKG, typed edits) / abstention
    # templates; asserts RECOMPUTED by the fixture executor (never typed)
bin/synth equiv [--max-pairs N]
    # P5 equivalence mining: uniform renames (EQUIVALENT) + observable
    # line deletions (DIFFERENT); the fixture executor COMPUTES verdicts
    # by comparing full execution deltas; undecidables counted & dropped
bin/synth conflict <base.ttl> <rev.ttl>
    # P19 contradiction mining: same (s,p), different objects across
    # sources → CONFLICT rows citing both files (memory governance)
bin/synth gbnf [--root response|document] [--out f]
    # R3 grammar export (src/ir-gbnf.py: one rule per line, lexer-pinned
    # facts — ops are = != < only; 130/130 property-test parse rate)
bin/synth sample --url <openai-endpoint> [--model m --k 4 --limit n]
    # R4 student-side RFT: serve → sample k → fleet-contract grade → keep
    # passes only; yield = the recursion's before/after meter (live nano0
    # baseline on the base model: 0.0 — rises with every trained round)
bin/synth grade <jsonl> [--out reward.jsonl]
    # R7 harness: offline (s, r) reward pairs via the same contract — the
    # exact function the GRPO loop calls; gold-side agreement 821/821
bin/synth mix --sources d:w,d:w --rows n --seed s --name X
    # R5 seeded weighted mix of PREPARED datasets → datasets/X/{mix.jsonl
    #   ,mix.manifest.json}; the mixing-law experiment primitive
bin/synth gaps --bench bench.json
    # P7/R10: failed bench probes → targeted `derive --codes` proposals
    # (the loop's propose step; scheduling it into a PR bot is pending)
bin/synth merge --base B --adapters a,b [--method ties|dare_ties|slerp]
    # R6 mergekit config generation (postproc/merge_loras.py) — compose
    # axis LoRAs offline into one adapter; pod-exec pending run volume
bin/synth rows [--task t]        # expand derived templates → jsonl
bin/synth manifest               # derived@h + oracle digests (L2/L3 lineage)
bin/synth bundle --domain cognition
    # derived → $CORPORA_DIR/datasets/cognition/{diagnosis,repair,abstain,
    # graph-qa,graph-hops,counterfactual,equivalence,conflict}.jsonl +
    # datasets/cognition-dpo/preference.jsonl → dataset@h →
    # bin/prepare-domain → S3 → qtrain submit. Proven live:
    # cognition@e184c8de 9,865 rows + cognition-dpo 821 pairs.
bin/synth audit                  # P28 near-dup + phrasing entropy, P29
                                 # coverage matrix, P3 re-check
```

MCP parity: `synth_derive` (background jobs + `synth_status`),
`synth_abstain`, `synth_graph`, `synth_equiv`, `synth_conflict`,
`synth_gbnf`, `synth_sample`, `synth_grade`, `synth_mix`, `synth_gaps`,
`synth_merge`, `synth_audit`, `synth_manifest`, `synth_bundle` — the
outer loop is agent-drivable end to end.

**Known rulebook gaps:** `E1_BAD_VARIABLE`, `E1_BAD_CARDINALITY`,
`E6_UNKNOWN_STATEMENT`, `E6_BAD_SCOPE` are **unreachable from text
mutation by construction** — the lexer/parser rejects the malformed shape
before Pass-1 can report its own code (an E0_PARSE_ERROR is what the
checker would receive). They stay as diagnosis targets for hand-written
scenarios or for the runtime lane, not for the engine.
`E3_USE_BEFORE_DEF` mid-program remains shape-dependent (the checker
classifies deleted/injected variants as E2_UNBOUND; swaps straddling a
RETURN double-fire E6 and are filtered) — honest yield, recorded in
stats.candidates vs stats.accepted.

Law: derived is **regenerable, not curated** — safe to delete and rebuild;
the curation tree (`generate/scenarios/`) stays human. When a derived
template proves pedagogically valuable, a human *promotes* it by copying it
into the curated tree with edits.
