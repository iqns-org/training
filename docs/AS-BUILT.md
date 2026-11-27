# The Cognition Loop — as built

Status: **AS-BUILT v1** (2026-11-25). The truth of what runs today, with
worked examples taken from the live corpus (not sketched — every number
and snippet below was produced by the commands shown). Companions:
`FRAMEWORK.md` (architecture/spec), `REGIMES.md` (execution backlog),
`SPEC.md` (training lane).

```
 THE ONE IDEA, IMPLEMENTED:  a model proposes → a deterministic oracle
 disposes → nothing unverified lands → the corpus compounds.

 raw doc ──► ttl@h ──► derived@h ──► cognition@h ──► adapter ──► bench
  (vault,   (funnel   (bin/synth     (bin/synth      (qtrain     (bench/
   rmcp)    gates)     16 rules       bundle         submit)      cognition
            + graph     + bridge)      + ids)                    .json)
```

---

## 1. Quickstart — the whole loop, real output

From `quadra/training` (needs `.venv`, sibling/`$CORPORA_DIR` corpora):

```bash
bin/synth derive --per-code 4     # mutate every valid IR program
bin/synth abstain --per-code 2    # NOT-DERIVABLE rows from graphs
bin/synth graph ../corpora/generate/examples/site-register.ttl --domain construction
bin/synth equiv                   # P5: computed equivalence pairs
bin/synth conflict <base.ttl> <rev.ttl>   # P19: contradiction rows
bin/synth gbnf --out tmp/ir.gbnf  # R3: guided-decoding grammar
bin/synth manifest                # derived@h + oracle digests
bin/synth bundle --domain cognition
bin/prepare-domain cognition      # the existing qtrain lane input
```

Optional (needs a served model / endpoints):

```bash
bin/synth sample --url http://nano0.lan:8000/v1 --model <id> --k 4 --limit 50
bin/synth grade tmp/synth/dpo.jsonl --out reward.jsonl   # RLVR (s,r) pairs
bin/synth mix --sources queer:0.3,cognition:0.6,construction:0.1 --rows 2000 --name mix-r1
bin/synth gaps --bench bench-out.json                    # failed probes -> synth cmds
bin/synth merge --base Qwen/Qwen3.5-0.8B --adapters a,b --method dare_ties --density 0.5
```

Actual yields (this build, curated tree of 200 hand-authored scenario
templates, 13,874 expansions):

| stage | number |
|---|---|
| sources scanned | 3,241 template-expansions |
| mutation candidates proposed | 42,711 |
| accepted by oracle (16 error classes) | 1,640 |
| **preflight failures** | **0** |
| derived templates (`derived@a5b1e82b`) | 675 |
| dataset rows (`cognition@e184c8de`) | 9,044 SFT (+821 DPO = 9,865) |
| equivalence pairs mined (computed verdicts) | 81 |
| conflict rows (example register vs rev2) | 6 |
| GBNF rules, parse-property samples | 41, 130/130 |
| reward-harness agreement (gold side) | 821/821 = yield 1.0 |
| base model on IR prompts (nano0, live) | yield 0.0 — the honest baseline |

The 42,711 -> 1,640 ratio is the point: **~96% of proposals were rejected
by the checker** (multi-error, warning-bearing, or wrong-code mutants).
That filter is what makes the loop safe to run without a human per row.
## 2. Data plane

### 2.1 What lives where (the boundary law, §4b)

```
quadra/corpora (private git)              quadra/training
  datasets/queer/**.jsonl    curated        src/synth.py       the engine
  datasets/construction/     curated        bin/synth          CLI wrapper
  datasets/cognition/        BUNDLED ✱      src/ir-compile.py  parser (oracle)
  datasets/cognition-dpo/    BUNDLED ✱      src/ir-ast-sanity  Pass-1 checker
  generate/scenarios/        curated        src/ir-check-scenarios  fixture+preflight
  generate/derived/          DERIVED ✱      mcp_trainer/server.py   control plane
  generate/examples/*.ttl    curated        orchestrator/      job lane
  recipes/<domain>/*.yaml    curated        docs/{SPEC,FRAMEWORK,REGIMES,AS-BUILT}
  bench/*.json               curated
✱ = gitignored: regenerable, not curated. Delete freely; rebuild anytime.
```

**Law:** the curation tree stays human. Derived artifacts earn their
existence by passing the oracle; promoting one means copying it into
`generate/scenarios/` with edits (it then becomes curated, reviewed,
permanent).

### 2.2 Content-addressing (L1/L2/L3 made real)

`bin/synth manifest` writes `generate/derived/manifest.json`:

```json
{
 "corpus_id": "derived@3b70db24b12ee683",
 "count": 673,
 "oracles": {
  "synth.py": "281319ba77349cc9",
  "ir-compile.py": "3426a482213faf2c",
  "ir-ast-sanity.py": "69f5b28748394289",
  "ir-check-scenarios.py": "3b400efab9e1cf2c"
 }
}
```

The **oracle digests are the load-bearing line**: edit the Pass-1 checker
and every downstream `derived@h` changes — so "the rules didn't change,
did they?" is a diff, not a memory (L3). `synth bundle` then gives the
dataset the same treatment:

```
cognition@327ea0682c611a20          (SFT, 6 files)
  diagnosis.jsonl   4,920   58e30d3b68a0e17b
  repair.jsonl      3,920   2d75af2bbd786da9
  abstain.jsonl        43   825c61a2d58cdb2a
  graph-qa.jsonl       30   3b54375538d08ffd
  graph-hops.jsonl     14   7027c5c0d1feca52
  counterfactual.jsonl 30   e34f75e1bf046cc5
cognition-dpo@0c45a542              (preference.jsonl 821 pairs)
```

`bin/prepare-domain cognition` → `tmp/cognition.jsonl` (8,957 lines) and
`prepare-domain cognition-dpo` → 821 lines — **both verified through the
exact lane that trains queer and construction today** (expand → S3 →
pod). No new plumbing between synthesis and GPU.

### 2.3 The three corpus axes, one pipeline

| axis | substrate | oracle | state |
|---|---|---|---|
| persona | chat envelope, `$`-grid, jinja slot | bench rules, ratio floors | live (kandi, 23k rows) |
| knowledge | TTL graphs (funnel-gated) | parse/SHACL/ASK + fixture recompute | bridge built; rule packs = REGIMES R7 |
| cognition | Graph IR programs over those graphs | parser + Pass-1 + fixture executor | live (16 classes) |

---

## 3. The synthesis engine — every family, with real examples

### 3.1 The rulebook (16 error classes, four waves)

`bin/synth derive` scans every scenario template whose final IR validates
under its own context, applies 16 mutation rules, and keeps a mutant
**only if the checker reports exactly the one intended error and zero
warnings**:

| wave | family | tier | accepted (live) |
|---|---|---|---|
| 1 text | E1_DUP_PREFIX | decl | 159 |
| 1 text | E2_UNBOUND | stmt | 112 |
| 1 text | E3_REDEFINE | stmt | 8 |
| 2 AST | E4_TYPE_CYCLE | decl | 4 |
| 2 AST | E4_TYPE_MISMATCH | stmt | **159** (wave 4) |
| 2 AST | E5_CARD_MISMATCH | stmt | **159** (wave 4) |
| 2 AST | E6_UNREACHABLE | stmt | 160 |
| 2 AST | E7_IMMUTABLE_SCOPE | stmt | 64 |
| 2 AST | E7_EFFECT_IN_PROJECTION | decl | 159 |
| 2 AST | E1_DUP_TYPE / _EVENT / _CAPABILITY | decl | 4 / 12 / 4 |
| 3 depth | E6_UNREACHABLE (nested blocks) | stmt | (in 160) |
| 3 inject | E6_PARALLEL_ARITY | stmt | 159 |
| 3 inject | E6_BAD_TIMEOUT | stmt | 159 |
| 4 shape-free | E1_DUP_PLAN / _PROJECTION | decl | 159 / 159 |

*Tier = where the violation lives: `stmt`-tier mutants get repairable
templates; `decl`-tier are diagnosis-only — `stats.no_repair` records the
split honestly instead of emitting broken repair contracts.*

**Worked example — wave 1, `E2_UNBOUND` (a rename at one use-site):**

```diff
- READ WORLD my:document ai:origin my:?first . ONE.
+ READ WORLD my:document ai:origin my:?qzx . ONE.
```
Oracle verdict (verbatim): `variable my:?first used in BIND right-hand
side list item but not visible here` — one error, so the row is kept.
The diagnosis template lands as `derived/invalid-program/nl-to-ir-01-publish--e2_unbound.json`
and the repair template (`input_error: E2_UNBOUND`, answer = the original
program) as `derived/repair/…` — verified: `fixed_valid: true`.

**Worked example — wave 4, shape-free inject `E5_CARD_MISMATCH`** (the
corpus had *zero* MATCH-AS handles, so this family was dead until the
rule synthesized the shape):

```text
MATCH WORLD {
    my:synke9 ai:p9 my:?x9 .
} AS my:?m9.                  ← m9 is MANY by definition
CALL ai:synkcap(my:?m9).       ← param declared ONE
```
Oracle: `CALL ai:synkcap argument 1 is MANY but parameter 1 requires ONE`.
Yield went 0 → 159. Same for `E4_TYPE_MISMATCH` (`BIND my:?e4 = 1.`
compared against `WHEN my:?e4 < "synth" THEN` → `incompatible literal
types in top WHEN 1: integer vs string`).

**Worked example — wave 3, inject-then-break `E6_PARALLEL_ARITY`:**
grammar quirk found the hard way — arms are `PARALLEL { { arm1 } { arm2 } }`
with *no trailing dot*; one arm → exactly `E6_PARALLEL_ARITY`. The corpus
contains no PARALLEL programs; the rule doesn't need them to.

### 3.2 DPO preference pairs (R1, shipped)

Every statement-tier mutant is also a preference pair — chosen = valid,
rejected = single-error. A real pair from `tmp/synth/dpo.jsonl`:

```text
user: "Return the document's origin and destination as a list, in that order."
chosen   …  READ … my:?first … READ … my:?second … BIND my:?pair = [?first,?second]. RETURN my:?pair.
rejected …  same program, PLUS one duplicated line:
            + BIND my:?pair = [my:?first, my:?second].
oracle:  chosen valid ✓   rejected errors = ['E3_REDEFINE'] ✓
```

A correctness gate worth knowing: pairs are only written when the mutant
is **ctx-identical** (`mutant_full == _full(original ctx, mutated stmts)`).
Inject rules that add a CAPABILITY declaration would make the rejected
side *valid* under the original context — silent data poisoning, caught
and excluded by construction.

Training: `synth bundle` writes `datasets/cognition-dpo/preference.jsonl`
(axolotl `chat_preference`: `{chosen:[msgs], rejected:[msgs]}`) as its
own dataset domain — **never beside the SFT files**, because
`prepare-domain` merges every jsonl in a directory. Recipe
`recipes/cognition/Qwen3.5-0.8B-dpo.yaml` is pref-only;
`dispatch.render_config` grew a `pref_datasets` branch that mounts the
domain file into the preference slot and refuses to fabricate an SFT
entry.

### 3.3 Abstention rows (P12)

`synth abstain` drops one fact from a graph and emits a NOT-DERIVABLE
row **only if the fixture executor proves non-derivability** (the
program's MATCH/READ yields 0 rows on the reduced graph — checked, not
asserted). A real bundled row:

```text
user:   "What is the riskscore of the site2?"   (site2's riskScore fact removed)
answer: NOT DERIVABLE — the fact required to answer is absent from the
        supplied graph. Name what is missing; do not invent facts.
```

### 3.4 The graph bridge (knowledge ↔ cognition, incl. CFKG R2)

`synth graph <file.ttl|dir>` turns **funnel-shaped turtle** — exactly
what rmcp-extract's SHACL/ASK gates emit — into four verified template
families. On `generate/examples/site-register.ttl` (10 facts):

| family | what it teaches | rows |
|---|---|---|
| graph-qa (P4) | read facts through IR | 30 |
| graph-hops (P6) | 2-hop chains (site1 → inspector7 → cert) | 14 |
| **counterfactual (R2)** | **follow the graph, not memory** | 30 |
| abstain (P12) | refuse when the fact is gone | 33* |

\* graph abstains merge into the shared `abstain.jsonl` count.

The counterfactual family (the strongest idea in the whole plan, now
real): one fact is *rewritten*, typed, deterministically —
`ai:yearsExp 12 → 112`, `ai:holdsCert "Class-A" → "Class-A-cf"`,
`ai:inspectedBy my:inspector7 → my:Active` — and the answer target is
**recomputed by the fixture executor on the edited graph**, verified in
this very doc:

```text
edited fact line:  my:inspector7 ai:yearsExp 112 .
qa program:        READ WORLD my:inspector7 ai:yearsExp my:?v . ONE.
                   RETURN my:?v.
fixture executes → returned: 112   expected: 112   MATCH: True
```

The counterfactual family (the strongest idea in the whole plan, now
real): one fact is *rewritten*, typed, deterministically —
`ai:yearsExp 12 → 112`, `ai:holdsCert "Class-A" → "Class-A-cf"`,
`ai:inspectedBy my:inspector7 → my:Active` — and the answer target is
**recomputed by the fixture executor on the edited graph**, verified in
this very doc:

```text
edited fact line:  my:inspector7 ai:yearsExp 112 .
qa program:        READ WORLD my:inspector7 ai:yearsExp my:?v . ONE.
                   RETURN my:?v.
fixture executes → returned: 112   expected: 112   MATCH: True
```

Why it matters: legal and construction knowledge *changes* (new
regulations, revised specs). CFKG rows train the model to **read the
supplied graph over its prior memory** — the exact failure mode that
otherwise makes retrieval-grounded serving silently lie. And the oracle
for "what the edited graph says" is free: it's just execution.

Grammar support built for this bridge (worth knowing): turtle subset =
`PREFIX`/`@prefix`, plain triple lines with qnames or literals (`a` →
`rdf:type`, prefixed automatically); predicate/object lists and full IRIs
are **skipped and counted** (`stats.skipped`) rather than guessed at —
the honest-low-fidelity law from mcp-extract, applied to synthesis.

Grammar support built for this bridge (worth knowing): turtle subset =
`PREFIX`/`@prefix`, plain triple lines with qnames or literals (`a` →
`rdf:type`, prefixed automatically); predicate/object lists and full IRIs
are **skipped and counted** (`stats.skipped`) rather than guessed at —
the honest-low-fidelity law from mcp-extract, applied to synthesis.

### 3.5 The audit (corpus health, P28/P29/P3)

`synth audit` runs the fleet's own preflight contract over the curated
tree. Real report highlights (`tmp/synth/audit.json`):

```
templates: 200 (10 task families × 20)          expansions: 13,874
derived:   invalid-program 412 · repair 246 · abstain 12 · nl-to-ir(bridge) 3
near-duplicate pairs ≥0.8 similarity: 20          (concrete-examples clusters)
lowest phrasing entropy: concrete-examples/01-publish
  → 350 expansions, 7 distinct user requests, ratio 0.02
```

That last line *is* the 2026-11-24 data-audit smoking gun ("variety is
combinatorial, not semantic") turned into a standing metric. The
near-dup clusters (e.g. `concrete-examples/01-publish` ≡ `05-memory-read`
≡ `08-broad-lookup` at similarity 1.0) are the next curation work list:
either differentiate them or collapse them and bank the budget.

---

## 4. The MCP control plane (agent-drivable)

`mcp_trainer` port 3022 — the training MCP grew a synthesis surface; the
CLI and the tools are the same engine (`src/synth.py`, lazy-loaded):

| tool | what | real return shape |
|---|---|---|
| `synth_derive(codes,systems,per_code,limit,background,out,tmp)` | mutations + DPO | job id → poll `synth_status` |
| `synth_abstain(per_code)` | NOT-DERIVABLE rows | `{templates, accepted, preflight_fail}` |
| `synth_graph(ttl_path, domain, max_facts, out)` | ttl → 4 families | `{files, templates, facts, skipped, preflight_fail, counterfactuals}` |
| `synth_manifest(out)` | derived@h + oracle digests | `{corpus_id, count, oracles}` |
| `synth_bundle(domain, out, dpo)` | templates → datasets | `{dataset, files:{name:{rows,sha256}}, rows}` |
| `synth_audit()` | coverage/entropy/near-dup | summary; full → `tmp/synth/audit.json` |
| `synth_status(job)` | background jobs | `{job, state, stats}` |

A real tool return (`synth_graph` on the example register):

```json
{"files": 1, "templates": 4, "facts": 10, "skipped": 0,
 "preflight_fail": [], "counterfactuals": 10}
```

Everything the loop needs is already callable from an agent session
(`mcp-agent` fleet member) — budget caps (`budget.py`) and release tags
(`domain-YYYY.NN`) remain the two human gates.

---

## 5. End-to-end: one TTL file → trained adapter → measured

The full story with the exact commands (each line proven; GPU steps are
the existing lane, not re-run here):

```bash
# 0. (future: rmcp-extract produces ttl@h from the vault; for now a file)
vim corpora/generate/examples/site-register.ttl

# 1. knowledge+cognition synthesis from the graph
bin/synth graph ../corpora/generate/examples/site-register.ttl --domain construction
#    → derived/nl-to-ir/site-register--{qa,hops,cfkg}.json,
#      derived/abstain/site-register--abstain.json   (all preflighted)

# 2. mutations over the curated scenario tree (cognition)
bin/synth derive --per-code 4 && bin/synth abstain --per-code 2

# 3. identity + dataset
bin/synth manifest                       # derived@3b70db24…
bin/synth bundle --domain cognition      # cognition@327ea068 (8,957 rows)
                                         # + cognition-dpo (821 pairs)

# 4. train (existing lane, nothing new)
bin/prepare-domain cognition             # verified today: 8,957 lines
qtrain submit --domain cognition --recipe cognition/Qwen3.5-0.8B-qlora --mini
qtrain submit --domain cognition --recipe cognition/Qwen3.5-0.8B-qlora
qtrain submit --domain cognition-dpo --recipe cognition/Qwen3.5-0.8B-dpo   # R1
qtrain stage merge --from <run>          # then register → nano0 vLLM

# 5. measure
qtrain bench --config $CORPORA_DIR/bench/cognition.json
#    probes: ir-translate / ir-diagnose / ir-repair / abstain / json-guard / ir-emit
```

Recipes shipped for the axis (tier-S/M): `SmolLM2-360M-qlora`,
`Qwen3.5-0.8B-qlora`, `Qwen3.5-0.8B-dpo` — the 0.8B ones carry the
text-path `lora_target_modules` (VLM trap) and, for DPO, `beta: 0.1`,
one epoch, LR 5e-6, no sample-packing (pairs must not pack together).

---

## 6. Extending it (the intended way)

**Add a mutation rule** — the contract is one function:

```python
def mut_your_rule(ctx, stmts):
    """E?_CODE: one-line mechanism."""
    # return [(code, mutant_full_text, mutated_stmts_or_None, var_or_None)]
    # mutated_stmts=None ⇒ declaration-tier ⇒ diagnosis-only, never repair.
```
Register in `RULEBOOK`, add the hint to `ERROR_HINTS`, and the oracle
does the rest: proposals that produce zero, two, or the wrong error are
discarded and counted (`stats.candidates` vs `accepted` is your rule's
honesty score). No candidate ever becomes data on its own say-so.

**Add a grammar rule (R3)** — one line in `CORE`/`DOC`, then extend the
mirror generator's matching production; the property test
(`test_response_samples_parse`) IS the agreement contract between the
two. The GBNF and the generator may never drift.

**Add a data family to the bridge** — `cmd_graph`/`cmd_equiv`/
`cmd_conflict` are template builders + `_verify_scenario` before every
write; a new family is a function, not a subsystem.

**Add a domain** — drop `.ttl` files (funnel-shaped) anywhere, point
`synth graph` at the directory, add `recipes/<domain>/*.yaml` +
`bench/<domain>.json`. The bridge needs no per-domain code.

**Add a bench probe** — `bench/<domain>.json` questions with rule checks
(`contains:`/`regex:`/`json`); the `synth_gaps` mapping table (question
id → targeted synth commands) turns every new probe into an automatic
curriculum proposal.

---

## 7. Serving lane, live truth (2026-11-26, verified against the fleet)

`vllm-qwen3-05` deployment (k3s ns quadra, node nano0 — tegra/arm64
kernel = Jetson; hostNetwork pod, so its IP IS the node's 10.9.0.2).
It answers on both ifaces for HTTP (10.9.0.2:8000 and LAN
192.168.20.139:8000 / nano0.lan:8000) while nano0's sshd answers ONLY
the direct-link iface (`me@10.9.0.2` from pi00 — LAN port 22: No route
to host). Image vllm/vllm-openai:latest — the engine crash-looped for
~40 min with `httpx2 ConnectError: Network is unreachable` during
startup (offline node reaching for HF) and recovered on its own; do not
assume the endpoint is up — poll /v1/models before measurement.
ROOT CAUSE of the later, worse crash-loop (3 restarts, exit 255): the
liveness probe (3 fails x 30s, no initialDelay) kills the container
MID-STARTUP — vLLM's HTTP layer answers /health transiently before the
GDN Triton warmup + cudagraph capture finish, so the startupProbe
passes, then liveness starves the warmup. Fleet fix: give liveness an
Orin-scaled grace (patched `initialDelaySeconds=600`; a later rollout
kept the 600 and reset failureThreshold to 3). The restarts that
continued after this were NOT the probe — see the OOM paragraph below. Cold
start on Orin measures ~8-10 min end-to-end; a rollout mid-measurement
shows up as HTTP 404s (wrong url form) then Connection-reset errors —
check `kubectl get events` before blaming the model.

Adapters are FLAT peft dirs at hostPath /home/me/models/<name>/
(copy route workstation -> me@pi00.lan -> me@10.9.0.2, sha256 verified
both ends). Serving a trained adapter needs NO merge stage: add
`--lora-modules NAME=/home/me/models/<name>` to the deployment —
SINGLE OCCURRENCE, space-separated list. vLLM parses `--lora-modules`
with nargs="+" STORE semantics, so two `--lora-modules=X` args silently
DROP the first: cognition r1 added this way evicted kandi from
/v1/models with no error anywhere (startup-args log shows only the
surviving LoRAModulePath). (BUILD THE PATCH FROM THE LIVE SPEC —
another session may be rolling it; re-poll after patch. This is how
cognition r1 went live, and how it died: the r2-era spec serves
kandi v3 + cognition in the CORRECT single-occurrence form (my r1 patch
killed kandi; the parallel session's 6867868c45 rollout fixed it). That
two-adapter config then OOMKilled (137) TWICE on Orin — both times
AFTER /health answered 200, i.e. not a probe failure: 1.53Gi model +
650M KV + TWO rank-32 adapter pools bust the 6Gi container limit at
cudagraph capture on unified memory (the limit can't be raised — node
is 7.5G total with the OS living in the same RAM). Fleet fix applied:
`--enforce-eager` — capture phase disappears, pod went 1/1 in ~7 min
with 0 restarts, stable; for a measurement engine (sequential bench
traffic) the graph speedup is irrelevant anyway. Eager also makes the
liveness race academic.
Served modules answer POST {"model": "NAME"} on
/v1/chat/completions. The old url-convention trap (bench wanted the
BARE base, synth wanted /v1 — wrong shape = silent 404s dressed as
model answers) is DEAD IN CODE: bench.normalize_url and
synth.openai_endpoint accept bare base, /v1, or the full endpoint and
canonicalize. And bench now PREFLIGHTs every target (/v1/models probe,
fail-fast, named transport error) before grading a single sample —
a dead socket is a transport finding, never a measurement (an
overnight tunnel death once produced a proud table built from
connection refusals).

Budget law bit: qtrain submit estimates worst-case at
max_runtime x max_price_hr and refuses PRE-SPEND if
balance - $2 reserve < estimate (r2 attempt: $3.96-$2.00 < $2.25).
Cheaper fix than topping up: pin --max-price to the real offer tier.

The guided-decoding incident (2026-11-27, first full --guided pass):
three hard truths. (1) /health 200 DOES NOT MEAN THE ENGINE IS ALIVE —
healthz only proves the API-server loop; EngineCore deadlocked while
probes stayed green for a full kubelet window. Real liveness =
generation_tokens_total ADVANCING, or a tiny probe request with a
client timeout. (2) On Orin + vLLM v0.34 + xgrammar, guided decode
costs minutes per 512-token generation (masking runs on CPU; measured
~3-5 min each while healthy) — engine-side throughput, not queue
depth, is the wall; keep --guided to single probes with small
--max-tokens until root-caused, and mine unguided (oracle filter is
the $0 safety net; yield 0.212 across 12 families measured).
(3) A CLIENT TIMEOUT NEVER CANCELS SERVER WORK: killing the synth run
left ~20 orphan generations queued — the engine finished 6 more
requests (log 200s) twelve minutes after the client died, then the
step loop stopped producing tokens entirely (counter frozen 15:20:53
-> 15:35, running=0, fresh 16-token request unanswered) — pod
restart required (kubectl -n quadra rollout restart
deploy/vllm-qwen3-05; kubectl runs on pi00, NOT on nano0 directly).

## 8. Honest edges

- **Unreachable error families (documented, not hidden):**
  `E1_BAD_VARIABLE`, `E1_BAD_CARDINALITY`, `E6_UNKNOWN_STATEMENT`,
  `E6_BAD_SCOPE` — the lexer/parser rejects the malformed shape before
  Pass-1 can report *its own* code (you'd get `E0_PARSE_ERROR`, a
  different lesson). Hand-written scenarios cover them; the engine
  doesn't pretend.
- **The checker is the ceiling.** Guided decoding (R3) and RLVR (R7)
  inherit its completeness: whatever Pass-1 misses is exactly what the
  model can learn to get wrong. Harden with held-out hand-written suites
  before training on the reward, not after.
- **E3_USE_BEFORE_DEF stays shape-coupled** (~8–few yield on the curated
  tree): the only use after a BIND in those programs is RETURN, and
  swapping past a RETURN double-fires E6. Honest yield, recorded in
  stats, fixable when Phase-1 graph scenarios arrive with mid-program
  uses.
- **The bridge is a turtle subset.** Semicolon/comma list subjects,
  full-IRI-only documents, blank nodes, and collections are skipped and
  counted. Funnel output (`rmcp-extract`) already fits the subset;
  arbitrary web-turtle does not.
- **Bench ≠ capability proof.** `cognition.json` is 6 mechanical probes;
  real generalization tests come with held-out domains (R5 grid + a
  train-on-literature / eval-on-legal split) — that's the next
  measurement step, not this one.
- **Nothing here trains a model on its own output yet.** Student-side
  RFT (`synth sample`, R4) is the recursion's missing half: until it
  ships, every row in `cognition@h` originates from the curated tree or
  from TTL graphs — never from the model.

## 8. Backlog → `REGIMES.md`

Shipped 2026-11-25/26: **R1** (DPO lane + format + recipe + pref-aware
render_config), **R2** (CFKG), **R3** (GBNF compiler + lexer-pinned
grammar + property tests), **R4** (student-side RFT sampler+grader,
live-verified vs nano0: base yield 0.0 = the before-meter), **R5** (`mix`
seeded grid primitive), **R6** (mergekit config tooling), **R7** (the
RLVR reward function `grade` — 821/821 gold agreement; pod loop pending),
**R11** (computed equivalence mining), **R12** (dual-source conflict
mining). Queued: R8 process supervision (needs parser line/col plumbing),
R9 self-model IR, R10 loop driver (the `gaps` propose step is shipped),
plus the dialect program (transpiler-as-oracle → any EBNF'd language).
