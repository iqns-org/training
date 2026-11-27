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

### R11 — Equivalence mining (P5) `M` · **DONE 2026-11-26**
- `synth equiv` mines (Program A, Program B, verdict) pairs over derived
  nl-to-ir programs: uniform variable renames must be EQUIVALENT; deleting
  observable BIND/EMIT/READ lines must be DIFFERENT. The fixture executor
  COMPUTES each verdict by comparing full execution deltas; unobservable
  or unexecutable drops are counted undecidable and dropped (34 of them
  on the live tree — honest yield). 27 EQUIVALENT pairs -> bundled as
  `equivalence` (81 rows). Verdicts are never typed.

### R12 — Contradiction mining (P19) `M` · **DONE 2026-11-26**
- `synth conflict <ttl> <other.ttl>` finds the same (subject, predicate)
  asserted with different objects across two sources -> CONFLICT rows
  that name both values and cite both files; ctx verified to parse with
  both facts coexisting. Example register vs rev2: 3 conflicts
  (status, inspector, riskScore) -> bundled as `conflict` (6 rows).
  The memory-governance skill retrieval serving needs, as corpus data.

## 2. Sequencing

```
SHIPPED  R1 R2 (DPO lane, CFKG) · R3 R4 R5 R6 R7harness R11 R12 (2026-11-26)
         R13 bench oracle grammar (rate:/not:) · R14 contrast factory ·
         R15 ablation/steering direction math + hard/soft tooling (2026-11-26)
NOW      serve-flag (R3 GBNF on vLLM), mix grid runs + merge pod-exec (need
         serving), GRPO loop (needs rollout wiring + checker hardening)
LIVE     cognition r1 trained+served+measured; r2 RELAUNCHED (04d2e9 died
         PRE-SPEND on the honest budget gate: balance $3.96 - reserve
         $2.00 < estimate $2.25 = runtime x worst-case price; a presign
         retry then failed for missing S3_* in this dev shell — creds
         live in the infra SOPS secret, so volume lane is the dev path;
         relaunch d30d5b with --max-price 0.60 -> TRAINING (pod Running
         on RTX A4500 since 11:13Z; ledger `state` stays `provisioning`
         forever — dispatch never writes `running`, cosmetic only;
         `qtrain watch <run_id>` reads the agent-lane status.json:
         step 106/892 loss 0.047 at ~1h, 4 epochs = 892 steps; scoped
         RunPod key cannot read pod logs — the agent lane is the only
         observability path. THE r2 POSTMORTEM (verified against lane +
         ledger, 2026-11-27): the submit watch hung AFTER ITS FIRST
         PING (11:14:39Z — 26 min before the harness reboot, so the
         reboot does NOT explain it; cause unresolved). Pod-side that
         meant: attempt 1 armed on the stale-but-readable ping, fired
         at +600s (SIGTERM 11:30:06Z, manifest stamped deadman:true)
         — then Runpod RESTARTED THE CONTAINER (no restartPolicy in
         v2 CreatePodRequest — checked the live OpenAPI), so attempts
         2,3,4,5 each booted fresh, re-armed on the same stale ping,
         fired again: the fail-safe became an unbounded billing loop.
         My 13:00Z ping-rescue proved the mechanism (attempt 4 crossed
         the death line: step 92 -> 290), but harness restarts killed
         the rescue daemons twice and the zombie ran 12h to a $0.207
         balance. Ended by hand: terminate + reconcile -> row closed
         `failed`, gate error `required artifact missing:
         adapter_model.safetensors` (correct — r2 never produced a
         model). ~$3.3 spent, zero artifacts, every mechanism logged
         the truth the whole time. LAWS LEARNED: (1) container exit !=
         pod stop on this provider — deadman amended (SPEC 14c
         POD-KILL-THEN-EXIT, agent env contract in; dispatch-side
         payload.json delivery is the pending hot-file wire);
         (2) every volume-lane submit runs alongside
         scripts/supervise_run.py until (1) lands; (3) ping staleness
         is now a first-class observable (progress.ping_age +
         PING-STALE warning line in watch).):
           r1 03760e — 2 epochs, loss 0.0886, $0.21, 53 min (~26.5
           min/epoch on RTX 2000 Ada). Served on nano0 vllm-qwen3-05 as a
           SECOND --lora-modules (no merge stage — registry vllm-lora
           true, proven live). bench 1/6 (base also 1/6; translate flips
           run-to-run -> suite pins repeats=3); sample yield 0.0/48:
           IR SHAPE learned (clean READ/BIND/RETURN + json-guard),
           CONTRACT not (abstain/repair/diagnose/emit all FAIL, verdicts
           in tmp/bench-cognition-r1.json). RE-MEASURED (canonical suite
           restored e0e1f9a, repeats=3, ledger --record on 03760e): stable
           1/6 — ir-translate 3/3, the other five 0/3, 34.8 tok/s med.
           The translate flip was single-sample noise, now pinned.
           Instrumented per-family sample (temp 0, k=1, --task), rerun
           clean after the engine settled (the first pass was wrecked
           by the parallel session's rollout — connection resets, not
           verdicts): INVALID-PROGRAM 6/6, ABSTAIN 5/6 — recognition
           families nearly perfect; craft families dead: concrete-
           examples 0/50 (129 AssertionError + 65 IRError + 6 Returned
           at k=4; Returned = RETURN leaked into the context block, a
           real misshape not a harness ghost), nl-to-ir 0/6, ir-to-nl
           1/6, repair 0/6, program-completion 0/6. So r1 learned to
           JUDGE the IR (verdict + abstain + bench translate 3/3) but
           not to CRAFT valid-and-correct programs — r3's data lever
           is the craft families specifically, not epochs, not the
           checker. The buckets also caught a tool lie:
           unfiltered cmd_sample walks sorted scenario files and --limit
           cuts mid-family, so "0.0/48" (r1) and "0.0/50" measured ONE
           family (concrete-examples) — rejects_by_system (315331c) is
           what exposed it; per-family --task is now the honest protocol.
           INTERLEAVED FULL-CORPUS MINING (post 53fffcc, temp 0.9, k 4,
           limit 48): yield 0.212 — NOT zero. The "corpus contract dead"
           story was the single-family artifact again; truth = verdict
           families keep at high temp, craft families never do. 115 DPO
           pairs mined across 12 systems (tmp/synth/rft/dpo-mining.jsonl;
           16 connection-reset errors when the tunnel died mid-run —
           errors list kept honest).
           HOLDOUT LAW v2 (13am amendment): synth profile measured the
           row-hash bucket LEAKING 1.0 — every templated holdout row had
           train siblings (diagnosis uniq 0.006: 3708 rows, ~22 golds —
           phrase memorization, and the "1/6 == 1/6" bench could not see
           it). is_holdout now buckets by TEMPLATE (normalized first
           message): re-audit leak 1.0 -> 0.0 at the same ~1.9% holdout
           rate; mirror test pins synth/prepare-domain agreement; the
           Returned-bucket fixture leak sealed (named AssertionError);
           paren/square hazard classes dropped (legal IR list literals =
           false positives). SPEC 13am amended (both clauses).
           GUIDED WEDGE + RECOVERY (8b3a7f4 laws, engine pod restarted):
           the first full --guided pass (limit 12, k 2, 512 tok) killed
           EngineCore on Orin. Root cause = client/server timeout
           mismatch: guided decodes ~0.5 s/token on masked CPU, 512 tok
           needs MINUTES, http_json's 120 s timeout fired while the
           server still worked — and a client timeout NEVER cancels
           server work. Twelve serial timeouts piled ~20 concurrent
           guided generations into one batch; the engine finished 6 of
           them, then produced ZERO tokens (counter frozen 15:20:53 ->
           15:35 while /health happily returned 200 — healthz only
           tests the API loop). Rollout restart recovered; post-restart
           single guided probe: 15 tok / 7.7 s, grammar-clean output —
           guided IS viable at small doses, dead at orphan-storm scale.
           Fix shipped: cmd_sample guided requests use a 600 s client
           timeout (the client must outlive the server, serially,
           always) + --guided help warns. Liveness protocol: watch
           generation_tokens_total ADVANCE, never /health alone.
           GUIDED VERDICT (safe protocol, limit 6 k 1, 256 tok): 0 of 6
           rejects were IRError — vs UNGUIDED where 44/115 (38%) were
           parser hazards. Grammar buys format purity (6 new DPO pairs
           of pure-semantics fuel, dpo-guided.jsonl) and buys nothing
           on craft (yield 0.0 at n=6 — small sample, honest bound).
           r2b PRECONDITION beyond f831b7 closing: QTRAIN_POD_STOP_URL
           is set by NOBODY yet (grep dispatch/runpod: zero refs) —
           until the peer wires it, a deadman fire = container revival
           (the exact r2 zombie drain); POD-STOP WIRE stays held.
           r2 = gaps output verbatim (derive+abstain --per-code 3 ->
           bundle 7482 rows -> 6727 train + 137 holdout, content-hash
           law verified) + num_epochs=4. Learnings landed in code:
           cmd_gaps eats the REAL bench shape (was fiction-tested),
           cmd_sample buckets rejects by cause, ledger evals record both
           legs (r1 vs base) — docs/AS-BUILT §7.
           r3 lever (from the CAUSE MAP, not bucket guesses — the
           reject dump + `pairs` tools, 1d52eae): ir-to-nl fails 4/4 on
           "final must be prose" (it answers NL asks WITH IR — a
           direction-confusion, not a capability gap); program-
           completion fails "must preserve supplied source" (rewrites
           instead of appending); repair fails E2_UNBOUND single-error
           signature mismatch; and the em-dash —
           `IRError: unexpected character '—'` recurs across THREE
           families: prose punctuation leaking into programs (persona-
           corpus bleed?). Fixes are cheap and targeted: direction-
           emphasized ir-to-nl rows, preserved-prefix completion rows,
           E-signature repair rows, plus a no-emoji/no-dash contrast;
           16 mined DPO pairs already sit in tmp/synth/rft/dpo-craft
           .jsonl (expand-lane shape, regenerable at temp 0). Verdict
           families (abstain 5/6, invalid-program 6/6) already pass —
           re-training them is waste. And an
           L3 event worth naming: the r2 rebuild from an UNCHANGED
           scenarios tree yielded cognition@569e46d5 (6864 SFT + 618
           DPO) vs r1's cognition@e184c8de (9044 + 821) — identical
           sources/candidates (3241/42711), FEWER accepted, same eight
           tags, shrinkage concentrated in diagnosis/repair. check/
           expand/ir-* files are untouched since r1; src/synth.py
           (where the rulebook lives) moved in the 7d1859c tree-sweep
           that merged two sessions' staged work into one commit —
           the acceptance delta points there. r3 pre-flight: bisect
           7d1859c's synth.py hunks for the mutation-rule change that
           dropped cells, then re-pin ids. (Candidates being equal at
           per-code 3 also says the cap rarely binds — growth lever is
           scenario count, not knobs.)
NEXT     R16 Datalog legal oracle · R17 pytest language oracle (coding axis) ·
         R18 concurrent-IR model checker · R20 calibration probes
         REGIME PLAN (agreed 2026-11-27, first tranche EXECUTED):
           P0 DONE — L3 canary (tests/test_rulebook_canary.py): 10 mutation
           rules + expansion math + unbound-READ subtlety pinned; drift
           injection (recode a rule) flips it red, verified. The silent
           −24% is now a loud test failure.
           P1 DONE — `synth profile` (distributional QA, dataset mode):
           ran on datasets/cognition: DIAGNOSIS 3708 rows but ~22 unique
           golds (uniq 0.006), equivalence 81→~1, abstain 43→1 — verdict
           families are phrase memorization in a 9k-row costume; repair
           2952→~95 unique, 288 paren + 252 square hazards in golds;
           holdout TEMPLATE LEAK = 1.0 in every templated family (13am
           row-hash split + template siblings ⇒ holdout NLL measures
           memorization — quantified, the number split-law v2 needs).
           Known follow-up: square/paren presence over-flags (IR has
           legal list literals) — switch to parse-based hazards.
           P4 VERDICT — split IS row-content-hash (bin/prepare-domain
           comment admits it); fix = family-level bucketing by
           messages[0] hash — LAW CHANGE, needs colleague sign-off
           (breaks their eval trend), propose with P1's numbers.
           P2 IN PROGRESS — base-model temp-0 per-family baselines
           recorded alongside r1's (behavioral table = the acceptance
           gate for every future corpus round; loss/NLL demoted to
           fit-only). Next: encode the law in SPEC + wire r2b/r3.
Held     qtrain stage ablate + synth_contrasts MCP wiring (mixed-file hunks
         wait for the concurrent holdout-lane session to land)
         POD-STOP WIRE (r2 law follow-up): the agent now POSTs
         QTRAIN_POD_STOP_URL on deadman fire (tested, no-regression);
         delivering that env at pod-create needs dispatch.py edits —
         THE concurrent session's hot file, so flagged not hand-edited.
         Meanwhile every volume-lane submit pairs with
         scripts/supervise_run.py (feeds ping, harvests fresh manifest,
         hard-terminates). Also open: why the submit watch's ping loop
         froze after its first cycle (11:14:39Z) — socket stall in
         _watch is the prime suspect, bisect needs a repro.
```

HANDOFF 2026-11-27 16:00 (+11), to the infra/serving lane —
r2b unlock, three items, in dependency order:
1. POD-STOP WIRE (the only hard gap). Agent side is SHIPPED and
   tested (app.py _pod_stop_best_effort, v0.5.0): env contract =
   QTRAIN_POD_STOP_URL (required), QTRAIN_POD_STOP_METHOD (default
   POST), QTRAIN_POD_STOP_TOKEN (optional Bearer). Nothing sets them:
   grep dispatch.py/runpod.py = zero refs. URL must be kill-capable
   and pod-reachable — the orchestrator's own terminate route is the
   natural target (scoped RunPod key lives there, not in the pod;
   mint per-run token like the existing agent-token flow). Why hard:
   CreatePodRequest has NO restartPolicy (OpenAPI-verified r2
   postmortem) -> deadman os._exit() without a pod kill = container
   REVIVAL = the r2 zombie drain (5 fires, ~$4). r2b must not launch
   before this lands.
2. f831b7: yours (5c5eda8 fixed the watch CLASS; the stale row is
   still open, provisioning since 15:08). It blocks r2b twice over:
   shared lane during an open provider anomaly, AND the budget gate
   counts open runs against margin. Close/reconcile it, then say go.
3. TREND BREAK — needs your call, not mine. Holdout law v2 (c3d7027)
   buckets by TEMPLATE now; every v1-era holdout NLL (kandi 0.9178 on
   621 rows vs base 4.1088 included — that split had train siblings:
   partly memorization) is NOT comparable with v2-era numbers.
   Options: (a) rebase by re-scoring kandi v3 + base on a fresh
   v2-prepared queer holdout (qtrain stage holdout, ~$0.15, no
   retrain), or (b) era-stamp the eval rows and accept a discontinuity
   at v2's first round. Proposing (a): the rebase number becomes the
   honest anchor for r2b's first v2 holdout NLL.

## 1b. R13-R15 evidence (2026-11-26)

- **R13** `orchestrator/bench.py`: check grammar = `contains:/regex:/one_of:/json/none` + `not:<check>` + `rate:<check>/<lo>-<hi>` (repeat-group fraction, `observed` recorded). Tests: 4 pure-grammar cases. The bench is now the soft-oracle harness for behavioral dials.
- **R14** `synth contrasts [--domain] [--behaviors] [--limit]` → `tmp/synth/contrasts/<behavior>.jsonl` ({behavior,label,messages}): ir-valid from DPO couples, abstain vs fabricated answers, self-name/emoji/json via bench-primitive labelers. Fuel for steering (ORACLES.md §2). Tests: 2.
- **R15** `postproc/ablate.py`: mean-diff + PCA(power-iter) direction, `W(I-vvᵀ)` hard projection, rank-1 steering-LoRA soft spec (rides vLLM multi-LoRA + R6 merge algebra — reversible by default). Tests: 8 incl. CLI end-to-end. Pod probe forward-pass + `qtrain stage ablate` wiring held behind the concurrent dispatch.py changes.
## 3. Anti-pattern register (enforced, not advice)

- LLM-judge in any corpus/reward gate — banned (FRAMEWORK §9.2)
- student-only data (collapse) — R4 must mix teacher-derived + student
- unsplit phrasing pools (eval leakage) — eval suites use unseen phrasing
- one-LoRA-trains-all — keep axes separable; composition is R6's job
- learned RM while an exact checker exists — only if R7 demands it
