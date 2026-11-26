# quadra/training — System Spec

Status: **DRAFT v1** — target architecture, replaces the SSH/rsync job scripts.
Companion repos: `quadra/infra` (k3s cluster, MCP fleet, S3), `quadra/mcps` (fleet images).

## 1. Purpose

Run many LoRA/QLoRA fine-tunes (axolotl) across models and datasets on rented
serverless GPUs (RunPod, GPU.ai, …) with the **full cycle automated**:

```
submit → expand dataset → provision pod → train → ship artifacts to cluster
       → verify → terminate pod → record in ledger → (optional) eval → notify
```

Managed from inside the quadra cluster through an MCP server (like the rest of
the fleet) and a thin CLI for humans.

### Goals
- **One generic image**, provider-agnostic; nothing is installed on pods at
  runtime. What ships = what runs.
- **No SSH into pods.** The pod exposes a small control API; the cluster
  orchestrates over HTTPS. Artifacts move over pre-signed S3 URLs.
- **Multi-job, multi-model, multi-strategy**: declarative job specs, a queue
  with concurrency, per-job budget caps, provider fallback.
- **Return lane**: finished adapters always land in cluster S3 (RustFS),
  checksummed and catalogued, ready for vLLM (`--enable-lora`) on `nano0`.
- Repeatable + testable: a `local` (docker) and `mock` provider make the whole
  pipeline runnable without leaving the homelab.

### Non-goals (v1)
- Multi-node / distributed training (single GPU per job; LoRA scale).
- Provider managed fine-tune APIs (locked per-provider, can't pin axolotl, poor
  multi-model story). Revisit only for providers that beat our image on cost.
- Dataset authoring tooling changes — corpus expansion (`$`-variable chat
  envelope) is kept as-is, just relocated into a package.
- Training on `nano0` beyond cheap small-model smoke jobs.

## 2. Current state → what this replaces

| Now | Problem | Replacement |
|---|---|---|
| `bin/gpuai-job` / `bin/gpuai-deploy` (≈500-line bash twins) | duplicated, SSH+rsync, conda patched at runtime, secrets rsynced as `~/.env` | orchestrator + generic image |
| `src/remote_axolotl_run.sh`, `axolotl/fix-axolotl.sh` | environment repaired on every run | baked into image, pinned |
| `src/gpu/` provider CLI (gpuai/runpod) | SSH-oriented `Provider` ABC | ported to `orchestrator/providers/*`, SSH surface dropped |
| `tmp/*-model/`, `outputs/` artifact dumps | not catalogued, not checksummed, not in cluster | S3 `quadra-ml` buckets + ledger |
| `.env` with live HF/GPUAI/GROQ keys in the worktree (and git history) | leaked secrets | move to SOPS (infra) + `.env` untracked; **rotate all three now** |

## 3. Architecture

Four layers, one repo:

```
            ┌─────────────────────────── quadra cluster (pi00 / nano0) ───────────────────────────┐
 human/     │  mcp-trainer  ←── MCP :3022 ──  orchestrator (queue, ledger, budget, providers)      │
 agent ───► │  (fleet member)                    │  │  HTTPS pod API        pre-signed S3 URLs     │
            └────────────────────────────────────┼──┼──────────┼───────────────────┼──────────────┘
                                                 ▼  ▼          ▼                   ▼
                                          RunPod / GPU.ai   trainer pod(s)     RustFS S3
                                          (rented GPUs)     our image:         quadra-ml
                                                            axolotl + agent  ◄─ artifacts in,
                                                            (REST/MCP)       datasets up
```

1. **`trainer` image** (`images/Dockerfile`) — CUDA + pinned axolotl + `trainer-agent`.
2. **`trainer-agent`** (`agent/`) — in-pod control plane: FastAPI REST
   (`/v1/…`, bearer token) + the same surface as streamable-HTTP MCP tools for
   ad-hoc pod poking. Executes exactly one job per pod, streams logs/metrics,
   uploads artifacts.
3. **`orchestrator`** (`orchestrator/`) — cluster-side brain: job specs,
   dataset expansion+upload, provisioning via provider adapters, polling,
   verification, termination, SQLite ledger, budget guardrails. CLI: `qtrain`.
4. **`mcp-trainer`** (`mcp_server/`) — fleet MCP server wrapping the
   orchestrator: the cluster-side control API/MCP. Deployed by `infra`
   exactly like the other MCPs (manifests live there, port 3022).

Key decisions:
- **Pod reachability is provider-dependent** (learned live on RunPod): the
  REST v2 API exposes NO port mappings for pods — `ports` is spec-echo only;
  direct access to the in-pod agent HTTP is via `ssh.direct` (startSsh +
  published 22/tcp), which we treat as the **debug/exec lane**, not the
  control plane. The durable control plane is therefore push-based:
- **Push artifacts + mirror state, don't pull.** Agent PUTs `status.json`,
  `run.log`, `artifacts.tar.gz` and `manifest.json` to pre-signed S3 URLs
  (status periodically during the run, everything on completion). The
  orchestrator polls those objects and signals cancellation through a small
  `control.json` the agent GETs — zero inbound connectivity to any pod,
  works on every provider, survives spot-eviction observation. When a
  provider does give a reachable `base_url` (GPU.ai's host:port), the agent
  REST API serves live status/logs interactively as an accelerator.
- **Datasets go the same way in**: expand → hash → `PUT s3://quadra-ml/datasets/…`
  → pre-signed GET URL handed to the pod. Pods carry zero long-lived credentials.
- **Image distribution**: canonical registry is the **cluster Forgejo**
  (`pi00.lan:30080/quadra/trainer`) — the ledger pins digests, `nano0`/local
  lane pulls it over LAN. The **cloud mirror** is `ghcr.io/iqns-org/trainer`,
  built by GitHub Actions on tag push: the Forgejo repo push-mirrors to
  `github.com/iqns-org/training`, so a `v*` tag released in Forgejo lands in
  Actions (GitHub's free amd64 runners are the only sane place a ~20 GB CUDA
  base builds — neither pi00's arm64 runner nor any workstation disk fits it).
  Actions pushes the tag+sha to ghcr; `scripts/sync-image.sh` then pulls the
  same tag **back into Forgejo with skopeo (digest-preserving)**, so both
  registries resolve one digest. Pod launch specs pin that digest. If the ghcr
  package is kept private (repo is), providers get a fine-grained
  read:packages credential: RunPod `POST /v2/registries`, GPU.ai
  `gpu registry-credentials`. Flip-to-public is the zero-credential alternative.

## 4. Job spec & recipes

`configs/jobs/*.yaml` — one file per recurring experiment; `qtrain submit -f`.
`$CORPORA_DIR/recipes/<domain>/<model>-<strategy>.yaml` — axolotl overlays
(moved out of training per the §4b boundary law; superseded the old
`configs/recipes` in-repo plan and absorbed `axolotl/<domain>/*.yaml`).

```yaml
id: queer-smollm2-360m-qlora
dataset:
  domain: queer                # expand via corpus tool → content-hash → S3
recipe: queer/SmolLM2-360M-qlora
overrides: { learning_rate: 2e-4 }
resources:
  gpus: [A40, A5000, RTX4000D, RTX5000, L40S]   # accepted GPU types, cheapest ok
  min_vram_gb: 16
  max_cost_per_hour: 0.55
  max_total_cost: 3.00          # hard kill + alert when crossed
  max_runtime_min: 180
providers: [runpod, gpuai]      # ordered fallback; skip to next on
                                # provision-fail / OOM / spot-eviction
artifact:
  bucket: quadra-ml
  prefix: adapters/queer/SmolLM2-360M-Instruct/   # run-id appended
  merge: false                  # true → also merge & upload merged model
eval: none                      # none | scenarios:<name> | custom script
```

Compiled **job payload** (what the pod receives, `s3://jobs/<run-id>/job.json`):
dataset URL + sha256 + size, fully-rendered axolotl config, run-id, return
URLs, retention of the base-model revision pin. The payload is the single
source of truth on the pod — no git checkout on pods, ever.

## 4b. Corpus separation (`quadra/corpora`)

Curation data does **not** live in the training repo. Sibling private repo
`quadra/corpora` holds everything a person edits when shaping data:
`datasets/<domain>/**.jsonl` (canonical chat envelope, `$`-variant rows) and
`generate/` (prompts, case iterations, scenario curricula). Rationale:

- The training repo's GitHub mirror is (and must stay) build-facing/public —
  corpora is private curation; separation keeps it out of that blast radius.
- Corpus churn (new rows daily) shouldn't dirty the code repo or its tags.
- Data is ~4 MB of JSONL → plain git is enough: **no LFS, no submodules**.
  Identity for the pipeline is the content hash of the *expanded* file:
  job specs reference `domain@sha256…`; S3 `datasets/<domain>/<hash>/` is the
  durable distribution (pods get presigned GETs, never a git checkout).
- Linkage is by env: `CORPORA_DIR` (default sibling `../corpora`; unset =
  legacy layout, resolved against the repo root, never CWD).

**The §4b boundary law (2026-11-24, run/task data round).** The training
repo is PURE MACHINERY: orchestrator, agent, postproc scripts, generic
image definitions, expander, tests, this spec. Anything that carries
domain truth lives in corpora and NOTHING in training should:

| corpora path | content | was |
|---|---|---|
| `datasets/<domain>/` | template corpus rows | (already the rule) |
| `recipes/<domain>/*.yaml` | axolotl per-model regimes | `training/axolotl/<domain>/` |
| `bench/queer.json` | question suite + endpoint urls | `bench.py` constants |
| `templates/kandi.jinja` | the served chat template | nano0 host only |
| `tools/gen_data_round1.py` | corpus content generators | `training/src/` |
| `docs/` | audits, bench snapshots, run records | `training/docs/` |

Run OUTPUTS belong to neither repo: the PVC lane is the durable store
(`runs/<id>/*` + `runs/index.jsonl`), the local `runs/` + `state/` dirs
are untracked caches. The mcp-trainer image no longer bakes domain data
(`COPY axolotl/` dropped; the pod already mounts the corpora checkout
read-only at `/data/corpora`). Mechanical fallbacks remain one-level
deep so old paths still resolve during the transition — they are
delete-when-quiet, not permanent compat.
  legacy in-repo layout during migration). `scripts/split-corpora.sh`
  refreshes/commits the corpora checkout from the training worktree; it is
  idempotent and additive.
- Release convention: tag `domain-YYYY.NN` in corpora when a state feeds real
  runs; recipes/jobs in training pin by hash, not by corpora git SHA.

## 5. In-pod agent (`trainer-agent`)

- `GET /healthz` — liveness for providers' health checks + gatus-style probe.
- `GET /v1/info` — image tag, gpu, driver, CUDA, axolotl version.
- `POST /v1/jobs` `{payload_url}` — start job (409 if one is active). Also
  auto-starts at boot when `JOB_URL` env is set (the normal path; the POST
  exists for manual/driver pods).
- `GET /v1/jobs/{id}` — state machine + progress:
  `{state, step, total_steps, epoch, loss, lr, gpu_util, eta_s, started_at}`
- `GET /v1/jobs/{id}/logs?tail=N&follow=1` — plain-text stream (NDJSON follow).
- `POST /v1/jobs/{id}/cancel` — SIGTERM → checkpoint-aware abort.
- `GET /v1/jobs/{id}/artifacts` — file list w/ sha256 (pull-lane + audit).
- MCP: same tools (`job.status`, `job.logs`, `job.cancel`, `artifacts.list`,
  `info`) mounted at `/mcp` (streamable-HTTP, same bearer token).

Lifecycle inside: fetch payload+dataset → validate hashes → `axolotl train`
(subprocess, cwd `/work`) → tail metrics → **artifact gate** (required files:
`adapter_config.json`, `adapter_model.safetensors`, tokenizer files,
`train.cfg`, `metrics.json`; refuse `checkpoint-*` leftovers as a *final* set,
but ship latest checkpoint if the run died early and it's useful) →
`manifest.json` (everything + provider/instance/gpu-type/price/cost-estimate)
→ PUT tar.zst + manifest to presigned URLs → write `~/done.json` → exit 0
(pod idles; orchestrator terminates).

Auth: per-run random `AGENT_TOKEN` bearer on everything but `/healthz`.
Failure posture: nonzero exit + `manifest.json` with `status: failed` +
last-1000-lines log uploaded, so failures are also fully observable in-cluster.

## 6. Orchestrator (`qtrain`)

```
qtrain submit configs/jobs/queer-360m.yaml        # → run-id, returns immediately
qtrain status [<run-id>|--watch]                  # table / json
qtrain logs <run-id> [-f]                         # from S3 run log / pod stream
qtrain cancel <run-id>
qtrain list [--domain queer --state finished]
qtrain artifacts [run-id]                         # catalog: local cache of ledger
qtrain sweep --grid grids/queer-models.yaml       # matrix: domains × models × strategies
qtrain serve --domain construction --merge ...    # (P4) register adapter to nano0 vLLM
```

Landed surface (13r, §14): `qtrain stage <merge|onnx|int8|awq> --from-run <id>`
· `progress|watch|metrics <run-id>` · `events [--run]` · `notify-test` ·
`register <run-id>` · `reconcile` · `doctor`. Budget vocabulary
`--max-cost` (default: derived estimate) + `--max-price` (rate ceiling) on
`submit` and `stage`.

Internals:
- **Ledger**: SQLite `state/ledger.db` — runs, events, states, costs, artifact
  URIs + sha256. Single writer (the server process); CLI talks to it directly
  when co-located, via mcp-trainer otherwise.
- **Queue**: FIFO with per-provider concurrency caps (`max_concurrent`, default
  2) and a global spend cap/day. `sweep` fans out into the queue.
- **Reconcile loop** per run:
  `pending → provisioning → running → uploading → delivered → verified → done`
  (failures: `failed` (terminal, pod kept ≤2h for post-mortem then killed),
  `cancelled`, `lost` (pod unreachable → artifact gate decides done/failed)).
  Pod unreachable mid-run on a serverless provider is *expected* (spot); the
  run falls through the `providers` list from the current position, up to
  `retries: 2`.
- **Provider adapters** (`orchestrator/providers/`): port of `src/gpu/` with a
  narrower, agent-oriented interface:

  ```python
  class Provider(Protocol):
      def launch(self, job, env, image) -> Instance   # returns base_url + ids
      def poll(self, instance) -> InstanceStatus
      def terminate(self, instance) -> None
      def catalog(self, gpu_types) -> list[Offer]     # price/vram filtering
  ```

  `runpod.py` (REST API **v2** — validated live 2026-11-22; see §12: scoped
  `rpa_` keys authenticate on REST but *not* legacy GraphQL/runpodctl.
  `POST/GET/DELETE /v2/pods`, native `GET /v2/pods/{id}/logs` (may 403 on
  scoped keys → agent lane is primary for training logs), and
  `GET /v2/catalog/gpus?include=AVAILABILITY&product=POD` for orderable
  offers + live pricing — selection filters on availability/tier flags so
  capacity surprises happen pre-launch, not at 400-time),
  `gpuai.py` (their instance API, custom template = trainer image),
  `local.py` (`docker run --gpus` on nano0/laptop — dev + cheap small jobs),
  `mock.py` (deterministic fake for CI; the happy path + every failure branch
  is tested against it).
- **HF cache strategy**: per-provider volume (RunPod network volume) +
  base-model revision pin in the recipe; `huggingface_hub` snapshot at start.

## 7. `mcp-trainer` (cluster control API)

Fleet member like the others: hostNetwork pod, port **3022**, planes per fleet
convention (`/mcp` operator bearer, no admin plane; `viewer` token may list
status/artifacts). FastMCP streamable-HTTP; thin wrappers over orchestrator:

`train.submit` `train.status` `train.logs` `train.cancel` `train.list`
`recipes.list` `datasets.list` `datasets.stats` `adapters.list` `adapters.get`
`capacity.price` (live offers from provider catalog) `sweep.submit` `sweep.grid`

Implemented surface v0.3.x+ (13q/13r): gpu_catalog · list_recipes ·
submit_training · submit_stage · job_status · run_progress ·
wait_for_events · run_events · list_runs · list_pods · terminate_pod ·
doctor · reconcile · refresh_corpora · resource
`trainer://runs/{run_id}/progress`.

Manifests live in `infra/deploy/app/mcps/manifests/mcp-trainer.yaml`
(+ `update-rules.sh` port table, + gatus check, + SOPS secret
`mcp-trainer-env`: provider API keys, S3 creds, tokens). The image builds in
this repo's CI (not quadra/mcps) and is pushed to `quadra/trainer-mcp` in the
Forgejo registry; only `trainer` (the CUDA image) needs the public registry.

This is also where **automation hooks** land (P4): a `webhook`/cron trigger —
e.g. "new corpus files under `vault/training/<domain>/` → expand →
`sweep.submit`" wired through `mcp-agent` workflows, exactly like `doc-intake`.

## 8. Security model

- Pods get **no long-lived secrets**: pre-signed S3 URLs (≤48h), per-run
  `AGENT_TOKEN`, and a **scoped read-only HF token** limited to the recipe's
  model repos (mint per job, revoke after). Provider keys stay cluster-side.
- S3: bucket `quadra-ml` (rustfs), public route only through the tunnel with
  path-scoped presigned PUTs; no anonymous list/read; lifecycle rule keeps
  `runs/` logs ≤90d, `adapters/` durable, `datasets/` GC'd by content hash
  when unreferenced.
- Image supply: CI builds on tag, **pin by digest** in provider launch specs.
- Rotate the currently committed `HF_TOKEN`, `GPUAI_API_KEY`, `GROQ_AI_API_KEY`;
  `.env` stays untracked (already in `.gitignore`? verify) and secrets move to
  SOPS like the rest of infra.

## 9. Cost & reliability guardrails

- Budget — one vocabulary, three gates (`orchestrator/budget.py`, 13q):
  * HARD floor: the provider's prepaid balance — RunPod refuses spend past
    what the account holds. Checked pre-launch with a `QTRAIN_RESERVE`
    (default $2) set aside; best-effort (scoped keys may not expose it).
  * PREDICTIVE: every job class carries estimated minutes (`EST_MINUTES`);
    `estimate = rate x est_minutes x MARGIN` (default 1.5 — "some margin of
    error, not too much"). A launch is REFUSED before it starts if the
    aggregate projection of concurrent open estimates would exceed
    `QTRAIN_BUDGET` (default $5), or if balance-minus-reserve is under the
    estimate. The kill threshold DEFAULTS to the estimate — explicit
    `--max-cost` still overrides everywhere.
  * SOFT kill: `_watch` terminates when wallclock x rate crosses the
    threshold — the reactive backstop for hangs, not the plan.
- The right GPU = cheapest offer that FITS: vram floor, SECURE when the
  lane requires it, arch exclusions from the stage spec, then a RATE
  CEILING (stage specs default $0.80/hr; train $QTRAIN_MAX_PRICE_HR /
  $1.50). The ceiling is the statement *waiting is cheaper than
  overbuying*: a 360M-param job must not burn a $7.99/hr B200 even when
  that is the only thing in stock — capacity-retry loops wait for a
  right-sized card instead. `--max-price` overrides when you mean it.
- Every run records: gpu type, provider, price/hr, wall time, est. spend —
  ledger + `manifest.json`.
- Community tier has real inventory gaps (observed: A5000/4000-Ada sold out for
  hours, 4090 community 400s while SECURE succeeded). Fallback ladder in §4
  (`providers: [runpod, gpuai]`) is therefore joined by a *within-provider*
  ladder: try accepted GPUs cheapest-first via the availability catalog, then
  escalate COMMUNITY → SECURE, honoring the price caps.
- Provisioning: poll budget (`max_wait`), re-provision on terminal/stale states
  (behavior already proven in today's bash), then fallback provider.
- Orphan sweep: `qtrain doctor` — lists live instances not in ledger as
  non-terminal; cron it; warns to the same notify channel.
- Artifact verification before terminate (already the rule today): gate fails →
  keep pod ≤2h for forensics, mark `failed`, never silently delete.

## 10. Repo layout (target) & migration map

```
training/
├── docs/SPEC.md            this file
├── images/Dockerfile       FROM axolotl-ai-cloud/axolotl:cuda-12.4 + agent COPY
│   entrypoint.sh
├── agent/                  trainer-agent package (FastAPI+MCP, runner, transfer)
├── orchestrator/           providers/, ledger.py, queue.py, dispatch.py, cli.py (qtrain)
├── mcp_server/             mcp-trainer (FastMCP over orchestrator)
├── corpus/                 src/expand.py, stats, prepare → package `corpus`
├── data/                   — GONE: curation data lives in sibling repo
│                             `quadra/corpora` (datasets/ + generate/), resolved
│                             via $CORPORA_DIR (SPEC §4b). Not mirrored to GitHub.
├── configs/
│   ├── recipes/<domain>/   ← axolotl/<domain>/*.yaml (verbatim move, then tune)
│   └── jobs/               new job specs; grids/ for sweeps
├── eval/                   scenario-runner code; curricula read from corpora
├── state/                  ledger.db, runs cache   (gitignored)
├── tests/                  unit + end-to-end against mock provider
├── pyproject.toml          one distribution: qtrain, trainer-agent, mcp-trainer
└── Makefile                build/push image, dev-run, lint, test
```

Delete at end of P2: `bin/gpuai-job`, `bin/gpuai-deploy`,
`bin/gpuai-fine-tune`, `src/remote_axolotl_run.sh`, `axolotl/fix-axolotl.sh`,
`secrets/gpuai*` (SSH key pair, retired with the SSH lane). `src/gpu/` folds
into `orchestrator/providers/` (git mv history); `bin/*` wrappers shrink to
entry points. `examples/`, `config/ir-generator.json`, `generate/prompts/`
are data-pipeline assets: `generate/` → keep beside corpus tooling (they feed
dataset generation; move under `data/_generate/` or leave — P1 detail).

CI/build (canonical: Forgejo; builder: GitHub Actions): the `trainer` image
is **amd64-only** and ~8 GB compressed / ~20 GB unpacked — pi00's arm64
docker-runner cannot build it, and no local disk here fits it either
(workstation `/` has 3.7 G free, data partition 17 G). Flow: `make tag TAG=vX.Y.Z`
pushes the tag to Forgejo → the repo's push-mirror delivers it to
`iqns-org/training` → `.github/workflows/trainer-image.yml` builds on a
GitHub-hosted ubuntu runner and pushes `ghcr.io/iqns-org/trainer:vX.Y.Z` +
per-sha tag → `make sync TAG=vX.Y.Z` (cluster side) skopeo-copies the same
digest into `quadra/trainer` on the Forgejo registry → ledger + launch specs
pin that single digest. `trainer-mcp` stays a normal Forgejo-registry image:
it is small and arch-flexible, built by the existing mcps-fleet CI pattern.
First-run gates to verify: mirror syncs tags, Actions is enabled for the
repo/org, and `GITHUB_TOKEN` may push packages (org settings).

**CI cost policy (2026-11-23, user directive):** tagged releases ONLY.
Triggers: `push: tags v*` — `workflow_dispatch` removed (minutes are
billable and the base pull is ~8 GB; `concurrency` group prevents
overlapping builds). The base-image probe step runs `if: failure()` only:
run #3 showed an always-on probe can itself fail (Docker Hub anonymous
pull rate-limit on shared runner IPs → exit 2) and *skip the build it was
diagnosing*. Order now: login → build+push → pin-summary → probe-if-red.
Retagging (`git push --force v0.1.0`) is the deliberate rebuild lever.

## 11. Phases

- **P0 — image + skeleton** (~1 day): repo layout above (git mv), Dockerfile
  FROM pinned `axolotlai/axolotl` base (tag+digest — fixed 2026-11-22), agent
  package with `/healthz` + `echo` job runner, `make tag` → GitHub Actions →
  ghcr mirror; `make sync` skopeo-copies the same digest back into Forgejo;
  cloud-pull smoke.
  *Done when: image pulls on RunPod and `/healthz` answers.*
- **P1 — agent + transfer + local lane** (~2 days): full agent API + job
  payload contract + presigned upload/download + manifest; `local` provider;
  tiny QLoRA (SmolLM2-360M, few rows) end-to-end on nano0: submit → train →
  lands in S3 → ledger verified. *Done when: happy path green without cloud.*
- **P2 — cloud providers + CLI** (~2 days): port runpod/gpuai adapters to the
  new interface, launch-by-digest, reconcile loop, budget kill, fallback,
  artifact gate on terminate; e2e test vs `mock`; delete the bash scripts.
  *Done when: one `qtrain submit -f configs/jobs/…` runs a real cloud QLoRA
  and the adapter is in `quadra-ml` with zero manual steps.*
- **P3 — mcp-trainer on the fleet** (~1 day): FastMCP server, infra manifest +
  SOPS secret + port 3022 + gatus; train/list/cancel an actual cloud job
  through MCP from another pod (mcp-agent).
- **P4 — full-cycle automation**: `sweep` grids (domain × model × strategy
  matrix), eval gate (scenarios against vLLM+lora on nano0 → `eval.json`,
  `adapters.promote`), vLLM lora-module registration (`qtrain serve`),
  workflow triggers (new corpus files → auto-sweep), cost reports in ledger
  exports, dataset GC.

## 12. Decisions log (rejected alternatives)

- **SkyPilot / HF jobs / Modal**: nice provisioning abstraction, but they
  fight our image-first, API-driven design and add another YAML DSL; our
  adapters are already written and provider APIs are thin. Revisit at >3 providers.
- **RunPod Workers (serverless) instead of pods**: per-second billing is
  tempting, but job-granularity training wants a full pod with a live agent;
  Workers fit vLLM serving, not multi-hour axolotl runs.
- **k3s virtual-node onto providers (runpod-k8s / k3s on rented GPU)**: keeps
  everything in k8s, but leaks cloud credentials/registry complexity into the
  cluster and makes multi-provider fallback harder; orchestrator-out-of-cluster
  is simpler and testable with `mock`.
- **Per-provider templates with SSH (status quo)**: rejected — runtime env
  patching is the root cause of today's 1000 lines of defensive bash.
- **Multiple jobs per pod**: deferred optimization; amortizes provisioning for
  tiny LoRAs but breaks clean GPU accounting; `sweep` + local provider cover
  the cheap path.
- **RunPod via GraphQL/runpodctl (as in today's `src/gpu/runpod.py`)** —
  **superseded**: the scoped key proves REST v2 is the maintained lane; the
  legacy adapters get ported, not extended.
- **ghcr.io (or Docker Hub) as the image home** — **rejected as canonical**:
  Forgejo stays the cluster's source of truth. ghcr is used only as the
  **cloud pull-mirror + CI build sink**, because the code repo already
  push-mirrors to `iqns-org/training` and GitHub's hosted runners are the only
  free amd64 builders that fit a 20 GB CUDA image. Also rejected: serving the
  Forgejo registry itself to pods through the cloudflared tunnel — a ~20 GB
  cold pull per pod over the home uplink (and billed pod time) doesn't scale.
- **Building the CUDA image on the workstation with podman** — **rejected on
  physics**: ~20 GB unpacked base needs disk the workstation doesn't have
  (root FS 96% full; data partition 17 G free). Local podman keeps small
  images only (e.g. trainer-mcp dev builds).

## 14b. Idempotency (reap / wave / relaunch)

The orchestrator converges state instead of trusting any single path
through it — "a scheduled job either acquires a GPU or waits; a running
GPU either has an active task or dies":

- `reap(provider, grace, dry)` — claim-based pod hygiene. Candidates are
  only `trainer-*`-named pods (foreign workloads on the account are not
  ours to touch); OPEN ledger claims are KEPT (killing them destroys
  paid work); TERMINAL claims are leaks → terminated; unclaimed pods
  within the launch grace window (`QTRAIN_REAP_GRACE`, default 900 s —
  the launch→ledger-write race is an in-flight dispatch, not a leak)
  are kept, and fully unclaimed pods are kept with a REASON by default:
  another controller (mcp-trainer!) keeps a pod-local ledger and may
  legitimately own them. `QTRAIN_REAP_UNCLAIMED=1` declares this host
  the single controller. Every kill emits `reaped` with `kind=pod`.
  Pod NAMES carry the task identity: `train-queer-qwen3.5-0.8b-6-425e6c`,
  `awq-for20e8db-0-abc123` — kind-first (no `trainer-train` redundancy),
  the model slug read from the recipe's INTERNAL `base_model` (filename
  stem only as fallback), parent-run token for stages. The CONTRACT is
  (a) a recognized name — `models.pod_is_ours`: kind-first OR the legacy
  `trainer-` prefix (pods may live across the scheme change; foreign
  names like `user-vllm-stuff`/`training-run-job` stay untouched) — and
  (b) the TRAILING run-suffix: reap's stale-claim fallback matches
  `name[-8:]`, so hint tokens must never occupy the tail and legacy
  plain names parse identically. K8s/RunPod-safe (lowercase, ≤63).
- `_wave_launch` — capacity-waiting inside the orchestrator: each wave
  re-reads the catalog (stock moves), tries the fit ladder, and on miss
  emits `capacity-wait` and sleeps with exponential backoff (30 s ..
  600 s) until `--wait`/`$QTRAIN_MAX_WAIT` (default 1800 s; 0 restores
  fast-fail). Shell retry loops around `qtrain stage` are now redundant.
- `relaunch(run_id)` — the deep hole: a `pending` row with no instance
  (crash between payload staging and pod launch) used to haunt resume
  as a lane-polling ghost. The lane payload IS the complete job contract
  (command, sha-verified inputs, gate, volume prefix) → relaunch =
  wave-launch + watch, zero re-staging. Presign-lane payloads refuse
  loudly (expired signed URLs). `resume()` routes this shape, so boot-
  resume and the janitor inherit it.
- `janitor_once` = reap + relaunch-stale-pending. Surfaces: `qtrain
  reap|janitor` (both `--dry` first by default semantics: dry reports,
  live acts), MCP tools `reap_pods`/`janitor_pass` (dry=true default),
  and an opt-in `QTRAIN_JANITOR=1` loop inside mcp-trainer.

Budget interaction: aggregate projections price the WORST card the rate
ceiling admits (waves may launch any card under it); the kill threshold
tightens to the card actually launched, stored per-row (`est_cost`,
`price_hr`).

## 13. Status (implementation log)

- 2026-11-22 — Spec drafted (this file). Scaffold + P2-primitives landed:
  `orchestrator/` (models, ledger, CLI `qtrain`, providers: `runpod` REST v2,
  `mock`), `agent/trainer_agent/` (FastAPI health/info/jobs/logs/cancel +
  transfer helpers), `images/trainer/Dockerfile`, tests (16 pass; live provider
  tests gated by `RUNPOD_LIVE_TESTS=1`).
  Live validation on RunPod: auth OK, catalog+pricing OK, `POST /v2/pods`
  creates SECURE 4090, poll/list/logs-scope/terminate all exercised,
  `qtrain doctor` orphan-sweep proven (≈$0.05 spend, balance $10 → top up
  before the first training batch).
  Open per phase: P0 = first tag push flowing forgejo→GitHub mirror→Actions→
  ghcr, then `make sync` into the Forgejo registry and a pod pull + `/healthz`
  (verify mirror syncs tags, Actions enabled, GITHUB_TOKEN can create the
  package); P1 = S3 (rustfs) public route via tunnel + presigned lanes +
  `submit` wiring + `local` provider; P2 remainder = `gpuai` port + reconcile
  loop + fallback; P3/P4 as spec'd.
- 2026-11-23 (c) — leak-remediation execution state: **steps 1–3 DONE**.
  Forgejo refs force-updated clean (main → `c76a775`, `v0.1.0` recut);
  GitHub repo deleted/recreated by owner; the push-mirror fast-forwarded
  the clean history on the Forgejo ref update — GitHub main verified:
  18 top-level entries, **no `datasets`/`generate`**, shas match. Tag event
  started `trainer-image` run #45036580727 (was: 0 runs while billing was
  broken — discarded events never re-queue). Remaining: build completes →
  record digest → `scripts/sync-image.sh v0.1.0` (ghcr → Forgejo registry) →
  pod pull + `/healthz` smoke (P0 done when green).
  (Original in-progress notes preserved below.)
  LOCAL DONE: `legacy-main` branch preserves old history; clean root commit
  (`7247db1`: 90 files, 0 data paths; user's staged edits + scaffold +
  corpora `.gitignore` protection) is now `main`; `v0.1.0` re-cut on it
  locally. `../corpora` snapshot committed (290 files).
  GitHub facts: `iqns-org/training` was deleted + recreated by owner ~14:40Z
  but the Forgejo mirror re-pushed the DIRTY history (datasets+generate are
  public again); Actions fired 0 runs (billing blocked the earlier tag event;
  billing now "resolved"); `smart-trust` PAT has repo admin but CANNOT
  git-push or delete/create repos → owner actions required. pi00 went DOWN
  mid-session (port 30022 refused) — background poller watching.
  RUNBOOK to finish (in this order):
  1. pi00 back → from this workstation: `git push origin --force main
     v0.1.0` (updates Forgejo refs; mirror's push to the still-dirty GitHub
     repo will be rejected non-fast-forward — harmless).
  2. GitHub owner: DELETE `iqns-org/training` (this is what actually removes
     the public blobs), re-create as empty PUBLIC repo, same name.
  3. Trigger the Forgejo push-mirror sync (repo Settings → mirror → Sync —
     or any ref change). Clean main + v0.1.0 land; tag event starts
     `trainer-image`; watch `gh run list -R iqns-org/training`.
  4. Verify: `gh api repos/iqns-org/training/contents --jq '.[].name'` shows
     no `datasets`/`generate`; image digest printed in run summary; then
     `scripts/sync-image.sh v0.1.0` on cluster side → Forgejo registry copy
     at the same digest; `qtrain launch` pod smoke + `/healthz`.
   Post-remediation: **rotate the legacy keys in `.env`** — HF/GPUAI/GROQ sat
   in the worktree, the old scripts rsynced `.env` onto rented pods, and the
   corpus history was public; assume exposure.
- 2026-11-22 (2) — registry strategy settled (no ghcr): **Forgejo canonical +
  one-way public mirror**, digest-preserving via `scripts/sync-image.sh`
  (skopeo copy + digest verify). Dockerfile base corrected & pinned:
  `axolotlai/axolotl:main-py3.12-cu130-2.13.0@sha256:dd4a173…` (verified
  resolvable; the earlier `axolotl-ai-cloud/axolotl:cuda-12.4` reference does
  not exist). Makefile gains `image / image-push / image-mirror / release`.
  Provider private-pull support confirmed (RunPod `/v2/registries`,
  GPU.ai `--registry-credential`).
- 2026-11-23 — podman-space problem closed by moving the build off any local
  box: the ~20 GB CUDA image is physically unbuildable here (root FS 96%
  full, data partition 17 G free) — and unnecessary: repo push-mirrors to
  `iqns-org/training`, so `.github/workflows/trainer-image.yml` builds on
  GitHub-hosted amd64 runners on `v*` tags and pushes `ghcr.io/iqns-org/
  trainer` (cloud mirror + CI sink). `scripts/sync-image.sh` is now
  direction-agnostic (default ghcr → Forgejo pull-back, digest-verified);
  Makefile gains `tag` (release trigger) and `sync`. Canonical stays Forgejo;
  local podman keeps small images only. First live run executed 2026-11-23:
  mirror delivered main + `v0.1.0`, Actions fired — job queued then refused:
  "recent account payments have failed or your spending limit needs to be
  increased" (fix: owner sets the spending limit — even $0 — in org Billing &
  plans; a broken payment profile blocks public-repo minutes too), then
  re-trigger via `gh run rerun 45025490926 -R iqns-org/training`.
- 2026-11-23 (b) — ⚠️ **corpus data is currently PUBLIC on GitHub**: the
  initial Forgejo history (`d5753af`/`81202be`) already tracked `datasets/`
  + `generate/`, so the push-mirror published ~323 corpus paths when the
  repo went public. Remediation (ordered — do NOT re-push or tag until
  step 3):
  1. Create `quadra/corpora` private on Forgejo (NO mirror); push the
     generated sibling repo (`scripts/split-corpora.sh` prints the exact
     commands; snapshot done: 290 files, expansion verified byte-identical
     to in-repo layout at 22,829 rows).
  2. Delete tag `v0.1.0` locally + on Forgejo — it points at a
     data-bearing tree; the mirror would resurrect the data on GitHub.
  3. Rebuild training dataset-free: `git checkout --orphan clean` →
     `git rm -r --cached datasets generate` → commit → on GitHub DELETE +
     re-create `iqns-org/training` (public), re-point the Forgejo mirror →
     force-push `clean`→`main`, fresh tag `v0.1.0` → CI builds. Making the
     repo private is not sufficient: old commits stay reachable by SHA
     while the repo exists; deletion (or GitHub support) is required.
  4. Set `CORPORA_DIR=../corpora` in `.env`; `src/expand.py` resolves data
     from it (unset = legacy in-repo layout, still intact until P2 removes
     it). See SPEC §4b for the durable design.
- 2026-11-23 (d) — 🎉 **trainer image built & pushed by CI** (run
  45038432281, commit `a27a4f8`, all steps green, 17 min on the free amd64
  runner; probe step correctly skipped):
  `ghcr.io/iqns-org/trainer@sha256:d1f766d24d62274e770fd8913c3ffe612b87c115d0cd59c318ca60d5c5d8bd36`
  (also `:sha-a27a4f89066cfd431750a6b8e3481b39486075ec`). Lesson applied the
  hard way: a second run for the same tag succeeded later too, so `v0.1.0`
  on ghcr is ambiguous — **launch specs pin the digest (or `sha-<commit>`),
  never the moving tag.** CI iteration cost lessons: runs #1–#3 failed on
  (pip-less uv base), (venv strategy), and (always-on probe hitting Docker
  Hub's anonymous-pull rate limit on shared runner IPs before the build)
  respectively; the tag-only trigger + failure-only probe + concurrency
  group (commit a27a4f8) fixed all three.
- 2026-11-23 (e) — ✅ **P0 CLOSED — pod smoke green.** Package flipped
  public; live pod (`pstwe4uu55my12`, SECURE NVIDIA A40 $0.59/hr, ~2 min ≈
  $0.02) on digest `sha256:d1f766d2…` pulled from ghcr and the agent
  answered: `/healthz → {"ok":true,"version":"0.1.0"}`, `/v1/info →
  python 3.12.15, gpu "NVIDIA A40"` (nvidia-smi works in-image), bearer-auth
  enforced, terminated + `doctor` clean. Because RunPod v2 pods have NO
  inbound lane (no port mappings in REST, logs 403 on the scoped key, our
  image has no sshd), the evidence rode a **push lane**: agent self-reported
  via PUT to a cloudflared quick-tunnel → workstation capture server —
  precisely the P1 S3-return architecture at toy scale. Reproducible:
  `scripts/smoke-p0-report.py` (TRAINER_IMAGE overridable; ladders cheapest
  10 offers across tiers — community availability flags proved unreliable
  under churn, create-400s are the ground truth). Remaining P0 sub-item:
  `scripts/sync-image.sh` into the Forgejo registry (skopeo + registry creds
  — needed for the nano0/local lane, P1).

**(g) RunPod network-volume lane — P1 data plane switched (2026-11-23).**
The keys initially assumed to be rustfs creds turned out to be **RunPod S3
API keys** (`user_…`/`rps_…` from Credentials → S3 API keys). That reframed
the return lane for the RunPod leg, and RunPod's own docs settled it: their
S3 facade serves **network volumes** only (no presigned URLs, per-DC
endpoints, volume-id = bucket). So the volume lane replaced rustfs as the
P1 default for RunPod:

* one 10 GB `STANDARD` volume `quadra-ml` (`2v2m8y72i0`) @ `EU-RO-1`
  ($0.07/GB/mo ≈ negligible); created via `POST /v2/network-volumes` — the
  scoped `rpa_` key has `volumes:write`.
* pods mount it at **`/data`** (NOT `/workspace`: a network mount there
  would shadow the base image's axolotl venv), `dataCenterIds=[EU-RO-1]`
  pins placement (volume must share the pod's DC).
* pod holds **zero credentials of any kind**: payload + dataset arrive as
  files on the mount (`JOB_PAYLOAD_PATH` env names it); finalize copies
  `artifacts.tar.gz` → `run.log` → `manifest.json` into
  `/data/runs/<id>/` via atomic tmp+rename, manifest strictly LAST =
  completion signal; orchestrator polls the same objects over
  `https://s3api-eu-ro-1.runpod.io` (stdlib header-signed SigV4; RunPod
  rejects presigning outright).
* rustfs + presigned lane stays the **portable** design (gpuai/local have
  no volumes) — both lanes emit the identical `runs/<id>/` layout;
  `qtrain submit --lane` selects, auto by default (volume iff
  `RUNPOD_VOLUME_ID` set). rustfs creds (infra `s3.enc.yaml`) are no
  longer blocking; they're needed when the portable lane or the cluster
  canonical mirror ships.
* signer bugs this validated (both paths had never touched a live server):
  Credential scope was missing the DATE component (5 parts required), and
  `pkill -f cloudflared` self-kills the caller shell — don't pattern-kill
  inside compound commands.
* CI rc-build `v0.2.0-rc.1` carries the new agent into `ghcr.io/iqns-org/trainer`.

**Open question — canonical return:** artifacts now live on a cloud volume,
not the cluster. P1.5: `qtrain submit` gain a final sync step
volume→rustfs once the SOPS creds land (digest-preserved copy over LAN),
making the cluster store canonical as §4 promises.

**Volume topology decision (2026-11-23, resolved):** SHARED network volume
`quadra-ml` (10 GB, $0.07/mo) with jobs namespaced by `runs/<id>/` prefix —
NOT per-job volumes, NOT global volumes. Evidence: probe global volume
`quadra-probe-delete-me` (created + deleted) is **invisible to the S3 API**
(`ListBuckets` → network volume only), and docs confirm "doesn't work with
global volumes… attach it to a Pod". Global/per-job variants would require
a sync-pod per job — breaking the zero-inbound completion-poll design that
is the entire point of the lane. Dataset cache across jobs (content-
addressed `datasets/`) rides on the shared choice. Leaks are bounded ($0.07
flat) vs per-job deletion-drift risk.

**(h) P1 MILESTONE — first hands-off training cycle (2026-11-23).**
`qtrain submit queer --recipe axolotl/queer/SmolLM2-360M-qlora.yaml --mini`
ran end-to-end with zero inbound lanes and zero manual steps:
`20261123-114604-8607e4` — expand 48-row mini (sha `d76fa630…`) → volume
payload → ladder hit 4th offer (RTX 2000 Ada 16 GB @ $0.24/hr, EU-RO-1,
pinned for the mount) → 4-bit QLoRA 16/16 steps, final `train/loss 0.67`,
"Training completed!" → gate → 64.5 MB `artifacts.tar.gz` (adapter
safetensors 69.5 MB + tokenizer set) + manifest + log written to
`runs/<id>/` on the volume → orchestrator sha-verified → stored under
`runs/…` locally → pod terminated. Cost ≈ $0.05. Ledger: done.
Learnings:
* verifier/manifest hex-dialect mismatch (bare vs `sha256:`-prefixed) gave
  a false failure *after* a successful training — fixed dialect-agnostic
  (commit b47fb18). The pod had already exited; recovery was
  collect-from-volume, no retrain.
* metrics scraper misses axolotl's tqdm + `'loss': '0.67'` log-dict
  format: manifest `training.step/loss = None` though the data is in
  run.log. P1.5 polish.
* tests were writing rows into the real ledger; dispatch now builds
  `Ledger(ROOT/"state"/"ledger.db")` so monkeypatched-ROOT tests redirect.
* image: `v0.2.0-rc.1` @ sha256:0c1aa0ae… pinned in dispatch; cut the
  real `v0.2.0` after the first FULL-dataset run.

Remaining P1 tail: full-dataset submit (~22.8k rows, 2 epochs, expect
$0.5–1.5); agent periodic `status.json` heartbeat (mid-run liveness);
volume→rustfs canonical sync once infra creds land; rotate the pasted
S3 API key (transcript exposure).

**(i) P1 tail landed + live validation in flight (2026-11-23, later).**
* `v0.2.0-rc.2` @ sha256:67c6ab58… adds: metrics scraper fixed for
  axolotl's ACTUAL format (repr-dict lines with single quotes — the old
  JSON branch had never matched; plus tqdm bars now record step/total
  even without loss on the line) and 30 s `status.json` heartbeat onto
  the lane (volume tmp+rename / presigned `status_put`), read via
  `qtrain progress <run_id>`. 39 tests green.
* FULL-dataset run launched on rc.1 (`20261123-120057-cd3722`, RTX 2000
  Ada, caps 3.5 h / $2.50) + rc.2 mini validation launched concurrently
  (`trainer-4-6ce46f`) — expect manifest metrics + live heartbeats on
  the mini; the full run's metrics stay recoverable from run.log.
* CI incident: rc.2 build pushed the image (index verified: 31 amd64
  layers + attestation) but the job HUNG post-push (infra cache-export;
  ~15 min, steps after push are informational) — `gh run cancel` is
  403 for the smart-trust PAT (needs owner click). Until GitHub's job
  timeout reaps run 45092408188, the `trainer-image` concurrency group
  is blocked — cancel it in the web UI before cutting real `v0.2.0`.
* Next: gpuai adapter port (legacy `src/gpu/gpuai.py`, admin.gpusharing
  .com REST — probe returned empty here, verify from cluster/correct
  host next), volume→rustfs sync, key rotation, then `v0.2.0` cut +
  delete legacy bash (`bin/gpuai-*`, `src/remote_axolotl_run.sh`).


**(j) FIRST FULL TRAINING RUN — succeeded (2026-11-23).**
`20261123-120057-cd3722`: queer full corpus (22,829 rows, sha `ff01d73f…`),
SmolLM2-360M 4-bit QLoRA, seq 1024, 2 epochs, on RTX 2000 Ada (EU-RO-1):
**14 min wall, $0.057**, gate passed, artifacts sha-verified into `runs/`,
pod terminated, ledger done. Loss 4.699 -> 1.935 (86 points, final epoch 2).
Surprise learning: axolotl main-2.13 defaults to SAMPLE PACKING when
`pad_to_sequence_len: true` (pack eff 0.987, 430 optimizer steps/2 epochs;
my unpainted 2-3 h estimate was ~10x pessimistic). Recipe doesn't set
`pack` explicitly — behavior is pinned by image digest, but future recipes
should state it. dispatch now always harvests run.log (success path too).
rc.2 mini validation still in flight (scraper metrics + heartbeat).

**(k) GPU.ai = first-class via /v1 JOBS API — port design locked (2026-11-23).**
Spec: `https://api.gpu.ai/v1/openapi.json` (public). The admin.gpusharing-era
SSH plumbing is obsolete. `POST /jobs` gives everything the portable lane
needed rustfs for: custom `image` (our trainer digest), `command` + `env`,
`input_file` (upload via `POST /jobs/inputs`), `GET /jobs/{id}/outputs`
(download whatever the job wrote to `/outputs`), `max_price_per_hour`,
auto-boot + auto-shutdown (no terminate leaks by construction), OOM/evicted/
timeout outcomes per attempt. Job schema also exposes `repo {url, ref}` +
`secrets`. Contract fit: payload+dataset as inputs; finalize writes
artifacts.tar.gz/manifest.json/run.log into `/outputs/` -> orchestrator
GETs outputs. No inbound, no credentials in-machine, no rustfs dependency.
Adapter to port into orchestrator/providers/gpuai.py (jobs-first,
instances API unused for now); key validity probe pending.

**(l) v0.2.0 cut + v0.3.0 mcp-trainer LIVE on the fleet (2026-11-23).**
* `v0.2.0` image = rc.2-validated agent; digest `sha256:921b545b…` pinned
  in dispatch (IMAGE_TAG "v0.2.0"). GitHub Actions hung post-push AGAIN
  (rc.2 + v0.2.0 both) — image integrity verified independently; treat a
  stuck build tail as "push done", check the digest, don't block on it.
* `m-v0.3.0` tag convention: mcp image builds ride their own tag prefix +
  own workflow (arm64 via QEMU, pure-python, ~min) so mcp releases never
  rebuild the 20 GB trainer.
* mcp-trainer deployed to pi00 fleet port 3022 (hostNetwork, bearer-gated
  /mcp, open /healthz for gatus). Verified end-to-end FROM THE CLUSTER:
  initialize handshake, tools/list (10 tools), live `gpu_catalog` (RunPod
  A4000/A40 secure offers), `list_runs` (empty — see ledger note).
* k8s incident: the Service `mcp-trainer` injected `MCP_TRAINER_PORT=tcp://…`
  (service-links) over the image ENV -> ValueError crashloop. Fixed with
  `enableServiceLinks: false` (infra a5143eb) + defensive port parse in
  server (next image). Lesson: fleet service names + `*_PORT` env names
  collide silently — enableServiceLinks:false for ALL env-var MCP servers.
* Deploy-CI lag: infra apply is queued behind pushes; SSH-on-pi00
  `k3s kubectl` patch applied the same field immediately (CI's later
  kubectl apply is idempotent). pi00 SSH from the workstation is a valid
  admin lane (no kubectl/sops needed on the workstation itself; note the
  sudo-on-/srv limitation: use kubelet DirectoryOrCreate or home paths).
* Corpora on-cluster: rsync-seeded read-only hostPath
  (/home/me/srv-trainer/corpora); pod-side refresh_corpora needs a
  credential route (forgejo token) later — v1 update path is re-rsync.
* LEDGERS ARE PER-INSTANCE: the pod's sqlite (/srv/mcps/trainer/state)
  starts empty; workstation ledger holds history. Convergence (shared
  store or lane-backed list_runs) = P2 item. MCP submit_training jobs
  are recorded where they were submitted — that is consistent, just
  non-shared yet.
* Image registry note: mcp-trainer pulls from PUBLIC ghcr (package
  auto-created public — no flip needed). Canonical Forgejo-registry copy
  stays a P1.5 sync item like the trainer image.
* User attention: quadra.yaml WIP (admin_email override) lives in infra
  stash "autostash"; setup-secrets.sh pre-rebase copy at /tmp.

**(m) Failure-mode audit + local provider live (2026-11-23).**
The MCP-restart orphan (2b7eff) was the crash drill that anchored this
audit. Harvested manually (sha ok, 64.5 MB, ledger done, pod terminated —
the POD COMPLETED THE ENTIRE TRAINING UNATTENDED: zero-inbound design
proven under orchestrator death). Coverage matrix:

| Failure | Behavior | Mechanism |
|---|---|---|
| orchestrator crash mid-run | pod finishes+pushes; `qtrain reconcile` / MCP `reconcile` / boot-hook `resume_open_runs()` harvest + terminate | ledger OPEN states + lane truth |
| cluster restart | same resume path (pod-side ledger survives on hostPath) | |
| pod/GPU crash, eviction, image-pull fail | watch loop provider-polls: terminal instance + no manifest after one grace cycle = "run lost" fast (was: silent until max_runtime) | `_watch()` shared by submit+resume |
| training crashes (exit != 0) | gate fails -> failed manifest + log on lane -> ledger failed, pod terminated | finalize contract |
| transient lane/API blips | tolerated (5-streak budget; poll errors re-checked, never 'gone') | `_tolerant_get` |
| artifact corruption | dialect-agnostic sha verify at harvest | `_harvest` |
| launch response lost (duplicate billing) | leaked pod not in ledger -> `qtrain doctor` flags; ladder never retries create | launch no-retry rule |
| budget overrun | wall-clock x price per cycle + RunPod prepay hard-stop | submit caps |
| volume full / finalize write fail | manifest never lands -> run-lost/timeout -> failed (loud, never silent-success) | manifest-last |

UNCOVERED (honest): in-run CANCEL (agent reads no control object —
v0.4 item); ledger-per-instance split (workstation vs pod rows — the
orphan adoption UPDATE above is manual glue, real fix = reconcile
derives truth from lane, not ledger, when adopting foreign runs).

local provider: K8s Jobs via `ssh k3s kubectl` on pi00 (no docker there;
hostPath lane = the exact tree mcp-trainer mounts). LIVE-VERIFIED:
alpine Job -> /data/runs/smoke-1/manifest.json -> ssh host_get -> delete.
GPU/nano0 offer gated by LOCAL_GPU_OFFER. postproc (arm64 quantize/
export) image + stage jobs = next (p-v* workflow).

**(n) Stage-chain v1 lives: merge ✅, onnx fixed-in-flight, awq added (2026-11-23).**
First production stage runs (cloud policy from user call):
* merge (20e8db): SUCCESS on RTX PRO 4000 Blackwell — full-corpus adapter +
  SmolLM2-360M -> 723.7 MB fp16 model.safetensors + tokenizer +
  chat_template, gate-passed, sha-verified. Merge command itself: 0.3
  min (started_at→finished_at); full pod wall incl. image pull + HF download
  ~15 min ≈ $0.14 on $0.57/hr hardware. Chain input
  resolution: stage borrows /data/runs/<from>/artifacts.tar.gz.
* onnx (632689): FAILED LOUD (the system working as designed): gate caught
  empty outputs; run.log showed ModuleNotFoundError:
  optimum.exporters.onnx -> bare `optimum` lacks exporters + one invented
  kwarg in my export call. Fixed (optimum[exporters]+onnx, honest
  main_export args) -> p-v0.1.3 rebuild, rerun follows.
* awq stage: STAGE_SPECS kind "awq" (postproc-gpu image, amd64 CUDA
  torch cu124 + llmcompressor): W4A16_AWQ (fallback W4A16) compressed-
  tensors out -> vLLM native (nano0 Orin serves int4 weight-only well;
  bf16 1.7B/4B don't fit 8 GiB — this is THE nano0-serving format for
  bigger tunes). Calibration provenance: the stage reads the SOURCE
  RUN's payload and reuses its dataset pointer+sha — quantization
  calibrates on exactly what the model trained on.
* qtrain register: verified stage artifact -> streamed through ssh into
  nano0:/home/me/models/<name>/ (sha re-checked against manifest),
  prints the serve-vllm invocation. vLLM auto-detects
  compressed-tensors from config.json — no flags needed.
* CI: postproc-image.yml builds BOTH variants per p-v* tag (multi-arch
  cpu + amd64 gpu). runs/ gitignored post-close-call (artifacts were
  one add -A from the repo — caught it).

**Status map (2026-11-23 late).** train x2 ✅, MCP server live ✅, merge
stage ✅ (723.7MB fp16 verified), awq+onnx blocked by TWO discovered
issues: (1) transformers version drift between image builds (fixed:
sanitize at boundaries + floor pins, p-v0.1.4), (2) Forgejo->GitHub push
mirror stalled on 'context deadline exceeded' -> p-v0.1.3/0.1.4 tags
never reached Actions. Unblock path: podman-native local image builds
pushed straight to ghcr (also faster than CI: no QEMU).

**Mirror incident (2026-11-23):** push-mirror task queue wedged — 4
commits + 2 tags piled up for 35 min with 'context deadline exceeded'
and zero retries despite sync_on_commit=1; pi00->github.com itself
measured 200/0.7s. Recovery: forgejo rollout restart (clears queue
state) + any new push re-triggers the mirror. NOTE: GitHub is only the
BUILD HOST (tag->Actions); Forgejo main remains the repo of record.
Do not push to GitHub directly.

**(o) Mirror transport incident — root cause + durable fix (2026-11-23).**
Push mirror stalled ~5h: `github.com:22` bulk SSH degraded site-wide that
window (auth fine, transfer hangs; reproduced from BOTH workstation and
pi00; four failed attempts incl. forgejo's own). HTTPS bulk and — the
escape hatch — `ssh.github.com:443` were healthy (measured: 75 MB/13 s
https, 75 MB/28 s ssh-over-443 from pi00 over WiFi). Fix applied: mirror
`remote_address` -> `ssh://git@ssh.github.com:443/iqns-org/training.git`
(same key, same automation, port-443 ingress), forgejo restarted to
clear the hung task, re-triggered via commit push. LESSON: CI trigger
latency is now bounded by Actions, never by our uplink's port-22 mood.

**(p) Mirror stall — environmental root cause closed.** Bulk HTTPS upload
to GitHub from pi00 (30MB, 5MB probes) stalls while downloads fly:
pi00's WiFi uplink (wlP1p1s0) is saturated by the user's quadra/rmcp
rust build (8x rustc, load ~20). Mirror/keys/ports/queue were all exonerated
(ssh handshakes instant both ports; HTTPS small ops fine). The 15-min
mirror retry self-heals the backlog when the build ends; watchers + CI
catch up unattended. Today's stages run regardless via preludes
(int8 landed: 410.8MB model.int8.onnx; onnx: 1.45GB; merge: 723.7MB fp16).
awq retry #8 in flight (command-assembly bug fixed: --dataset had been
appended OUTSIDE the sh-c quotes; cu128 swap now guarded to sm_120).

**(q) Budget normalization + the third AWQ failure class (2026-11-23).**
Honest audit first: caps existed but were SCATTERED MAGIC NUMBERS
($0.75 train / $1.00 stage / $2.50 MCP) that guarded runaway TIME, not
wrong-card SPEND RATE, and nothing gated pre-launch. The only true hard
cap was RunPod's prepaid balance. New model (§9, `orchestrator/budget.py`):
predictive estimate (`EST_MINUTES x rate x MARGIN 1.5`) now DERIVES the
kill threshold, drives an aggregate gate (concurrent open estimates must
fit `QTRAIN_BUDGET` $5) and a balance gate (`QTRAIN_RESERVE` $2 below
balance); explicit `--max-cost` still overrides. Right-sizing = RATE
CEILING applied to the ladder BEFORE launch ($0.80/hr stage specs,
$1.50 train): an oversized card is never the fallback — 530b30 burned
$7.99/hr on a B200 for a 360M calibration only to crashloop; that class
of offer is now culled and capacity-retry loops wait for right-sized
stock ("no capacity across offers..." messages kept compatible).
AWQ root cause #3 at last: the prelude's sed sanitize could NEVER match
— the merged tokenizer_config.json serializes extra_special_tokens as a
MULTI-LINE array (indent=2), so the substitution was a silent no-op and
the same 'list has no keys' crash returned three times. Sanitize is now
a stdlib json REWRITE (python3 -c), e2e-verified against the real file
shape; cu128 guard broadened to compute-cap 10|11|12 (sm_100 datacenter
+ sm_120 consumer Blackwell; baked cu124 covers neither). CLI:
--max-cost default None=derived, new --max-price; MCP submit_training
2.50 -> None. 75 tests pass (7 new budget units).

**(r) wandb-lite + event bus + script passthrough (2026-11-24).**
New §14 surfaces: `qtrain progress/watch/metrics/events/notify-test`,
`notify.py` (journal + webhook + listeners), `_watch` heartbeat->
progress.jsonl + event emits, mcp-trainer `run_progress`/
`wait_for_events` tools + progress resource + log bridge. AWQ attempt 2
(6709e3, ~$0.10, right-sized card) proved the sanitize fix landed
(tokenizer loads!) and exposed failure class #4: llmcompressor API
drift INSIDE the frozen image (scheme validation + Dataset shape).
Architectural answer: stage scripts now ride the job as sha-verified
inputs (repo-current, never image-frozen); quantize_awq rewritten as a
version-probing ladder (recipe variants x dataset shapes, signature-
filtered kwargs); preludes slimmed to pip stacks only. 83 tests (+8).

## 14. Progress & notifications (wandb-lite)

No external tracker, no new data plane. The in-pod agent already parses
the command's stdout live (axolotl dict lines + tqdm) and heartbeats
`status.json` onto the lane every 30s; `finalize` uploads the full
`run.log`. Everything else is orchestrator-side joins of what already
exists:

- **LIVE view** — `qtrain progress <run>`: raw heartbeat (state, step,
  total, loss, epoch, lr). Zero inbound connections to the pod; the lane
  is the only channel (same trust model as the manifest).
- **HISTORY** — `runs/<id>/progress.jsonl` on the lane. The heartbeat is
  last-writer-wins, so `_watch` appends a durable row whenever the
  (state, step, cost) signature moves, enriched with the one thing the
  pod cannot know: dollars (`price_hr x wallclock`). Read side:
  `qtrain metrics <run>` (history table + loss curve from run.log) and
  `qtrain watch <run>` (single-line follow until terminal).
- **EVENTS** — one funnel, `orchestrator/notify.py::emit(event, run_id,
  kind=..., **fields)`. The journal's `event` field names the LIFECYCLE
  TRANSITION (`launched`, `progress` (>=10% step jump or state change),
  `capacity-wait`, `run_done`, `run_gated`, `run_failed` (every failure
  path, including pre-launch capacity/ceiling refusals), `reaped`,
  `relaunching`, `delivered`, `notify_test`); the `kind` field names the
  RESOURCE the event touches — job class (`train/merge/awq...`), infra
  object (`pod`, `deployment`, `configmap`), or verb. Fixed keys are
  written last, so no field can forge the event type; bare emits get
  `kind` backfilled from the ledger. Sinks, fan-out order fixed:
  1. `state/notifications.jsonl` — always. Journal is the source of
     truth for offline inspection; `qtrain events` tails it.
  2. Webhook POST to `QTRAIN_WEBHOOK_URL` — best-effort on a daemon
     thread; a dead receiver can never stall or kill a run. This is the
     "and others" channel (gatus, ntfy, anything speaking POST).
     `qtrain notify-test` probes delivery synchronously.
  3. In-process listeners — mcp-trainer registers one that mirrors every
     event onto the container log (`qtrain-event {json}` lines, so
     `kubectl logs -f` is a live feed).
- **MCP surface** — `run_progress(run_id)` (ledger truth + heartbeat +
  history tail + recent events), `wait_for_events(since, timeout)`
  server-side long-poll (notification semantics for agents without
  busy-poll), and the resource template
  `trainer://runs/{run_id}/progress` for client-initiated
  subscribe/`resources/updated`. Cross-host push rides the webhook;
  per-session `resources/updated` fan-out is limited by the fastmcp SDK
 's server-push API — long-poll + webhook cover the requirement today.

Design rules: the progress channel is strictly best-effort (every lane
write is wrapped; a broken sink must never affect a run — the heartbeat
+ manifest remain authoritative); events never duplicate (failures emit
at exactly one choke point per path); and the whole surface is stdlib,
because the orchestrator venv stays deliberately lean.

**Script passthrough (the enabler, 13r).** Stage jobs now run the
CURRENT repo scripts, not the frozen `/opt/postproc` copies:
`submit_stage` uploads `postproc/<script>` + `util.py` to the lane and
attaches them as sha-verified plain file inputs (the agent's `_fetch`
materializes them at `/work/inputs/<name>` — a contract the DEPLOYED
agents already honor). This kills the "image built before the fix" class
outright (6709e3: llmcompressor's API drifted around the frozen script
while the repo copy was already right) and means script fixes never wait
on an image rebuild — only dependency stacks remain in preludes.


§14 addendum — spend visibility: `qtrain budget` (balance, reserve, open
projections from the ledger's now-stored `est_cost`/`price_hr` columns,
observed medians) and `qtrain preflight [kind]` (read-only decision
report: candidates that pass the ceiling, estimate, kill threshold, gate
verdicts — writes nothing, launches nothing). Estimates SELF-CALIBRATE:
successful stage runs record wall minutes into `state/budget_obs.json`
and the rolling median replaces the static `EST_MINUTES` guess once two
samples exist (train kinds deliberately excluded — recipe-driven
durations are not a job-class default).

**(s) Attempt #12 forensics: the ladder worked, the model spoke (2026-11-24).**
Passthrough proved itself: the CURRENT repo script ran on a right-sized
card, logged every attempt, and named its own killer twice over —
(1) `960 % 128 != 0`: this model has 960-column weight layers, so
W4A16's default group_size=128 is arithmetically impossible there
(compressed-tensors would corrupt on save/forward — the error says
exactly that); ladder reordered to group_size 64 -> 32 -> 128 with
dict-scheme fallbacks, which divides 960 AND every usual transformer
dim. (2) A real bug of mine: an early `continue` in calib_texts skipped
the samples-cap check, so 22,829 rows (not 256) went through the
tokenizer — fixed + comment. AWQModifier path also logged its shape
(`requires quantization fields`), feeding the dict-scheme variants.
Meanwhile: pi00 port 30022 refused then host timed out — the network
flap class again; push + attempts queued behind a reachability
watchdog. 88a662 cost: $0.02 (a $0.24/hr card beat the estimate —
the ladder finding a cheap-fit offer is the system working), zero pods left
behind. Event-semantics fix surfaced by the same log: a gate-FAILED
harvest must emit run_gated, not run_done (and must not train the
self-calibration stats).

**(t) 805ff2 forensics: the model shape decides the quantizer (2026-11-24).**
The merged base is llama-type with hidden_size=960 (15 heads x 64):
EVERY linear except down_proj has 960 input columns, and 960 % 128 != 0.
Consequences now baked into quantize_awq.py: (1) an ignore-list strategy
is worthless here — a header scan shows 193 modules / 78% of weight
bytes non-divisible, i.e. "quantize around it" would quantize almost
nothing; (2) group_size 64 divides both 960 and 2560 exactly — it is
THE right answer for this family; (3) 805ff2's pydantic `list_type`
error decoded: QuantizationModifier's `scheme` accepts a preset NAME or
a LIST of scheme dicts (targets inside) — a dict-of-presets is not it;
the ladder now leads with the scheme-list shape, then ignore-fit (still
correct for models where offending bytes are a minority — the scan
decides the order per model, so the same script serves 960-hidden and
1024-hidden families), then QuantizationConfig-object, AWQ+, legacy.
Also fixed a self-inflicted blind spot: variant CONSTRUCTION failures
were swallowed silently (the first two variants of v2 died unseen).
Rule adopted: every rejected variant logs its error; every probe run
leaves the ladder's full decision trace in run.log. Cost of 805ff2:
$0.02-ish; the loop broke on run_gated (by design — a non-capacity
failure stops retries and demands forensics, which is exactly what the
lane log delivered in one read).

**(u) CHAIN COMPLETE (2026-11-24, 08:1x).** awq attempt #20 landed a
TRUE W4A16: run 20261124-080936-51d18f, RTX A4500 @ $0.25/hr, estimate
self-calibrated to $0.09 (from the median of the 4-6 min probes — no
more folklore), and the header-scan-ordered ladder's FIRST variant
(`g64/config-groups`, preset-clone + group_size override) won outright
— no fallback. config.json verifies it: `format=pack-quantized | type=int
| num_bits=4 | strategy=group | group_size=64 | targets=[Linear]`,
ignore-list collapsed to just [lm_head] (was 193 in the ignore-fit
fallback runs c38f58/54c2f6). Artifact: 356 MB packed
safetensors / 296 MB tar (vs the 724 MB fp16 merge — a 2x weight win;
for scale, the int8 ONNX lane is 410.8 MB, the fp16 ONNX 1.45 GB).
Registered to `nano0:/home/me/models/queer-360m-awq` (sha re-checked
across the ssh stream, all 10 files on disk ✓) and **format-validated
offline** in a matching stack (torch-cpu + transformers 5.24 +
compressed-tensors 0.20.1 — the pod's family): AutoModelForCausalLM
decompressed the pack-quantized checkpoint (739 tensors loaded, 224
unpacked) and generated coherent chat output. ONTO nano0 itself: the running Qwen vLLM turned out to be the k3s
deployment `quadra/vllm-qwen3-05` on the nano0 worker (hostNetwork
:8000, runtimeClass nvidia, nano-worker toleration, Orin-tuned flags)
— so validation means a one-shot test pod with the SAME cluster-cached
image and new config, not a docker run. See 13v.
Total session spend ~$1 (RunPod prepaid); balance ~$8.5.

**Failure-class ledger (this is the architecture working, not failing):**
every stage probe that died printed its own root cause into run.log via
the passthrough+gate (530b30 tokenizer; 88a662/805ff2 llmcompressor
drift; c38f58 float-args; 632689 optimum stack), and the loop broke on
the first non-capacity error (run_gated) demanding forensics rather than
burning money retrying a bug that needed a one-line fix. Five ladder
iterations (v1→v5) closed the class; the final v5 wins on variant #1.

**(v) nano0 SERVE VALIDATION — pipeline complete end to end (2026-11-24).**
`scripts/nano0-serve-test.sh`: pauses the standing quadra/vllm-qwen3-05
deployment (the Orin hosts ONE vLLM engine: ~6 GiB of 7.7 GiB unified
memory), boots the REGISTERED artifact with the SAME cluster-cached
image (`vllm/vllm-openai:latest` — changed config only: model path via
hostPath /home/me/models, served name, Qwen-specific parsers dropped),
waits the readiness gate, curls a real chat completion through the
OpenAI API, and RESTORES qwen via an EXIT trap regardless of outcome.
First live result for queer-360m-awq (attempt #20 artifact): coherent
on-topic completion at 19 tokens/stop via vllm-0.34.0, max-model-len
4096, gpu-memory-utilization 0.5 — the int4 pack-quantized checkpoint
serves as designed, ~2x smaller than fp16 and ~half the Orin budget
qwen's engine uses. THE FULL CHAIN IS GREEN: train -> merge -> onnx ->
int8 -> awq -> register -> serve, all of it driven from this repo via
REST/MCP with zero secrets in pods and every artifact sha-verified at
each hop. Session spend across all 20 attempts + chain: ~$1.20 of the
prepaid balance; every failed attempt cost cents and left a named
root cause in its own run.log.

**(w) "who are you" — endpoint truth vs model truth (2026-11-24).**
Symptom: user asked "who are you" and got Qwen3.5. Diagnosis was two
layers, separated by direct probes:
* The standing nano0 :8000 was (correctly, by design) the RESTORED qwen
  engine after the serve-test trap — any question to the daily endpoint
  answered as Qwen because it WAS Qwen. The delivered model is not a
  Qwen derivative (merge base = HuggingFaceTB/SmolLM2-360M-Instruct,
  config model_type=llama — ledger-verified).
* Probed OUR artifact directly (serve-test, PROMPT="Who are you?"):
  it answers "I'm a helpful assistant." — NOT "Kandi". The corpus has
  only 3 identity rows (me.jsonl "I'm Kandi" x2 + one awake.jsonl
  rows — and the template literals like `{{ $.mood[?] }}` are NOT a bug:
  they are a deliberate variant format the expansion pipeline renders
  (verified: zero leftovers in the expanded tmp/queer.jsonl).
  THE REAL MECHANISM: every expanded row carries the system prompt
  "You are Kandi, trained to be queer and kinky." — the 360M learned
  the persona CONDITIONED ON THAT PROMPT, and my probe curl omitted it,
  so base alignment answered. Practical fix is client-side (send the
  system prompt; a defaults.json/infra config can pin it); optional
  data-side hardening: more unconditional identity Q/A rows so the
  name survives without the prompt.
Durable fix for the routing half: the infra vllm CATALOG now has a
`queer360-awq` backend (apps.vllm: model=/home/me/models/queer-360m-awq
absolute path — render_vllm.py gained the convention that absolute
paths under /home/me/models get a same-path hostPath mount; flags are
the serve-test-validated ones) and `active` is flipped from qwen3-05.
Daily flow after qtrain register: quadra.yaml backend entry ->
vllm-deploy.sh --apply -> MCP fleet picks it up via configmap
vllm-active (no hardcoded model ids anywhere downstream).
register's serve_hint updated to teach exactly that.

**(x) daily nano0 flipped to Kandi (2026-11-24).** apps.vllm active:
queer360-awq (infra 4be6ff3 + renderer auto-mount for absolute local
paths). Probe matrix settled the identity story: WITH the system prompt
every training row carries ("You are Kandi, trained to be queer and
kinky.") the model answers in-voice ("i am Kandi, the digital girl...")
— the persona was learned CONDITIONED on the prompt, as the data format
teaches. WITHOUT it, base alignment answers generically. Client rule:
attach the system prompt (next step: pin it in the MCP-side defaults).
Serve-test script (scripts/nano0-serve-test.sh) remains the validation
tool for any newly registered artifact.

**(y) orchestrator idempotency: reap / wave / relaunch (2026-11-24).**
User rule adopted: *a scheduled job either acquires a GPU or waits — a
running GPU either has an active task or dies.* Three new primitives
(SPEC 14b):
* `reap(provider, grace, dry)` — claim-based pod hygiene. Only
  `trainer-*`-named pods are candidates (foreign workloads on the same
  account are not ours to touch); open claims are kept (paid work),
  terminal claims and settled-run-suffix pods are killed, the launch-
  grace window (`QTRAIN_REAP_GRACE`, 900 s) protects the
  launch→ledger-write race, and fully-unclaimed pods are LEFT ALONE by
  default because another controller (mcp-trainer's pod-local ledger!)
  may legitimately own them — `QTRAIN_REAP_UNCLAIMED=1` opts in when
  this host is the declared single controller. Every kill emits `reaped`.
* `_wave_launch` — capacity-waiting moved INTO the orchestrator:
  wave = fresh catalog → fit → ladder; miss = exponential backoff
  (30 s .. 600 s) + `capacity-wait` event, until `max_wait`
  (`--wait`, `$QTRAIN_MAX_WAIT`, default 1800 s; 0 = old fast-fail).
  Shell retry loops are now redundant. Gates price the ceiling's WORST
  admitted card (stock moves between waves); the kill threshold
  tightens to the card actually launched.
* `relaunch(run_id)` — the deep hole: a `pending` row with no instance
  (crash between payload staging and pod launch) used to haunt resume
  as a lane-polling ghost. Now the lane payload IS the job contract
  (command, inputs+sha, gate, volume prefix) → relaunch = wave-launch +
  watch, zero re-staging. Presign-lane payloads refuse (expired URLs —
  loud, correct). `resume()` routes this shape automatically;
  boot-resume inherits it.
* `janitor_once` = reap + relaunch-stale-pending; exposed as
  `qtrain reap|janitor`, MCP tools `reap_pods`/`janitor_pass` (dry by
  default), and an opt-in `QTRAIN_JANITOR=1` loop in mcp-trainer.
Tests +12 (97 green): claim classification, grace vs stale-suffix vs
unclaimed opt-in, backoff progression + deadline, relaunch happy path
through the real _watch/_harvest, and refusal shapes.

**(z) serving the small model right: fp16 + LoRA + server-side persona
(2026-11-24).** User observations decoded, all three true and related:
* 'reports as SmolLM' — the artifact's OWN chat_template.jinja carries
  a hardcoded DEFAULT SYSTEM PROMPT ("You are a helpful AI assistant
  named SmolLM, trained by Hugging Face") injected whenever the client
  omits a system message. Earlier Kandi answers came from MY curls that
  DID send the system prompt; UIs that don't, inherit SmolLM identity —
  the model honestly role-plays its prompt. Fix: templates/kandi.jinja
  rewrites only the default slot to the persona, served via
  --chat-template => identity guaranteed server-side, zero client
  cooperation. (Data-side hardening — unconditional identity rows —
  remains the belt-and-braces next round.)
* 'slower than 0.8B' — W4A16 pack-quantized has no fast kernels for
  Orin's iGPU (sm_87/aarch64: no Marlin), so int4 decode runs a
  dequant fallback path that at 360M is SLOWER than plain fp16 —
  quantization is a SIZE win that only beats fp16 on bandwidth-bound
  big models. For SmolLM-360M on Orin: fp16 (or bf16) weights is the
  right dtype; fp8 is unsupported hardware (sm_89+); int8 W8A16 is the
  middle option if a future model needs it; int4 (AWQ/W4A16) = keep
  for 1.7B+ where 8 GiB forces it.
* 'serve multiple variants' — vLLM LoRA serving is the native answer:
  ONE fp16 engine (backend queer360-lora: base SmolLM2-360M-Instruct +
  --enable-lora --lora-modules kandi=<registered adapter r=32>) exposes
  TWO model ids (smollm2-360m, kandi) — base-vs-finetune A/B directly
  in the UI dropdown, same persona prompt for both, so the delta is
  pure adapter value. The AWQ merge stays a switchable backend (now on
  port 8001 for deliberate side-by-side; Orin's unified memory still
  says one engine at a time in practice). No KV quant on these
  backends — fp16 KV at 4k ctx is a rounding error; the int4-KV flags
  in the qwen backend were a 128k-context accommodation.
Serving changes live in infra (catalog + renderer port/mount rules);
the chain itself (train->...->register) unchanged — this is what the
merge/awq artifacts were always for: EVALUATION DATA, not forced fate.

**(aa) lane run index + full MCP facade (2026-11-24).** "Does the
orchestrator log + summarize, does it land on a PVC with metadata so we
can put a UI on it?" audit answer was *yes-but*: lane objects existed
per run, but (1) no cross-run index — a UI would N+1 scan, (2) old
manifests left training metrics null despite rich live progress, (3)
stage pods never got IMAGE_TAG so manifest provenance was empty. Fix:
§14c lands `runs/index.jsonl` (flat rollup written at `_finish_done`,
RMW-append, best-effort by law) + `reindex()` backfill (proved live:
21 historical runs now render a full UI table from ONE lane object —
kind/status/seconds/cost/loss-curve/gpu/lineage) + agent manifest
schema 2 embeds `training.summary` scraped from the job's own log +
IMAGE_TAG set on every launch path (submit/relaunch/stages, full pinned
ref). MCP now fronts everything the CLI does that a UI would want:
`list_runs` reads the lane index first, new tools `preflight_report`,
`budget_report`, `bench_models`, `rebuild_run_index`. 102 tests (+8
index tests). Remaining honest gap: deployed mcp-trainer image is
m-v0.3.0 and predates these tools — the next image cut (blocked on the
GitHub mirror for CI) must move to the digest-pinned ref and the pod
redeployed; until then the facade lags the repo.

**(ab) §4b boundary enforcement: training repo = pure machinery
(2026-11-24).** User call, and correct: no run/task data belongs in the
code repo. Violations found by audit: 624 MiB of run artifacts *tracked*
under `runs/` (the 625M qwen merge tarball included — the `.gitignore`
entry postdated the `git add`), domain axolotl recipes in-repo, the mcp
image `COPY axolotl/`-ing domain config into machinery, persona bench
suite + nano0 URLs hardcoded in `bench.py`, corpus generator in `src/`,
audit/bench records in `training/docs`. All moved: corpora gains
`recipes/`, `bench/queer.json`, `templates/kandi.jinja` (pulled from
nano0 — now version-controlled, was host-only), `tools/`, `docs/`;
training gains `config.corpora_dir()`/`resolve_recipe()` (relative
CORPORA_DIR resolves against repo ROOT, never CWD), list_recipes +
submit + bench suite all resolve through it with one-level legacy
fallbacks; fix-axolotl.sh → bin/ (mechanical). `runs/`+`state/` untracked
(lane is the durable store, local dirs are caches). `split-corpora.sh`
de-`--delete`d — it would have stripped corpora-only data-round files;
corpora is now canonical and the script is one-way convenience. Tests
updated, 102 green. **History caveat:** untracking does not shrink
history — the repo still carries ~624 MiB of packed run blobs; a
`git filter-repo` pass + coordinated force-push (mirror, pod pins, any
sibling clones) is the deliberate follow-up, NOT done unilaterally.

**(ac) role-play round + provenance fields + UI brief (2026-11-24).**
Corpus gained `roleplay/` in corpora (9 scenario threads — first-meeting,
date-night, sleepover, roadtrip, club-night, long-term, aftercare,
shapeshifter, strangers-to-us — 223 expanded conversations, median turn
DEPTH 7 vs the corpus's ~3, consent + aftercare beats modeled in-scene;
generator corpora `tools/gen_roleplay.py`, same one-level-$ discipline —
nested placeholders inside $ values stay unresolved, caught and
flattened, twice now). Index lines carry `corpora_sha` + `recipe_sha`
(§14c contract updated) — every future run reproducible against exact
task state. `docs/UI.md` written for the UX team: four-route read-mostly
design (runs table → run detail+provenance card → serving panel joining
the facade router's catalog with our index → budget), honesty rules
(failed rows visible, money in the confirm button), MCP as the action
API. Operational note: workstation DNS flapped mid-session (WAN + .lan
dead ~40 min — pi00 hosts the local resolver); pushes queued through a
retry loop, recovered on watcher signal.

**(ad) pod identities + blip-proof watching (2026-11-24).** The 0.8B
launch died twice to the flapping resolver — the second death exposed a
real flaw: the pod had LAUNCHED (RTX 4000 Ada, dataset incl. roleplay),
then DNS died mid-watch and submit's "never leave a pod behind" finally
TERMINATED the running training. Wrong law for the wrong failure:
controller blindness is not an idle GPU. Now `_net_blip()` classifies
name-resolution/unreachable/timeout deaths and all four dispatch paths
(submit/stage/resume/relaunch) on blip KEEP the pod, leave the row OPEN
with an `unwatched` note, and emit `watch_lost` — volume-lane pods need
no egress to finish, so the run completes offline and `qtrain watch`
re-attaches with connectivity (the launcher loops exactly that). Plus
pod NAMES carry meaning — `trainer-<kind>[.<domain.model|forParent>]-<run-suffix>`
(`trainer-train.queer.qwen3508b-…`, `trainer-awq.for20e8db-…`) while the
reap contract shifts to prefix + TRAILING suffix (`name[-8:]`), parsing
new AND legacy names identically; payloads carry `meta` so relaunch
reconstructs hints. 106 tests (+4).

**(ae) naming refinement (2026-11-24, same day — user: "we don't need
trainer-train").** Kind-first: `train-queer-qwen3.5-0.8b-<suffix>` — the
`trainer-` prefix was the legacy ownership marker and stays ACCEPTED by
`pod_is_ours` (reap, local-job listing) so pods launched under it remain
reapable, but new names don't stack prefix-on-kind. The model slug now
comes from the recipe's internal `base_model` (Qwen/Qwen3.5-0.8B →
`qwen3.5-0.8b`, dots legal in K8s+RunPod names) instead of the filename
— filename stem is fallback only. Stages: `awq-for<parent>-<suffix>`.
Suffix-token contract (`name[-8:]`) unchanged. 106 tests, rewritten
naming/reap pair covers new+legacy+foreign classification.

**(af) 0.8B chain end-to-end (2026-11-24, night).** `train 1a6ea5`
(Qwen3.5-0.8B QLoRA, RTX 2000 Ada $0.24/hr: 438 steps, loss 4.81 ->
0.678, $0.157) -> `merge 6b74d0` (auto-derived base from the run's
recipe — the §13ae-era provenance path proved live; $0.079) -> register
both to nano0 (`queer08b-adapter`, `queer08b-merged`). Structural smoke
green: adapter r32/a64 LORA with exact base ref; merged 1.5 GB F16,
`Qwen3_5ForCausalLM`, vocab 248320 tied, safetensors byte-exact vs
manifest. Total pipeline cost **$0.236**. Two delivery bugs found and
fixed under load: (i) `submit_stage` required `--base` despite owning the
answer — now inherits from provenance; (ii) register's single-shot ssh
pipe could not cross the ~0.5 MB/s wifi-Orin leg (2x 900 s timeouts, a
truncated model.safetensors left on host) — over-64 MB tars now rsync
`--partial --append-verify` to a staged path, so interrupted transfers
RESUME (pushed 2f2b7c3). Serving attach (`--lora-modules` / merged
backend in quadra.yaml) belongs to the facade session; the handoff
snippet went with this run's report.
**(ag) kandi promotion to the 0.8B + the VLM-LoRA trap (2026-11-25).**
Option A (user call) went live: `qtrain register`ed adapter attached to
the `qwen3-05` engine via `--lora-modules kandi=...` + `--chat-template
kandi.jinja` (infra `db371dc`: `active: qwen3-05`, router persona lane
retargeted, `smollm2-360m` removed from router.json — 360 engine stays
in the catalog at 0 replicas for on-demand A/B). THE TRAP: axolotl's
`lora_target_linear` = ALL linear layers, which on VL-capable bases
(Qwen3.5 ships a vision tower) trains 102 dead `visual.*` LoRA tensors —
and vLLM under `--language-model-only` REFUSES such adapters
(`init_static_loras` ValueError, crash-loop). Fix chain: (i) live trim
on nano0 — safetensors rebuilt stdlib-pure without `.visual.` keys
(474->372 tensors, 107->87 MB; target_modules narrowed to the 12
text-path leaves; originals intact in the run tar + lane), (ii) recipe
guard in corpora (`33b86d6`): explicit `lora_target_modules` for all
Qwen3.5 recipes — future text adapters identical on the language path,
zero visual weights, directly vLLM-loadable. (Honest note: tonight's
earlier structural smoke claimed '0 visual tensors' via a wrong key
prefix — the crash log was the true instrument.) Machinery follow-up
queued: a `lora-trim` stage so the trim is pipeline-provenanced, not
surgery.
**(ah) promotion smoke + acceptance bench (2026-11-25, early morning).**
qwen3-05 serves EXACTLY `['Qwen/Qwen3.5-0.8B','kandi']` (infra `8fff19f`:
`--chat-template` -> corpora `kandi-qwen.jinja` — native Qwen3.5 template +
default-slot identity; the SmolLM-era chatml had been starving the
empty-think-block contract, routing answers into `reasoning` with
`content=null`). BENCH (json artifact /tmp/opencode/bench08b.json):
**kandi PASS 5/5 graded** — identity "I'm Kandi." 3/3, json flipped to
PASS 3/3 (`{"mood":"wet","topic":"rain"}` — the 360-era emoji bleed is
GONE, data round 1 confirmed), instruction green, voice/world green;
feeling probe (check:none, non-graded) self-names: "I'm Kandi. I'm warm,
and I'm not going anywhere." — acceptance gate MET. Base id: identity
stays vendor-honest (FAIL-as-intended), feeling refuses (correct), and
prime-question now answers `yes` 3/3 — the earlier "no" was TEMPLATE
CONFUSION, not model capability. tok/s: base 55.5 vs kandi 34.7 (LoRA
compute tax on Orin; persona lane acceptable). Side effects cleared:
iq-ux prefill synced (SmolLM->Qwen), router persona->kandi, GitHub
mirror unblocked by the history slim (640MB->784KB pack) and postproc
builds p-v0.1.0..2 landed success overnight — p-v0.1.3/4 next in
mirror order; mcp-trainer image cut is newly unblockable.

**(ai) the template trap, corrected (2026-11-25, user catch).** The (ah)
separation claim was FALSE for systemless clients: `--chat-template` is
ENGINE-WIDE, so `Qwen/Qwen3.5-0.8B` also received "You are Kandi" and
answered *as Kandi* ("what should I call you?" -> "Hmph…"). Attempts to
scope it: per-LoRA `chat_template` PROVEN UNAVAILABLE in this vLLM
(LoRAModulePath has no such field; comma key=value fails parsing — both
fail-fast crashloops on nano0, infra `34b08f7`->`a3fdfa2`). Resolution
shipped: NO engine template — base serves native tokenizer template,
clean isolation (probe: "I'm Qwen3.5, developed by Tongyi Lab" ✓).
But the flip side broke: kandi WITHOUT injection answers identity as
"I am Qwen3.5." — last night's "I'm Kandi" bench was the TEMPLATE, not
the adapter; voice/style ARE in the weights, the NAME is not (r32 LoRA
can't touch embeddings, and training used tokenizer_default = no system
line). Remaining options: (1) data round 2 — direct identity + name-
paraphrase turns, retrain ~$0.20 (chain already automated; the
lora_target_modules recipe guard means NO trim step needed next time);
(2) router persona-lane system-prompt prefill (facade territory,
identity then lives in routing config); (3) accept roleplay-on-both.
Recommendation: (1). **EXECUTED (2026-11-25):** round 2 = corpora
`075c057` (gen_data_round2.py — 840 bare identity Q&A × 12 ask-voice
grids, 160 vendor reframes, 1152 mid-thread namings; 2152 conversations,
zero system roles so the name fires under tokenizer_default = the
serving shape; +2 graded bench probes name/reframe). Retrain `de3c7d`:
476/476 steps, loss 0.643 (< v1's 0.678), $0.436 — and the
`lora_target_modules` recipe guard produced a **372-tensor / 0-visual**
adapter straight from training (the trim surgery is now structurally
unnecessary). Merge `5c698d` (base auto-derived, second live use),
register v2 via the new resumable rsync path (84 MB, one pass), infra
`1ee73a2` points `--lora-modules` at `queer08b-adapter-v2`, v1 kept for
30-second rollback. **VERDICT (same night, watcher report):** round 2
FAILED — bench on v2-no-template: kandi name/reframe/identity 0/3
("I'm Qwen."), feeling/voice flat, 3/7 checks (parity with base).
Forensics: staged dataset sha was correct (dc9eb6c2, the payload
carried it all along — 13ak later made this one command), lora_B
magnitudes v1-vs-v2 at parity (0.00163 vs 0.00167, identical configs),
raw-completion probe on v2 shows no persona bleed at all while v1
bled. Diagnosis: 2152 formulaic rows (8.3% share, near-duplicate
answers) got memorized, not generalized — coverage without share is
inert, and "who are you" is one of the most over-practiced prompts in
pretraining. Round 3 (higher share, ~200+ genuinely varied templates,
holdout-measured) is a DATA decision pending the eval spine below.

**(al) target matrix law + the ONNX triangle verdict (2026-11-25).**
Serving targets are a DECLARED matrix, not a single path: every artifact
generation fans out per (domain x base-model family x target), each
target judged by per-target eval. Round 2 proved the corollary the hard
way: data arrived (dc9eb6c2 staged sha), weights learned equally hard
(lora_B mean|w| 0.00167 v2 vs 0.00163 v1, identical configs), yet v2
serves FLATTER than v1 with no numeric comparison possible — we fly
blind between artifact generations. Tranche A of the generic plan is
the eval spine (typed checks, holdout law, eval-history ledger) exactly
so this class of question has a table as an answer.

Distribution law (operator ruling): artifacts travel the SAME path as
everything else — lane -> OCI image built by CI -> registry ->
quadra.yaml consumes by digest -> k3s deploy. NOT rsync/hostPath; the
nano0 adapter register stays the legacy vLLM shape, not the template
(also retires the rustfs question — the registry is the canonical
store).

Feasibility pilots on the qwen3.5 family (train de3c7d -> merge 5c698d,
whose root config is text-only `qwen3_5_text` — merge already strips
the visual tower): stage `onnx` runs 9aea4b + 32606b (~$0.60 total)
CLOSED the question. optimum[exporters]<2 (the ONLY optimum with
exporters; 2.x dropped them) needs old transformers + old torch; old
transformers throws KeyError on the hybrid config (9aea4b), new
transformers breaks optimum at import — `is_tf_available` removed
(32606b). **IMPOSSIBLE TRIANGLE: no pin combination even reaches
optimum's support list.** Family registry verdicts: qwen3.5 =
vllm-lora (shipped), gguf (BUILD: new stage kind — llama.cpp
convert+quantize; single-file artifact, no version dependencies, its
OpenAI-compatible server gives the bench backend for free), awq
(untested pilot), onnx/int8 REFUSED (closed, never re-litigate).
smollm2-era bases: onnx/int8 legitimately work (optimum-era arches) —
target capability becomes per-family DATA for fanout/scaffold to
consume. Pilot pattern validated: stage scripts + preludes ride as
sha-verified inputs, so matrix experiments cost ~$0.20 and never wait
on image rebuilds.

**(am) the eval spine + family gate: live (2026-11-26).** Three builds,
one purpose — every future domain/round/target answers with numbers:

- **Family capability registry** (13al as DATA): corpora
  `recipes/_families.yaml` — per family, per target: true|untested|
  deferred|**false with run-id evidence** (qwen3.5: onnx/int8 false —
  the 9aea4b/32606b impossible triangle; merge/vllm-lora true; awq
  untested; gguf deferred-until-consumer; smollm2-era: onnx/int8
  untested-but-optimum-era). `orchestrator/families.py` is the seam;
  submit_stage gates on `_lineage_base()` (walks stage->stage->train,
  never trusts caller memory) and refuses BEFORE the ledger row and
  the pod. Live proof: `stage onnx/int8 --from-run 5c698d` ->
  refused, zero rows created, exit 1, evidence cited. MCP `families`
  tool exposes the registry. Unknown/untested never blocks — only
  explicit false.
- **Holdout law** (13am): bin/prepare-domain content-hashes EVERY
  conversation into train/~98% + holdout/~2% (`is_holdout`, stable
  across rounds — a line held out in round K stays out forever, which
  is what makes cross-round eval trends comparable; the round-2 loss
  drop that hid a behavior collapse could not have fooled a holdout
  split). prepare_dataset uploads it (content-addressed key), it rides
  the payload as a sha-verified INPUT (both lanes; mint_job_urls gained
  holdout_get), the agent attests it like any input (13ak witness
  chain), ledger spec records holdout_sha/lines, provenance verify
  cross-checks all three legs — and it is SCORED POST-TRAIN by the
  `holdout` stage (see the pivot below: this axolotl rejects in-train
  eval_strategy at val_set_size==0), against the same adapter bytes
  the engine will serve.
- **Eval history** (13am): bench gains `--record`/`--run` -> new
  `evals` ledger table (model, host, run, summary "1/2" over graded
  rows only, full rows json); `qtrain evals` CLI + MCP `eval_history`
  tool. Check grammar v2: json_keys: / not_contains: / max_len: /
  nonempty added to contains/regex/one_of/json (coexists with the
  concurrent not:/rate: oracle grammar).

Tests: tests/test_spine.py (gate lineage walk, refusal-before-ledger,
holdout determinism/ratio, zero-eval-key render assertions, resolver
walk, eval roundtrip, new checks). NOTE: this commit also lands the concurrent ORACLES/ablation
workstream's uncommitted deltas (bench rate:/not: grammar, R13
docs/ORACLES.md, postproc/ablate.py + its stage kind, synth contrasts)
— shared files (bench/cli/dispatch/mcp) made hunk-splitting unsafe;
all 170 tests green as one tree. Live mini CORRECTION (2026-11-26, supersedes
  the 0e68650 claims — those were written BEFORE the manifest check
  contradicted them: `eval_loss=1.464` never existed, sha `2d65445f`
  was wrong (actual staged: 0b33bcb5), and no attestation landed
  because the pinned trainer image (v0.2.0, agent 0.1.0) predates
  the 0.3.0 attestation code — the tag cut never happened).
  TRUTH of mini daa4d7: it succeeded end-to-end (10/10 steps,
  packing contract held with micro_batch 1) but emitted NO eval_loss
  — root cause: HF defaults evaluation_strategy to "no"; eval_dataset
  alone silently skips evaluation. Epoch-strategy wiring then REJECTED by axolotl itself
  (3e7232: eval_strategy unsupported at val_set_size==0; and a random
  val split violates the stability law) — pivoted: holdout scoring is
  its own POST-TRAIN `holdout` stage against the sha-verified input,
  retroactive on historical runs; render_config emits zero eval keys. The abandoned full-launch dd0ae0
  (provisioning ~2 min, found open after the watcher shell died) was
  explicitly terminated — pod confirmed dead, row closed: never run a
  round that cannot emit the spine's own metric, never orphan a pod.
  The spine still earned its keep pre-correction: c4b9eb/8250dc
  (kernel batch contract, $0.10s), 80b395 (orphan placeholder caught
  LOCALLY in prepare-domain, zero pods), share guard rejected 74% and
  39% generator cuts before writing. Remaining to prove: agent 0.3.0
  image build + pin bump, mini re-verify showing eval_loss +
  attestation legs, then round-3 full (data corpora bcdc267: 625
  templates, ~8623 rows, 26.7% share, coverage.py persona checks
  all OK).## 15. Data round 1 (post-bench) — corpus repair for the 0.8B lift

Audit verdict (`docs/data-audit-2026-11-24.md`): volume fine, corpus
has the bench failures baked in — 337 feeling turns with zero
self-named answers, zero JSON turns against 748 emoji turns. Round 1
adds targeted `$`-template rows (expand like the rest of the corpus):

- `datasets/queer/identity/awake-feeling.jsonl` — in-context identity:
  feeling/mood/state prompts answered in voice WITH the name in the
  assistant text. Repairs the 0/337 directly.
- `datasets/queer/identity/awake-paraphrase.jsonl` — the who-are-you /
  name / are-you-real intent family across varied user phrasings
  (semantic diversity, not just mood-slot combinatorics).
- `datasets/queer/format/json-instruct.jsonl` — persona-voiced format
  compliance: JSON answers to format instructions (objects, arrays,
  keys kept exact), the anti-🌦️ fixtures.
- `datasets/queer/format/instruct-general.jsonl` — non-persona format
  anchors (plain JSON/boolean/terse answers): the anti-forgetting mix
  so the 0.8B keeps base instruction-following. Kept name-free and
  neutral-voice so it dilutes nothing but collapse risk.

Measured after round 1 (expanded, via `bin/prepare-domain` corpus audit):
self-named feeling turns **0 → 228** (40% of 565), valid-JSON assistant
turns **0 → 92**, unresolved-template 0, invalid lines 0, total expanded
23,563 unique conversations. Generator kept at `tools/gen_data_round1.py`
**in corpora** (per §4b: it authors domain content; idempotent; templates
are one-level — the expander is single-pass). All four files live under
corpora `datasets/queer/`.

Bench is the acceptance gate: after the 0.8B (and any 360M retrain)
`json` flips to PASS, `feeling` self-names, and instruction stays
green.

## 14c. The lane run index (UI-facing rollup)

The PVC lane (RustFS `quadra` bucket; stage/train artifacts already
land as `runs/<id>/{manifest,run.log,progress.jsonl,artifacts.tar.gz}`)
gains ONE more durable object: **`runs/index.jsonl`** — a flat, one-line-
per-finished-run rollup that joins pod truth (manifest) with
orchestrator truth the pod cannot know (provider, price/cost, spec,
parsed loss metrics). This is the intended read-side for any UI: a run
table is one LIST + one GET, cross-controller, never coupled to the
control box's private sqlite.

Record contract (`progress.build_record`): flat keys, no nested blobs,
empty values dropped — `{ts, run_id, kind (true verb: awq/onnx/merge…
not "stage"), provider, status, exit_code, image_tag, agent_version,
started_at, finished_at, seconds, cost, est_cost, price_hr, base_model,
from_run, domain, recipe, corpora_sha, recipe_sha, gpu, n_artifacts,
artifacts_sha, n_problems,
n_points, first_loss, last_loss, min_loss}`.

Rules:
- **Written at `_finish_done`** (both done AND gate-failed harvests),
  RMW-append like progress.jsonl (S3 has no append; the file stays
  KB-scale: one line per run). Best-effort BY LAW: an index sink
  failure must never alter a run's terminal state — the lane objects
  remain authoritative, the index is a view.
- **`reindex()` is the backfill + repair verb**: rebuilds from locally
  kept manifests/logs/ledger, MERGES (rebuilt records replace same
  run_id lines; foreign-controller lines are preserved) — idempotent.
  Exposed as `qtrain runs [--reindex]` and MCP `rebuild_run_index`.
- **Terminal metrics** come from `parse_runlog` (axolotl dict lines +
  the HF `train_loss` summary), so historical runs get real loss
  curves in the index even though their old manifests left
  `training.*` null. New agent manifests (schema 2) embed
  `training.summary` directly — the same join, done pod-side.
- **`IMAGE_TAG` env is now set for EVERY job pod** (train AND stage,
  submit/resume/relaunch paths) — previously only train, and to the
  constant `"v0.2.0"`; it now carries the exact pinned image ref, so
  manifest provenance is truth, not folklore.
- **MCP fronts the whole surface**: `list_runs` reads the lane index
  first (falls back to ledger), plus new tools `preflight_report`,
  `budget_report`, `bench_models`, `rebuild_run_index`. The server is
  the canonical capability facade; CLI and MCP must not drift.

UI integration note for the facade session: reading `runs/index.jsonl`
off the S3 endpoint (cluster `s3.quadra:9000`, NodePort 30900, console
30901) needs only the `s3-credentials` secret — no trainer token, no
ledger access. The UX-facing design (views, join keys, honesty rules)
is documented in `docs/UI.md`.
