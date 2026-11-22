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
`configs/recipes/<domain>/<model>-<strategy>.yaml` — axolotl overlays
(absorbs today's `axolotl/<domain>/*.yaml`).

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

- Every run records: gpu type, provider, price/hr, wall time, est. spend —
  ledger + `manifest.json`. `max_total_cost` enforced by reconcile loop
  (hard terminate), not just hopeful timeouts. Provider account balance is a
  second, absolute ceiling (RunPod: prepaid top-ups are the cap — keep it small,
  `GET /v2/billing/balance` is exposed via the adapter).
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
