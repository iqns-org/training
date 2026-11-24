# UI design — training fleet + serving observability

Audience: UX team. Status: read-side is **already implementable today**
with zero new backend work; write-side actions are one MCP connection
away. This doc is the map, not a wish list.

## 1. What exists (the surfaces a UI stands on)

| plane | source | protocol / location | freshness |
|---|---|---|---|
| run history | `runs/index.jsonl` — one flat JSON line per finished run | S3 (RustFS) bucket `quadra`, cluster svc `s3.quadra:9000`, NodePort **30900**; console **30901** | written at run terminal (`_finish_done`), plus `qtrain runs --reindex` repair |
| live run | `runs/<id>/status.json` (30 s heartbeat: step/total/loss/epoch/cost) + `progress.jsonl` (durable history) | same bucket | 30 s |
| run provenance | `runs/<id>/manifest.json` (artifacts+sha256, hardware, image tag, problems) + `corpora_sha`/`recipe_sha` in the index line | same bucket | terminal |
| event feed | notification journal (launched/progress/run_done/run_failed/delivered) | MCP `wait_for_events` (long-poll + cursor) or webhook push | live |
| capabilities (writes) | mcp-trainer server: `submit_training`, `submit_stage`, `reconcile`, `reap_pods`, `janitor_pass`, `preflight_report`, `budget_report`, `bench_models`, `terminate_pod`, `doctor` | MCP streamable-http, bearer token (`MCP_TRAINER_TOKEN`) | on demand |
| model catalog (serving) | each engine's `/v1/models`; the router facade (in development by the infra session) aggregates engines | OpenAI-compatible HTTP on nano0 / cluster | live |

Key property: **the index line is flat by contract** (no nested blobs) —
a table view is `JSON.parse(line)` away, no schema archaeology. Run
`curl <s3>/quadra/runs/index.jsonl` (signed or anon-read) and you have
the whole fleet history in ~9 KB.

## 2. Proposed views (priority order)

### v0 — Runs dashboard (read-only, shippable immediately)
The lane index rendered as a table + tiny charts.

- columns: status chip (green succeeded / red failed·gated / gray
  verified-dry), kind chip (train/merge/onnx/int8/awq), run_id, age,
  duration, **$ cost** (actual or `~est`), loss micro-spark (first→min
  →last), GPU model, and a provenance tooltip (`corpora_sha`,
  `image_tag`, dataset hash from `recipe_sha`).
- sort by finished_at desc; filter kind/status/domain; search run_id.
- auto-refresh the single GET every 15–30 s (file is tiny; no websocket
  needed).
- the failed rows are the feature: 15 of 21 historical runs are
  failures, and the table + `problems` field is what makes the fleet's
  health legible. Do not hide them.

### v1 — Run detail page (still mostly static reads)
Deep-link `/runs/<id>`; three zones:

1. **live/terminal telemetry** — heartbeat + `progress.jsonl` curve
   (steps vs loss, cost ramp), progress % while state=running; ETA from
   step cadence. (This is the wandb-lite view `qtrain metrics` renders
   in text — same data, drawn.)
2. **provenance card** — base model, recipe (sha + link to corpora
   commit), dataset content-hash → expandable to the corpora git SHA
   (exact data state reproducible), image digest, agent version, GPU +
   price. "Why did this run turn out like this" should be answerable
   from this card alone.
3. **artifact + action rail** — artifact list with sizes/shas,
   presigned download, ledger states, and (v2) action buttons wired to
   MCP tools: *submit next stage* (the chain train→merge→onnx→int8→awq
   is linear — show it as a pipeline stepper with per-stage cost/time),
   *register to serving*, *reap*, *terminate*.

Also on this page: **lineage strip** — `from_run` chains give a
train→merge→stage thread; render as breadcrumbs. The UI's most
satisfying trick: from any merged/AWQ artifact, one click back to the
exact dataset SHA that produced it.

### v2 — Serving panel
Depends on the router facade (in development).

- model catalog from the router/engines (`/v1/models`): every served
  variant (base, LoRA aliases, AWQ, qwen refs) as cards — engine,
  dtype, port, replicas (from the infra catalog), memory budget.
- **persona guarantee badge**: the served chat-template is what
  carries identity (SPEC 13z) — show template source/version per model,
  not just weights. This was a real mystery ("why does it say SmolLM")
  the UI should make impossible.
- chat sandbox: pick any listed model, stream, see TTFT + tok/s
  measured client-side; A/B split view (kandi vs base) is the core
  evaluation UX and needs no new backend (both ids come off one engine).
- **bench scorecard**: `bench_models` returns `{table, rows}` — render
  the matrix (question × model, PASS/FAIL chips, tok/s, TTFT) and pin
  the latest per-run snapshot beside the run that produced the model.
  The acceptance gate for data rounds, visible: "json 0/2 → 2/2".

### v3 — Budget & capacity panel
- live balance + open projections vs cap (`budget_report`), per-kind
  observed medians (self-calibrating), last 30 days spend sparkline.
- pod strip: live RunPod instances with claim state (MCP
  `list_pods`/`doctor`); reap/janitor buttons with confirm modals.
- preflight widget: pick kind+recipe → `preflight_report` shows cards,
  estimate, gate verdicts — a "what would this cost" button that
  launches nothing.

## 3. Interaction & honesty rules

- **Confirm-modal everything that burns money** (submit/stage/relaunch);
  show the preflight estimate inline in the confirm button ("launch
  ~$0.20"). Budget is a hard gate server-side already; mirror it in UX.
- Dry-run parity: the backend distinguishes staged (verified, $0) from
  launched — render the difference.
- Never poll the provider API from the UI; capacity truth comes from
  MCP tools (it rate-limits and reconciles there).
- Event timeline (journal) beside every run: launched → progress ticks
  → run_done → delivered; failures link the `error` text + run.log
  tail (`job_logs`).
- Cost shown per-run AND aggregate; idle-GPU kills (reap) logged as
  their own event class so users learn the money law.

## 4. Tech suggestions (non-binding)

- v0/v1 need only: static SPA + one tiny read proxy for the bucket
  (sign URLs server-side; the browser never sees S3 creds) —
  a ~50-line Go/Node service in `quadra` ns, or the RustFS console as
  a stopgap. Poll interval 15 s.
- v2/v3 speak MCP (streamable-http + bearer). Any MCP-capable client
  works; a thin in-house panel can call the same JSON-RPC directly.
- Charts: progress.jsonl rows are ≤ a few hundred points/run — plain
  SVG/canvas, no charting framework required.
- Deep-link scheme: `/runs/<id>` (detail), `/runs` (table),
  `/serve` (catalog), `/money` (budget). That's the whole app — four
  routes, one table first.

## 5. Integration notes for the facade session (their router, our index)

The router aggregates engines for chat traffic; this UI's serving panel
should consume **their** `/v1/models` (single catalog truth) and
**our** `runs/index.jsonl` (single run-history truth). The join key is
the model/alias name registered at `qtrain register` time — worth a
shared convention: `name = <domain>-<size>[-<stage>]` (already the
pattern: queer360-lora, queer-360m-awq).

## 6. Aesthetic (kandi's house style)

Dark, dense, terminal-adjacent. Monospace numbers, colored status
chips, 3am-texting restraint — the product of the model this fleet
trains. The mascot panel (persona card with template provenance) is the
one place for warmth.
