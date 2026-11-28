"""mcp-trainer — cluster MCP server exposing the qtrain capability set
(SPEC §4 mcp layer, v0.3.0). Fleet port 3022, bearer-gated.

Tools mirror `qtrain` 1:1 so any MCP client (desktop agents, the mcp-agent
fleet service) can drive training without shell access. submit_training is
NON-BLOCKING: it returns the run_id the moment the ledger registers it and
the full cycle proceeds on a worker thread; job_status reads truth from
the ledger + lane heartbeat.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import yaml

from fastmcp import FastMCP

from orchestrator import dispatch, notify, progress, storage
from orchestrator.config import ROOT, corpora_dir, load_dotenv, resolve_recipe
from orchestrator.ledger import Ledger
from orchestrator.providers import get_provider

from . import __version__
from .auth import BearerGate

mcp = FastMCP(
    "trainer",
    instructions=(
        "Automated LoRA/QLoRA/FFT fine-tuning on rented GPUs. Typical flow: "
        "list_recipes -> gpu_catalog/preflight_report -> submit_training "
        "(returns run_id; pair= co-trains two models on one pod) -> "
        "run_progress/job_status (poll; heartbeat mid-run) -> artifacts "
        "land in runs/<id>/ on the PVC lane + one rollup line in "
        "runs/index.jsonl (list_runs reads that index — the UI-facing "
        "table). bench_models scores any OpenAI-compatible endpoint. "
        "terminate_pod / reap_pods / janitor_pass manage capacity; "
        "budget_report watches spend. READ trainer://laws BEFORE your "
        "first submit — it is the fleet's operational law with run-id "
        "citations (money caps, deadman, holdout, Qwen3.5 traps, pair "
        "mode, scrub discipline)."),
)

LEDGER = "state/ledger.db"


def _led() -> Ledger:
    return Ledger(ROOT / LEDGER)


# ------------------------------------------------------------------- lore
# THE LAWS SURFACE: an agent driving this fleet must not need tribal
# knowledge (r2 died from a silent packaging law that lived only in a
# human's head; f831b7 from a deadman law that lived only in a README).
# Every law here has a run-id or commit citation — no vibes. Keep in
# sync with SPEC; the pinned tests (test_mcp_server) fail if a law
# leaves the surface.
@mcp.resource("trainer://laws")
def laws_resource() -> str:
    """Operational laws of the training fleet, machine-readable."""
    return """\
FLEET LAWS (each cited; full text SPEC.md, tests are the enforcement):

RECEIPTS & PROVENANCE
- Every claim cites a run id answerable by `qtrain verify`. The lane
  payload IS the job contract — relaunch reuses staged bytes, never
  re-stages (dispatch relaunch).
- 13ak: dataset/image/config bytes are attested IN-POD after sha-verify
  (agent materialize -> attest.json -> manifest). Absence of complaint
  is not proof.
- HOLDOUT LAW v2 (13am): rows bucket by first-message template hash;
  a template held out in round K is held out forever. Row-hash splitting
  leaked ~100% (templated families share variants — synth measured).
  val_set_size is forced 0 by render_config; eval happens post-train
  against the served bytes.

MONEY
- Caps from history, not hope; jobs die at max_cost/max_runtime with a
  stated reason. Over-ceiling offers are never launched: waiting beats
  overbuying.
- 13an DEADMAN: volume-lane pods watch the orchestrator ping; silence
  past grace = save artifacts, stamp deadman, manifest, ask the
  PLATFORM to stop the pod, then exit (r2 law: container exit alone
  does NOT stop RunPod billing — restart loop, ~$3.3, 5 zombies).
  The wiring itself is pinned by test_arm_deadman_wires_the_watchdog —
  an indented-line refactor once SILENTLY DISARMED this while every
  other test stayed green (fixed 86d5e9a). Trust wiring tests, not
  docstrings.

TRAINING
- Qwen3.5 family: text-path LoRA targets ONLY (1a6ea5 VLM trap —
  lora_target_linear bakes dead visual tensors and vLLM then refuses
  the adapter); micro_batch_size 1 (425e6c: fla kernels reject batch>1
  with packing cu_seqlens); per-LoRA chat templates do NOT exist fleet
  wide (13ai) — identity/behavior lives in weights.
- Tool-calling domains: `axolotl preprocess --debug` must show the
  template's tools branch emitting tool-call TOKENS before any spend.
  Otherwise the model pays full price to learn prose ABOUT tools.
- Datasets carry per-row `tools` (field_tools); one arg key = one type
  corpus-wide (HF datasets casting is fatal; extract_opencode.py
  stringifies collisions and RECORDS it).
- Teacher quality is a corpus receipt, not a hope: name the source
  model in the extraction report (qwen3.8-flash self-distillation is
  capability-PROOF, not capability-CEILING).
- Corpus rows are LITERAL: expand.py scans row JSON for $-grid markers,
  and transcripts ABOUT the corpus carry live grid syntax inside tool
  outputs AND tool inputs (receipt 2026-11-28, three steps to truth:
  prepare-domain died "missing $ arrays"; defusing outputs-only left it
  dying — the survivors were inside edit/write ARGUMENTS, the same
  place secrets hid). The extractor deep-scrubs every string a row can
  contain, SELF-TESTS the defuser against expand's exact matcher at
  every start, and VERIFY-FAILS (exit 3) if any live marker survives
  the written file. Generated corpora must be expand-idempotent BY
  CHECK, not by hope.
- Tool NAME whitelist: transcripts carry teacher decoding damage
  (`ash`, `shll`, `tool_use`, `invalid`) — drop, never schema-ify;
  training hallucinated tool names INTO a small model is the tier-1
  failure (extractor receipt 2026-11-27: 19 such calls).

CO-TENANT PAIR MODE (one pod, two recipes)
- MEMORY-GATED, not free: full-CE legs peak ~21.3 GiB EACH at 8192
  (d90d9a — the old launch-bound "fills the idle SM" premise is dead,
  the pair OOMed both legs at step 0). Concurrent pair needs the 48 GB
  class; that class can be unsourceable in a volume's DC (d009dd: 8
  waves / 1800s, no instances) while 24 GB is plentiful — so
  pair_mode="sequential" chains legs one-at-a-time (PAIR_SEQ_MIN_VRAM
  28 — quiet-card 25.2/leg + headroom, c4f1ec): one run_id, one
  manifest, same gate, serialized wall-clock.
- One run_id, one manifest (file map), per-slug output trees + per-slug
  prepared dirs (shared prepared dir = HF fingerprint race). Gate
  requires BOTH trees: half-success fails HONESTLY.
- Never pair two sustained-memory-heavy legs (FFT vs big qlora): an OOM
  kills both models in one manifest. Proven-fit pilot pair: 0.8B-lora +
  4B-qlora.

SAFETY SPINE
- Corpora mounts read-only into RENTED pods: secret scrubbing runs at
  extraction, with an alarm rate that FAILS the build (exit 2). Secrets
  never reach the lane, the pod, or the product.
- Every terminal path terminates the pod; keep-alive only when bytes
  worth money sit on the instance (TrainingError keep_alive).
- Family registry (recipes/_families.yaml, 13al): targets proven false
  are REFUSED at submit_stage for free; re-asking needs new evidence.

QUANT & SERVING
- The train recipe returns NO quant: QLoRA's 4-bit base is training-time
  math, never a product. Artifacts are bf16 adapters; quants are STAGE
  children of a MERGED model (train -> merge -> awq|fp8), each its own
  run row + gate. One train run may fan out to many quant products.
- vLLM (2026 status, maintainer table): FP8+LoRA dynamic switching ✅
  but W8A8 kernels need Ada/Hopper (Ampere dead); AWQ+LoRA ✅ with the
  #50059 caveat — r=32 all-layer adapters over int4 W4A16 go weak/non-
  reproducible on some stacks. bnb-4bit base + adapter switching ❌ —
  never serve a training quant as a product.
- HOLDOUT LAW EXTENSION: adapter holdout NLL does NOT certify the
  quantized product. The QUANTIZED ENDPOINT benches before register
  flips traffic. Quantize-on-hybrid (qwen3.5 GDN) expects PARTIAL
  products — linears without kernels stay bf16; measure the delta.

GUIDANCE vs LAW (assumption audit, 2026-11-27)
- Recipes carry PRIORS, not ceilings: lora_r=32, alpha=64, lr=1e-4,
  epochs=2, adamw_bnb_8bit are tuned defaults that travel by copy —
  NOTHING enforces them; per-task tuning is expected (r scales with
  skill breadth, sequence_len is always task-measured, never tradition).
  Read recipe_report() for the per-knob truth before inventing laws.
- A REFUSAL exists only where money burned: qwen3.5 micro_batch>1
  (fla/cu_seqlens, 425e6c) and adapter-without-target-list (VLM trap,
  1a6ea5). Every refusal cites its run and names its bypass code —
  force_guards=["code"] proceeds knowingly, and the choice is recorded
  on the ledger row (guards spec field). No silent assumptions.
- Silent forcing is a bug: render_config forces val_set_size=0 (law —
  clash recorded even against overrides), logging_steps/save_total_limit
  (soft — recipe-declared value RECORDED in the row when different).
  The packaging bug survived months because the books could not see
  what actually ran; visibility first, refusal only for burns.

PIPELINE DEFAULTS (stage/budget audit, 2026-11-27 — read pipeline_report())
- The silent min_vram=10 stage floor is REMOVED: CPU-only kinds (onnx,
  int8) declare 0 (the floor was EXCLUDING their cheapest fits), and GPU
  kinds declare labeled PRIORS (merge/ablate/holdout 12, awq 16 — the
  4B pilots correct these numbers; DEFAULT_MIN_VRAM fallback is reported
  as an assumption in preflight, never hidden).
- Preflight must quote the SAME kill math as launch: max_runtime is
  SECONDS (/3600, the 425e6c units law) — the /60-vs-/3600 bug class is
  why preflight output is test-pinned against the launch path.
- Estimates name their basis: observed-median / static-prior /
  FALLBACK-30 (unknown kind = a guess the gate runs on — declare it in
  EST_MINUTES or see it in the report). Calibration-contamination law:
  obs samples expire with EST_VERSION, never deleted (cb059d).
- DEADMAN stale-warning threshold derives from QTRAIN_DEADMAN_GRACE_S
  (70% of grace), not a hardcoded 420 — knobs and their watchers cannot
  drift apart.
- HOLDOUT modulo (1-in-50) is a NAMED constant with a law note: changing
  it reshuffles every historical split — that is a HOLDOUT_LAW version
  bump + rebase (c3d7027), never a quiet edit.
- Money gates (QTRAIN_BUDGET/_MARGIN/_RESERVE/QTRAIN_REAP_GRACE) are all
  env knobs WITH defaults stated in pipeline_report — the aggregate gate
  refuses BEFORE launch, the soft kill acts during, the deadman covers
  watcher death. Three layers, one vocabulary.
- Remote tuning has a surface: fleet_settings (read values + provenance)
  and knob_set (audited writes). Whitelist exhaustive; money ceilings
  and rate caps DOWNWARD-only over the API, safety floors UPWARD-only —
  a compromised surface can tighten the purse and raise the floors,
  never the reverse. Writes are process-lifetime; persistence is git's
  job (env file/infra, reviewed like code). Image pins and the holdout
  law are git-only by design.
- Idempotence (DUP-POD LAW): every train row carries work_key = sha of
  (dataset sha x rendered configs). Re-submitting identical work that
  already DONE — artifacts still on hand — returns prior_run and buys NO
  pod; the check itself is journaled (state 'unchanged'). retrain=True
  is the deliberate re-roll. A done row whose artifacts were GC'd is a
  receipt for nothing, so then a fresh run is the honest answer.
- Watcher death is a class, now with three machines (da62ed): the agent
  IDLES on boot-revival when a terminal manifest is already on the lane
  (RunPod REVIVES exited containers — re-running was the r2 billing
  loop); launches that arm a deadman without a kill endpoint journal
  'deadman_inert' instead of failing silently; and the janitor HEALS
  open rows past their pod deadline — harvest the lane's truth, else
  close the row and terminate the zombie it claims. An unrun janitor is
  folklore: bin/supervise carries the timer.

- Recovery rides the CONTRACT (9ddeb5): relaunch reads the pair shape
  (configs[]) off the lane payload, so the VRAM floor applies on the
  recovery path too — a janitor-healed pair on single-leg thinking
  picked a 16GB card and re-burned the OOM; and a live pod matching the
  run-name suffix is ATTACHED, never double-booked (crash between
  pod-create and the ledger write leaves 'pending + no instance').

SEQUENCE LAW (smoke 1ffca0, pinned by the window guard)
- The WINDOW belongs to the DATA (rows + template render, including
  per-sample tools schemas). --mini trims ROWS and epochs — NEVER
  sequence_len: forcing 256 under ~4.6K-token agentic rows produced
  num_samples=0 and step-0 death for both pair legs. submit refuses a
  starved window (bypass code window_starved) BEFORE any pod launches.

MEMORY LAW (receipts da62ed + d90d9a + c4f1ec + 2a7b95)
- Full-materialization cross-entropy costs ~25.2 GiB PER LEG at the
  8192 window x 151,936 Qwen vocab — QUIET-CARD measurement (c4f1ec:
  17.62 in-use + 7.58 attempt on a 23.43 GiB Blackwell; the earlier
  21.3 was a collision-corrupted number). Consequences: the 24 GB class
  cannot hold ANY single leg at 8192; sequential pairs need the 32 GB
  class (floor 28); concurrent pairs need 48 — and d009dd showed the
  volume's DC may not source 48 AT ALL despite catalogs (DC alignment,
  not market absence: preflight later listed A40/A6000/L40S at $0.59+).
- The pinned axolotl has NO CE-optimization path, PROVEN BY TWO
  BEHAVIORAL PROBES: chunks_per_batch (d90d9a) and use_liger (2a7b95)
  both rode the config, left ZERO trace in the pod log, and the OOM
  allocation came back byte-identical at 7.58 GiB — silent drops, no
  error, no effect. Boot dumps also omit ACCEPTED keys
  (gradient_checkpointing lived yet never appears in serialization), so
  absence-from-dump is not schema truth — absence-plus-inert-behavior
  is. The machine: unknown_keys guard on the never-observed signature
  (bypassable, receipted — 2a7b95 BYPASSED it deliberately to run this
  very probe). v2 = dump the schema from the image itself (O4.3).
- The 'sample_packing & sdpa => 0.0 loss' warning is neutralized in
  this build by the sdpa_varlen cu_seqlens patch; attention stays sdpa.
"""


KNOBS: dict[str, dict] = {
    # name -> {env, default, cast, clamp, note}. Whitelist IS the law:
    # set_knob knows ONLY these. Money ceilings are DOWNWARD-ONLY over
    # remote (raise requires the env/git path); floors are upward-only
    # (raising a safety floor is always allowed). Pins and holdout law
    # are absent by design — those move via commit+CI, never a socket.
    "QTRAIN_BUDGET": {"env": "QTRAIN_BUDGET", "default": 5.0, "cast": float,
                      "range": None, "dir": "down",
                      "note": "concurrent-estimate ceiling ($)"},
    "QTRAIN_BUDGET_MARGIN": {"env": "QTRAIN_BUDGET_MARGIN", "default": 1.5,
                             "cast": float, "range": (1.0, 3.0), "dir": "down",
                             "note": "estimate padding (lower = tighter caps)"},
    "QTRAIN_RESERVE": {"env": "QTRAIN_RESERVE", "default": 2.0, "cast": float,
                       "range": (0.0, None), "dir": "up",
                       "note": "balance floor left untouched ($)"},
    "QTRAIN_MAX_PRICE_HR": {"env": "QTRAIN_MAX_PRICE_HR", "default": 1.50,
                            "cast": float, "range": (0.0, None), "dir": "down",
                            "note": "rate ceiling $/hr (right-sizing, 13q)"},
    "QTRAIN_PAIR_MIN_VRAM": {"env": "QTRAIN_PAIR_MIN_VRAM", "default": 48,
                             "cast": int, "range": (8, None), "dir": "up",
                             "note": "concurrent-pair VRAM floor GB "
                                     "(2 legs full-CE + 2 ctx alive at "
                                     "once, d90d9a/c4f1ec measured)"},
    "QTRAIN_PAIR_SEQ_MIN_VRAM": {
        "env": "QTRAIN_PAIR_SEQ_MIN_VRAM", "default": 28, "cast": int,
        "range": (8, None), "dir": "up",
        "note": "sequential-pair VRAM floor GB (25.2/leg quiet-card "
                "measured c4f1ec + headroom; targets 32GB class)"},
    "QTRAIN_DEADMAN_GRACE_S": {"env": "QTRAIN_DEADMAN_GRACE_S", "default": 600,
                               "cast": int, "range": (60, None), "dir": "up",
                               "note": "watcher-silence grace before pod "
                                       "self-harm (13an)"},
}


def fleet_settings_impl() -> dict:
    """Current value + provenance of every remote-tunable knob: 'env'
    (something set it) vs 'default' (nobody overrode the code value).
    Read via MCP; writes go through set_knob (whitelisted + audited)."""
    out = {}
    for name, spec in KNOBS.items():
        raw = os.environ.get(spec["env"])
        out[name] = {"value": (spec["cast"](raw) if raw not in (None, "")
                               else spec["default"]),
                     "source": "env" if raw not in (None, "") else "default",
                     "direction": spec["dir"],
                     "range": spec.get("range"),
                     "note": spec["note"]}
    return {"knobs": out,
            "law": "whitelist is exhaustive; image pins & holdout law are "
                   "git-only (commit+CI), never remote-settable",
            "audit": "events.db knob rows carry old/new + ts + host user"}


def set_knob_impl(name: str, value) -> dict:
    """Audited runtime knob change (assumption-audit doctrine, 2026-11-28):
    enumerated whitelist, direction-biased clamps (money ceilings DOWN-only,
    safety floors UP-only), every write lands an 'knob' event row. The
    change applies to THIS orchestrator process for values read at call
    time (budget module reads os.environ live via module attrs — those
    are re-synced); launcher-spawned children inherit the new env."""
    spec = KNOBS.get(name)
    if spec is None:
        return {"error": f"knob not whitelisted (have: "
                         f"{', '.join(sorted(KNOBS))}); image pins and "
                         "holdout law are git-only by design"}
    cast = spec["cast"]
    try:
        v = cast(value)
    except (TypeError, ValueError):
        return {"error": f"value {value!r} is not a {cast.__name__}"}
    lo, hi = spec.get("range") or (None, None)
    if lo is not None and v < lo:
        return {"error": f"{name} >= {lo} ({spec['note']})"}
    if hi is not None and v > hi:
        return {"error": f"{name} <= {hi} ({spec['note']})"}
    old = os.environ.get(spec["env"])
    old_v = cast(old) if old not in (None, "") else spec["default"]
    if spec["dir"] == "down" and v > old_v:
        return {"error": f"{name} is DOWNWARD-only over the API "
                         f"(current {old_v:g}); raising it is an env/git "
                         "act, so a compromised surface cannot loosen "
                         "the purse"}
    if spec["dir"] == "up" and v < old_v:
        return {"error": f"{name} is UPWARD-only over the API: lowering a "
                         f"safety value (current {old_v:g}) is an env/git act"}
    os.environ[spec["env"]] = str(v)
    # budget.py read MARGIN/TOTAL/RESERVE at import — resync so a knob
    # set actually MOVES the gate, not just the string (drift here was a
    # real failure class: mint/check)
    _budget_resync()   # same-module function — `from . import` here would
                       # import from the PACKAGE and silently no-op via
                       # the except below (the drift class itself)
    notify.emit("knob", kind="knob", knob=name,
                old=None if old is None else str(old), new=str(v),
                source=str(os.environ.get("USER", "unknown")),
                law="runtime-set; restart does NOT persist (env is "
                    "ephemeral by design — persistence is git's job)")
    return {"knob": name, "old": old_v, "new": v, "persisted": False,
            "note": "process-lifetime change; durable tuning belongs in "
                    "the server's env file / infra, reviewed like code"}


def _budget_resync() -> None:
    from orchestrator import budget
    budget.MARGIN = float(os.environ.get("QTRAIN_BUDGET_MARGIN", "1.5"))
    budget.TOTAL = float(os.environ.get("QTRAIN_BUDGET", "5.0"))
    budget.RESERVE = float(os.environ.get("QTRAIN_RESERVE", "2.0"))


@mcp.tool
def fleet_settings() -> dict:
    """Current value + provenance ('env' vs 'default') of every
    remote-tunable fleet knob (budget, rate/pair/deadman tuning) — read
    before set_knob. Image pins and the holdout law are ABSENT by
    design: those move via commit+CI so pin drift stays git-visible."""
    return fleet_settings_impl()


@mcp.tool
def knob_set(name: str, value) -> dict:
    """Audited runtime tuning, whitelist-only: enumerated knobs,
    direction-biased clamps (money ceilings/price caps DOWNWARD-only
    over the API; safety floors UPWARD-only), every write journal-
    audited (event 'knob', old/new/source) and process-lifetime ONLY —
    persistence is git's job. Unlisted names refuse with evidence."""
    return set_knob_impl(name, value)


@mcp.tool
def pipeline_report(kind: str | None = None) -> dict:
    """Read the pipeline's declared defaults BEFORE tuning a stage or
    trusting an estimate (assumption-audit doctrine, see trainer://laws
    PIPELINE DEFAULTS): per-stage gate contracts, declared vs fallback
    min_vram, estimate basis (observed | static-prior | FALLBACK-30),
    budget knobs, deadman/holdout law notes. Delegates to
    dispatch.pipeline_report — the CLI `qtrain defaults` shows the
    identical object."""
    return dispatch.pipeline_report(kind)


@mcp.resource("trainer://versions")
def versions_resource() -> str:
    """Runtime truth of this surface: banner + pinned laws present."""
    return (f"mcp-trainer {__version__}; laws resource present; "
            "submit_training pair-mode present")


def _lane_get(key: str, provider: str = "runpod") -> bytes | None:
    if provider == "local":
        return storage.host_get(key)
    vol = storage.vol_cfg()
    return storage.vol_get(key) if vol else storage.get_object(key)


# ------------------------------------------------------------------ tools

@mcp.tool
def gpu_catalog(gpus: list[str] | None = None, tier: str | None = None,
                provider: str = "runpod") -> dict:
    """Live GPU offers, cheapest first. gpus: name filters (regex-ish
    substring); tier: 'community'|'secure'."""
    offers = get_provider(provider).catalog(gpus, tier)
    return {"offers": [{"gpu_id": o.gpu_id, "vram_gb": o.vram_gb,
                        "price_hr": o.price_hr, "secure": o.secure,
                        "availability": o.availability}
                       for o in offers[:20]]}


@mcp.tool
def list_recipes(domain: str | None = None) -> dict:
    """Training recipes — task config lives in corpora (SPEC §4b boundary):
    $CORPORA_DIR/recipes/<domain>/<model>.yaml, with the legacy in-repo
    axolotl/<domain>/ as read-fallback until fully retired."""
    out = []
    bases = [(corpora_dir() / "recipes", ""), (ROOT / "axolotl", "axolotl/")]
    for base, prefix in bases:
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir() or (domain and d.name != domain):
                continue
            for f in sorted(d.glob("*.yaml")):
                out.append({"domain": d.name, "model": f.stem,
                            "recipe": prefix + f.relative_to(base).as_posix()})
    return {"recipes": out}


@mcp.tool
def submit_training(domain: str, recipe: str, mini: bool = True,
                    overrides: dict | None = None,
                    gpus: list[str] | None = None, tier: str | None = None,
                    max_runtime: int = 3600, max_cost: float | None = None,
                    dry: bool = False, provider: str = "runpod",
                    pair: str | None = None,
                    pair_mode: str = "concurrent",
                    force_guards: list[str] | None = None,
                    retrain: bool = False) -> dict:
    """Launch a full training cycle in the background; returns as soon as
    the run is registered (seconds). mini=True is a cheap lane-validation
    run (48 rows, 1 epoch) — ALWAYS pilot mini before a real spend.
    provider: runpod | local ($0 K8s Jobs).
    Budget: max_cost None = derived (runtime x rate); offers above the
    rate ceiling are never launched — waiting beats overbuying.
    Track with job_status(run_id).

    IDEMPOTENCE (dup-pod law): identical data + configs that already
    SUCCEEDED (artifacts on hand) returns {status: "unchanged",
    prior_run} without buying a pod — re-submitting an unchanged corpus
    costs $0. retrain=True is the deliberate re-roll.

    pair (recipe name): CO-TENANT — one pod trains both recipes.
    LAWS (receipts in dispatch.submit + agent tests):
    - MEMORY IS THE CONSTRAINT, NOT LAUNCH: micro_batch_size 1 is a
      kernel law (425e6c) but d90d9a proved co-running is NOT free at
      the 8192 window — full-CE legs peak ~21.3 GiB EACH (13.75 in-use
      + 7.58
      attempt on a 19.55 GiB card, both legs dead at step 0). The old
      "~1.5x throughput, one pod bill" premise was FALSE: the pair OOMs
      before it computes. Concurrent pair therefore needs the 48 GB
      class (2 legs + 2 CUDA ctx).
    - pair_mode "concurrent" (default) enforces PAIR_MIN_VRAM=48 and
      takes the capacity risk; "sequential" chains legs one-at-a-time
      under PAIR_SEQ_MIN_VRAM=24 — same run_id, same both-trees gate,
      same combined exit, serialized wall-clock. d009dd proved the 48 GB
      class can be UNAVAILABLE (8 waves / 1800s, no instances) while 24
      GB is plentiful, so sequential is the mode that actually ships.
    - ONE run_id / manifest / budget; per-model output trees under
      /work/outputs/<slug>; the artifact gate requires BOTH trees, so a
      half-success fails honestly. One crash never masks the other
      (per-pid waits in the sh -c chain).
    - DO NOT pair the FFT regime (no `adapter:` key) against another
      big leg: full-weight training is 10-12 GB sustained and the pair
      OOMs — an OOM kills BOTH legs in one manifest. 0.8B-lora + 4B-qlora
      is the proven-fit pilot pair.
    - Deadman (13an) is re-armed as of 86d5e9a — a dead orchestrator
      saves work and stops billing; before that fix, co-tenancy would
      have doubled the r2 zombie exposure. Long pair runtimes are safe.

    Agentic/tool domains (opencode): before ANY real submit, the
    preprocess --debug gate must show the tokenizer template's tools
    branch rendering tool calls into tokens — if the template has no
    tools branch the model learns prose ABOUT tool calls instead of the
    tokens, at full price. Teacher quality ships with the corpus receipt
    (extract_opencode.py report), not vibes.

    Guards: submit REFUSES only burned footguns (qwen3.5 batch>1, adapter
    without target-modules — each refusal cites its run). Everything else
    in a recipe is a PRIOR you may move: tune via overrides, see
    recipe_report() for the per-knob guidance, force_guards=[codes] to
    proceed past a refusal knowingly (recorded on the ledger row)."""
    recipe_path = resolve_recipe(recipe)
    if not recipe_path.is_file():
        return {"error": f"recipe not found: {recipe}"}
    pair_path = None
    if pair:
        pair_path = resolve_recipe(pair)
        if not pair_path.is_file():
            return {"error": f"pair recipe not found: {pair}"}
    holder: dict = {}
    gate = threading.Event()

    def _on_run_id(rid: str) -> None:
        holder["run_id"] = rid
        gate.set()

    def _bg() -> None:
        try:
            holder["result"] = dispatch.submit(
                domain, recipe_path, mini=mini, overrides=overrides or {},
                gpus=gpus, tier=tier, max_runtime=max_runtime,
                max_cost=max_cost, dry=dry, provider=provider,
                pair=pair_path, force_guards=force_guards,
                pair_mode=pair_mode,
                retrain=retrain,
                on_run_id=_on_run_id)
        except Exception as e:                          # noqa: BLE001
            holder["error"] = f"{type(e).__name__}: {e}"
        finally:
            gate.set()

    threading.Thread(target=_bg, daemon=True,
                     name="submit-training").start()
    deadline = time.time() + 240
    while time.time() < deadline:
        if "run_id" in holder:
            return {"run_id": holder["run_id"], "state": "submitted",
                    "note": "full cycle continues in background; poll "
                            "job_status()"}
        if "error" in holder or "result" in holder:      # failed pre-ledger
            return {"error": holder.get("error"),
                    "result": holder.get("result")}
        time.sleep(0.25)
    return {"error": "staging did not register a run within 240s"}


@mcp.tool
def recipe_report(recipe: str, pair: str | None = None,
                  overrides: dict | None = None) -> dict:
    """Read-only guidance view of a resolved config (recipe + overrides):
    per-knob GUIDE text (which values are PRIORS you may move vs which
    have burned evidence), the guard verdict (refusals + advisories), and
    the exact bypass codes. Read this BEFORE inventing constraints — and
    before overriding anything the fleet refuses."""
    rp = resolve_recipe(recipe)
    if not rp.is_file():
        return {"error": f"recipe not found: {recipe}"}
    ov = dict(overrides or {})
    merged = {**(yaml.safe_load(rp.read_text()) or {}), **ov}
    merged_pair = None
    if pair:
        pp = resolve_recipe(pair)
        if not pp.is_file():
            return {"error": f"pair recipe not found: {pair}"}
        merged_pair = {**(yaml.safe_load(pp.read_text()) or {}), **ov}
    refusals, advisories = dispatch.config_guards(merged, merged_pair)
    return {"recipe": str(rp), "base_model": merged.get("base_model", ""),
            "family": (dispatch.families.family_for(
                str(merged.get("base_model", ""))) or ("unknown", {}))[0],
            "resolved": {k: merged.get(k) for k in
                         ("adapter", "lora_r", "lora_alpha",
                          "learning_rate", "micro_batch_size",
                          "num_epochs", "sequence_len", "load_in_4bit")},
            "refusals": refusals, "advisories": advisories,
            "bypassable": [r["code"] for r in refusals],
            "knob_guide": dispatch.KNOB_GUIDE}


@mcp.tool
def job_status(run_id: str) -> dict:
    """Truth for one run: ledger state + live heartbeat (mid-run) +
    terminal manifest summary when done."""
    row = next((r for r in _led().list()
                if r["run_id"] == run_id), None)
    if row is None:
        return {"error": f"no such run: {run_id}"}
    out = {"run_id": run_id, "state": row["state"], "kind": row["kind"],
           "instance_id": row["instance_id"],
           "artifact_uri": row["artifact_uri"], "error": row["error"],
           "updated_at": row["updated_at"]}
    try:
        prov = row["provider"] or "runpod"
        hb = _lane_get(f"runs/{run_id}/status.json", prov)
        if hb:
            out["heartbeat"] = json.loads(hb)
        man = _lane_get(f"runs/{run_id}/manifest.json", prov)
        if man:
            m = json.loads(man)
            out["manifest"] = {
                "status": m.get("status"), "training": m.get("training"),
                "artifacts": sorted(m.get("artifacts", {}))[:12],
                "problems": m.get("problems")}
    except Exception as e:                              # noqa: BLE001
        out["lane_error"] = str(e)[:200]
    return out


@mcp.tool
def submit_stage(kind: str, from_run: str, base_model: str | None = None,
                 provider: str = "runpod", max_runtime: int = 3600,
                 max_cost: float | None = None, dry: bool = False,
                 fresh_holdout: bool = False,
                 score_base: bool = False) -> dict:
    """Post-process a finished run: kind merge (needs base_model) | onnx
    | int8 | awq (GPU; calibrates on the source run's dataset). Cloud
    (runpod) by default; provider=local is the opt-in $0 fallback (cpu
    kinds only). Budget defaults are derived per stage (estimate = rate x
    est-minutes x margin); the rate ceiling refuses oversized cards.
    Non-blocking; track with job_status(). Stage outputs chain as the
    next stage's from_run. fresh_holdout (holdout kind): re-split the
    lineage's domain corpus under the CURRENT bucketing law and score
    that — the trend rebase across the v1->v2 seam (c3d7027)."""
    holder: dict = {}
    gate = threading.Event()

    def _on_run_id(rid: str) -> None:
        holder["run_id"] = rid
        gate.set()

    def _bg() -> None:
        try:
            holder["result"] = dispatch.submit_stage(
                kind, from_run, base_model=base_model, provider=provider,
                max_runtime=max_runtime, max_cost=max_cost, dry=dry,
                fresh_holdout=fresh_holdout, score_base=score_base,
                on_run_id=_on_run_id)
        except Exception as e:                          # noqa: BLE001
            holder["error"] = f"{type(e).__name__}: {e}"
        finally:
            gate.set()

    threading.Thread(target=_bg, daemon=True, name="submit-stage").start()
    deadline = time.time() + 60
    while time.time() < deadline:
        if "run_id" in holder:
            return {"run_id": holder["run_id"], "state": "submitted"}
        if "error" in holder or "result" in holder:
            return {"error": holder.get("error"), "result": holder.get("result")}
        time.sleep(0.25)
    return {"error": "stage did not register within 60s"}


def row_provider(run_id: str) -> str:
    row = next((r for r in _led().list() if r["run_id"] == run_id), None)
    return (row or {}).get("provider") or "runpod"


@mcp.tool
def job_logs(run_id: str, tail: int = 100) -> dict:
    """Training log: prefer local harvest (runs/<id>/run.log), else the
    lane copy. tail = last N lines."""
    local = ROOT / "runs" / run_id / "run.log"
    raw = local.read_bytes() if local.is_file() else _lane_get(
        f"runs/{run_id}/run.log", row_provider(run_id))
    if raw is None:
        return {"error": "no run.log on disk or lane (still running on an "
                         "older image? heartbeat has progress meanwhile)"}
    lines = raw.decode(errors="replace").splitlines()
    return {"run_id": run_id, "lines": lines[-tail:]}


@mcp.tool
def list_runs(limit: int = 20) -> dict:
    """Recent runs: lane index first (UI-facing, cross-controller truth),
    local ledger as fallback for runs the lane never saw."""
    try:
        from orchestrator import progress as prog
        rows = prog.read_index("runpod", limit=limit)
        if rows:
            return {"source": "lane_index", "runs": list(reversed(rows))}
    except Exception:                              # noqa: BLE001
        pass
    rows = _led().list()[-limit:]
    return {"source": "ledger", "runs": [{"run_id": r["run_id"], "state": r["state"],
                      "provider": r["provider"], "instance_id": r["instance_id"],
                      "artifact_uri": r["artifact_uri"], "error": r["error"]}
                     for r in rows]}


@mcp.tool
def list_pods(provider: str = "runpod") -> dict:
    """Live provider instances (capacity census; compare with list_runs)."""
    return {"pods": [{"id": i.id, "status": i.status, "name": i.name,
                      "gpu": i.gpu_type, "price_hr": i.price_hr}
                     for i in get_provider(provider).list_instances()]}


@mcp.tool
def terminate_pod(instance_id: str, provider: str = "runpod") -> dict:
    """Force-terminate one instance (budget emergency / orphan cleanup)."""
    return {"id": instance_id,
            "terminated": get_provider(provider).terminate(instance_id)}


@mcp.tool
def doctor(provider: str = "runpod") -> dict:
    """Orphan sweep: live pods with no ledger claim (billable leaks)."""
    live = {i.id: i for i in get_provider(provider).list_instances()}
    claimed = {r["instance_id"] for r in _led().list()
               if r["instance_id"]}
    orphans = [{"id": i, "gpu": p.gpu_type, "price_hr": p.price_hr}
               for i, p in live.items() if i not in claimed]
    return {"live_pods": len(live), "orphans": orphans}


@mcp.tool
def refresh_corpora() -> dict:
    """git pull the corpora checkout (new data -> fresh expansion next
    submit). CORPORA_DIR must point at a git clone."""
    d = os.environ.get("CORPORA_DIR", "")
    if not d or not (Path(d) / ".git").is_dir():
        return {"error": f"CORPORA_DIR={d!r} is not a git checkout"}
    r = subprocess.run(["git", "-C", d, "pull", "--ff-only"],
                       capture_output=True, text=True, timeout=120)
    return {"rc": r.returncode, "out": (r.stdout or r.stderr).strip()[:300]}


@mcp.tool
def reconcile(run_id: str | None = None) -> dict:
    """Re-attach the orchestrator to an OPEN run (state not terminal) and
    finish it: poll lane -> harvest -> terminate instance. Use after a
    crash/restart dropped the watch loop; run_id None = most recent open."""
    holder: dict = {}
    gate = threading.Event()

    def _bg() -> None:
        try:
            holder["result"] = dispatch.resume(run_id=run_id)
        except Exception as e:                          # noqa: BLE001
            holder["error"] = f"{type(e).__name__}: {e}"
        finally:
            gate.set()

    threading.Thread(target=_bg, daemon=True, name="reconcile").start()
    gate.wait(15)
    if "result" in holder and holder["result"].get("resumed") is None:
        return holder["result"]                          # no open runs: fast
    if "error" in holder:
        return {"error": holder["error"]}
    return {"state": "reconciling",
            "note": "watch resumed in background; poll job_status()",
            "run_id": run_id or "most-recent-open"}


def resume_open_runs() -> int:
    """Boot hook: fleet restarts mid-job must not orphan runs. Any ledger
    row in an open state gets a watcher thread (harvest + terminate)."""
    try:
        rows = [r for r in _led().list() if r["state"] in dispatch.OPEN_STATES]
    except Exception:                                    # noqa: BLE001
        return 0
    for row in rows:
        threading.Thread(target=dispatch.resume,
                         kwargs={"run_id": row["run_id"]}, daemon=True,
                         name=f"resume-{row['run_id'][-6:]}").start()
        print(f"mcp-trainer: resumed watch for open run {row['run_id']} "
              f"(state={row['state']})", flush=True)
    return len(rows)


@mcp.tool
def run_progress(run_id: str) -> dict:
    """wandb-lite view (SPEC 14): ledger state + live heartbeat + durable
    progress history + recent lifecycle events. Cheap to poll; the
    heartbeat (status.json) is in-pod truth, progress.jsonl is the trail."""
    snap = progress.snapshot(run_id)
    snap["events"] = notify.recent(limit=12, run_id=run_id)
    return snap


@mcp.tool
def verify(run_id: str | None = None, live: bool = False,
           recompute: bool = False) -> dict:
    """Sanity check one run (default: most recent): does what the pod
    TRAINED ON match what we THINK it trained on? Cross-checks ledger
    record -> lane payload -> agent-attested hashes (dataset sha/lines,
    config, exact image), plus corpora/recipe drift. 'ok' is False iff a
    leg disagrees. recompute re-expands the domain first (slow, exact).
    """
    from orchestrator import provenance
    return provenance.verify_run(run_id, live=live, recompute=recompute)


@mcp.tool
def families(base_model: str | None = None) -> dict:
    """Family capability registry (SPEC 13al): what serving targets each
    base-model family supports, as gated data with run-id evidence. With
    base_model: that family's verdicts; without: the whole registry."""
    from orchestrator import families as fam
    if base_model:
        f = fam.family_for(base_model)
        return {"base_model": base_model, "family": f[0] if f else None,
                "targets": f[1].get("targets") if f else None,
                "notes": f[1].get("notes") if f else None,
                "refusals": {k: v for k, v in (f[1].get("targets") or {}).items()
                             if str(v).lower() == "false"} if f else {}}
    return {"families": fam.registry()}


@mcp.tool
def eval_history(model: str | None = None, limit: int = 20) -> dict:
    """Recorded bench scores (SPEC 13am): the cross-round table — every
    `qtrain bench --record` lands here keyed by (model, host, run)."""
    return {"evals": _led().list_evals(model=model, limit=limit)}


@mcp.tool
def wait_for_events(since: float = 0.0, timeout: int = 300,
                   run_id: str | None = None) -> dict:
    """Long-poll the lifecycle feed: blocks until events newer than `since`
    (a prior response's cursor) appear, or timeout. MCP clients get
    notification semantics without busy-polling job_status; webhooks
    (QTRAIN_WEBHOOK_URL) carry the same events across hosts."""
    deadline = time.time() + min(max(timeout, 0), 600)
    while True:
        fresh = [e for e in notify.recent(limit=100, run_id=run_id)
                 if e["ts"] > since]
        if fresh or time.time() >= deadline:
            return {"events": fresh,
                    "cursor": fresh[-1]["ts"] if fresh else since,
                    "timed_out": not fresh}
        time.sleep(2)


@mcp.resource("trainer://runs/{run_id}/progress")
def progress_resource(run_id: str) -> str:
    """Progress snapshot as a subscribable resource: clients may
    resources/subscribe and receive updated notifications when this
    server's watch loop emits them."""
    return json.dumps(progress.snapshot(run_id), default=str)


@mcp.tool
def reap_pods(provider: str = "runpod", dry: bool = True) -> dict:
    """Idempotency hygiene: terminate live GPUs with no active task
    behind them (terminal claims, stale run-suffix pods). DRY by
    default; only pods OUR naming scheme made (kind-first train-/awq/…,
    or legacy trainer-) are ever candidates; unclaimed
    pods are kept unless QTRAIN_REAP_UNCLAIMED=1 declares a single
    controller."""
    return dispatch.reap(provider, dry=dry)


@mcp.tool
def janitor_pass(provider: str = "runpod", dry: bool = True) -> dict:
    """One idempotency sweep: reap stray pods, relaunch stale
    never-launched runs, and HEAL open rows past their pod deadline
    (harvest a terminal lane, else fail the row + terminate its zombie
    pod). This is the backstop for a dead watcher while the deadman's
    kill endpoint stays inert; bin/supervise runs it on a timer."""
    return dispatch.janitor_once(provider, dry=dry)


@mcp.tool
def preflight_report(kind: str = "train", provider: str = "runpod",
                     max_price_hr: float | None = None,
                     max_runtime: int = 3600,
                     pair: bool = False,
                     pair_mode: str = "concurrent") -> dict:
    """Read-only launch decision (SPEC 13p): fitting cards, cost estimate,
    kill threshold, budget gate verdicts. Launches nothing, writes
    nothing — answers 'what would this cost, on what' before committing.
    pair=True shows the CO-TENANT fit. pair_mode selects the floor:
    "concurrent" (48GB — 2 legs + 2 CUDA ctx alive at once) or
    "sequential" (24GB — legs chained one at a time, serialized wall
    clock). d009dd: the 48GB class returned NO INSTANCES across 8 waves
    / 1800s in this volume's DC while 24GB had supply, so sequential is
    the mode that actually gets a pod."""
    return dispatch.preflight(kind, provider=provider,
                              max_price_hr=max_price_hr,
                              max_runtime=max_runtime, pair=pair,
                              pair_mode=pair_mode)


@mcp.tool
def budget_report(provider: str = "runpod") -> dict:
    """Spend visibility: live balance, open-run projections vs the cap,
    per-kind observed medians (self-calibrated from real runs)."""
    from orchestrator import budget as bud
    led = _led()
    bal = None
    if provider == "runpod":
        try:
            bal = get_provider(provider).balance()
        except Exception:                          # noqa: BLE001
            pass
    opens = [r for r in led.list() if r["state"] in bud.BILLED]
    import statistics
    return {"balance": bal, "reserve": bud.RESERVE, "budget_total": bud.TOTAL,
            "margin": bud.MARGIN, "est_version": bud.EST_VERSION,
            "open_runs": [{"run_id": r["run_id"], "kind": r["kind"],
                           "state": r["state"],
                           "est_cost": round(r.get("est_cost") or 0, 3)}
                          for r in opens],
            "open_projected": round(sum(r.get("est_cost") or 0
                                        for r in opens), 2),
            "observed_median_minutes":
                {k: round(statistics.median(v), 1)
                 for k in bud._obs() if (v := bud.obs_minutes(k))}}


@mcp.tool
def lane_hygiene(bad_shas: list[str] | None = None,
                 apply: bool = False) -> dict:
    """Lane quota housekeeping (SPEC (ar)): delete ONLY proven junk —
    orphan .part objects under terminal runs and artifacts.tar.gz whose
    manifest records a sha256 in bad_shas. Manifests are evidence, never
    candidates; every delete is verified (HEAD must answer 404).
    apply=False (default) returns the plan and touches nothing."""
    from orchestrator import hygiene
    return hygiene.sweep(bad_shas=set(bad_shas or ()), apply=apply)


@mcp.tool
def bench_models(urls: list[str], repeats: int = 2, timeout: int = 90,
                 max_tokens: int = 64) -> dict:
    """Question-matrix benchmark against OpenAI-compatible endpoints
    (identity/feeling/voice/instruction/json/world + rule checks, TTFT +
    tok/s from the stream, models auto-discovered via /v1/models).
    Returns a markdown table + raw rows — run it on the facade URL to
    cover every model behind the router in one pass."""
    from orchestrator import bench
    cfg = {"urls": urls, "repeats": repeats, "timeout": timeout,
           "max_tokens": max_tokens}
    rows = bench.run(cfg)
    return {"table": bench.table(rows), "rows": rows}


@mcp.tool
def rebuild_run_index(provider: str = "runpod") -> dict:
    """Rebuild runs/index.jsonl on the PVC lane from locally kept copies
    (manifests + run.log + ledger). Idempotent; preserves index lines for
    runs this controller never held. The UI's list view is that one
    object — see SPEC 14c."""
    return progress.reindex(provider)


# ------------------------------------------------------- synthesis (FRAMEWORK §10)

_SYNTH = None
_SYNTH_JOBS: dict = {}


def _synth():
    """Lazy-load src/synth.py (the deterministic IR synthesis engine —
    zero LLM, every artifact oracle-gated before it lands)."""
    global _SYNTH
    if _SYNTH is None:
        import importlib.machinery
        import importlib.util
        path = ROOT / "src" / "synth.py"
        spec = importlib.util.spec_from_file_location(
            "synth", str(path),
            loader=importlib.machinery.SourceFileLoader("synth", str(path)))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SYNTH = mod
    return _SYNTH


def _synth_run(argv: list[str]) -> dict:
    """Drive synth.main() and return its JSON report (stdout contract)."""
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        _synth().main(argv)
    out = buf.getvalue().strip()
    try:
        return json.loads(out)
    except Exception:                                    # noqa: BLE001
        return {"raw": out[:2000]}


@mcp.tool
def synth_derive(codes: list[str] | None = None, systems: list[str] | None = None,
                 per_code: int = 4, limit: int = 0, background: bool = True,
                 out: str | None = None, tmp: str | None = None) -> dict:
    """P2/P23 mutation synthesis: valid IR → exactly-one-error mutants
    (15-rule rulebook, E1–E7), emitted as derived invalid-program/repair
    templates + dpo.jsonl preference pairs. Every template is preflighted
    by the Pass-1 checker + fixture executor before it lands — yield
    report in stats. background=True (default): full-corpus runs take
    minutes; poll synth_status(job_id)."""
    argv = ["derive", "--per-code", str(per_code), "--limit", str(limit)]
    if codes:
        argv += ["--codes", ",".join(codes)]
    if systems:
        argv += ["--systems", ",".join(systems)]
    if out:
        argv += ["--out", out]
    if tmp:
        argv += ["--tmp", tmp]
    if not background:
        return _synth_run(argv)
    job = f"synth-{int(time.time()) % 100000:05d}"

    def _bg() -> None:
        try:
            _SYNTH_JOBS[job] = {"state": "done", "stats": _synth_run(argv)}
        except Exception as e:                           # noqa: BLE001
            _SYNTH_JOBS[job] = {"state": "error", "error": str(e)[:400]}

    _SYNTH_JOBS[job] = {"state": "running", "argv": argv}
    threading.Thread(target=_bg, daemon=True, name=job).start()
    return {"job": job, "state": "running",
            "note": "poll synth_status(job)"}


@mcp.tool
def synth_abstain(per_code: int = 2, out: str | None = None) -> dict:
    """P12: NOT-DERIVABLE rows — drop one fact from nl-to-ir graphs and
    emit a refusal row only where the fixture executor proves the program
    can no longer derive the answer (0 rows). Anti-hallucination data."""
    argv = ["abstain", "--per-code", str(per_code)]
    if out:
        argv += ["--out", out]
    return _synth_run(argv)


@mcp.tool
def synth_graph(ttl_path: str, domain: str | None = None,
                max_facts: int = 50, out: str | None = None) -> dict:
    """P4/P6/P12 over REAL knowledge: a .ttl file (or directory of them,
    e.g. rmcp-extract funnel output) becomes graph QA / hop-ladder /
    abstention scenario templates — every answer is RECOMPUTED by the
    closed-world fixture executor, never typed. This is the knowledge↔
    cognition bridge: the same graph teaches facts and reasoning."""
    if not Path(ttl_path).exists():
        return {"error": f"no such ttl path: {ttl_path}"}
    argv = ["graph", ttl_path, "--max-facts", str(max_facts)]
    if domain:
        argv += ["--domain", domain]
    if out:
        argv += ["--out", out]
    return _synth_run(argv)


@mcp.tool
def synth_audit() -> dict:
    """P28/P29/P31 corpus health: template coverage by task family,
    phrasing-entropy ratios (combinatorial-vs-semantic variety),
    final-IR near-duplicate pairs, derived inventory."""
    return _synth_run(["audit"])


@mcp.tool
def synth_manifest(out: str | None = None) -> dict:
    """Phase-0 kernel lineage: content-address every derived template +
    the oracle source digests (FRAMEWORK L3: checker changes invalidate
    downstream rows). Writes derived/manifest.json, returns derived@h."""
    argv = ["manifest"]
    if out:
        argv += ["--out", out]
    return _synth_run(argv)


@mcp.tool
def synth_bundle(domain: str = "cognition", out: str | None = None) -> dict:
    """Close the synthesis→training lane: expand all derived templates into
    $CORPORA_DIR/datasets/<domain>/*.jsonl (diagnosis/repair/abstain/graph)
    + preference.jsonl, and report the dataset@hash ids the qtrain pipeline
    consumes. Then submit_training(domain=..., recipe=...)."""
    argv = ["bundle", "--domain", domain]
    if out:
        argv += ["--out", out]
    return _synth_run(argv)


@mcp.tool
def synth_status(job: str) -> dict:
    """Result of a background synth job (synth_derive). Also lists live
    jobs when job is empty."""
    if not job:
        return {"jobs": {k: v.get("state") for k, v in _SYNTH_JOBS.items()}}
    return {"job": job, **_SYNTH_JOBS.get(job, {"state": "unknown"})}


@mcp.tool
def synth_sample(url: str, model: str = "kandi", k: int = 4, limit: int = 50,
                 task: str | None = None, dataset: str | None = None,
                 temperature: float = 0.8, max_tokens: int = 512) -> dict:
    """R4 student-side RFT: prompts from derived templates, completions
    from a served OpenAI-compatible endpoint, kept only if the fleet
    oracle accepts them (check_record per task family). The recursion
    half: student output becomes data only through the verifier."""
    return _synth_run(["sample", "--url", url, "--model", model,
                       "--k", str(k), "--limit", str(limit),
                       "--temperature", str(temperature),
                       "--max-tokens", str(max_tokens)]
                      + (["--task", task] if task else [])
                      + (["--dataset", dataset] if dataset else []))


@mcp.tool
def synth_mix(sources: str, name: str, rows: int = 4000, seed: int = 7) -> dict:
    """R5 data-mixing regression primitive: weighted merge of PREPARED
    datasets (run prepare-domain per source) into datasets/<name>/ with a
    seeded, content-addressed manifest. The mixing-law experiment tool."""
    return _synth_run(["mix", "--sources", sources, "--rows", str(rows),
                       "--seed", str(seed), "--name", name])


@mcp.tool
def synth_grade(path: str, out: str | None = None) -> dict:
    """R7-ready RLVR reward oracle: grade a jsonl of {messages} rows
    against the derived templates' fleet contract — (s, r) pairs WITHOUT
    model weights. This is the exact reward function the GRPO loop will
    reuse."""
    argv = ["grade", path]
    if out:
        argv += ["--out", out]
    return _synth_run(argv)


@mcp.tool
def synth_gaps(bench_path: str) -> dict:
    """P7/R10 gap analysis: failed bench questions -> targeted synthesis
    proposals (exact bin/synth commands). The outer loop's propose step."""
    return _synth_run(["gaps", "--bench", bench_path])


@mcp.tool
def synth_gbnf(out: str | None = None) -> dict:
    """R3: emit the Graph IR GBNF grammar (one rule per line, llama.cpp/
    vLLM compatible) for guided decoding. Syntax becomes a decode-time
    guarantee; serve with --guided-decoding-backend or sampling_params
    extra_body guided_grammar."""
    return _synth_run(["gbnf", "--out", out] if out else ["gbnf"])


@mcp.tool
def synth_merge(base: str, adapters: str, method: str = "ties",
                density: float = 0.5) -> dict:
    """R6 merge-algebra config generator (mergekit: ties/dare_ties/
    slerp): compose persona x domain x cognition LoRA tracks into ONE
    merged adapter = single-LoRA serving. Returns the config; the
    execution rides the stage/pod lane."""
    return _synth_run(["merge", "--base", base, "--adapters", adapters,
                       "--method", method, "--density", str(density)])


@mcp.tool
def synth_contrasts(domain: str = "queer", behaviors: str | None = None,
                    limit: int = 400) -> dict:
    """R14 contrast-set factory (ORACLES.md §2): labeled doing/not-doing
    pairs from verified artifacts (ir-valid from DPO couples, abstain from
    refusal rows) and persona datasets via bench-grammar checks
    (self-name/emoji/json). This is the fuel the ablate/steering stage
    consumes — behaviors to remove, add, or dial."""
    argv = ["contrasts", "--domain", domain, "--limit", str(limit)]
    if behaviors:
        argv += ["--behaviors", behaviors]
    return _synth_run(argv)


# ------------------------------------------------------------------ serve

def build_app():
    app = mcp.http_app(path="/mcp")
    # gate ALWAYS wraps: refuse-all is the safe default without a token
    app.add_middleware(BearerGate)
    return app


# ---------------------------------------------------------------------------
# POD-STOP KILL ROUTE (deadman cost guard; SPEC 13an + r2 postmortem).
# RunPod's CreatePodRequest has NO restartPolicy (OpenAPI-verified): an
# exited container is REVIVED, so a deadman fire must end in a real pod
# kill. The pod may not hold the scoped RunPod key (it lives HERE), so
# dispatch hands each pod a per-run capability token:
#     token = sha256(f"{run_id}|{MCP_TRAINER_TOKEN}")[:32]
# Possibility of killing = knowledge of token + global secret; the token
# binds to exactly one run (run_id is enumerable, NOT a credential; the
# secret never leaves the orchestrator/server envs). BearerGate lets the
# /pod-stop/ prefix reach THIS handler, which authenticates fail-closed:
# no MCP_TRAINER_TOKEN set -> every call 401s (refuse-all chassis law).
# Cross-controller honest: the server ledger is pod-local, so the kill
# resolves by POD NAME prefix (trainer-<run8>-*, the same convention
# doctor resolves by suffix) — a workstation-launched run is killable
# without its row existing in this ledger.

import hashlib as _hashlib
import hmac as _hmac

from starlette.responses import JSONResponse as _JSONResponse

# SINGLE SOURCE: dispatch owns the per-run kill-token derivation
# (workstation mints via _pod_stop_env, this route verifies) — drift
# between mint and check silently disarms the deadman kill, which is
# exactly the r2 failure mode this route exists to end.
pod_stop_token = dispatch.pod_stop_token


async def _pod_stop(request):
    run_id = (request.path_params.get("run_id") or "").strip()
    if len(run_id) < 15:                     # 20261127-150613-d2886a shape
        return _JSONResponse({"error": "bad run id"}, status_code=400)
    got = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    want = pod_stop_token(run_id)
    if not want or not _hmac.compare_digest(got or "", want):
        return _JSONResponse({"error": "unauthorized"}, status_code=401)
    # ledger is corroboration only (pod-local DB — a workstation-
    # launched row may not exist here at all; the TOKEN is the claim)
    row = next((r for r in _led().list() if r["run_id"] == run_id), None)
    if row is not None and row["state"] not in dispatch.OPEN_STATES \
            and not row.get("instance_id"):
        return _JSONResponse({"stopped": None,
                              "note": f"already terminal ({row['state']})"})
    provider = (row or {}).get("provider") or "runpod"
    # _pod_name contract (SPEC 14b): names END with run_id[-8:] (kind
    # prefix + hint before it) — reap/doctor map by that tail, so this
    # route must too. startswith on run_id[:8] matches NOTHING.
    tail = run_id[-8:]
    killed = []
    try:
        insts = get_provider(provider).list_instances()
    except Exception as e:                   # noqa: BLE001
        return _JSONResponse({"error": f"list failed: {str(e)[:120]}"},
                             status_code=502)
    for i in insts:
        if (i.name or "").endswith(tail):
            try:
                if get_provider(provider).terminate(i.id):
                    killed.append(i.id)
            except Exception as e:           # noqa: BLE001
                return _JSONResponse(
                    {"error": f"terminate {i.id}: {str(e)[:120]}"},
                    status_code=502)
    print(f"pod-stop: run {run_id} -> killed {killed or 'nothing live'}",
          flush=True)
    notify.emit("pod_stopped", run_id, killed=killed, source="pod-stop-route")
    # idempotent: no live pod is a SUCCESS for a pod that wants to die now
    return _JSONResponse({"stopped": bool(killed) or None,
                          "killed": killed,
                          "note": "" if killed else "no live pod matched"})


mcp.custom_route("/pod-stop/{run_id}", methods=["POST"])(_pod_stop)



def _port() -> int:
    raw = os.environ.get("MCP_TRAINER_PORT", "3022")
    try:
        return int(raw)
    except ValueError:      # e.g. k8s service-link tcp://... if re-enabled
        return 3022


def _janitor_loop() -> None:
    """Periodic reap+relaunch sweep for controllers that own the lane.
    OFF unless QTRAIN_JANITOR=1: the server ledger is pod-local, so
    enabling it is a declaration that this process is THE janitor."""
    interval = int(os.environ.get("QTRAIN_JANITOR_INTERVAL", "900"))
    while True:
        try:
            out = dispatch.janitor_once(dry=False)
            interesting = {k: out.get(k) for k in ("reaped", "relaunched",
                                                   "healed")
                           if out.get(k)}
            if interesting:
                print("qtrain-janitor "
                      + json.dumps(interesting, default=str)[:600],
                      flush=True)
        except Exception as e:                              # noqa: BLE001
            print(f"qtrain-janitor error: {e}", flush=True)
        time.sleep(interval)


def main(argv=None) -> int:
    import uvicorn
    load_dotenv()
    # events -> container log as machine-parseable lines: `kubectl logs -f
    # mcp-trainer` becomes a live feed for humans and cluster tooling
    notify.register_listener(
        lambda e: print("qtrain-event "
                        + json.dumps(e, default=str), flush=True))
    if not os.environ.get("MCP_TRAINER_TOKEN"):
        print("WARNING: MCP_TRAINER_TOKEN unset — REFUSING all tool traffic "
              "except /healthz (set MCP_TRAINER_ALLOW_OPEN=1 for local dev)",
              flush=True)
    port = _port()
    n = resume_open_runs()
    if n:
        print(f"mcp-trainer: {n} open run(s) re-attached", flush=True)
    if os.environ.get("QTRAIN_JANITOR", "") == "1":
        threading.Thread(target=_janitor_loop, daemon=True,
                         name="janitor").start()
        print("mcp-trainer: janitor loop active "
              f"(every {os.environ.get('QTRAIN_JANITOR_INTERVAL', '900')}s)",
              flush=True)
    print(f"mcp-trainer {__version__} listening :{port} (/mcp, /healthz)",
          flush=True)
    uvicorn.run(build_app(), host="0.0.0.0", port=port, log_level="info")
    return 0
