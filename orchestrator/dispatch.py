"""Job dispatch loop (SPEC §4/§6, P1): expand -> upload -> launch -> watch.

Two storage lanes, chosen per submit (SPEC §13 g):
  volume   — RunPod network volume mounted into the pod at RUNPOD_VOLUME_MOUNT
             (/data). Pod writes files, orchestrator reads them via the
             per-DC S3 API. Zero pod credentials, zero public cluster route,
             zero presigning. Manifest appearing == job finished.
  presign  — rustfs + presigned URLs: the portable lane for providers
             without volumes (gpuai, local). Same push semantics.

In both lanes the orchestrator learns completion by polling
runs/<id>/manifest.json; the return lane IS the ledger.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml

from . import budget, families, notify, progress, storage
from .config import ROOT, corpora_dir, corpora_git_sha, resolve_recipe
from .ledger import Ledger
from .models import POD_PREFIX, ProviderError, pod_is_ours
from .providers import get_provider

MINI_LINES = 48


def _tolerant_get(get_run, key: str, state: dict, limit: int = 5):
    """Lane reads during the watch loop can blip (network, provider API);
    tolerate a few consecutive failures before declaring the lane dead.
    `state` tracks the streak across loop iterations."""
    try:
        data = get_run(key)
        state["errors"] = 0
        return data
    except Exception:
        state["errors"] = state.get("errors", 0) + 1
        if state["errors"] >= limit:
            raise
        return None


def _progress_tick(run_id: str, get_run, put_run, price_hr, started: float,
                   memo: dict) -> None:
    """wandb-lite (SPEC 14): join the live heartbeat into a durable history
    row (progress.jsonl — S3 has no append, so read-modify-write a file
    that stays KB-small), and emit progress events on state change or a
    >=10% step jump. Everything here is best-effort BY DESIGN: a broken
    history sink must never affect the run; the heartbeat + manifest
    remain the authoritative signals."""
    sstate = memo.setdefault("status", {})
    raw = _tolerant_get(get_run, f"runs/{run_id}/status.json", sstate)
    if raw is None:
        return
    try:
        st = json.loads(raw.decode() if isinstance(raw, (bytes, bytearray))
                        else raw)
    except (ValueError, UnicodeDecodeError):
        return
    if not isinstance(st, dict):
        return
    row = {"ts": round(time.time(), 3)}
    for k in ("state", "step", "total", "loss", "epoch"):
        if st.get(k) is not None:
            row[k] = st[k]
    if price_hr and started:
        row["cost"] = round(price_hr * (time.time() - started) / 3600, 4)
    sig = (row.get("state"), row.get("step"), row.get("cost"))
    if sig == memo.get("sig"):
        return
    memo["sig"] = sig
    if put_run:
        try:
            key = f"runs/{run_id}/progress.jsonl"
            prev = get_run(key) or b""
            if isinstance(prev, str):
                prev = prev.encode()
            put_run(key, prev + (json.dumps(row) + "\n").encode())
        except Exception:                              # noqa: BLE001
            pass
    pct = (100.0 * row["step"] / row["total"]) if row.get("total") else None
    prev_pct = memo.get("pct")
    jumped = pct is not None and (prev_pct is None or abs(pct - prev_pct) >= 10)
    if row.get("state") != memo.get("state") or jumped:
        memo["pct"], memo["state"] = pct, row.get("state")
        notify.emit("progress", run_id,
                    pct=round(pct, 1) if pct is not None else None,
                    **{k: v for k, v in row.items() if k != "ts"})


def _index_write(run_id: str, get_run, put_run, *, cost, price_hr,
                 started, led=None) -> None:
    """Append this run's flat rollup to runs/index.jsonl on the lane
    (SPEC 14c). Manifest + run.log are re-read from the lane (they were
    just written; cheap and lane-agnostic) so the record works for harvests
    from ANY controller — the local sqlite stays orchestrator-private, the
    index is the shared UI view. Best-effort: index failure never touches
    the run's terminal state."""
    try:
        man_raw = get_run(f"runs/{run_id}/manifest.json")
        if not man_raw:
            return
        man = json.loads(man_raw.decode() if isinstance(
            man_raw, (bytes, bytearray)) else man_raw)
        row = led.get(run_id) if led else None
        log = get_run(f"runs/{run_id}/run.log") or b""
        if isinstance(log, (bytes, bytearray)):
            log = log.decode(errors="replace")
        progress.append_index(
            progress.build_record(
                run_id, man, row, provider=(row or {}).get("provider")
                or "runpod", seconds=time.time() - started, cost=cost,
                price_hr=price_hr, log_text=log), get_run, put_run)
    except Exception:                              # noqa: BLE001
        pass


def _keepalive_public_lane() -> None:
    """Quick tunnels idle-expire; during training the pod sends no traffic,
    so the orchestrator pings the public endpoint each poll cycle (presign
    lane only; no-op when public == LAN or volume lane)."""
    import httpx
    pub = os.environ.get("S3_PUBLIC_ENDPOINT", "").rstrip("/")
    lan = os.environ.get("S3_ENDPOINT", "").rstrip("/")
    if not pub or pub == lan:
        return
    try:
        httpx.get(f"{pub}/minio/health/live", timeout=10)
    except Exception:
        pass  # keepalive only — the presigned calls surface real errors


def _watch(run_id: str, get_run, result: dict, led: Ledger, poll: int,
           started: float, max_runtime: int, prov=None, inst_id: str = "",
           price_hr: float | None = None, max_cost: float = 0.0,
           keepalive: bool = False, put_run=None) -> dict:
    """Shared watch loop (submit + resume): poll the lane for the manifest,
    tolerate transient blips, fail fast when the instance died without one,
    respect budget + wall-clock ceilings. Returns `result` after harvest;
    raises ProviderError on budget/timeout/run-lost. Side channel (SPEC 14):
    heartbeat -> progress.jsonl history + events; failures are emitted by
    the CALLER's except (single choke point — no double events)."""
    lane_state: dict = {}
    memo: dict = {}
    gone = 0               # consecutive DEFINITE-absence cycles (f831b7)
    deadline = started + max_runtime
    inst = result.get("instance")
    inst_fields = dict(inst) if isinstance(inst, dict) else (
        {"instance": inst} if inst else {})
    notify.emit("launched", run_id, job=result.get("kind") or "train",
                lane=result.get("lane", ""), **inst_fields,
                max_cost=round(max_cost, 4) if max_cost else None)
    while time.time() < deadline:
        if max_cost and price_hr:
            hours = (time.time() - started) / 3600
            if hours * price_hr > max_cost:
                raise ProviderError(f"budget cap ${max_cost} crossed")
        if keepalive:
            _keepalive_public_lane()
        if put_run:
            # DEADMAN PING (SPEC 13an): every cycle rewrites a fresh token
            # the pod-side watchdog reads beside its own heartbeat mirror.
            # Content-change protocol — no clock ever compared across
            # machines. Best-effort: a lane-blip write failure just lets
            # the pod age toward grace; the WATCHER surviving blips is
            # what the poll loop already handles.
            try:
                put_run(f"runs/{run_id}/ping",
                        f"{time.time():.6f}".encode())
            except Exception:                               # noqa: BLE001
                pass
        manifest_raw = _tolerant_get(get_run, f"runs/{run_id}/manifest.json",
                                     lane_state)
        if manifest_raw:
            _harvest(run_id, manifest_raw, get_run, led, result)
            _finish_done(run_id, started, price_hr, get_run, put_run,
                         kind=result.get("kind") or "",
                         status=result.get("status") or "succeeded",
                         led=led)
            return result
        if prov and inst_id:
            try:
                st = prov.poll(inst_id).status
            except Exception as e:
                # DEFINITE ABSENCE is not a blip (f831b7: killed pod,
                # watch re-checked the corpse until max_runtime). The
                # pod may be briefly missing during creation — same
                # boot-change discipline as the deadman — so a DEFINITE
                # not-found must PERSIST two cycles to count as death;
                # NET_MARKERS blips are never death.
                if _gone_says_absent(e) and not _net_blip(e):
                    gone += 1
                    st = "terminated" if gone >= 2 else ""
                else:
                    st = ""          # provider API blip: re-check next cycle
            else:
                gone = 0
            if st in {"terminated", "deleted", "failed", "error", "stopped",
                      "exited"} or st == "" and gone >= 2:
                time.sleep(poll)  # grace: agent may still be flushing
                manifest_raw = get_run(f"runs/{run_id}/manifest.json")
                if manifest_raw:
                    _harvest(run_id, manifest_raw, get_run, led, result)
                    _finish_done(run_id, started, price_hr, get_run, put_run,
                         kind=result.get("kind") or "",
                         status=result.get("status") or "succeeded",
                         led=led)
                    return result
                raise ProviderError(
                    f"instance {inst_id} {st} without manifest — run lost")
        _progress_tick(run_id, get_run, put_run, price_hr, started, memo)
        time.sleep(poll)
    raise ProviderError("max runtime exceeded without manifest")


def _finish_done(run_id: str, started: float, price_hr, get_run=None,
                 put_run=None, kind: str = "", status: str = "succeeded",
                 led=None) -> None:
    """Terminal signal for the progress channel: final history row (best
    effort) + a run_done or run_gated event. Gate-FAILED harvests are NOT
    'done' — they delivered a manifest whose artifact gate refused, so
    they must not train the self-calibration stats either (a failed run's
    duration is not the duration of success)."""
    cost = (round(price_hr * (time.time() - started) / 3600, 4)
            if price_hr else None)
    if get_run and put_run:
        try:
            key = f"runs/{run_id}/progress.jsonl"
            prev = get_run(key) or b""
            if isinstance(prev, str):
                prev = prev.encode()
            put_run(key, prev + (json.dumps(
                {"ts": round(time.time(), 3), "state": status,
                 "cost": cost}) + "\n").encode())
        except Exception:                              # noqa: BLE001
            pass
    if get_run and put_run:
        _index_write(run_id, get_run, put_run, cost=cost,
                     price_hr=price_hr, started=started, led=led)
    if status == "succeeded":
        if kind:
            budget.record_observed(kind, (time.time() - started) / 60)
        notify.emit("run_done", run_id, cost=cost,
                    seconds=round(time.time() - started))
    else:
        notify.emit("run_gated", run_id, status=status, cost=cost,
                    seconds=round(time.time() - started))


JOB_KINDS = ("train", "merge", "onnx", "int8", "awq", "ablate",
             "holdout")   # noqa: F841 re-export


def _pod_hint(domain=None, recipe=None, parent=None) -> str:
    """Human tag for pod names: domain + model slug. The slug prefers the
    recipe's INTERNAL `base_model` (Qwen/Qwen3.5-0.8B -> qwen3.5-0.8b —
    reads like the thing it trains) over the filename; filename stem is
    the fallback. Stages tag their parent run (`for<token>`)."""
    import yaml
    bits = []
    if domain:
        bits.append(re.sub(r"[^a-z0-9.-]+", "", str(domain).lower())[:12])
    if recipe:
        slug = ""
        try:
            bm = str((yaml.safe_load(Path(str(recipe)).read_text())
                      or {}).get("base_model", ""))
            slug = bm.rsplit("/", 1)[-1]
        except Exception:                              # noqa: BLE001
            pass
        if not slug:
            slug = re.sub(r"[-_ ]?qlora\b", "",
                          Path(str(recipe)).stem, flags=re.I)
        slug = re.sub(r"[^a-z0-9.-]+", "", slug.lower())[:24].strip("-.")
        if slug and slug not in bits:
            bits.append(slug)
    if parent:
        tok = re.sub(r"[^a-z0-9]+", "", str(parent).split("-")[-1].lower())[:8]
        if tok:
            bits.append(f"for{tok}")
    return "-".join(b for b in bits if b)


def _pod_name(kind: str, run_id: str, hint: str = "") -> str:
    """CONTRACT (SPEC 14b): starts with the kind (legacy trainer- names
    remain reapable) and ENDS with run_id[-8:] — reap maps pods to ledger
    rows by that trailing token, so hint parts must never occupy the
    tail. Total length stays k8s-safe (<= 63)."""
    tail = f"{run_id[-8:]}"
    return "-".join(p for p in (kind.lower(), hint.lower().strip("-."),
                                tail) if p)[:63].rstrip("-.")


# ---------------------------------------------------------------------------
# POD-STOP WIRING (r2 postmortem / REGIMES handoff item 1). RunPod's
# CreatePodRequest has NO restartPolicy: an exited container is REVIVED,
# so deadman's finalize+exit must be followed by a real kill. The pod may
# not hold the scoped RunPod key (lives HERE), so dispatch hands it a
# per-run capability token and the mcp server exposes /pod-stop/{run_id}
# that checks it (single source: pod_stop_token — mint/check drift means
# the deadman's kill 401s, i.e. the exact r2 failure mode, silently).
# INERT BY DEFAULT: without QTRAIN_POD_STOP_URL_BASE + MCP_TRAINER_TOKEN
# in the controller env, no pod-kill env is set and the agent keeps
# logging 'exit alone will NOT stop billing' (v0.5.0 law). The base URL
# must be reachable FROM the provider pod and point at an mcp-trainer
# (>= m-v0.5.2) that carries the route — an infra-plane decision, not a
# dispatch assumption.

def pod_stop_token(run_id: str, secret: str | None = None) -> str:
    """sha256(run_id|secret)[:32]; empty secret/env -> empty token. The
    run_id is enumerable, never a credential — the secret binding is
    what makes this a kill CAPABILITY scoped to exactly one run."""
    s = (os.environ.get("MCP_TRAINER_TOKEN", "") if secret is None
         else secret)
    if not s:
        return ""
    return hashlib.sha256(f"{run_id}|{s}".encode()).hexdigest()[:32]


def _pod_stop_env(run_id: str) -> dict:
    base = os.environ.get("QTRAIN_POD_STOP_URL_BASE", "").rstrip("/")
    tok = pod_stop_token(run_id)
    if not base or not tok:
        return {}
    return {"QTRAIN_POD_STOP_URL": f"{base}/pod-stop/{run_id}",
            "QTRAIN_POD_STOP_TOKEN": tok}

# states whose work is OVER — a pod still alive under one of these is a
# leak (crash between harvest and terminate), never a live task.
SETTLED = {"done", "failed", "cancelled", "verified", "delivered", "lost"}


def _max_wait(explicit: int | None) -> int:
    if explicit is not None:
        return max(0, int(explicit))
    return int(os.environ.get("QTRAIN_MAX_WAIT", "1800"))


def _train_pool(prov, gpus, tier, vol, ceiling):
    """Fresh catalog pass through the train fit: SECURE when the lane
    needs it, then the rate ceiling. Raises the capacity error so
    _wave_launch treats 'nothing fits right now' as retryable stock."""
    offers = prov.catalog(gpus, tier)
    if vol:
        # network volumes exist only in the SECURE cloud; community
        # launches that ask for one 400 as "no instances available"
        offers = [o for o in offers if o.secure]
    within = [o for o in offers if (o.price_hr or 0.0) <= ceiling]
    if not within:
        best = min((o.price_hr for o in offers if o.price_hr), default=None)
        cheap = f"${best:.2f}/hr" if best is not None else "none"
        raise ProviderError(
            f"no capacity across offers within ceiling ${ceiling:.2f}/hr "
            f"(cheapest in stock: {cheap})")
    return within[:8]


def _stage_pool(prov, spec, provider, ceiling):
    """Stage fit (SPEC 13q): SECURE-only for the volume lane, arch
    exclusions, vram floor, rate ceiling — an oversized card is never
    the fallback. Re-evaluated per wave: stock moves."""
    if provider == "local":
        return prov.catalog()
    pool = [o for o in prov.catalog() if o.secure]
    for pat in spec.get("arch_exclude", []):
        pool = [o for o in pool if pat.lower() not in o.gpu_id.lower()]
    ok, best = budget.fits(pool, spec.get("min_vram", 10), ceiling)
    if not ok:
        cheap = f"${best:.2f}/hr" if best is not None else "none"
        raise ProviderError(
            f"no capacity across offers within ceiling ${ceiling:.2f}/hr "
            f"(cheapest vram-fit in stock: {cheap})")
    return ok[:12]


def _wave_launch(prov, offers_fn, *, name, image, env, lk, run_id, kind,
                 max_wait, backoff0=30, backoff_cap=600, sleep=time.sleep):
    """Idempotent GPU acquisition (SPEC 14): try every fitting offer, and
    when stock refuses, BACK OFF and retry against a FRESH catalog until
    the deadline — capacity-waiting lives in the orchestrator, not in
    shell loops around it. max_wait=0 keeps the old fast-fail semantics
    (one wave) for janitor/test use. Returns (instance, offer)."""
    deadline = time.time() + max_wait
    delay = backoff0
    wave = 0
    while True:
        wave += 1
        last_err = None
        try:
            offers = offers_fn()
        except ProviderError as e:
            offers, last_err = [], e
        if not offers and last_err is None:
            raise ProviderError("no capacity across offers: catalog empty")
        for o in offers:
            try:
                inst = prov.launch(name=name, image=image, gpu_id=o.gpu_id,
                                   cloud="SECURE" if o.secure else "COMMUNITY",
                                   env=env, **lk)
                return inst, o
            except ProviderError as e:
                last_err = e
        if time.time() >= deadline:
            raise ProviderError(
                f"no capacity across offers: {last_err} "
                f"(waited {max_wait}s / {wave} wave(s))")
        notify.emit("capacity-wait", run_id, job=kind, wave=wave,
                    retry_in=int(delay), err=str(last_err)[:160])
        sleep(delay)
        delay = min(delay * 2, backoff_cap)


def reap(provider: str = "runpod", *, grace: int | None = None,
         dry: bool = False) -> dict:
    """Idempotent GPU hygiene (SPEC 14): terminate live pods that have NO
    ACTIVE TASK behind them. Rules, in order:

      * only pods OUR naming scheme produced are candidates — kind-first
        (train-/merge-/onnx-/int8-/awq-) or the legacy trainer- prefix;
        other workflows sharing the account are not ours to touch;
      * claimed by THIS ledger with an open state -> KEEP (live run;
        terminating it would destroy paid work);
      * claimed by THIS ledger in a settled state -> leak from a crashed
        watch cycle: terminate;
      * unclaimed HERE but matching a settled run's id suffix in the pod
        name -> stale pod whose claim row moved on: terminate;
      * fully unclaimed -> it may live in ANOTHER controller's ledger
        (mcp-trainer keeps its own). Kept with a reason unless
        QTRAIN_REAP_UNCLAIMED=1 declares this the single controller.

    Grace protects the launch->ledger-write window (a dispatch in flight
    is not a leak)."""
    if grace is None:
        grace = int(os.environ.get("QTRAIN_REAP_GRACE", "900"))
    unclaimed_ok = os.environ.get("QTRAIN_REAP_UNCLAIMED", "") == "1"
    led = Ledger(ROOT / "state" / "ledger.db")
    prov = get_provider(provider)
    claims = {r["instance_id"]: r for r in led.list() if r["instance_id"]}
    by_suffix = {r["run_id"][-8:]: r for r in led.list()}
    killed, kept = [], []
    for inst in prov.list_instances():
        if not inst.alive or not pod_is_ours(inst.name or ""):
            continue
        row = claims.get(inst.id)
        why = ""
        if row is None:
            # identity = trailing run-suffix token (works for hint-bearing
            # names trainer-<kind>.<hint>-<suffix> AND legacy plain ones)
            row2 = by_suffix.get(inst.name[-8:])
            if row2 is not None and row2["state"] in SETTLED:
                row, why = row2, (f"stale pod of settled run "
                                  f"{row2['run_id'][-8:]} "
                                  f"(state={row2['state']})")
            else:
                age = inst.created_epoch
                if age is not None and time.time() - age < grace:
                    kept.append({"id": inst.id, "why": "within launch grace"})
                    continue
                if not unclaimed_ok:
                    kept.append({"id": inst.id, "why": "unclaimed here — "
                                 "another controller may own it "
                                 "(QTRAIN_REAP_UNCLAIMED=1)"})
                    continue
                why = "unclaimed beyond grace"
        elif row["state"] in SETTLED:
            why = f"terminal claim (state={row['state']})"
        else:
            kept.append({"id": inst.id,
                         "why": f"open run {row['run_id'][-8:]} "
                                f"({row['state']})"})
            continue
        item = {"id": inst.id, "gpu": inst.gpu_type, "why": why,
                "run_id": row["run_id"] if row is not None else ""}
        if dry:
            item["dry"] = True
            killed.append(item)
            continue
        try:
            prov.terminate(inst.id)
            killed.append(item)
            notify.emit("reaped", item["run_id"], pod=inst.id, why=why)
        except Exception as e:                              # noqa: BLE001
            item["error"] = str(e)[:160]
            killed.append(item)
    return {"provider": provider, "killed": killed, "kept": kept,
            "grace_s": grace, "unclaimed_kill": unclaimed_ok}


def relaunch(run_id: str, *, poll: int = 20, max_runtime: int = 4 * 3600,
             max_wait: int | None = None) -> dict:
    """Bring the GPU up for a STAGED-BUT-NEVER-LAUNCHED run (crash
    between payload staging and pod launch). Idempotent by construction:
    the lane payload IS the complete job contract — same run_id, prefix,
    gate — so nothing is re-staged; wave-launch + watch, done. Defaults
    to the env capacity-wait: recovery SHOULD wait for stock, because
    fast-failing here flips the row terminal and LOSES the run — exactly
    the non-idempotency this path exists to remove."""
    led = Ledger(ROOT / "state" / "ledger.db")
    row = led.get(run_id)
    result = {"run_id": run_id}
    if row is None:
        return {"status": "error", "error": f"no such run: {run_id}"}
    if row["instance_id"]:
        return {"status": "error", "error": "already has an instance — "
                "use resume, not relaunch"}
    if row["state"] != "pending":
        return {"status": "error", "error": f"state={row['state']}, "
                "not pending — nothing to relaunch"}
    provider = row["provider"] or "runpod"
    spec = json.loads(row["spec_json"] or "{}")
    is_stage = row["kind"] == "stage"
    pod_id = ""
    try:
        if provider == "local":
            mount = os.environ.get("LOCAL_MOUNT", "/data")
            get_run = storage.host_get
            put_run = lambda k, d, ct="": storage.host_put(k, d)  # noqa: E731
            vol = None
            lane = "host"
        elif provider == "runpod":
            vol = storage.vol_cfg()
            if not vol:
                raise ProviderError("volume lane not configured")
            mount = vol["mount"]
            get_run = storage.vol_get
            put_run = lambda k, d, ct="": storage.vol_put(k, d)  # noqa: E731
            lane = "volume"
        else:
            raise ProviderError(f"relaunch: provider {provider!r} "
                                "unsupported")
        raw = get_run(f"runs/{run_id}/payload.json")
        if raw is None:
            raise ProviderError("payload missing on lane — nothing to "
                                "relaunch")
        payload = json.loads(raw)
        meta = payload.get("meta") or {}
        if not payload.get("volume"):
            raise ProviderError("presign-lane payloads cannot relaunch "
                                "(signed URLs expire) — resubmit instead")
        result["lane"] = lane
        kind = (spec.get("kind") if is_stage else "train") or "train"
        if is_stage and kind not in STAGE_SPECS:
            raise ProviderError(f"stage kind {kind!r} unknown")
        token = hashlib.sha256(run_id.encode()).hexdigest()[:24]
        if is_stage:
            s = STAGE_SPECS[kind]
            image = _stage_image(s)
            ceiling = s.get("max_price_hr", 1.50)
            offers_fn = lambda: _stage_pool(prov, s, provider, ceiling)  # noqa: E731
            rate_bound = 0.0 if provider == "local" else ceiling
        else:
            s = None
            image = _image_ref()
            ceiling = float(os.environ.get("QTRAIN_MAX_PRICE_HR", "1.50"))
            offers_fn = lambda: _train_pool(prov, None, None, vol, ceiling)  # noqa: E731
            rate_bound = ceiling
        env = {"JOB_PAYLOAD_PATH":
               f"{payload['volume']['mount']}/runs/{run_id}/payload.json",
               "AGENT_TOKEN": token, "IMAGE_TAG": image,
               "DEADMAN_GRACE_S": os.environ.get("QTRAIN_DEADMAN_GRACE_S",
                                            "600"),
               **_pod_stop_env(run_id)}
        prov = get_provider(provider)
        lk = ({"volume_id": vol["bucket"], "volume_path": vol["mount"],
               "data_center": vol["dc"]} if provider == "runpod" else {})
        result["kind"] = kind
        result["estimate"] = round(budget.estimate(rate_bound, kind), 2)
        notify.emit("relaunching", run_id, kind=kind, lane=lane)
        inst, offer = _wave_launch(prov, offers_fn,
                                   name=_pod_name(
                                       kind, run_id,
                                       (_pod_hint(meta.get("domain"),
                                                  meta.get("recipe"))
                                        if not is_stage else
                                        _pod_hint(parent=meta.get("from_run")))),
                                   image=image, env=env, lk=lk,
                                   run_id=run_id, kind=kind,
                                   max_wait=_max_wait(max_wait))
        pod_id = inst.id
        result["instance"] = {"id": inst.id, "gpu": offer.gpu_id,
                              "price_hr": offer.price_hr}
        mc = (budget.estimate(offer.price_hr or rate_bound, kind) if is_stage
              else (offer.price_hr or 0.0) * max_runtime / 60)
        led.update(run_id, state="provisioning", instance_id=inst.id,
                   base_url=getattr(inst, "base_url", "") or "",
                   est_cost=round(mc, 4), price_hr=offer.price_hr or 0.0)
        _watch(run_id, get_run, result, led, poll, started=time.time(),
               max_runtime=max_runtime, prov=prov, inst_id=inst.id,
               price_hr=offer.price_hr, max_cost=mc, keepalive=False,
               put_run=put_run)
        return result
    except Exception as e:                                  # noqa: BLE001
        if pod_id and _net_blip(e):
            led.update(run_id, error=f"watch lost (network): "
                       f"{str(e)[:140]} — pod kept; `qtrain watch` re-attaches")
            result.update({"status": "unwatched", "error": str(e)[:200],
                           "pod": pod_id})
            notify.emit("watch_lost", run_id, pod=pod_id, kind=kind,
                        relaunch=True, error=str(e)[:160])
        else:
            led.update(run_id, state="failed", error=str(e)[:400])
            result.update({"status": "error", "error": str(e)})
            notify.emit("run_failed", run_id, error=str(e)[:300],
                        relaunch=True)
        return result
    finally:
        if pod_id and not result.get("unwatched"):
            try:
                result["terminated"] = bool(
                    get_provider(provider).terminate(pod_id))
            except Exception as e:                          # noqa: BLE001
                result["terminate_error"] = str(e)


def janitor_once(provider: str = "runpod", *, grace: int | None = None,
                 stale_after: int = 900, dry: bool = False) -> dict:
    """One idempotency sweep (SPEC 14): reap pods with no active task,
    then relaunch stale pending-but-unlaunched runs. Deliberately does
    NOT re-watch open provisioning/running rows — those belong to a live
    dispatch or to boot-resume; a second watcher would race the harvest."""
    out: dict = {"provider": provider}
    try:
        r = reap(provider, grace=grace, dry=dry)
        out["reaped"] = r["killed"]
        out["kept"] = r["kept"]
    except Exception as e:                                  # noqa: BLE001
        out["reap_error"] = str(e)[:200]
    led = Ledger(ROOT / "state" / "ledger.db")
    # A pending row is only a GHOST once its original dispatch can no
    # longer be launching: capacity-waits run up to QTRAIN_MAX_WAIT
    # (default 1800 s). Relaunching inside that window = double launch.
    horizon = max(stale_after,
                  int(os.environ.get("QTRAIN_MAX_WAIT", "1800")) + 120)
    cutoff = time.time() - horizon
    out["relaunched"] = []
    for row in led.list():
        if (row["state"] == "pending" and not row["instance_id"]
                and row["created_at"] < cutoff
                and (row["provider"] or "runpod") == provider):
            if dry:
                out["relaunched"].append({"run_id": row["run_id"],
                                          "dry": True})
                continue
            r2 = relaunch(row["run_id"])
            out["relaunched"].append({"run_id": row["run_id"],
                                      "status": r2.get("status")
                                      or "watched to completion"})
    return out


def _sha(path: Path) -> str:
    return storage.sha256_file(path)


def prepare_dataset(domain: str, mini: bool = False,
                    vol: dict | None = None) -> tuple[str, Path, str, int, dict | None]:
    """Run corpus expansion (via prepare-domain), upload to the lane, return
    (key, local_path, sha256, lines, holdout|None). vol=None -> presign lane;
    vol dict -> volume lane; vol["local"] set -> host lane.

    HOLDOUT LAW (SPEC 13am): prepare-domain content-hashes every row into
    train/~98% + holdout/~2% — stable across rounds, so cross-run eval
    trends compare apples to apples. The holdout trains on NOTHING and
    rides the payload as a sha-verified input feeding axolotl's
    eval_dataset."""
    subprocess.run([sys.executable, str(ROOT / "bin/prepare-domain"),
                    domain], check=True, capture_output=True, text=True)
    src = ROOT / "tmp" / f"{domain}.jsonl"
    if not src.is_file():
        raise ProviderError(f"prepare-domain produced no {src}")
    lines = 0
    if mini:
        all_lines = src.read_text().splitlines()[:MINI_LINES]
        mini_path = src.with_name(f"{domain}.mini.jsonl")
        mini_path.write_text("\n".join(all_lines) + "\n")
        src = mini_path
        lines = len(all_lines)
    sha = _sha(src)
    if not lines:
        with open(src, "rb") as f:
            lines = sum(1 for _ in f)
    key = f"datasets/{domain}/{sha[:16]}/{src.name}"
    data = src.read_bytes()
    if vol and vol.get("local"):
        storage.host_put(key, data)
    elif vol:
        storage.vol_put(key, data)
    else:
        storage.ensure_bucket()
        storage.put_object(key, data, "application/x-ndjson")
    hold = None
    hpath = ROOT / "tmp" / f"{domain}.holdout.jsonl"
    if hpath.is_file() and hpath.stat().st_size:
        hsha = _sha(hpath)
        hkey = f"datasets/{domain}/{hsha[:16]}/{hpath.name}"
        hdata = hpath.read_bytes()
        if vol and vol.get("local"):
            storage.host_put(hkey, hdata)
        elif vol:
            storage.vol_put(hkey, hdata)
        else:
            storage.put_object(hkey, hdata, "application/x-ndjson")
        hold = {"key": hkey, "sha256": hsha, "name": hpath.name,
                "lines": sum(1 for _ in open(hpath, "rb"))}
    return key, src, sha, lines, hold


def render_config(recipe: Path, overrides: dict, dataset_name: str,
                  slug: str | None = None,
                  report: dict | None = None) -> str:
    """Render a recipe into a pod-ready config. slug (co-tenant pair mode)
    isolates BOTH output_dir and dataset_prepared_path per model — two
    concurrent axolotl procs sharing a prepared dir race HF fingerprint
    writes; two models sharing an output dir silently overwrite saves and
    the gate would lie about what exists.

    TRANSPARENCY (2026-11-27 assumption audit): every soft-forced key the
    recipe declared differently is RECORDED in report["forced"] — a
    recipe that lies about what runs is how the packaging bug survived
    months. And LAW_KEYS are re-applied AFTER overrides: an operator knob
    must not silently break the holdout law; the clash is recorded, not
    hidden (refusing the whole submit stays possible at the caller)."""
    cfg = yaml.safe_load(recipe.read_text())
    name = f"/work/datasets/{dataset_name}"
    if cfg.get("pref_datasets"):
        # preference regimes (DPO/ORPO/SimPO, REGIMES R1): the domain file
        # IS the preference set (synth bundle writes datasets/<domain>-dpo
        # exclusively) — mount it into the pref slot, never invent an SFT
        # datasets entry (axolotl reads chosen/rejected fields).
        cfg["pref_datasets"][0]["path"] = name
        cfg["pref_datasets"] = [cfg["pref_datasets"][0]]
        cfg.pop("datasets", None)
    elif cfg.get("datasets"):
        cfg["datasets"][0]["path"] = name
        cfg["datasets"] = [cfg["datasets"][0]]
    else:
        cfg["datasets"] = [{"path": name, "type": "chat_template",
                            "ds_type": "json", "field_messages": "messages",
                            "roles_to_train": ["assistant"]}]
    # HOLDOUT LAW (SPEC 13am) — in-train eval is NOT this pipeline's
    # mechanism: this axolotl build REJECTS eval_strategy at
    # val_set_size == 0 (3e7232's pydantic refusal), and its random val
    # split would violate the content-hash stability law anyway. The
    # stable holdout rides the payload as a sha-verified input and is
    # scored POST-TRAIN by the `holdout` stage, against the same adapter
    # bytes the engine will serve.
    # soft forces the lane needs (heartbeat cadence / disk economics): the
    # recipe's declared value is RECORDED as overridden, not hidden
    soft_forced = {"logging_steps": 5, "save_total_limit": 1}
    declared = {k: cfg[k] for k in soft_forced if k in cfg}
    cfg.update({
        "dataset_prepared_path": (f"/work/prepared/{slug}" if slug
                                  else "/work/prepared"),
        "output_dir": (f"/work/outputs/{slug}" if slug else "/work/outputs"),
        "val_set_size": 0.0,
        "bf16": cfg.get("bf16", "auto"),
        **soft_forced,
    })
    cfg.update(overrides)
    # LAW_KEY: val_set_size stays 0 even against overrides — an operator
    # knob tunes preferences; it does not vote the holdout law out
    # (13am stability beats convenience; the clash is on the record)
    if cfg.get("val_set_size") not in (0, 0.0):
        if report is not None:
            report.setdefault("law_clash", []).append(
                f"val_set_size={cfg['val_set_size']!r} overridden to 0.0 "
                "(holdout law 13am: random val split breaks content-hash "
                "stability; use the holdout STAGE for eval)")
        cfg["val_set_size"] = 0.0
    if report is not None and declared:
        report["forced"] = {k: {"declared": v, "forced": cfg[k]}
                            for k, v in declared.items() if cfg[k] != v}
    return yaml.safe_dump(cfg, sort_keys=False)


def _harvest(run_id: str, manifest_raw: bytes, get_run, led: Ledger,
             result: dict) -> None:
    """Terminal-lane logic shared by submit() and resume(): verify, store,
    ledger. Raises on artifact corruption (state stays failed-able)."""
    manifest = json.loads(manifest_raw)
    ok = manifest.get("status") == "succeeded"
    local = ROOT / "runs" / run_id
    local.mkdir(parents=True, exist_ok=True)
    (local / "manifest.json").write_bytes(manifest_raw)
    if ok:
        tar = get_run(f"runs/{run_id}/artifacts.tar.gz")
        assert tar is not None
        (local / "artifacts.tar.gz").write_bytes(tar)
        log = get_run(f"runs/{run_id}/run.log")   # forensic trail
        if log:
            (local / "run.log").write_bytes(log)
        got = hashlib.sha256(tar).hexdigest()
        want = str(manifest.get("artifacts_tar_sha256", "")).removeprefix("sha256:")
        if want and got != want:
            raise ProviderError(f"artifact sha mismatch: {got} != {want}")
        led.update(run_id, state="done",
                   artifact_uri=f"runs/{run_id}/artifacts.tar.gz")
        result.update({"status": "succeeded", "local": str(local),
                       "artifacts": len(manifest.get("artifacts", {}))})
    else:
        log = get_run(f"runs/{run_id}/run.log") or b""
        (local / "run.log").write_bytes(log[-20000:])
        led.update(run_id, state="failed",
                   error=str(manifest.get("problems"))[:400])
        result.update({"status": "failed", "problems": manifest.get("problems")})


OPEN_STATES = ("pending", "verified", "provisioning", "running", "uploading")

# DNS/connectivity blips on the control box are a recurring environmental
# fact (SPEC 13ac). They are the CONTROLLER's problem, not the job's:
# a volume-lane pod needs zero egress to finish training. Watch-death by
# network must never terminate a running pod — the money law kills pods
# WITHOUT active tasks, and a training pod HAS one.
NET_MARKERS = ("Temporary failure in name resolution",
               "Name or service not known",
               "[Errno -3]", "[Errno -2]", "[Errno -101]", "[Errno -110]",
               "Network is unreachable", "timed out", "URLError",
               "Connection aborted")


def _net_blip(e: BaseException) -> bool:
    return any(m in str(e) for m in NET_MARKERS)


def _gone_says_absent(e: BaseException) -> bool:
    """Definite absence per the provider's documented law (terminate():
    '404 anywhere = gone'). Distinct from a NET blip — a corpse must run
    the death path, not be re-checked until max_runtime (f831b7)."""
    s = str(e).lower()
    return "404" in s or "not found" in s


def resume(run_id: str | None = None, poll: int = 30,
           max_runtime: int = 4 * 3600) -> dict:
    """Re-attach to an open (non-terminal) run: resume lane polling,
    harvest, terminate the instance — EXCEPT on a controller-side network
    blip, which keeps the pod (unwatched) for a later re-attach. Recovery
    path for crashed orchestrators / fleet restarts mid-job (2026-11-23
    incident)."""
    led = Ledger(ROOT / "state" / "ledger.db")
    # 'verified' + no instance = DRY run (staged, never launched): nothing
    # to re-attach to; polling its lane would burn max_runtime for $0 work
    rows = [r for r in led.list() if r["state"] in OPEN_STATES
            and not (r["state"] == "verified" and not r["instance_id"])]
    if run_id:
        exact = [r for r in rows if r["run_id"] == run_id]
        # suffix ids are the house convention everywhere else (verify/
        # register resolve them, doctor matches pods by name suffix);
        # resume silently answering "no open runs" to a suffix of a LIVE
        # run is a no-op wearing success (f831b7 tripped it)
        rows = exact or [r for r in rows if r["run_id"].endswith(run_id)]
    if not rows:
        if run_id:
            return {"resumed": None, "run_id": run_id,
                    "error": f"no OPEN run matching {run_id!r} — `qtrain "
                             "runs` shows states; terminal rows are not "
                             "resumable"}
        return {"resumed": None, "note": "no open runs"}
    row = rows[-1]
    run_id = row["run_id"]
    provider = row["provider"] or "runpod"
    inst_id = row["instance_id"]
    if row["state"] == "pending" and not inst_id:
        # staged but never launched (crash between staging and launch):
        # relaunch IS the resume path for this shape (SPEC 14)
        return relaunch(run_id, poll=poll, max_runtime=max_runtime)
    result = {"run_id": run_id, "resumed_from": row["state"],
              "kind": row["kind"], "instance": inst_id}
    if provider == "local":
        get_run = storage.host_get
        put_run = lambda k, d, ct="": storage.host_put(k, d)
    else:
        vol = storage.vol_cfg() if provider == "runpod" else None
        get_run = (lambda k: storage.vol_get(k)) if vol else storage.get_object
        put_run = ((lambda k, d, ct="": storage.vol_put(k, d)) if vol
                   else (lambda k, d, ct="": storage.put_object(k, d, ct)))
    result["lane"] = "volume" if (provider == "runpod" and vol) else provider
    try:
        prov = get_provider(provider) if inst_id else None
        # DEADLINE RIDES THE ROW (485a22 law): anchor elapsed to the ROW'S
        # creation and use the budget recorded at submit (fallback: the
        # caller's default). time.time() here made every re-attach reset
        # the clock — a supervisor looping resume() never reaches the
        # ceiling, and the pod bills forever (the 90-min zombie holdout
        # was held alive by exactly this reset). max_runtime is a POD
        # lifetime, not a watcher's viewing time.
        _watch(run_id, get_run, result, led, poll,
               started=row["created_at"],
               max_runtime=int(json.loads(row["spec_json"] or "{}")
                               .get("max_runtime") or max_runtime),
               prov=prov, inst_id=inst_id, put_run=put_run)
        return result
    except Exception as e:
        if inst_id and _net_blip(e):
            led.update(run_id, error=f"watch lost (network): "
                       f"{str(e)[:140]} — pod kept; `qtrain watch` re-attaches")
            result.update({"status": "unwatched", "error": str(e)[:200],
                           "pod": inst_id})
            notify.emit("watch_lost", run_id, pod=inst_id,
                        kind=result.get("kind") or "train",
                        error=str(e)[:160])
        else:
            led.update(run_id, state="failed", error=str(e)[:400])
            result.update({"status": "error", "error": str(e)})
            notify.emit("run_failed", run_id, error=str(e)[:300])
        return result
    finally:
        if inst_id and not result.get("unwatched"):
            try:
                result["terminated"] = bool(
                    get_provider(provider).terminate(inst_id))
            except Exception as e:
                result["terminate_error"] = str(e)


# artifact gate law: recipes WITHOUT `adapter:` are full fine-tunes — the
# produced bytes are model*.safetensors shards + config, not an adapter
# pair. Agent finalize defaults (REQUIRED_FILES) are adapter-shaped, so a
# successful FFT round would die at the gate without this override.
_FULL_GATE = ["config.json", "model*.safetensors",
              "tokenizer.json", "tokenizer_config.json"]
_ADAPTER_GATE = ["adapter_config.json", "adapter_model.safetensors",
                 "tokenizer.json", "tokenizer_config.json"]


def _gate_required(recipe: Path, slug: str = "") -> list[str]:
    base = (_ADAPTER_GATE if (yaml.safe_load(recipe.read_text()) or {})
            .get("adapter") else _FULL_GATE)
    return [f"{slug}/{f}" for f in base] if slug else list(base)


def _cfg_slug(recipe: Path) -> str:
    """Filename-safe per-model slug for pair mode (co-tenant outputs)."""
    hint = _pod_hint(recipe=recipe) or recipe.stem
    return re.sub(r"[^a-z0-9._-]+", "-", hint.lower()).strip("-")


# ---------------------------------------------------------------------------
# GUIDANCE vs LAW (operator principle, 2026-11-27): recipes carry PRIORS
# (r=32, lr=1e-4, adamw_bnb_8bit, epochs=2 — tuned defaults that travel via
# copy, tunable per task via `overrides`, enforced by NOTHING). Refusals
# exist ONLY for footguns with burned evidence, and every refusal names
# the run that proved it + the bypass code. Silent forcing is a bug: what
# render_config overrides is now RECORDED on the ledger row.
GUARD_BYPASS = "pass force_guards=[code] to override a refusal"

KNOB_GUIDE = {
    "lora_r": "PRIOR not law: 32 is corpus convention (works 360M-4B). "
              "Scale to task capacity: format/persona skills often "
              "survive 8-16 (cheaper serving, less overfit); broad "
              "behavior may want 64. Nothing enforces 32.",
    "lora_alpha": "PRIOR: 2x r convention. r/alpha ratio moves effective "
                  "LR more than either alone — tune the ratio, not ritual.",
    "learning_rate": "PRIOR 1e-4 (lora/qlora), 2e-5 (FFT: full weights "
                     "cook at LoRA rates). Advisory refusal only for FFT "
                     ">1e-4; otherwise per-task judgement.",
    "micro_batch_size": "SPLIT CATEGORY: qwen3.5 family >1 = REFUSED on "
                        "BURNED evidence (425e6c step-0 death via fla "
                        "cu_seqlens; c4b9eb/8250dc re-proved via "
                        "overrides). Other families: free (smollm2 runs 2).",
    "num_epochs": "PRIOR 2; preference/FFT regimes often want 1 "
                  "(cognition dpo note: overfits fast). Holdout law is "
                  "what catches overtraining — not this number.",
    "optimizer": "PRIOR adamw_bnb_8bit (fleet-wide, memory-cheap, no "
                 "known harm). adamw_torch fine for small LoRAs.",
    "sequence_len": "TASK-DERIVED, never tradition: rows + template "
                    "render (incl. tools schemas) must fit p99 or you "
                    "train truncation artifacts. opencode agentic = 8192 "
                    "BY MEASUREMENT, chat domains 1024 BY MEASUREMENT.",
    "lora_target_modules": "GUARDED: text-path list is the 1a6ea5 VLM-Trap "
                           "fix (lora_target_linear bakes dead visual "
                           "tensors -> vLLM refuses the adapter). The "
                           "MODULE LIST guards; r stays free.",
    "val_set_size": "always forced to 0 (holdout law 13am) — the recipe "
                    "value is decorative; forcing is RECORDED per row.",
}


def config_guards(cfg: dict, pair_cfg: dict | None = None) -> tuple[list[dict], list[str]]:
    """(refusals, advisories) for a RESOLVED train config (recipe+overrides).
    Refusals: only burned footguns; each = {code, reason} and is bypassable
    via force_guards=[code]. Advisories: strong opinions, loosely held."""
    refusals: list[dict] = []
    advisories: list[str] = []
    for tag, c in (("main", cfg), ("pair", pair_cfg)):
        if not c:
            continue
        bm = str(c.get("base_model", ""))
        fam = families.family_for(bm)
        fam_name = fam[0] if fam else ""
        mb = c.get("micro_batch_size", 1)
        if fam_name == "qwen3.5" and isinstance(mb, int) and mb > 1:
            refusals.append({"code": "fla_batch", "leg": tag, "reason":
                f"{tag}: qwen3.5 + micro_batch_size={mb} — fla gated-delta "
                "kernels reject batch>1 when sample_packing sets cu_seqlens "
                "(425e6c died at step 0; c4b9eb/8250dc re-proved). "
                "Keep 1 and raise gradient_accumulation_steps, or pay for "
                "new evidence. " + GUARD_BYPASS})
        if c.get("adapter"):
            tgts = c.get("lora_target_modules")
            linear = c.get("lora_target_linear") or tgts in (None, "all-linear")
            if fam_name == "qwen3.5" and linear:
                refusals.append({"code": "vlm_lora", "leg": tag, "reason":
                    f"{tag}: qwen3.5 adapter without an explicit "
                    "text-path lora_target_modules targets the vision tower "
                    "(1a6ea5: dead visual.* tensors -> vLLM refuses the "
                    "merged/adapter load). Pin the list. " + GUARD_BYPASS})
        lr = float(c.get("learning_rate") or 0)
        if c.get("adapter"):
            if lr and lr > 2e-4:
                advisories.append(
                    f"{tag}: learning_rate={lr:g} above the fleet band "
                    "(<=1e-4 observed for LoRA at these sizes) — identity "
                    "cooks; not refused (no burned run), but holdout will "
                    "tell the truth.")
            if int(c.get("num_epochs", 1)) > 4:
                advisories.append(f"{tag}: num_epochs={c['num_epochs']} — "
                                  "the holdout law is what catches "
                                  "memorization at this altitude; 5+ passes "
                                  "usually trade generalization for loss "
                                  "curves (round-2 shape).")
        elif lr > 5e-5:
            advisories.append(f"{tag}: FULL FT at lr={lr:g} — full weights "
                              "cook at LoRA rates; fleet FFT prior is 2e-5 "
                              "(~20x under LoRA). Not refused: no burned "
                              "run yet — this number is YOURS to prove.")
        if fam_name == "qwen3.5" and "0.8" in bm.lower() and c.get("load_in_4bit"):
            advisories.append(f"{tag}: 0.8B on qwen3.5 with load_in_4bit — "
                              "bf16 base fits 24GB many times over; 4-bit "
                              "buys nothing and adds quant noise under the "
                              "adapter (2026-11-27 call, recipes/opencode).")
    if cfg and pair_cfg:
        heavy = [t for t, c in (("main", cfg), ("pair", pair_cfg))
                 if c and not c.get("adapter")]
        if heavy:
            advisories.append("pair: FFT leg(s) "
                              f"{heavy} hold sustained full-weight memory — "
                              "co-tenancy risks an OOM that kills BOTH legs "
                              "in one manifest. Pair lora-legs only.")
    return refusals, advisories
# ---------------------------------------------------------------------------


def submit(domain: str, recipe: Path, *, model: str = "", mini: bool = False,
           overrides: dict | None = None, provider: str = "runpod",
           gpus: list[str] | None = None, tier: str | None = None,
           dry: bool = False, max_runtime: int = 3600,
           max_cost: float | None = None, poll: int = 20,
           force_lane: str | None = None, max_price_hr: float | None = None,
           max_wait: int | None = None,
           pair: "Path | str | None" = None,
           force_guards: "list[str] | None" = None,
           on_run_id: "callable | None" = None) -> dict:
    """Full cycle; returns ledger-shaped result dict. NEVER leaves a pod
    behind: every terminal path terminates. on_run_id (if given) fires the
    instant the ledger registers the run — lets async callers (mcp-trainer)
    return before staging finishes.

    CO-TENANT (pair=): one pod, two recipes, two concurrent axolotl procs
    (micro_batch_size=1 regimes are kernel-launch-bound — co-running is
    nearly free throughput). One run_id, one manifest, per-slug artifact
    trees + per-slug prepared dirs; the gate requires BOTH models' files,
    so a half-success fails honestly. Relaunch-safe: the lane payload IS
    the contract. Budget: price is per pod-hour, so the pair does two
    models' work under one cap — that is the POINT, not a leak; the wave
    loop still dies at max_runtime/max_cost as always.
    """
    led = Ledger(ROOT / "state" / "ledger.db")  # ROOT-scoped: tests redirect
    cdir = corpora_dir()
    rp = Path(recipe)
    try:
        rrecipe = str(rp.relative_to(cdir))      # portable provenance
    except ValueError:
        rrecipe = str(rp)
    spec = {"domain": domain, "recipe": rrecipe,
            "mini": mini, "overrides": overrides or {},
            "corpora": corpora_git_sha(),
            "recipe_sha": _sha(recipe)[:12],
            # DEADLINE SURVIVES THE CONTROLLER (485a22 law): the watch
            # budget is part of the run contract, not a caller-local —
            # resume() must be able to enforce the SAME ceiling after a
            # crash/restart, or every re-attach silently grants a fresh
            # 4 h while the pod keeps billing (the 90-min zombie-holdout
            # incident, 2026-11-27).
            "max_runtime": int(max_runtime)}
    rpair = ""
    if pair is not None:
        pp = Path(pair)
        if not pp.is_file():
            # fail BEFORE the row exists: a missing recipe is operator
            # error at zero cost — no dangling open run to reconcile
            raise SystemExit(f"pair recipe not found: {pp}")
        try:
            rpair = str(pp.relative_to(cdir))
        except ValueError:
            rpair = str(pp)
        spec["pair_recipe"] = rpair
        spec["pair_recipe_sha"] = _sha(pp)[:12]
    run_id = led.create("train", spec, provider=provider)
    if on_run_id:
        on_run_id(run_id)
    result = {"run_id": run_id}
    pod_id = ""
    if provider == "local":
        mount = os.environ.get("LOCAL_MOUNT", "/data")
        vol = {"mount": mount, "local": True}
        lane = "host"
        get_run = storage.host_get
        put_run = lambda k, d, ct="": storage.host_put(k, d)
    else:
        vol = storage.vol_cfg() if provider == "runpod" else None
        if force_lane == "presign":
            vol = None
        elif force_lane == "volume" and not vol:
            led.update(run_id, state="failed", error="volume lane not configured")
            return {"run_id": run_id, "status": "error",
                    "error": "RUNPOD_VOLUME_ID/RUNPOD_S3_* unset — volume lane off"}
        lane = "volume" if vol else "presign"
        get_run = (lambda k: storage.vol_get(k)) if vol else storage.get_object
        put_run = ((lambda k, d, ct="": storage.vol_put(k, d)) if vol
                   else (lambda k, d, ct="": storage.put_object(k, d, ct)))
    result["lane"] = lane
    try:
        if lane == "volume" and tier == "community":
            # CONFIG CONTRADICTION FAILS FAST (120c28 receipt: 8 waves /
            # 1800s retrying an empty-by-construction pool — RunPod
            # network volumes exist ONLY in the SECURE cloud, so a
            # community catalog + the secure filter in _train_pool can
            # never yield an offer). Capacity-wait is for market
            # flakiness, not impossibilities.
            raise ProviderError(
                "volume lane + --tier community is impossible: RunPod "
                "network volumes are SECURE-cloud only — drop --tier "
                "or use --lane presign")
        ov = dict(overrides or {})
        if mini:
            # micro_batch_size stays 1 — the recipe's own lesson (packing
            # + cu_seqlens: FLA kernel requires batch dim 1; 425e6c died
            # at step 0, and c4b9eb/8250dc re-proved it via the override)
            ov.update({"sequence_len": 256, "num_epochs": 1,
                       "gradient_accumulation_steps": 1,
                       "saves_per_epoch": 1})
        # GUARDS (2026-11-27 assumption audit): refuse burned footguns on
        # the RESOLVED config (recipe+overrides) BEFORE staging a single
        # byte; everything else is guidance the operator keeps. Verdicts
        # ride the ledger row (merge_spec) — the row answers "what did we
        # know and choose" without archaeology.
        merged = {**(yaml.safe_load(rp.read_text()) or {}), **ov}
        merged_pair = ({**(yaml.safe_load(Path(pair).read_text()) or {}),
                        **ov} if pair is not None else None)
        refusals, advisories = config_guards(merged, merged_pair)
        bypass = set(force_guards or [])
        blocked = [r for r in refusals if r["code"] not in bypass]
        if blocked:
            raise ProviderError(
                "config guards refused: "
                + " | ".join(r["reason"] for r in blocked)
                + f" (bypass knowingly: force_guards="
                  f"{sorted({r['code'] for r in blocked})})")
        dataset_key, dataset_path, dataset_sha, dataset_lines, hold = \
            prepare_dataset(domain, mini, vol)
        # --- train-job shaping: single vs co-tenant pair, gate by recipe ---
        train_cmd = "axolotl train /work/job/train.yaml"
        cfg_report: dict = {}
        job_fields: dict = {"config_yaml":
                            render_config(recipe, ov, dataset_path.name,
                                          report=cfg_report)}
        gate: dict | None = None
        if pair is not None:
            s1, s2 = _cfg_slug(recipe), _cfg_slug(pp)
            if s1 == s2:
                s2 += "-2"   # identical slugs = two models sharing an
                             # output dir = the gate would lie
            c1 = render_config(recipe, ov, dataset_path.name, slug=s1,
                               report=cfg_report)
            c2 = render_config(pp, ov, dataset_path.name, slug=s2,
                               report=cfg_report)
            # per-pid waits so ONE crash cannot orphan the other's finish;
            # sh -c because the runner execs shlex.split — no shell (law,
            # same chain trick as stage preludes)
            chain = (f"axolotl train /work/job/{s1}.yaml & p1=$!; "
                     f"axolotl train /work/job/{s2}.yaml & p2=$!; "
                     "r=0; wait $p1 || r=1; wait $p2 || r=1; exit $r")
            train_cmd = "sh -c " + shlex.quote(chain)
            job_fields = {"configs": [
                {"name": s1, "config_yaml": c1},
                {"name": s2, "config_yaml": c2}]}
            gate = {"required_files": _gate_required(recipe, s1)
                    + _gate_required(pp, s2)}
        elif not (yaml.safe_load(job_fields["config_yaml"]) or {}).get(
                "adapter"):
            # FULL FT round: adapter-shaped default gate would fail an
            # otherwise-perfect run — declare what the recipe produces
            gate = {"required_files": _gate_required(recipe)}
        meta = {"domain": domain, "recipe": str(recipe)}
        if pair is not None:
            meta["pair"] = str(pp)

        if vol:  # pod sees the dataset THROUGH the mount
            pod_ds_path = f"{vol['mount']}/{dataset_key}"
            payload = {"run_id": run_id,
                       "command": train_cmd,
                       **job_fields,
                       "meta": meta,
                       **({"gate": gate} if gate else {}),
                       "dataset": {"name": dataset_path.name,
                                   "path": pod_ds_path, "sha256": dataset_sha},
                       # holdout rides as a sha-verified input (agent
                       # materializes + attests it, SPEC 13am)
                       "inputs": ([{"name": hold["name"],
                                     "path": f"{vol['mount']}/{hold['key']}",
                                     "sha256": hold["sha256"]}]
                                  if hold else []),
                       "outputs_dir": "/work/outputs",
                       "urls": {},
                       "volume": {"mount": vol["mount"],
                                  "prefix": f"runs/{run_id}"}}
            payload_url = ""
            payload_env = {"JOB_PAYLOAD_PATH": f"{vol['mount']}/runs/{run_id}/payload.json"}
        else:
            urls = storage.mint_job_urls(run_id, dataset_key,
                                         holdout_key=hold["key"] if hold else None)
            payload = {"run_id": run_id,
                       "command": train_cmd,
                       **job_fields,
                       "meta": meta,
                       **({"gate": gate} if gate else {}),
                       "dataset": {"name": dataset_path.name,
                                   "url": urls["dataset_get"],
                                   "sha256": dataset_sha},
                       "inputs": ([{"name": hold["name"],
                                     "url": urls["holdout_get"],
                                     "sha256": hold["sha256"]}]
                                  if hold else []),
                       "outputs_dir": "/work/outputs",
                       "urls": urls}
            payload_url = urls["payload_get"]
            payload_env = {"JOB_PAYLOAD_URL": payload_url}
        put_run(f"runs/{run_id}/payload.json", json.dumps(payload).encode(),
                "application/json")
        result.update({"dataset_key": dataset_key, "dataset_sha256": dataset_sha,
                       "payload_key": f"runs/{run_id}/payload.json"})
        # provenance INTO the ledger (SPEC 13ak): the row must exist before
        # staging (async callers grab the id), so the staged contract is
        # merged after — dataset sha/lines + exact image bytes. Without
        # this, "did the pod train on the data we think?" is unanswerable
        # (de3c7d post-mortem, 2026-11-25).
        led.merge_spec(run_id, dataset_sha=dataset_sha,
                       dataset_lines=dataset_lines, image=_image_ref(),
                       guards={"advisories": advisories,
                               "bypassed": sorted(bypass),
                               "forced": cfg_report.get("forced") or {},
                               "law_clash": cfg_report.get("law_clash") or []},
                       **({"holdout_sha": hold["sha256"],
                           "holdout_lines": hold["lines"]} if hold else {}))
        if dry:
            led.update(run_id, state="verified", note=f"dry-run: {lane} lane staged")
            result["dry"] = True
            return result

        prov = get_provider(provider)
        # right-sizing (SPEC 13q): a rate ceiling keeps small models off
        # B200/H200-class hardware even when that is all that's in stock —
        # waiting is cheaper than overbuying. Pre-launch gates price
        # against the WORST card the ceiling admits, because stock moves
        # between waves; _wave_launch retries with fresh catalogs.
        ceiling = (max_price_hr if max_price_hr is not None
                   else float(os.environ.get("QTRAIN_MAX_PRICE_HR", "1.50")))
        rate_bound = 0.0 if provider == "local" else ceiling
        # THE ESTIMATE MUST MATCH THE WORK PROFILE (f44f27 receipt): a
        # --mini is 48 lines x 1 epoch — the 'smoke' profile, not a full
        # train. Estimating mini as train (60min -> $2.25) refused a
        # ~$0.25 validation at the balance gate; the 'smoke' row in
        # EST_MINUTES was DEAD WEIGHT, selected by nothing. Under-
        # estimating kills good runs at the cap (cb059d); over-estimating
        # refuses cheap runs at the gate — both are calibration lies.
        est = budget.estimate(rate_bound, "smoke" if mini else "train")
        why = budget.aggregate_gate(
            [r.get("est_cost") or 0.0 for r in led.list()
             if r["state"] in budget.BILLED and r["run_id"] != run_id], est)
        if not why and provider == "runpod":
            try:
                bal = prov.balance()
            except Exception:
                bal = None                       # scoped key: prepaid anyway
            why = budget.balance_gate(bal, est)
        if why:
            raise ProviderError(why)
        result["estimate"] = round(est, 2)
        token = hashlib.sha256(run_id.encode()).hexdigest()[:24]
        lk = ({"volume_id": vol["bucket"], "volume_path": vol["mount"],
               "data_center": vol["dc"]}
              if (vol and not vol.get("local")) else {})
        inst, offer = _wave_launch(
            prov, lambda: _train_pool(prov, gpus, tier, vol, ceiling),
            name=_pod_name("train", run_id, _pod_hint(domain, recipe)),
            image=_image_ref(),
            env={**payload_env, "AGENT_TOKEN": token,
                 "IMAGE_TAG": _image_ref(),
                 # DEADMAN knob (SPEC 13an): flows to agent os.environ;
                 # always explicit (default 600) so pod forensics sees it
                 "DEADMAN_GRACE_S": os.environ.get("QTRAIN_DEADMAN_GRACE_S",
                                            "600"),
                 **_pod_stop_env(run_id)},
            lk=lk, run_id=run_id, kind="train",
            max_wait=_max_wait(max_wait))
        pod_id = inst.id
        result["instance"] = {"id": inst.id, "gpu": offer.gpu_id,
                              "price_hr": offer.price_hr}
        # kill threshold: explicit wins; else the runtime cap expressed in
        # dollars (wall-clock already bounds cost at this rate). NOTE:
        # max_runtime is SECONDS — 425e6c exposed a /60-vs-/3600 units bug
        # that made the derived cap 60x too high ($33 for a $0.50 job).
        mc = (max_cost if max_cost is not None
              else (offer.price_hr or 0.0) * max_runtime / 3600)
        led.update(run_id, state="provisioning", instance_id=inst.id,
                   base_url=inst.base_url, est_cost=round(mc, 4),
                   price_hr=offer.price_hr or 0.0)
        _watch(run_id, get_run, result, led, poll, started=time.time(),
               max_runtime=max_runtime, prov=prov, inst_id=inst.id,
               price_hr=offer.price_hr, max_cost=mc,
               keepalive=(lane == "presign"), put_run=put_run)
        return result
    except Exception as e:
        if pod_id and _net_blip(e):
            led.update(run_id, error=f"watch lost (network): "
                       f"{str(e)[:140]} — pod kept; `qtrain watch` re-attaches")
            result.update({"status": "unwatched", "error": str(e)[:200],
                           "pod": pod_id})
            notify.emit("watch_lost", run_id, pod=pod_id, kind="train",
                        error=str(e)[:160])
        else:
            led.update(run_id, state="failed", error=str(e)[:400])
            result.update({"status": "error", "error": str(e)})
            notify.emit("run_failed", run_id, error=str(e)[:300])
        return result
    finally:
        if pod_id and not result.get("unwatched"):
            try:
                result["terminated"] = bool(
                    get_provider(provider).terminate(pod_id))
            except Exception as e:
                result["terminate_error"] = str(e)


STAGE_SPECS = {
    "merge": {"script": "merge_adapter.py", "needs_base": True,
              "required": ["config.json", "model.safetensors",
                           "merge_stats.json"],
              "max_price_hr": 0.80},
    # R15 (docs/ORACLES.md): contrast activations -> behavior direction;
    # soft mode emits a steering-LoRA spec (rides the vLLM lane), hard
    # mode projects W <- W(I - vv^T). Fuel: `synth contrasts`. The lane
    # ships the pod payload; probe forward-passes run torch-side when
    # the image provides it — direction math is pure python (tested).
    "ablate": {"script": "ablate.py", "needs_base": True,
               "required": ["steer.json"], "max_price_hr": 0.80},
    # preludes now carry ONLY the dependency stack (pip drift-fixes until
    # p-v0.1.4+ images land via CI); the stage scripts + util.py ride along
    # as sha-verified file inputs (see submit_stage), so script fixes NEVER
    # wait on an image rebuild.
    "onnx": {"script": "export_onnx.py", "required": ["model.onnx"],
             "max_price_hr": 0.80,
             # torch 2.6: newer torches dropped the symbolic_opset14
             # symbols optimum 1.x imports; optimum 2.x dropped the
             # exporters extra entirely.
             # PROBE CLOSED 2026-11-25 (9aea4b + 32606b): for qwen3_5_text
             # this pin-set is an IMPOSSIBLE triangle — new transformers
             # breaks optimum 1.x at import (is_tf_available removed),
             # old transformers cannot LOAD the hybrid config. No pin
             # reaches optimum's support list at all. Edge target for the
             # qwen3.5 family is GGUF (SPEC 13al); ONNX here stays for
             # optimum-era families (smollm2 etc.).
             "prelude": "pip install --no-cache-dir -q --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.*'"
                        " && pip install --no-cache-dir -q 'optimum[exporters]<2' onnx"},
    "int8": {"script": "quantize_onnx.py", "required": ["model.int8.onnx"],
             "max_price_hr": 0.80,
             "prelude": "pip install --no-cache-dir -q onnx"},
    # HOLDOUT STAGE (SPEC 13am): teacher-forced NLL on the content-hashed
    # unseen split, against the train run's adapter bytes. Post-train by
    # design — this axolotl rejects eval_strategy at val_set_size==0 and
    # a random val split would violate the stability law. Retroactive:
    # any historical train run can be scored after the fact.
    # image: "trainer" — 45efe0 died KeyError 'qwen3_5' on the
    # postproc-gpu stack (transformers 4.57.6 predates the hybrid arch);
    # the trainer image knows this generation AND ships peft (axolotl
    # dependency), so no prelude. e7712c (missing peft) taught the same
    # lesson from the other side.
    "holdout": {"script": "holdout_loss.py", "needs_base": True, "gpu": True,
                "image": "trainer", "required": ["holdout.json"],
                "max_price_hr": 0.85},
    "awq": {"script": "quantize_awq.py", "required": ["config.json"],
            "gpu": True, "needs_dataset": True, "max_price_hr": 0.80,
            # cu128 swap on Blackwell only — datacenter sm_100 (compute
            # 10.x) AND consumer sm_120 (12.x); baked cu124 lacks both.
            # Guard skips the 3.5GB download on Ada/Ampere (530b30's
            # B200 would now be rate-culled before launch anyway).
            "prelude": "if nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -qE '^(10|11|12)'; then pip install --no-cache-dir -q --index-url https://download.pytorch.org/whl/cu128 torch; fi"},
}

POSTPROC_PIN = ("ghcr.io/iqns-org/postproc"
                "@sha256:e61e6254628793f5003b3a5aeb61b37d913065f11c474380e871fcfda5d361ac")
POSTPROC_GPU_PIN = ("ghcr.io/iqns-org/postproc-gpu"
                    "@sha256:905f4fe0bc64cd6eaca90f87f8d9a418b8b84e2dd17b0862eedf2bd0bec12b2d")


def _postproc_image() -> str:
    return os.environ.get("POSTPROC_IMAGE", POSTPROC_PIN)


def _postproc_gpu_image() -> str:
    return os.environ.get("POSTPROC_GPU_IMAGE", POSTPROC_GPU_PIN)


def _stage_image(spec: dict) -> str:
    """Stage image selection (SPEC 13am): 'trainer' rides the trainer
    digest (its transformers knows the current generation's archs and
    peft ships inside); everything else stays on the postproc pair."""
    if spec.get("image") == "trainer":
        return _image_ref()
    return _postproc_gpu_image() if spec.get("gpu") else _postproc_image()


def _trace_dataset(get_run, run_id: str, depth: int = 4) -> dict | None:
    """Walk the artifact lineage upstream (stage -> stage -> train) until a
    payload carries a dataset reference — calibration provenance is
    inherited, never guessed."""
    rid = run_id
    for _ in range(depth):
        raw = get_run(f"runs/{rid}/payload.json")
        if not raw:
            return None
        p = json.loads(raw)
        ds = p.get("dataset") or {}
        if ds.get("path") or ds.get("url"):
            return ds
        nxt = None
        for inp in p.get("inputs", []):
            if inp.get("name") == "model":
                src = inp.get("path") or inp.get("url") or ""
                m = re.search(r"runs/([^/]+)/artifacts", src)
                if m:
                    nxt = m.group(1)
        if nxt is None:
            return None
        rid = nxt
    return None


def _recipe_base_from_row(run_id: str) -> str:
    """Best-effort base_model for stage auto-resolution: the from-run's
    recorded recipe (corpora-relative since the provenance fix) is the
    same file that drove training, so its base_model IS the base —
    requiring --base to repeat it invites drift. Swallows everything:
    unknown/unreachable ledger is simply 'no base', and the caller's
    validation error (which must fire BEFORE any I/O) is actionable."""
    try:
        led = Ledger(ROOT / "state" / "ledger.db")
        row = led.get(run_id)
        sp = json.loads((row or {}).get("spec_json") or "{}")
        rec = sp.get("recipe")
        if not rec:
            # stage rows record the RESOLVED base_model on their own
            # spec (auto-resolution at their submit) — the ledger's
            # word beats a hand-typed --base (same anti-drift law);
            # this closes the stage-of-stage dead-end that made
            # `stage holdout --from-run <merge>` refuse honest inputs
            return str(sp.get("base_model", ""))
        p = resolve_recipe(rec)
        return str((yaml.safe_load(p.read_text()) or {}).get(
            "base_model", ""))
    except Exception:                                  # noqa: BLE001
        return ""


def _lineage_base(led: Ledger, run_id: str, depth: int = 4) -> str:
    """Walk the artifact lineage (stage -> stage -> train) until a base
    model is known: stage rows record base_model, train rows via their
    recipe. Registry gating uses this — the family must come from the
    actual lineage, never from a caller's memory."""
    cur, seen = run_id, set()
    while cur and depth > 0 and cur not in seen:
        seen.add(cur)
        row = led.get(cur)
        if not row:
            break
        spec = json.loads(row["spec_json"])
        if spec.get("base_model"):
            return spec["base_model"]
        if spec.get("recipe"):
            b = _recipe_base_from_row(cur)
            if b:
                return b
        cur = spec.get("from_run", "")
        depth -= 1
    return ""


# BAKING STATUS: the (ar) audit asks whether laws are machines or
# folklore. Three were folklore; they are now machines (13ar/av):
# EST_VERSION (budget contamination), merge_law_error (no-op refusal),
# hygiene.classify (proven-junk-only deletion). The pod-kill env
# (QTRAIN_POD_STOP_*) and the holdout bucketing law below still carry
# their audit pointers in (av)/(au) until wired — a constant is the
# first draft of a machine.
HOLDOUT_LAW = 2      # 1 = row-content hash (c3d7027: leaked templated
                     # families); 2 = template-key bucketing


def _holdout_fresh(led: Ledger, get_run, from_run: str) -> str:
    """Domain at the head of the lineage (train row's spec carries it —
    stage rows reference the parent). The rebase input: re-split NOW."""
    cur = from_run
    for _ in range(6):
        row = next((r for r in led.list(include_terminal=True)
                    if r["run_id"] == cur), None)
        if row is None:
            break
        spec = json.loads(row["spec_json"] or "{}")
        if spec.get("domain"):
            return spec["domain"]
        cur = spec.get("from_run") or ""
    raise ProviderError(f"no domain in the lineage of {from_run}")


def _lineage_holdout(get_run, run_id: str, depth: int = 5) -> dict | None:
    """Walk the artifact lineage (stage -> stage -> train) to the payload
    carrying the holdout input (SPEC 13am). Volume-lane payloads only:
    a presign-era holdout URL is expired. None -> caller refuses
    honestly (never a silent zero-eval)."""
    cur = run_id
    while cur and depth > 0:
        depth -= 1
        try:
            raw = get_run(f"runs/{cur}/payload.json")
            payload = json.loads(raw) if raw else {}
        except Exception:                              # noqa: BLE001
            payload = {}
        for i in payload.get("inputs", []):
            name = str(i.get("name", ""))
            if "holdout" in name and i.get("sha256") and i.get("path"):
                return {"name": "holdout.jsonl", "path": i["path"],
                        "sha256": i["sha256"]}
        cur = payload.get("meta", {}).get("from_run", "") \
            or payload.get("from_run", "")
    return None


def submit_stage(kind: str, from_run: str, *, base_model: str | None = None,
                 provider: str = "runpod", max_runtime: int = 3600,
                 max_cost: float | None = None, poll: int = 15,
                 dry: bool = False, max_price_hr: float | None = None,
                 max_wait: int | None = None,
                 fresh_holdout: bool = False,
                 score_base: bool = False,
                 on_run_id: "callable | None" = None) -> dict:
    """Post-process a finished run's artifacts (merge|onnx|int8).

    Routing policy (2026-11-23, user call): cloud boxes are stronger in
    every dimension (RAM/CPU/NVMe/pull speed) — stages default to the
    RunPod volume lane; provider="local" remains as the opt-in $0
    fallback/dev lane. Input = runs/<from_run>/artifacts.tar.gz (the
    volume lane's own copy — every cloud run's output chains there);
    outputs pack through the same finalize contract with a stage gate.
    """
    if kind not in STAGE_SPECS:
        return {"status": "error",
                "error": f"unknown stage {kind!r} (have: "
                         f"{', '.join(STAGE_SPECS)})"}
    spec = STAGE_SPECS[kind]
    if spec.get("needs_base") and not base_model:
        # provenance auto-resolution: the from-run's recipe knows its own
        # base (merge must use EXACTLY that tokenizer/weights — passing it
        # by hand invites drift)
        base_model = _recipe_base_from_row(from_run) or None
    if spec.get("needs_base") and not base_model:
        return {"status": "error", "error":
                "merge stage needs base_model — pass --base or the "
                "from-run must record a recipe that declares one"}
    if provider not in ("local", "runpod"):
        return {"status": "error",
                "error": f"stage provider {provider!r} not wired yet"}
    # FAMILY GATE (SPEC 13al): a stage whose target is registry-proven
    # impossible for the lineage's family is refused BEFORE the ledger row
    # and before any pod — 9aea4b/32606b bought this law for ~$0.60, and
    # re-probing is a corpora-registry decision, not a dispatch accident.
    fam_base = (base_model or "").strip()
    if not fam_base:
        fam_base = _lineage_base(Ledger(ROOT / "state" / "ledger.db"),
                                 from_run)
    reason = families.refuse_reason(fam_base, kind)
    if reason:
        return {"status": "error", "kind": kind, "from_run": from_run,
                "base_model": fam_base, "refused": "family-registry",
                "error": reason}
    led = Ledger(ROOT / "state" / "ledger.db")
    run_id = led.create("stage", {"kind": kind, "from_run": from_run,
                                  "base_model": (base_model or "").strip()
                                  or fam_base,
                                  "corpora": corpora_git_sha(),
                                  # deadline rides the row (train law,
                                  # same incident): re-attach enforces
                                  # the submitted ceiling
                                  "max_runtime": int(max_runtime)},
                        provider=provider)
    if on_run_id:
        on_run_id(run_id)
    result = {"run_id": run_id, "kind": kind, "from_run": from_run}
    pod_id = ""

    if provider == "local":
        mount = os.environ.get("LOCAL_MOUNT", "/data")
        lane = "host"
        get_run = storage.host_get
        put_run = lambda k, d, ct="": storage.host_put(k, d)
    elif provider == "runpod":
        vol = storage.vol_cfg()
        if not vol:
            led.update(run_id, state="failed", error="volume lane off")
            return {"run_id": run_id, "status": "error",
                    "error": "RUNPOD_VOLUME_ID/RUNPOD_S3_* unset"}
        mount = vol["mount"]
        lane = "volume"
        get_run = storage.vol_get
        put_run = lambda k, d, ct="": storage.vol_put(k, d)
    else:
        return {"status": "error",
                "error": f"stage provider {provider!r} not wired yet"}
    result["lane"] = lane

    in_key = f"runs/{from_run}/artifacts.tar.gz"
    # ARTIFACT PREFLIGHT (07bea4 law): every stage reads the from-run's
    # artifacts.tar.gz as its `model` input — if that object is not on
    # the lane the pod boots, the agent fetches nothing, and the run
    # dies at max_runtime having spent the whole GPU budget for zero
    # signal (the 2026-11-27 base-half rebase burned 45 min + ~$0.43
    # discovering the c7cde2 tar had been GC'd from evidence). Refuse
    # BEFORE any pod, same shape as the family gate: a missing input is
    # an operator/corpus fact known at dispatch, never something to pay
    # a live card to learn. vol_head is the (ar) verify-after-delete
    # witness — 200 = present, anything else = absent/unverifiable.
    present = True   # dry runs rehearse against planned lineages: no pod,
                     # no $, so the input may legitimately not exist yet
    if not dry:
        try:
            present = (storage.host_exists(in_key) if lane == "host"
                       else storage.vol_head(in_key) == 200)
        except Exception:                              # noqa: BLE001
            present = False                            # uncheckable -> refuse
    if not present:
        err = (f"from-run artifact missing on lane: {in_key} — the base/"
               "merged bytes this stage must read are gone (evidence GC "
               "or wrong from_run); refusing before any pod launch")
        led.update(run_id, state="failed", error=err[:400])
        notify.emit("run_failed", run_id, error=err[:300],
                    reason="artifact-preflight")
        return {"status": "error", "run_id": run_id, "error": err,
                "refused": "artifact-preflight"}
    cmd = (f"python /work/inputs/{spec['script']} "
           f"--in /work/inputs/model --out /work/outputs")
    if spec.get("needs_base"):
        cmd += f" --base {shlex.quote(base_model)}"
    inputs: list[dict] = [{"name": "model", "path": f"{mount}/{in_key}",
                           "extract": True}]
    if spec.get("needs_dataset"):
        # calibration must match what the model was TRAINED on: the dataset
        # is INHERITED from the artifact lineage (merge -> train), never
        # guessed or re-supplied
        ds = None
        try:
            ds = _trace_dataset(get_run, from_run)
        except Exception as e:                       # noqa: BLE001
            err = f"lineage read failed: {e}"
            led.update(run_id, state="failed", error=err[:400])
            notify.emit("run_failed", run_id, error=err[:300])
            return {"status": "error", "run_id": run_id, "error": err}
        ds = ds or {}
        ds_ref = {"name": "data"}
        if ds.get("path"):
            ds_ref["path"] = ds["path"]
        elif ds.get("url"):
            ds_ref["url"] = ds["url"]
        else:
            err = f"no dataset in lineage of {from_run}"
            led.update(run_id, state="failed", error=err)
            return {"status": "error", "run_id": run_id, "error": err}
        if ds.get("sha256"):
            ds_ref["sha256"] = ds["sha256"]
        inputs.append(ds_ref)
        cmd += " --dataset /work/inputs/data"
    if kind == "holdout":
        # HOLDOUT RESOLVER (SPEC 13am): re-attach the stable split from
        # the lineage; refuse honestly when it is absent (pre-13am runs).
        # fresh_holdout (TREND-SEAM FIX, (av)-2 / c3d7027): bucket law v2
        # changed split semantics, so a v1-era lineage split scores
        # memorization siblings — rebase by re-running the CURRENT splitter
        # (bin/prepare-domain, template-key bucketing) against the lineage's
        # domain corpus and score THAT. The fresh file is content-addressed
        # to the same datasets/<domain>/<sha>/ convention, so comparisons
        # land on identical bytes or they do not compare (sha rides the
        # row, and the row now records the LAW too).
        if fresh_holdout:
            domain = _holdout_fresh(led, get_run, from_run)
            _k, _p, _s, _n, hold = prepare_dataset(domain, vol=vol)
        else:
            hold = _lineage_holdout(get_run, from_run)
        if hold is None:
            err = (f"no holdout input in the lineage of {from_run} — "
                   "pre-13am train runs carry no stable split; submit "
                   "again to stage one (or --fresh-holdout to rebase)")
            led.update(run_id, state="failed", error=err)
            return {"status": "error", "run_id": run_id, "error": err}
        if "path" not in hold:               # fresh: key -> mount path
            hold = {**hold, "path": f"{mount}/{hold['key']}"}
        inputs.append({"name": "holdout.jsonl", "path": hold["path"],
                       "sha256": hold["sha256"]})
        cmd += " --holdout /work/inputs/holdout.jsonl"
        if score_base:
            # ONE pod, BOTH pair halves (07bea4 law): no-op-merge tars are
            # not guaranteed on the lane (they get GC'd), so the base
            # reference must come from the same run that scores the
            # persona bytes — identical split, identical guards.
            cmd += " --score-base"
        # RECORD the resolved sha on the row (4eed0a law: the books must
        # carry what the pod actually received — verify's holdout leg
        # reads spec['holdout_sha'], and "?" there is a recorder bug,
        # not a data bug) + the bucketing LAW, so trend seams are
        # self-documenting instead of archaeological (c3d7027).
        led.merge_spec(run_id, holdout_sha=hold["sha256"],
                       holdout_law=HOLDOUT_LAW, score_base=score_base)
    # Run CURRENT repo scripts, not the frozen /opt copies baked into the
    # p-v images: those are stranded at pre-fix builds (mirror stall), and
    # 6709e3 proved the failure class — llmcompressor's API drifted around
    # the baked script while the repo one was already fixed. Plain file
    # inputs materialize at /work/inputs/<name> (agent _fetch + sha check);
    # util.py rides along so the script's `from util import ...` resolves.
    scripts: dict[str, bytes] = {}
    for fname in (spec["script"], "util.py"):
        fpath = ROOT / "postproc" / fname
        if not fpath.is_file():
            err = f"stage script missing in repo: postproc/{fname}"
            led.update(run_id, state="failed", error=err)
            notify.emit("run_failed", run_id, error=err)
            return {"status": "error", "run_id": run_id, "error": err}
        data = fpath.read_bytes()
        scripts[fname] = data
        inputs.append({"name": fname,
                       "path": f"{mount}/runs/{run_id}/script/{fname}",
                       "sha256": hashlib.sha256(data).hexdigest()})
    if spec.get("prelude"):
        # wrap LAST: the runner execs shlex.split(command) — no shell — so
        # the whole chain must arrive as ONE quoted `sh -c` invocation
        cmd = "sh -c " + shlex.quote(f"{spec['prelude']} && {cmd}")
    payload = {"run_id": run_id, "command": cmd, "config_yaml": "",
               "inputs": inputs,
               "meta": {"kind": kind, "from_run": from_run},
               "outputs_dir": "/work/outputs",
               "gate": {"required_files": spec["required"]},
               "urls": {},
               "volume": {"mount": mount, "prefix": f"runs/{run_id}"}}
    payload_env = {"JOB_PAYLOAD_PATH":
                   f"{mount}/runs/{run_id}/payload.json"}
    try:
        for fname, data in scripts.items():
            put_run(f"runs/{run_id}/script/{fname}", data)
        put_run(f"runs/{run_id}/payload.json",
                json.dumps(payload).encode())
        result["payload_key"] = f"runs/{run_id}/payload.json"
        # same discipline as train (SPEC 13ak): what exact bytes this
        # stage job was handed, recorded in the ledger spec.
        led.merge_spec(run_id,
                       # 4eed0a law, FINALLY enforced: the row carries the
                       # SAME image the launch boots — the legacy ternary
                       # missed image:"trainer" kinds (holdout/ablate), so
                       # the books said postproc digest, the pod ran the
                       # trainer. One helper, two readers.
                       image=_stage_image(spec),
                       inputs=[{"name": i.get("name", ""),
                                "sha256": i.get("sha256", "")}
                               for i in inputs if i.get("sha256")])
        if dry:
            led.update(run_id, state="verified", note=f"dry stage {kind}")
            result["dry"] = True
            return result

        prov = get_provider(provider)
        token = hashlib.sha256(run_id.encode()).hexdigest()[:24]
        lk = ({"volume_id": vol["bucket"], "volume_path": vol["mount"],
               "data_center": vol["dc"]} if provider == "runpod" else {})
        ceiling = (max_price_hr if max_price_hr is not None
                   else spec.get("max_price_hr", 1.50))
        # normalized budgeting (SPEC 13q): gates price the WORST card the
        # ceiling admits — stock moves between waves, so nothing assumes
        # an offer that may not be there when a wave actually runs. The
        # kill threshold tightens to the card ACTUALLY launched unless
        # an explicit max_cost says otherwise. No magic numbers anywhere.
        rate_bound = 0.0 if provider == "local" else ceiling
        est = budget.estimate(rate_bound, kind)
        result["estimate"] = round(est, 2)
        why = budget.aggregate_gate(
            [r.get("est_cost") or 0.0 for r in led.list()
             if r["state"] in budget.BILLED and r["run_id"] != run_id], est)
        if not why and provider == "runpod":
            try:
                bal = prov.balance()
            except Exception:
                bal = None                       # scoped key; prepaid anyway
            why = budget.balance_gate(bal, est)
        if why:
            raise ProviderError(why)
        stage_image = _stage_image(spec)
        inst, offer = _wave_launch(
            prov, lambda: _stage_pool(prov, spec, provider, ceiling),
            name=_pod_name(kind, run_id, _pod_hint(parent=from_run)),
            image=stage_image,
            env={**payload_env, "AGENT_TOKEN": token,
                 "IMAGE_TAG": stage_image,
                 "DEADMAN_GRACE_S": os.environ.get("QTRAIN_DEADMAN_GRACE_S",
                                            "600"),
                 **_pod_stop_env(run_id)}, lk=lk,
            run_id=run_id, kind=kind, max_wait=_max_wait(max_wait))
        mc = (max_cost if max_cost is not None
              else budget.estimate(offer.price_hr or rate_bound, kind))
        pod_id = inst.id
        result["instance"] = {"id": inst.id, "gpu": offer.gpu_id,
                              "price_hr": offer.price_hr}
        led.update(run_id, state="provisioning", instance_id=inst.id,
                   est_cost=round(mc, 4), price_hr=offer.price_hr or 0.0)

        _watch(run_id, get_run, result, led, poll,
               started=time.time(), max_runtime=max_runtime,
               prov=prov, inst_id=inst.id, price_hr=offer.price_hr,
               max_cost=mc, keepalive=False, put_run=put_run)
        result["output"] = f"runs/{run_id}/artifacts.tar.gz"
        return result
    except Exception as e:
        if pod_id and _net_blip(e):
            led.update(run_id, error=f"watch lost (network): "
                       f"{str(e)[:140]} — pod kept; `qtrain watch` re-attaches")
            result.update({"status": "unwatched", "error": str(e)[:200],
                           "pod": pod_id})
            notify.emit("watch_lost", run_id, pod=pod_id, kind=kind,
                        error=str(e)[:160])
        else:
            led.update(run_id, state="failed", error=str(e)[:400])
            result.update({"status": "error", "error": str(e)})
            notify.emit("run_failed", run_id, error=str(e)[:300])
        return result
    finally:
        if pod_id and not result.get("unwatched"):
            try:
                result["terminated"] = bool(
                    get_provider(provider).terminate(pod_id))
            except Exception as e:
                result["terminate_error"] = str(e)


def preflight(kind: str = "train", *, provider: str = "runpod",
              max_price_hr: float | None = None,
              max_runtime: int = 3600) -> dict:
    """Read-only decision report (the plan's `--preflight`): which cards
    pass the fit, what the estimate/kill threshold would be, gate verdicts
    — launches NOTHING, writes NOTHING. Answers 'what would this job
    cost, on what' before committing a cent."""
    led = Ledger(ROOT / "state" / "ledger.db")
    prov = get_provider(provider)
    out: dict = {"kind": kind, "provider": provider}
    try:
        raw_catalog = prov.catalog()
    except Exception as e:                             # noqa: BLE001
        out["decision"] = f"refuse: catalog unavailable: {str(e)[:160]}"
        return out
    is_stage = kind in STAGE_SPECS
    if is_stage:
        spec = STAGE_SPECS[kind]
        pool = [o for o in raw_catalog if o.secure]
        for pat in spec.get("arch_exclude", []):
            pool = [o for o in pool if pat.lower() not in o.gpu_id.lower()]
        ceiling = (max_price_hr if max_price_hr is not None
                   else spec.get("max_price_hr", 1.50))
        offers, best = budget.fits(pool, spec.get("min_vram", 10), ceiling)
    else:
        ceiling = (max_price_hr if max_price_hr is not None
                   else float(os.environ.get("QTRAIN_MAX_PRICE_HR", "1.50")))
        all_ = raw_catalog
        if storage.vol_cfg():
            all_ = [o for o in all_ if o.secure]
        offers = sorted((o for o in all_
                         if (o.price_hr or 0.0) <= ceiling),
                        key=lambda o: o.price_hr or 0.0)
        best = min((o.price_hr for o in all_ if o.price_hr), default=None)
    out["ceiling_hr"] = ceiling
    out["candidates"] = [{"gpu": o.gpu_id, "price_hr": o.price_hr,
                          "vram_gb": o.vram_gb} for o in offers[:5]]
    if not offers:
        cheap = f"${best:.2f}/hr" if best is not None else "none"
        out["decision"] = (f"refuse: no in-stock offer within "
                           f"${ceiling:.2f}/hr (cheapest fit: {cheap})")
        return out
    est = budget.estimate(offers[0].price_hr or 0.0,
                          kind if is_stage else "train")
    kill = est if is_stage else (offers[0].price_hr or 0.0) * max_runtime / 60
    opens = [r.get("est_cost") or 0.0 for r in led.list()
             if r["state"] in budget.BILLED]
    why = budget.aggregate_gate(opens, est)
    bal = None
    if provider == "runpod":
        try:
            bal = prov.balance()
        except Exception:                            # noqa: BLE001
            pass
    why = why or budget.balance_gate(bal, est)
    out.update({"balance": bal, "estimate": round(est, 2),
                "kill_threshold": round(kill, 2),
                "open_projected": round(sum(opens), 2),
                "decision": why or
                f"launch {offers[0].gpu_id} @ ${offers[0].price_hr}/hr"})
    return out


IMAGE_PIN = ("ghcr.io/iqns-org/trainer"
             "@sha256:a188c08ba767362dfbd0b951de868c06e715bd94bbac9c5ba00f059372d980d1")
IMAGE_TAG = "v0.5.0"


def _image_ref() -> str:
    return os.environ.get("TRAINER_IMAGE", IMAGE_PIN)


def _image_tag() -> str:
    return os.environ.get("TRAINER_IMAGE_TAG", IMAGE_TAG)
