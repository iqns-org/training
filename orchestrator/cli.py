"""qtrain — orchestrator CLI (SPEC §6). P2 slice: cloud lane primitives.

    qtrain catalog [--gpu A40 ...]        live GPU offers + pricing
    qtrain launch --image X --gpu ID ...  raw pod (smoke/driver); ledger row
    qtrain pods                           live instances across providers
    qtrain status ID | logs ID | terminate ID
    qtrain doctor                         live pods vs ledger (orphan check)

`submit` (full train jobs: dataset→S3, presigned URLs, reconcile loop) lands
with P1/P2 completion; the primitives here are exactly what it builds on.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import ROOT, load_dotenv, resolve_recipe
from . import dispatch, hygiene, storage
from .ledger import Ledger
from .models import ProviderError
from .providers import available_providers, get_provider


def _provider(args) -> str:
    return getattr(args, "provider", "runpod") or "runpod"


def _prov(args):
    load_dotenv()
    return get_provider(_provider(args))


def _print(obj):
    print(json.dumps(obj, indent=2, default=str, ensure_ascii=False))


def cmd_catalog(args):
    offers = _prov(args).catalog(args.gpu or None, args.tier)
    if args.json:
        _print([o.__dict__ for o in offers])
        return
    width = max((len(o.gpu_id) for o in offers), default=10)
    for o in offers[: args.limit]:
        print(f"{o.gpu_id:<{width}}  {o.vram_gb:>3}GB  "
              f"${o.price_hr:>5.2f}/hr  {'secure' if o.secure else 'community':<9} "
              f"{o.availability}")


def cmd_launch(args):
    env = dict(kv.split("=", 1) for kv in (args.env or []))
    inst = _prov(args).launch(
        name=args.name, image=args.image, gpu_id=args.gpu,
        cloud=args.cloud, env=env, args=args.cmd, disk_gb=args.disk,
        ssh=args.ssh)
    if args.record:
        led = Ledger()
        led.create("smoke", {"image": args.image, "gpu": args.gpu},
                   provider=_provider(args), run_id=args.record)
        led.update(args.record, instance_id=inst.id, base_url=inst.base_url)
    _print({"instance": inst.id, "status": inst.status,
            "gpu": inst.gpu_type, "base_url": inst.base_url})


def _wait_ready(args):
    prov = _prov(args)
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        inst = prov.poll(args.id)
        if inst.status == "running" and inst.base_url:
            _print({"instance": inst.id, "status": inst.status,
                    "base_url": inst.base_url, "gpu": inst.gpu_type})
            return
        if inst.status in {"terminated", "failed", "error", "stopped"}:
            raise ProviderError(f"pod {args.id} died: {inst.status}")
        print(f"… {inst.status}", file=sys.stderr, flush=True)
        time.sleep(args.poll)
    raise ProviderError(f"pod {args.id} not ready within {args.timeout}s")


def cmd_pods(args):
    for inst in _prov(args).list_instances():
        print(f"{inst.id}  {inst.status:<12} {inst.gpu_type:<34} {inst.name}")


def cmd_status(args):
    _print(_prov(args).poll(args.id).__dict__)


def cmd_logs(args):
    for line in _prov(args).logs(args.id, tail=args.tail):
        print(line)


def cmd_terminate(args):
    ok = _prov(args).terminate(args.id)
    print(f"{args.id}: {'terminated' if ok else 'FAILED to terminate'}")
    raise SystemExit(0 if ok else 1)


def cmd_submit(args):
    import json as _json
    overrides = {}
    for kv in args.override or []:
        k, _, v = kv.partition("=")
        try:
            overrides[k] = _json.loads(v)
        except _json.JSONDecodeError:
            overrides[k] = v
    res = dispatch.submit(
        args.domain, resolve_recipe(args.recipe), mini=args.mini,
        overrides=overrides,
        provider=args.provider, gpus=args.gpu or None, tier=args.tier,
        dry=args.dry, max_runtime=args.max_runtime, max_cost=args.max_cost,
        force_lane=args.lane, max_price_hr=args.max_price,
        max_wait=args.wait)
    _print(res)
    raise SystemExit(0 if res.get("status") in ("succeeded", None) and not res.get("error") else 1)


def cmd_stage(args):
    res = dispatch.submit_stage(
        args.kind, args.from_run, base_model=args.base,
        provider=args.provider, dry=args.dry, max_runtime=args.max_runtime,
        max_cost=args.max_cost, max_price_hr=args.max_price,
        max_wait=args.wait)
    _print(res)
    raise SystemExit(0 if res.get("status") in ("succeeded", None)
                     and not res.get("error") else 1)


def cmd_register(args):
    from . import register
    res = register.register(args.run_id, name=args.name, host=args.host,
                            dest_dir=args.dest, dry=args.dry)
    _print(res)
    raise SystemExit(0 if res.get("status") in ("registered", None)
                     and not res.get("error") else 1)


def cmd_reconcile(args):
    res = dispatch.resume(run_id=args.run_id, poll=args.poll)
    _print(res)
    raise SystemExit(0 if res.get("status") in ("succeeded", None)
                     and not res.get("error") else 1)


def cmd_progress(args):
    vol = storage.vol_cfg()
    key = f"runs/{args.run_id}/status.json"
    raw = storage.vol_get(key) if vol else storage.get_object(key)
    if raw is None:
        _print({"run_id": args.run_id, "heartbeat": None,
                "note": "no status.json on lane yet (pre-boot or old image)"})
        return
    _print(json.loads(raw))


def cmd_evals(args):
    led = Ledger(ROOT / "state" / "ledger.db")
    rows = led.list_evals(model=args.model or None, limit=args.limit)
    if args.json:
        _print(rows)
        return
    if not rows:
        print("no eval history yet (bench --record populates it)")
        return
    print(f"{'when':<16} {'model':>20}  {'score':>6}  {'run/host':<28} checks")
    for r in rows:
        ident = next((c["q"] for c in r["rows"] if c["q"] == "identity"), "")
        ident_pass = next((c["pass"] for c in r["rows"]
                           if c["q"] == "identity"), "—")
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ts"]))
        print(f"{ts:<16} {r['model']:>20}  {r['summary']:>6}  "
              f"{(r['run_id'] or '-')[:12]}/{r['host']:<14.14} "
              f"identity={ident_pass if ident else '—'}")


def cmd_verify(args):
    from . import provenance
    res = provenance.verify_run(args.run_id, live=args.live,
                                recompute=args.recompute)
    if args.json:
        _print(res)
    else:
        print(f"run {res.get('run_id')}  kind={res.get('kind','?')}  "
              f"state={res.get('state','?')}  "
              f"agent={res.get('agent_version','') or '?'}")
        for c in res.get("checks", []):
            line = (f"  [{c['status']:>4}] {c['check']:<28} "
                    f"{str(c['expected'])[:24]:>24}  "
                    f"{str(c['actual'])[:24]}")
            if c.get("note"):
                line += f"   # {c['note']}"
            print(line)
        if res.get("error"):
            print("  error:", res["error"])
    sys.exit(0 if res.get("ok") else 1)


def cmd_watch(args):
    from . import progress as prog
    while True:
        snap = prog.snapshot(args.run_id)
        if "error" in snap:
            _print(snap)
            raise SystemExit(1)
        print(prog.fmt_line(snap), flush=True)
        if snap.get("state") in ("done", "failed", "cancelled"):
            raise SystemExit(0 if snap.get("state") == "done" else 1)
        time.sleep(args.interval)


def cmd_metrics(args):
    from . import progress as prog
    led = Ledger()
    row = led.get(args.run_id)
    if not row:
        _print({"error": f"no such run: {args.run_id}"})
        raise SystemExit(1)
    prov = row["provider"] or "runpod"
    hist = prog.history(args.run_id, provider=prov)
    print(f"{args.run_id}  kind={row['kind']}  state={row['state']}  "
          f"provider={prov}")
    for h in hist:
        step = f"{h.get('step')}/{h.get('total')}" if h.get("step") is not None else "-"
        print(f"  {time.strftime('%H:%M:%S', time.localtime(h.get('ts', 0)))}"
              f"  {str(h.get('state') or '-'):<12} step {step:<10} "
              f"loss {h.get('loss', '-'):<8} cost ${h.get('cost', 0)}")
    log = prog.runlog_local(args.run_id)
    if log:
        curve = prog.parse_runlog(log)
        losses = [c["loss"] for c in curve if c.get("loss") is not None]
        if losses:
            print(f"  loss curve: {len(losses)} pts, first {losses[0]} "
                  f"-> last {losses[-1]}, min {min(losses)}")


def cmd_runs(args):
    """UI-facing run table from the lane index (SPEC 14c) — single
    source of truth, works across controllers, no local-sqlite coupling."""
    from . import progress as prog
    if args.reindex:
        _print(prog.reindex(args.provider))
        return
    rows = prog.read_index(args.provider, limit=args.limit)
    if not rows:
        _print({"runs": [], "note": f"lane index empty on provider "
              f"{args.provider!r} — backfill once with `qtrain runs "
              "--reindex`"})
        return
    hdr = (f"{'run_id':<28} {'kind':<6} {'status':<10} {'secs':>6} "
           f"{'cost':>7} {'loss':>8} {'gpu':<20} {'ref':<20}")
    print(hdr)
    print("-" * len(hdr))
    for r in reversed(rows):
        loss = r.get("last_loss")
        ref = (r.get("base_model") or r.get("from_run")
               or r.get("recipe") or r.get("domain") or "")
        print(f"{r['run_id']:<28} {r.get('kind', '-'):<6} "
              f"{r.get('status', '-'):<10} {r.get('seconds') or '-':>6} "
              + (f"${r['cost']:.3f} " if r.get("cost") is not None
                 else (f"~${r['est_cost']:.2f}" if r.get("est_cost")
                       else "      -  "))
              + (f"{float(loss):>8.3f} " if loss is not None else "       - ")
              + f"{(r.get('gpu') or '-'):<20} {str(ref)[:20]:<20}")


def cmd_budget(args):
    from . import budget as bud
    import statistics
    led = Ledger()
    bal = None
    if args.provider == "runpod":
        try:
            bal = _prov(args).balance()
        except Exception:
            pass
    opens = [r for r in led.list() if r["state"] in bud.BILLED]
    obs_k = list(bud._obs())
    out = {"balance": bal, "reserve": bud.RESERVE,
           "budget_total": bud.TOTAL, "margin": bud.MARGIN,
           "est_version": bud.EST_VERSION,
           "open_runs": [{"run_id": r["run_id"], "kind": r["kind"],
                          "state": r["state"],
                          "est_cost": round(r.get("est_cost") or 0, 3)}
                         for r in opens],
           "open_projected": round(sum(r.get("est_cost") or 0
                                       for r in opens), 2),
           "observed_median_minutes":
               {k: round(statistics.median(v), 1)
                for k in obs_k if (v := bud.obs_minutes(k))}}
    _print(out)


def cmd_hygiene(args):
    res = hygiene.sweep(bad_shas=set(args.bad_sha or ()),
                        apply=args.apply)
    _print(res)


def cmd_preflight(args):
    res = dispatch.preflight(args.kind, provider=args.provider,
                             max_price_hr=args.max_price,
                             max_runtime=args.max_runtime)
    _print(res)
    raise SystemExit(0 if "launch" in str(res.get("decision", "")) else 1)


def cmd_events(args):
    from . import notify
    for e in notify.recent(limit=args.limit, run_id=args.run or None):
        ts = time.strftime("%m-%d %H:%M:%S", time.localtime(e["ts"]))
        extra = " ".join(f"{k}={v}" for k, v in e.items()
                         if k not in ("ts", "event", "kind", "run_id"))
        print(f"{ts}  {e.get('event', e.get('kind', '?')):<13} "
              f"{(e.get('kind') or '-'):<12} {e.get('run_id', ''):<24} "
              f"{extra}")


def cmd_reap(args):
    res = dispatch.reap(args.provider, dry=args.dry)
    _print(res)
    raise SystemExit(0 if not res.get("error") else 1)


def cmd_janitor(args):
    res = dispatch.janitor_once(args.provider, dry=args.dry)
    _print(res)


def cmd_bench(args):
    from . import bench
    argv = []
    if args.config:
        argv += ["--config", args.config]
    for url in (args.url or []):
        argv += ["--url", url]
    for m in (args.model or []):
        argv += ["--model", m]
    for q in (args.question or []):
        argv += ["--question", q]
    if args.repeats is not None:
        argv += ["--repeats", str(args.repeats)]
    if args.max_tokens is not None:
        argv += ["--max-tokens", str(args.max_tokens)]
    if args.temperature is not None:
        argv += ["--temperature", str(args.temperature)]
    if args.json:
        argv += ["--json", args.json]
    if args.csv:
        argv += ["--csv", args.csv]
    if args.record:
        argv += ["--record"]
    if args.run:
        argv += ["--run", args.run]
    raise SystemExit(bench.main(argv))


def cmd_notify_test(args):
    from . import notify
    ev = notify.emit("notify_test", note="qtrain notify-test")
    ok, msg = notify.deliver_sync(ev)
    _print({"journal": str(notify.JOURNAL), "webhook": notify.webhook_url()
            or None, "delivered": ok, "detail": msg})
    raise SystemExit(0 if ok else 1)


def cmd_doctor(args):
    prov = _prov(args)
    led = Ledger()
    known = led.live_instances(_provider(args))
    orphans, tracked = [], []
    for inst in prov.list_instances():
        if not inst.alive:
            continue
        (tracked if inst.id in known else orphans).append(inst)
    for i in orphans:
        print(f"ORPHAN {i.id}  {i.status:<12} {i.gpu_type:<30} {i.name}  "
              f"(not in ledger — terminate or record it)")
    for i in tracked:
        print(f"tracked {i.id}  {i.status:<12} {i.gpu_type:<30} {i.name}")
    if orphans:
        print(f"\n{len(orphans)} orphan pod(s) burning budget.")
        raise SystemExit(1)


def build_parser():
    p = argparse.ArgumentParser(prog="qtrain", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    def prov_arg(s, default="runpod"):
        s.add_argument("--provider", default=default, choices=available_providers())

    s = sub.add_parser("catalog", help="live GPU offers/prices")
    prov_arg(s)
    s.add_argument("--gpu", action="append", help="substring filter, repeatable")
    s.add_argument("--tier", choices=["community", "secure"], default=None)
    s.add_argument("--limit", type=int, default=15)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_catalog)

    s = sub.add_parser("launch", help="launch a raw pod (smoke test)")
    prov_arg(s)
    s.add_argument("--image", required=True)
    s.add_argument("--gpu", required=True)
    s.add_argument("--name", default="qtrain-smoke")
    s.add_argument("--cloud", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    s.add_argument("--env", action="append", metavar="K=V")
    s.add_argument("--cmd", default=None, help="container command override")
    s.add_argument("--disk", type=int, default=30)
    s.add_argument("--record", default=None, metavar="RUN_ID",
                   help="also store in ledger under this run id")
    s.add_argument("--ssh", action="store_true",
                   help="startSsh + publish 22/tcp (exec/debug lane)")
    s.set_defaults(func=cmd_launch)

    s = sub.add_parser("wait", help="block until pod runs and exposes an URL")
    prov_arg(s)
    s.add_argument("id")
    s.add_argument("--timeout", type=int, default=600)
    s.add_argument("--poll", type=int, default=10)
    s.set_defaults(func=_wait_ready)

    s = sub.add_parser("pods", help="list live instances")
    prov_arg(s)
    s.set_defaults(func=cmd_pods)

    s = sub.add_parser("status", help="one instance record")
    prov_arg(s)
    s.add_argument("id")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("logs", help="provider-native pod logs")
    prov_arg(s)
    s.add_argument("id")
    s.add_argument("--tail", type=int, default=100)
    s.set_defaults(func=cmd_logs)

    s = sub.add_parser("terminate", help="delete an instance")
    prov_arg(s)
    s.add_argument("id")
    s.set_defaults(func=cmd_terminate)

    s = sub.add_parser("submit", help="full-cycle training job (SPEC §6)")
    prov_arg(s)
    s.add_argument("domain")
    s.add_argument("--recipe", required=True,
                   help="axolotl YAML (corpora recipes/<domain>/… or path)")
    s.add_argument("--mini", action="store_true",
                   help=f"tiny dataset ({dispatch.MINI_LINES} rows, seq 256, 1 epoch) "
                        "— lane validation, not a real training run")
    s.add_argument("--override", action="append", metavar="K=JSON",
                   help="axolotl config override, repeatable")
    s.add_argument("--gpu", action="append", help="GPU id filter, repeatable")
    s.add_argument("--tier", choices=["community", "secure"], default=None)
    s.add_argument("--dry", action="store_true",
                   help="stage dataset+payload to the lane, do not launch a pod")
    s.add_argument("--lane", choices=["volume", "presign"], default=None,
                   help="storage lane (default: volume when RunPod volume "
                        "is configured, else presign)")
    s.add_argument("--max-runtime", type=int, default=3600)
    s.add_argument("--max-cost", type=float, default=None,
                   help="kill threshold $ (default: runtime cap expressed "
                        "in dollars at the chosen card's rate)")
    s.add_argument("--max-price", type=float, default=None,
                   help="rate ceiling $/hr for offer selection "
                        "(default: $QTRAIN_MAX_PRICE_HR or 1.50)")
    s.add_argument("--wait", type=int, default=None,
                   help="capacity-wait seconds (default: $QTRAIN_MAX_WAIT "
                        "or 1800; 0 = fast-fail single wave)")
    s.set_defaults(func=cmd_submit)

    s = sub.add_parser("stage", help="post-process a run's artifacts "
                        "(merge|onnx|int8) — $0 local K8s Jobs")
    s.add_argument("kind", choices=sorted(dispatch.STAGE_SPECS))
    s.add_argument("--from-run", required=True, help="run id whose "
                    "artifacts.tar.gz is the input")
    s.add_argument("--base", help="HF base model id (merge)")
    s.add_argument("--provider", default="runpod",
                   choices=available_providers())
    s.add_argument("--dry", action="store_true")
    s.add_argument("--max-runtime", type=int, default=5400)
    s.add_argument("--max-cost", type=float, default=None,
                   help="kill threshold $ (default: est-minutes x rate x "
                        "margin from orchestrator/budget.py)")
    s.add_argument("--max-price", type=float, default=None,
                   help="rate ceiling $/hr (default: stage spec, 0.80)")
    s.add_argument("--wait", type=int, default=None,
                   help="capacity-wait seconds (default: $QTRAIN_MAX_WAIT "
                        "or 1800; 0 = fast-fail single wave)")
    s.set_defaults(func=cmd_stage)

    s = sub.add_parser("register", help="deliver a run/stage artifact to a "
                        "serving host (nano0 vLLM)")
    s.add_argument("run_id")
    s.add_argument("--name", help="model dir name (default kind-runid)")
    s.add_argument("--host", default=None,
                   help="serve host (default: $QTRAIN_SERVE_HOST)")
    s.add_argument("--dest", default=None,
                   help="model dir on host (default: $QTRAIN_SERVE_DIR)")
    s.add_argument("--dry", action="store_true")
    s.set_defaults(func=cmd_register)

    s = sub.add_parser("reconcile", help="re-attach to an open run "
                        "(post-crash harvest + terminate)")
    s.add_argument("run_id", nargs="?")
    s.add_argument("--poll", type=int, default=30)
    s.set_defaults(func=cmd_reconcile)

    s = sub.add_parser("progress", help="live heartbeat from the lane")
    s.add_argument("run_id")
    s.set_defaults(func=cmd_progress)

    s = sub.add_parser("budget", help="spend visibility: balance, open "
                       "projections, caps, observed medians")
    prov_arg(s)
    s.set_defaults(func=cmd_budget)

    s = sub.add_parser("preflight", help="read-only launch decision "
                       "report: cards, estimate, gates (writes nothing)")
    s.add_argument("kind", nargs="?", default="train",
                   choices=["train"] + sorted(dispatch.STAGE_SPECS))
    prov_arg(s)
    s.add_argument("--max-price", type=float, default=None)
    s.add_argument("--max-runtime", type=int, default=3600)
    s.set_defaults(func=cmd_preflight)

    s = sub.add_parser("watch", help="follow a run until terminal state")
    s.add_argument("run_id")
    s.add_argument("--interval", type=int, default=30)
    s.set_defaults(func=cmd_watch)

    s = sub.add_parser("metrics", help="progress history + loss curve "
                        "(the wandb-lite view)")
    s.add_argument("run_id")
    s.set_defaults(func=cmd_metrics)

    s = sub.add_parser("runs", help="lane run index (the UI-facing table): "
                       "runs/index.jsonl; --reindex backfills it")
    prov_arg(s)
    s.add_argument("--limit", type=int, default=25)
    s.add_argument("--reindex", action="store_true",
                   help="rebuild the lane index from local runs + ledger")
    s.set_defaults(func=cmd_runs)

    s = sub.add_parser("events", help="tail the notification journal")
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--run", help="filter by run id")
    s.set_defaults(func=cmd_events)

    s = sub.add_parser("notify-test",
                       help="emit + probe QTRAIN_WEBHOOK_URL delivery")
    s.set_defaults(func=cmd_notify_test)

    s = sub.add_parser("bench", help="benchmark model backends (identity + "
                       "speed matrix over OpenAI-compatible endpoints)")
    s.add_argument("--config", help="json: urls/targets/questions/params")
    s.add_argument("--url", action="append", help="endpoint, repeatable "
                        "(default: nano0 :8000/:8001)")
    s.add_argument("--model", action="append",
                   help="pin model ids with --url; default = everything "
                        "the endpoint's /v1/models lists")
    s.add_argument("--question", action="append",
                   help='"id|prompt|check" — check: contains:X regex:P '
                        "one_of:a,b json none; repeatable")
    s.add_argument("--repeats", type=int, default=None)
    s.add_argument("--max-tokens", type=int, default=None)
    s.add_argument("--temperature", type=float, default=None)
    s.add_argument("--record", action="store_true",
                   help="persist scores to the eval-history ledger")
    s.add_argument("--run", default="",
                   help="key the recording on an artifact run id")
    s.add_argument("--json", dest="json", help="write rows json here")
    s.add_argument("--csv", dest="csv", help="write rows csv here")
    s.set_defaults(func=cmd_bench)

    s = sub.add_parser("evals", help="eval-history ledger: recorded bench "
                       "scores per model/run (SPEC 13am)")
    s.add_argument("--model", help="filter by model id")
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_evals)

    s = sub.add_parser("verify", help="provenance cross-check: ledger vs "
                                      "lane payload vs agent attestation "
                                      "(SPEC 13ak); non-zero exit on FAIL")
    s.add_argument("run_id", nargs="?", help="default: most recent run")
    s.add_argument("--live", action="store_true",
                   help="also ask the running pod /v1/sanity")
    s.add_argument("--recompute", action="store_true",
                   help="re-expand the domain now and compare (slow)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("reap", help="terminate live GPUs with no active "
                       "task behind them (idempotency hygiene)")
    prov_arg(s)
    s.add_argument("--dry", action="store_true",
                   help="report what would die, kill nothing")
    s.set_defaults(func=cmd_reap)

    s = sub.add_parser("janitor", help="one full idempotency sweep: reap "
                       "+ relaunch stale staged-never-launched runs")
    prov_arg(s)
    s.add_argument("--dry", action="store_true")
    s.set_defaults(func=cmd_janitor)

    s = sub.add_parser("hygiene", help="lane quota housekeeping: delete "
                       "ONLY proven junk (orphan .part under terminal "
                       "runs / artifacts whose manifest records a "
                       "--bad-sha blob), verify-after-delete")
    s.add_argument("--bad-sha", action="append", default=[],
                   metavar="SHA256",
                   help="artifact sha256 proven junk (repeatable); the "
                        "manifest is the proof — no download needed")
    s.add_argument("--apply", action="store_true",
                   help="actually delete (default: print the plan only)")
    s.set_defaults(func=cmd_hygiene)

    s = sub.add_parser("doctor", help="orphan sweep: live pods vs ledger")
    prov_arg(s)
    s.set_defaults(func=cmd_doctor)
    return p


def main(argv=None):
    load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except ProviderError as e:
        print(f"qtrain: error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
