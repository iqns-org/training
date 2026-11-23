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

from .config import load_dotenv
from . import dispatch
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
        args.domain, Path(args.recipe), mini=args.mini, overrides=overrides,
        provider=args.provider, gpus=args.gpu or None, tier=args.tier,
        dry=args.dry, max_runtime=args.max_runtime, max_cost=args.max_cost,
        force_lane=args.lane)
    _print(res)
    raise SystemExit(0 if res.get("status") in ("succeeded", None) and not res.get("error") else 1)


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
    s.add_argument("--recipe", required=True, help="axolotl YAML path")
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
    s.add_argument("--max-cost", type=float, default=0.75)
    s.set_defaults(func=cmd_submit)

    s = sub.add_parser("doctor", help="orphan sweep: live pods vs ledger")
    prov_arg(s)
    s.set_defaults(func=cmd_doctor)
    return p


def main(argv=None):
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
