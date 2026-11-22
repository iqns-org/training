"""Rich command line for GPU compute: ``bin/gpu``.

    bin/gpu [--provider gpuai|runpod] <command> [args]

    provision   create (or reuse) an instance            → provision/creation
    status      show instance(s), optionally watch       → monitoring
    install     local venv or remote training setup      → software installer
    upload      copy local path to instance              → data upload
    run         schedule axolotl training + monitor      → job scheduling
    download    fetch artifacts to outputs/<domain>      → data download
    terminate   delete an instance                       → gpu termination
    keys        create/list SSH keys
    fine-tune   legacy thin wrapper (dataset+model+method)

Run `bin/gpu <command> --help` for per-command flags.
"""

import argparse
import json
import sys
from pathlib import Path

from . import available_providers, get_provider
from .common import GpuError, job_file, load_dotenv

DEFAULT_PROVIDER = "gpuai"


def _dump(obj):
    print(json.dumps(obj, indent=2, default=str))


def _provider(args, **extra):
    load_dotenv()
    return get_provider(args.provider, **extra)


def cmd_provision(args):
    from . import common

    provider = _provider(args)
    job = common.job_file(args.provider, args.app, args.domain) if args.domain else None
    res = provider.provision(app=args.app, name=args.name, ssh_key_id=args.ssh_key_id,
                             job_file=str(job) if job else None,
                             force=args.force)
    _dump(res)


def cmd_status(args):
    provider = _provider(args)
    if args.instance:
        _dump(provider.status(args.instance))
    else:
        _dump(provider.list_instances())


def cmd_install(args):
    from . import install as installer

    if args.remote:
        provider = _provider(args)
        _dump(installer.install_remote(
            provider, args.remote, conda_env=args.conda_env,
            fix_script=args.fix_script))
    else:
        _dump(installer.install_local(
            python=args.python, venv=args.venv, torch_index=args.torch_index))


def cmd_check(args):
    from . import install as installer

    _dump(installer.check_local(venv=args.venv))


def cmd_upload(args):
    provider = _provider(args)
    _dump(provider.upload(args.instance, args.src, args.dst))


def cmd_run(args):
    from . import axolotl as workflow

    root = Path(__file__).resolve().parents[2]
    names = [args.provider] + [f.strip() for f in (args.fallback or "").split(",")
                               if f.strip()]
    named = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        named.append((name, get_provider(name),
                      job_file(name, workflow.APP, args.domain)))
    if len(named) == 1:
        name, provider, job = named[0]
        res = workflow.run_training(
            provider, root, args.domain, model=args.model,
            strategy=args.strategy, quick=args.quick, job_file=job,
            timeout=args.timeout, resume=args.resume)
        res["provider"] = name
        _dump(res)
    else:
        _dump(workflow.run_with_fallback(
            named, root, args.domain, model=args.model,
            strategy=args.strategy, quick=args.quick,
            timeout=args.timeout, resume=args.resume))


def cmd_download(args):
    from . import axolotl as workflow

    provider = _provider(args)
    root = Path(__file__).resolve().parents[2]
    _dump(workflow.download_artifacts(
        provider, root, args.domain,
        instance_id=args.instance, out=args.out))


def cmd_terminate(args):
    provider = _provider(args)
    _dump(provider.terminate(args.instance, wait_timeout=args.wait_timeout))


def cmd_keys(args):
    provider = _provider(args)
    if args.keys_action == "create":
        _dump(provider.keys_create(args.name, args.public_key_file))
    else:
        res = provider.keys_list()
        _dump(res)


def cmd_fine_tune(args):
    provider = _provider(args)
    if not hasattr(provider, "fine_tune"):
        raise GpuError(f"provider {args.provider!r} has no fine-tune API; "
                       f"use `bin/gpu run` instead")
    _dump(provider.fine_tune(args.dataset, args.model, method=args.method,
                             max_budget=args.max_budget,
                             wait_timeout=args.wait_timeout,
                             lora_alpha=args.lora_alpha,
                             lora_dropout=args.lora_dropout, lora_r=args.lora_r))


def build_parser():
    p = argparse.ArgumentParser(prog="gpu", description=__doc__.splitlines()[0])
    p.add_argument("--provider", default=DEFAULT_PROVIDER,
                   choices=available_providers(),
                   help="compute backend (default: %(default)s)")
    sub = p.add_subparsers(dest="command", required=True, metavar="<command>")

    s = sub.add_parser("provision", help="provision/create an instance")
    s.add_argument("--app", default="axolotl-finetuning")
    s.add_argument("--name", default=None)
    s.add_argument("--domain", default=None, help="track job file per domain")
    s.add_argument("--ssh-key-id", default=None)
    s.add_argument("--force", action="store_true", help="ignore saved job file")
    s.set_defaults(func=cmd_provision)

    s = sub.add_parser("status", help="show instance(s)")
    s.add_argument("instance", nargs="?", default=None)
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("install", help="software installer (local venv or remote setup)")
    s.add_argument("--remote", default=None, metavar="INSTANCE",
                   help="run remote setup instead of local install")
    s.add_argument("--python", default=None)
    s.add_argument("--venv", default=None)
    s.add_argument("--torch-index", default=None)
    s.add_argument("--conda-env", default="py3.11")
    s.add_argument("--fix-script", default=None)
    s.set_defaults(func=cmd_install)

    s = sub.add_parser("check", help="report local torch/cuda status")
    s.add_argument("--venv", default=None)
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("upload", help="data upload to instance")
    s.add_argument("instance")
    s.add_argument("src")
    s.add_argument("dst")
    s.set_defaults(func=cmd_upload)

    s = sub.add_parser("run", help="schedule axolotl training + monitor to completion")
    s.add_argument("domain")
    s.add_argument("model", nargs="?", default="SmolLM2-360M-Instruct")
    s.add_argument("strategy", nargs="?", default="qlora")
    s.add_argument("-q", "--quick", action="store_true",
                   help="skip dataset upload, refresh config/scripts only")
    s.add_argument("--timeout", type=int, default=86400,
                   help="remote training timeout in seconds (default: %(default)s)")
    s.add_argument("--resume", action="store_true",
                   help="reuse the live saved instance and skip dataset "
                        "re-upload; starts fresh if none is live")
    s.add_argument("--fallback", default="",
                   metavar="PROVIDERS",
                   help="comma-separated providers to try in order after "
                        "--provider, e.g. --fallback runpod "
                        "(skipped automatically on keep-alive failures)")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("download", help="data download of trained artifacts")
    s.add_argument("domain", nargs="?", default=None)
    s.add_argument("--instance", default=None)
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_download)

    s = sub.add_parser("terminate", help="gpu termination")
    s.add_argument("instance")
    s.add_argument("--wait-timeout", default="5m")
    s.set_defaults(func=cmd_terminate)

    s = sub.add_parser("keys", help="SSH keys")
    s.add_argument("keys_action", choices=["create", "list"], metavar="action")
    s.add_argument("--name", default=None)
    s.add_argument("--public-key-file", default="secrets/gpuai.pub")
    s.set_defaults(func=cmd_keys)

    s = sub.add_parser("fine-tune", help="legacy thin fine-tune wrapper (gpuai)")
    s.add_argument("--dataset", required=True)
    s.add_argument("--model", default="SmolLM2-360M-Instruct")
    s.add_argument("--method", default="qlora")
    s.add_argument("--max-budget", default=1, type=float)
    s.add_argument("--wait-timeout", default="2h")
    s.add_argument("--lora-alpha", default=32, type=float)
    s.add_argument("--lora-dropout", default=0.05, type=float)
    s.add_argument("--lora-r", default=16, type=float)
    s.set_defaults(func=cmd_fine_tune)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except GpuError as e:
        print(f"gpu: error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("gpu: interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
