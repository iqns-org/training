#!/usr/bin/env python3
"""R6: LoRA merge-algebra tooling — mergekit config generation (REGIMES R6).

Persona, knowledge and cognition train as separate LoRA tracks (the axis
law); composition is mergekit's job: TIES/DARE density merges of adapter
artifacts into ONE merged adapter = single-LoRA serving, no runtime
stacking. This module generates the config; the pod `stage` lane executes
it (pending — needs an adapter-rich run set; dry-run config proven).

Usage:
  python postproc/merge_loras.py --base Qwen/Qwen3.5-0.8B \
      --adapters /models/kandi-lora,/models/cognition-lora \
      --method ties --density 0.5 --out tmp/merge.yaml
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:                            # stdlib fallback: pod lane has PyYAML
    yaml = None


def gen_merge_config(base, adapters, method="ties", density=0.5,
                     out_path="./merged", merge_params=None):
    """mergekit config dict. adapters ordered: base FIRST then LoRAs."""
    assert adapters, "need at least one adapter"
    models = [{"base_model": base}]
    for a in adapters:
        models.append({"model": a})
    cfg = {
        "models": models,
        "merge_method": method,          # ties | dare_ties | dare_linear | slerp
        "parameters": {
            "normalize": False,
            "int8 masking": False,
            "density": density,          # sparsity fraction kept (DARE/TIES)
            **(merge_params or {}),
        },
        "dtype": "bfloat16",
        "output_mode": "lora",           # adapter output, serve via vLLM --enable-lora
        "base_model": base,
        "out_path": out_path,
    }
    return cfg


def to_yaml(cfg):
    if yaml:
        return yaml.safe_dump(cfg, sort_keys=False)
    raise SystemExit("PyYAML required for yaml output (use --json)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="merge_loras",
                                description="mergekit config generator (R6)")
    p.add_argument("--base", required=True)
    p.add_argument("--adapters", required=True, help="comma-separated paths/run dirs")
    p.add_argument("--method", default="ties",
                   choices=("ties", "dare_ties", "dare_linear", "slerp"))
    p.add_argument("--density", type=float, default=0.5)
    p.add_argument("--param", action="append", default=[],
                   help="extra merge param k=v (int overrides)")
    p.add_argument("--out", default="-")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    extra = {}
    for kv in args.param:
        k, _, v = kv.partition("=")
        try:
            extra[k] = json.loads(v)
        except Exception:
            extra[k] = v
    cfg = gen_merge_config(args.base,
                           [a.strip() for a in args.adapters.split(",") if a.strip()],
                           args.method, args.density, out_path=args.out if args.out != "-" else "./merged",
                           merge_params=extra)
    text = json.dumps(cfg, indent=1) if args.json or not yaml else to_yaml(cfg)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text)
        print(json.dumps({"out": args.out, "adapters": len(cfg["models"]) - 1,
                          "method": cfg["merge_method"],
                          "density": cfg["parameters"]["density"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
