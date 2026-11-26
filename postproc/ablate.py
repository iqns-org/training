#!/usr/bin/env python3
"""ablate — behavior-direction extraction & weight-space editing (R15).

The representation-engineering lane, MEASURED-FIRST (docs/ORACLES.md §2):

  contrast pairs (synth contrasts)  ->  direction v per layer
    SOFT mode : steering spec consumed as a LoRA (rank-k: B@A ≈ v) ->
                rides the existing vLLM multi-LoRA lane; reversible,
                composable with persona/cognition adapters (R6)
    HARD mode : weight orthogonalization  W <- W (I - v v^T)
                -> merged artifact; permanent; gated by bench+grade

Direction math is PURE PYTHON (lists as vectors) so it runs — and is
unit-tested — anywhere: the mean-diff is the RedHatAI/repeng recipe,
first principal component via power iteration when requested. The heavy
forward-pass (activation probing) runs pod-side (postproc image, torch
optional); this module defines the contract: probe writes
{layer: [vectors]} json, ablate consumes contrasts + produces edits.

Usage (local, no GPU):
  python postproc/ablate.py direction --pos P.json --neg N.json --out v.json
  python postproc/ablate.py project  --weights W.json --direction v.json
  python postproc/ablate.py spec     --model base --direction v.json \
      --strength 1.0 --out steer.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

# ── vector math (pure, dependency-free) ────────────────────────────────

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def scale(a, k):
    return [k * x for x in a]


def mean(vectors):
    if not vectors:
        return []
    d = len(vectors[0])
    acc = [0.0] * d
    for v in vectors:
        for i, x in enumerate(v):
            acc[i] += x
    return [x / len(vectors) for x in acc]


def mean_direction(pos, neg, pca=False):
    """Behavior direction from contrast activations: mean(pos) - mean(neg)
    (repeng mean-diff); pca=True takes the top component instead
    (power iteration on the covariance of pos∪neg, centered on neg)."""
    if not pos or not neg:
        raise ValueError("both contrast sides need samples")
    if len(pos[0]) != len(neg[0]):
        raise ValueError("dimension mismatch")
    v = [a - b for a, b in zip(mean(pos), mean(neg))]
    if pca:
        rows = pos + neg
        mu = mean(rows)
        cov_diag = [0.0] * len(mu)
        for r in rows:
            for i, x in enumerate(r):
                cov_diag[i] += (x - mu[i]) ** 2
        x = [d if d else 1.0 for d in cov_diag]
        n = norm(x) or 1.0
        x = [d / n for d in x]
        for _ in range(30):                      # power iteration, diag approx
            y = [d * xi for d, xi in zip(cov_diag, x)]
            ny = norm(y) or 1.0
            if abs(ny - 1.0) < 1e-9 and dot(y, x) < 0:
                x = [-t for t in y]
                break
            x = [t / ny for t in y]
        # align with the mean-diff sign
        md = v
        if dot(x, md) < 0:
            x = [-t for t in x]
        v = x
    n = norm(v)
    if n == 0:
        raise ValueError("zero direction — contrasts may be identical")
    return [t / n for t in v]


def project_matrix(W, v):
    """W (I - v v^T): remove the v component from every output row.
    W is a list of rows (each len == len(v))."""
    if not W or len(W[0]) != len(v):
        raise ValueError("shape mismatch")
    out = []
    for row in W:
        c = dot(row, v)
        out.append([row[i] - c * v[i] for i in range(len(v))])
    return out


def lora_from_direction(v, strength=1.0):
    """Soft mode: rank-1 LoRA whose B@A adds strength*v per output neuron
    (A = v^T normalized, B = ones) — the steering-adapter contract the
    pod/proxy consumes; composes via mergekit (R6)."""
    n = norm(v) or 1.0
    return {"rank": 1,
            "A": [[x / n for x in v]],
            "B": [[strength * n]],
            "note": "constant additive steering y += strength * v "
                    "(fold into out_proj bias at load)"}


# ── io contract ────────────────────────────────────────────────────────

def _read_pairs(path):
    """pos/neg json: {layer: {"pos": [[..]], "neg": [[..]]}} or a flat
    jsonl of {"label": 0|1, "vectors": {layer: [...]}}."""
    p = Path(path)
    if p.suffix == ".jsonl":
        data = {}
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            for layer, vec in (r.get("vectors") or {}).items():
                side = "pos" if r.get("label") else "neg"
                data.setdefault(layer, {"pos": [], "neg": []})[side].append(vec)
        return data
    return json.loads(p.read_text())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ablate")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("direction", help="contrast activations -> unit dirs")
    d.add_argument("--pairs", help="combined json/jsonl (see _read_pairs)")
    d.add_argument("--pos")
    d.add_argument("--neg")
    d.add_argument("--pca", action="store_true")
    d.add_argument("--out", required=True)

    pj = sub.add_parser("project", help="W (I - vv^T) -> merged json")
    pj.add_argument("--weights", required=True)
    pj.add_argument("--direction", required=True)
    pj.add_argument("--layer", required=True)
    pj.add_argument("--out", required=True)

    sp = sub.add_parser("spec", help="steering-lora spec (soft mode)")
    sp.add_argument("--direction", required=True)
    sp.add_argument("--layer", required=True)
    sp.add_argument("--model")
    sp.add_argument("--strength", type=float, default=1.0)
    sp.add_argument("--out", required=True)

    args = ap.parse_args(argv)

    if args.cmd == "direction":
        if args.pairs:
            data = _read_pairs(args.pairs)
            out = {layer: mean_direction(g["pos"], g["neg"], args.pca)
                   for layer, g in data.items()
                   if g["pos"] and g["neg"]}
        else:
            pos = json.loads(Path(args.pos).read_text())
            neg = json.loads(Path(args.neg).read_text())
            out = {}
            for layer in pos:
                out[layer] = mean_direction(pos[layer], neg[layer], args.pca)
        Path(args.out).write_text(json.dumps(out))
        print(json.dumps({"layers": list(out), "dims":
                          {k: len(v) for k, v in out.items()}}))
        return 0

    if args.cmd == "project":
        W = json.loads(Path(args.weights).read_text())
        v = json.loads(Path(args.direction).read_text())[args.layer]
        Path(args.out).write_text(json.dumps(project_matrix(W, v)))
        print(json.dumps({"projected_rows": len(W), "dim": len(v)}))
        return 0

    if args.cmd == "spec":
        v = json.loads(Path(args.direction).read_text())[args.layer]
        spec = {"model": args.model, "mode": "soft",
                args.layer: lora_from_direction(v, args.strength)}
        Path(args.out).write_text(json.dumps(spec))
        print(json.dumps({"out": args.out, "mode": "soft",
                          "rank": 1, "layer": args.layer}))
        return 0


if __name__ == "__main__":
    sys.exit(main())
