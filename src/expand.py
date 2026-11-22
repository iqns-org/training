#!/usr/bin/env python3
import itertools, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Corpus data may live in a sibling checkout (quadra/corpora). Unset CORPORA_DIR
# keeps the legacy layout (data inside this repo) working untouched.
import os

DATA_ROOT = Path(os.environ.get("CORPORA_DIR") or ROOT)
PAT = re.compile(r"\{\{\s*\$\.([A-Za-z0-9_]+)\[\?\]\s*\}\}|\$\{([A-Za-z0-9_]+)\[\?\]\}")


def _key_from_match(match):
    return match.group(1) or match.group(2)


def subst(obj, mapping):
    if isinstance(obj, str):
        return PAT.sub(lambda m: str(mapping[_key_from_match(m)]), obj)
    if isinstance(obj, list):
        return [subst(x, mapping) for x in obj]
    if isinstance(obj, dict):
        return {k: subst(v, mapping) for k, v in obj.items()}
    return obj


def load_default(domain: Path):
    default_path = domain / "default.json"
    if not default_path.is_file():
        return []
    with default_path.open(encoding="utf-8") as f:
        default = json.load(f)
    return default.get("messages", [])


def expand_message(message):
    dollar = message.get("$") or {}
    body = {k: v for k, v in message.items() if k != "$"}
    keys = sorted(set(_key_from_match(m) for m in PAT.finditer(json.dumps(body, ensure_ascii=False)) if _key_from_match(m)))
    if not keys:
        return [body]
    miss = [k for k in keys if k not in dollar]
    if miss:
        raise SystemExit(f"missing $ arrays: {miss}")
    return [
        subst(body, dict(zip(keys, combo)))
        for combo in itertools.product(*[dollar[k] for k in keys])
    ]


def expand_row(row, default_messages=None, default_dollar=None):
    default_messages = [msg for default in (default_messages or []) for msg in expand_message(default)]
    row_dollar = dict(row.get("$") or {})
    body = {k: v for k, v in row.items() if k != "$"}
    body["messages"] = default_messages + body.get("messages", [])
    keys = sorted(
        {
            _key_from_match(match)
            for match in PAT.finditer(json.dumps(body, ensure_ascii=False))
            if _key_from_match(match)
        }
    )
    if not keys:
        return [body]
    miss = [k for k in keys if k not in row_dollar]
    if miss:
        raise SystemExit(f"missing $ arrays: {miss}")
    return [
        subst(body, dict(zip(keys, combo)))
        for combo in itertools.product(*[row_dollar[k] for k in keys])
    ]


def resolve_src(name: str) -> Path:
    name = name.removesuffix(".jsonl")
    for c in (
        Path(f"{name}.jsonl"),
        DATA_ROOT / f"{name}.jsonl",
        DATA_ROOT / "datasets" / f"{name}.jsonl",
        ROOT / f"{name}.jsonl",
        ROOT / "datasets" / f"{name}.jsonl",
        Path("datasets") / f"{name}.jsonl",
    ):
        if c.is_file():
            return c
    raise SystemExit(f"not found: {name}.jsonl")


def resolve_domain(domain: str) -> Path:
    for c in (
        Path(domain),
        DATA_ROOT / domain,
        DATA_ROOT / "datasets" / domain,
        ROOT / domain,
        ROOT / "datasets" / domain,
        Path("datasets") / domain,
    ):
        if c.is_dir():
            return c
    raise SystemExit(f"not found: domain {domain}")


def expand_file(src: Path, dst: Path, default_messages=None, default_dollar=None) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open(encoding="utf-8") as f, dst.open("w", encoding="utf-8") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for row in expand_row(json.loads(line), default_messages, default_dollar):
                g.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    print(f"{src} -> {dst} ({n})")
    return n


def main_one(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        raise SystemExit("usage: expand {name}")
    src = resolve_src(argv[0])
    domain = src.parent if src.parent.name != "datasets" else src.parent.parent
    default_messages = load_default(domain)
    expand_file(src, ROOT / "tmp" / src.name, default_messages)


def main_all(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        raise SystemExit("usage: expand-all {domain}")
    domain = resolve_domain(argv[0])
    default_messages = load_default(domain)
    files = sorted(domain.rglob("*.jsonl"))
    if not files:
        raise SystemExit(f"no jsonl in {domain}")
    total = 0
    for src in files:
        rel = src.relative_to(domain)
        total += expand_file(src, ROOT / "tmp" / domain.name / rel, default_messages)
    print(f"{len(files)} files, {total} rows")
