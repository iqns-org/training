#!/usr/bin/env python3
"""Expand IR curriculum templates into concrete training records.

Takes a domain filter (or 'all'), generates variants, writes outputs to
{out}/ir/{system}/{example}-{variant}.jsonl with interpolated messages,
mirroring generate/scenarios/{system}/{example}.json → {out}/ir/{system}/{example}-{variant}.jsonl.
Optional --system injects the relevant system prompt from generate/prompts.
Optional --out specifies base output folder (defaults to ./tmp/).

Usage:
  ir-scenarios-expand [domain] [--system] [--out DIR]
  ir-scenarios-expand all --system --out ./tmp
  ir-scenarios-expand book --system
"""
import argparse
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*)\}")

# candidate prompt roots (supports both layouts)
PROMPT_ROOTS = [
    ROOT / "generate" / "prompts" / "ir",
    ROOT / "prompts" / "generate" / "ir",
    ROOT / "prompts" / "ir",
    ROOT / "generate" / "prompts",
]
SCENARIO_ROOTS = [
    ROOT / "generate" / "scenarios",
    ROOT / "scenarios",
]
CONVERSATIONS_CANDIDATES = [
    ROOT / "generate" / "prompts" / "scenarios" / "conversations.md",
    ROOT / "prompts" / "generate" / "scenarios" / "conversations.md",
    ROOT / "generate" / "prompts" / "scenarios.md",
]

def find_dir(candidates):
    for p in candidates:
        if p.is_dir():
            return p
    return candidates[0]

def find_file(candidates):
    for p in candidates:
        if p.is_file():
            return p
    return None

PROMPT_DIR = find_dir(PROMPT_ROOTS)
SCENARIO_DIR = find_dir(SCENARIO_ROOTS)
CONVERSATIONS = find_file(CONVERSATIONS_CANDIDATES)

def expand(value, choices, stack=()):
    if isinstance(value, str):
        def resolve(match):
            key = match[1]
            if key in stack:
                raise ValueError(f"cyclic placeholder {key}")
            result = choices
            for part in key.split("."):
                result = result[part]
            # If result is a dict with iri, return the iri string for string interpolation
            if isinstance(result, dict) and "iri" in result and len(result) == 1:
                return result["iri"]
            if isinstance(result, dict) and "iri" in result:
                # For dicts like {"iri": "my:X", "text": "x"}, return iri for string context
                # But for entire placeholder, return the dict as is for assert handling
                # We need to distinguish: if entire placeholder, return dict; else return iri string
                pass
            return expand(result, choices, (*stack, key))
        entire = PLACEHOLDER.fullmatch(value)
        if entire:
            res = resolve(entire)
            # For entire placeholder, if it's a dict with iri, return iri string (for assert triples)
            if isinstance(res, dict) and "iri" in res:
                return res["iri"]
            return res
        def repl(m):
            res = resolve(m)
            if isinstance(res, dict) and "iri" in res:
                return res["iri"]
            return str(res)
        return PLACEHOLDER.sub(repl, value)
    if isinstance(value, list):
        return [expand(v, choices, stack) for v in value]
    if isinstance(value, dict):
        return {k: expand(v, choices, stack) for k, v in value.items()}
    return value

def load_system_prompt(system: str) -> str | None:
    # try generate/prompts/ir/{system}/system.md then prompt.md
    for base in PROMPT_ROOTS:
        for name in ("system.md", "prompt.md"):
            p = base / system / name
            if p.is_file():
                c = p.read_text(encoding="utf-8")
                if CONVERSATIONS and CONVERSATIONS.is_file():
                    base_text = CONVERSATIONS.read_text(encoding="utf-8")
                    return base_text.strip() + "\n\n" + c.strip()
                return c.strip()
    return None

def ensure_prompts_have_scenarios():
    prompts = set()
    if PROMPT_DIR.is_dir():
        for child in PROMPT_DIR.iterdir():
            if child.is_dir():
                # check has system.md
                if any((child / n).is_file() for n in ("system.md", "prompt.md")):
                    prompts.add(child.name)
    scenarios = set()
    if SCENARIO_DIR.is_dir():
        for child in SCENARIO_DIR.iterdir():
            if child.is_dir():
                scenarios.add(child.name)
    missing = sorted(prompts - scenarios)
    extra = sorted(scenarios - prompts)
    return prompts, scenarios, missing, extra

def sanitize_domain(domain_choices: dict) -> str:
    # Try to find a human-readable domain key: subject.text > subject.type > first domain value text
    if not domain_choices:
        return "generic"
    # domain_choices is like {"domain": {...}} where {...} is a domain fixture
    # But our choices dict is flat: {"domain": {...}, "phrasing": "...", "prefix": {...}}
    # We need to extract domain fixture's subject text
    dom = domain_choices.get("domain")
    if isinstance(dom, dict):
        subj = dom.get("subject", {})
        if isinstance(subj, dict) and "text" in subj:
            return re.sub(r"[^a-z0-9]+", "-", subj["text"].lower()).strip("-") or "generic"
        # fallback to first text field in domain
        for v in dom.values():
            if isinstance(v, dict) and "text" in v:
                return re.sub(r"[^a-z0-9]+", "-", v["text"].lower()).strip("-")
        # fallback to iri
        if "subject" in dom and isinstance(dom["subject"], dict) and "iri" in dom["subject"]:
            return re.sub(r"[^a-z0-9]+", "-", dom["subject"]["iri"].split(":")[-1].lower()).strip("-")
    return "generic"

def main(argv=None):
    parser = argparse.ArgumentParser(description="Expand IR curriculum templates")
    parser.add_argument("domain", nargs="?", default="all", help="domain filter (e.g., 'book' or 'all')")
    parser.add_argument("--system", action="store_true", help="inject relevant system prompt as system message")
    parser.add_argument("--out", dest="out", default=str(ROOT / "tmp"), help="base output folder (defaults to ./tmp/)")
    parser.add_argument("--scenarios", default=str(SCENARIO_DIR), help="scenarios directory")
    parser.add_argument("--dry-run", action="store_true", help="don't write files, just report")
    args = parser.parse_args(argv)

    domain_filter = args.domain.lower().strip()
    out_base = Path(args.out)
    scenarios_dir = Path(args.scenarios)
    if not scenarios_dir.is_dir():
        scenarios_dir = SCENARIO_DIR

    prompts, scenarios, missing, extra = ensure_prompts_have_scenarios()
    if missing:
        print(f"warning: prompts without scenarios: {', '.join(missing)}", file=sys.stderr)
        # we consider this an error if --system is used, otherwise just warn
        # but spec says ensure all prompts have corresponding scenarios
        if args.system:
            print("hint: create generate/scenarios/{prompt}/ for each prompt", file=sys.stderr)
    if extra:
        print(f"note: scenarios without prompts: {', '.join(extra)}", file=sys.stderr)

    files = sorted(scenarios_dir.rglob("*.json"))
    if not files:
        print(f"no scenario files in {scenarios_dir}", file=sys.stderr)
        return 1

    # cache system prompts
    system_cache = {}
    if args.system:
        for p in prompts:
            sp = load_system_prompt(p)
            if sp:
                system_cache[p] = sp
            else:
                print(f"warning: no system prompt for {p}", file=sys.stderr)

    total_expansions = 0
    written = 0
    skipped = 0

    for path in files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"skip {path}: {e}", file=sys.stderr)
            continue
        system = record.get("system", "unknown")
        id_ = record.get("id", str(path.stem))
        # id is like "valid-program/01-lookup"
        example = id_.replace("/", "-")
        variants = record.get("variants", {})
        # validate pools
        for k, pool in variants.items():
            if not isinstance(pool, list) or not pool:
                print(f"skip {path}: empty pool {k}", file=sys.stderr)
                continue
        # iterate product
        keys = list(variants.keys())
        pools = [variants[k] for k in keys]
        for variant_idx, selected in enumerate(itertools.product(*pools)):
            choices = dict(zip(keys, selected))
            # Build flat choices for placeholder expansion (choices dict with nested dicts)
            # expand needs choices as dict of top-level keys mapping to selected dict/string
            # For domain filtering, check if domain filter matches any domain fixture text/iri
            if domain_filter != "all":
                # check against domain fixture and phrasing and prefix
                hit = False
                # if filter matches system name, allow
                if domain_filter in system.lower() or domain_filter in id_.lower():
                    hit = True
                else:
                    # check domain fixture fields
                    dom = choices.get("domain")
                    if isinstance(dom, dict):
                        # flatten and search
                        def flat(d, pre=""):
                            out = {}
                            for kk, vv in d.items():
                                if isinstance(vv, dict):
                                    out.update(flat(vv, pre+kk+"."))
                                else:
                                    out[pre+kk] = str(vv).lower()
                            return out
                        flat_dom = flat(dom)
                        for v in flat_dom.values():
                            if domain_filter in v:
                                hit = True
                                break
                        # also check phrasing choice if present
                        phr = choices.get("phrasing")
                        if isinstance(phr, str) and domain_filter in phr.lower():
                            hit = True
                    else:
                        # generic fallback: check string representation of choices
                        if domain_filter in json.dumps(choices).lower():
                            hit = True
                if not hit:
                    skipped += 1
                    continue

            # expand messages and assert
            try:
                messages_expanded, assert_expanded = expand(record["messages"], choices), expand(record["assert"], choices)
                # messages is list of {role, content} - expand already did
                # but expand for messages expects list of messages with placeholders
                # we have already expanded via expand on the list
                # Do full deep expand with choices (handles nested placeholders)
                # The above expand on messages does recursive; ensure it handles {phrasing} etc.
                pass
            except Exception as e:
                print(f"skip variant {variant_idx} of {id_}: expand error {e}", file=sys.stderr)
                continue

            # Ensure system injection
            final_messages = list(messages_expanded)
            if args.system:
                sp = system_cache.get(system)
                if sp:
                    final_messages = [{"role": "system", "content": sp}] + final_messages
                else:
                    # fallback: try load per-file system
                    sp2 = load_system_prompt(system)
                    if sp2:
                        final_messages = [{"role": "system", "content": sp2}] + final_messages

            # build output record (training record) – messages + meta envelope (id inside meta only)
            example_stem = path.stem
            variant_id = f"{id_}-{variant_idx}"
            out_record = {
                "meta": {
                    "id": variant_id,
                    "domain": system,
                    "scenario": example_stem,
                },
                "messages": final_messages,
            }

            # output path: {out}/ir/{system}/{example_stem}-{variant}.jsonl
            # Follows original structure: generate/scenarios/{system}/{example}.json -> {out}/ir/{system}/{example}-{variant}.jsonl
            example_stem = path.stem
            out_dir = out_base / "ir" / system
            out_path = out_dir / f"{example_stem}-{variant_idx}.jsonl"
            if args.dry_run:
                total_expansions += 1
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(out_record, ensure_ascii=False, indent=2) + "\n")
            written += 1
            total_expansions += 1

    print(f"expanded {total_expansions} variants ({written} written, {skipped} skipped by domain filter)")
    if not args.dry_run:
        print(f"output base: {out_base / 'ir'}")
    if missing:
        print(f"missing scenarios for prompts: {missing}", file=sys.stderr)
        # ensure we return non-zero if missing and --system? keep warning but not fail
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
