#!/usr/bin/env python3
"""
ir-generator: combine domain/case/turn fixtures with generator prompts,
call Groq LLM, validate JSONL {messages: []}, write structured tree.

Config: config/ir-generator.json
Env: GROQ_AI_API_KEY via .env

Usage:
  ir-generator [--compact] [--config path] [--domains path] [--cases path] [--turns path] [--output path]
  ir-generator --help

Behavior:
  - First prompt in config.prompts is system prompt, rest concatenated as user prompt
  - Injects {{domain}}, {{case}}, {{turn}} with JSON dumps of each fixture
  - Iterates over all domain/case/turn combinations (cartesian product)
  - Calls LLM at config.endpoint with config.model etc.
  - Validates each response line is JSON with {messages: [{role, content}]}
  - Writes to {output}/{domain.id}/... or with --compact to {output}/{domain.id}.jsonl
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
import itertools

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "ir-generator.json"

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

def estimate_tokens(text: str) -> int:
    """Estimate tokens for text. Try tiktoken, else heuristic."""
    if not text:
        return 0
    # Try tiktoken with cl100k_base (gpt-4) or o200k_base
    try:
        import tiktoken
        try:
            enc = tiktoken.get_encoding("o200k_base")
        except:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except:
        pass
    # Fallback: heuristic ~4 chars per token, or 0.75 tokens per word
    # Use max of both to be safe
    c_est = len(text) / 4
    w_est = len(text.split()) * 1.3
    return int(max(c_est, w_est) + 0.5)

def load_dotenv():
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line=line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k,v = line.split("=",1)
        k=k.strip(); v=v.strip().strip('"').strip("'")
        if k not in os.environ:
            os.environ[k]=v

def resolve_env_template(s: str) -> str:
    # handles {{env.VAR}} and {{env.VAR}} inside strings
    def repl(m):
        var=m.group(1)
        return os.environ.get(var, "")
    return re.sub(r"\{\{\s*env\.([A-Za-z0-9_]+)\s*\}\}", repl, s)

def load_config(path: Path):
    cfg=json.loads(path.read_text(encoding="utf-8"))
    # resolve env templates recursively
    def rec(v):
        if isinstance(v, str):
            return resolve_env_template(v)
        if isinstance(v, dict):
            return {k: rec(x) for k,x in v.items()}
        if isinstance(v, list):
            return [rec(x) for x in v]
        return v
    return rec(cfg)

def load_prompts(prompt_paths):
    # prompt_paths: list of paths from config – strictly obey config
    texts=[]
    for p in prompt_paths:
        cand = ROOT / p.lstrip("./")
        if not cand.is_file():
            # try alternative with comma typo
            alt = Path(str(cand).replace("system.md", "system,.md"))
            if alt.is_file():
                cand = alt
            else:
                # try resolving relative to ROOT
                cand2 = Path(p)
                if cand2.is_file():
                    cand = cand2
                else:
                    print(f"warning: prompt not found {p}", file=sys.stderr)
                    continue
        texts.append(cand.read_text(encoding="utf-8"))
    if not texts:
        raise SystemExit("no prompts loaded")
    system_prompt = texts[0]
    user_prompt = "\n\n".join(texts[1:])
    return system_prompt, user_prompt

def ensure_prompts_have_scenarios():
    prompts = set()
    if PROMPT_DIR.is_dir():
        for child in PROMPT_DIR.iterdir():
            if child.is_dir():
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

def load_jsonl(path: Path):
    items=[]
    with path.open(encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def render_template(template: str, injects: dict) -> str:
    # Replace {{domain}}, {{case}}, {{turn}}, {{instruct}}, etc. with JSON dumps
    # Handle any key present in injects, also support {{ domain }} with spaces
    def repl(m):
        key=m.group(1).strip()
        if key in injects:
            val=injects[key]
            # If val is already a string, return as is, else json dump
            if isinstance(val, str):
                return val
            return json.dumps(val, ensure_ascii=False, indent=2)
        return m.group(0)
    # Build regex from available keys + generic fallback
    # Use generic pattern that matches any word inside {{ }}
    return re.sub(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}", repl, template)

def validate_final_message(content: str, expected_result: str) -> tuple[bool, str | None]:
    """Validate final assistant message per turn's expected result.
    - 'ir': must be valid Graph IR per BNF, checked via ir-ast-sanity
    - 'nl': must contain markdown links [text](iri) and those iris must be syntactically valid
    Returns (success, error_message)
    """
    content = content.strip()
    if not content:
        return False, "empty final message"
    if expected_result == "ir":
        # Validate as Graph IR using ir-ast-sanity (RDF-blind)
        try:
            # Lazy load to avoid circular deps
            import importlib.util
            from pathlib import Path
            root = Path(__file__).resolve().parents[1]
            # Try ir-ast-sanity, fallback to ir-compile
            for mod_name, fname in [("ir_sanity", "ir-ast-sanity.py"), ("ir_sanity", "ir-ast-sanity.py"), ("ir_compile", "ir-compile.py")]:
                p = root / "src" / fname
                if p.is_file():
                    spec = importlib.util.spec_from_file_location(mod_name, p)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    # Try sanity first
                    if hasattr(mod, "validate_source"):
                        res = mod.validate_source(content, strict=False)
                        # res has valid, errors
                        if res.get("valid"):
                            return True, None
                        else:
                            errs = res.get("errors", [])
                            first = errs[0].get("message") if errs else str(res)
                            return False, f"ir-ast-sanity: {first}"
                    elif hasattr(mod, "parse"):
                        try:
                            mod.parse(content)
                            return True, None
                        except Exception as e:
                            return False, f"parse: {e}"
            # Fallback: try ir-compile
            p2 = root / "src" / "ir-compile.py"
            if p2.is_file():
                spec2 = importlib.util.spec_from_file_location("ir_compile2", p2)
                m2 = importlib.util.module_from_spec(spec2)
                spec2.loader.exec_module(m2)
                m2.parse(content)
                return True, None
            return False, "no validator available"
        except Exception as e:
            return False, f"ir validation error: {e}"
    elif expected_result == "nl":
        # Must contain at least one markdown link [text](iri)
        links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)
        if not links:
            return False, "nl: missing markdown link [text](iri)"
        # Validate each link's iri is plausible IRI (prefixed or absolute)
        # Use a simple BNF-ish check: prefix:local or <iri> or absolute
        iri_pat = re.compile(r"^(?:[A-Za-z_][A-Za-z0-9_-]*:[A-Za-z_][A-Za-z0-9_\-]*|<[^>]+>|https?://\S+|my:[^ \t\n]+|ai:[^ \t\n]+)$")
        for text, iri in links:
            iri = iri.strip()
            # Allow my:xxx, ai:xxx, prefix:local, <...>
            if not iri_pat.match(iri):
                # Try more lenient: must contain colon or be <...>
                if ":" not in iri and not (iri.startswith("<") and iri.endswith(">")):
                    return False, f"nl: link iri invalid per BNF: {iri}"
            # Also try to validate via sanity's IRI check if available
            # For now, just ensure it looks like an IRI
        return True, None
    else:
        # Unknown result type, consider valid if non-empty
        return True, None

def validate_jsonl_response(text: str):
    # Try to handle code fences
    if text.startswith("```"):
        # strip fences
        text = re.sub(r"^```(?:jsonl?|json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text=text.strip()
    lines=[]
    # If response is a JSON array, treat as lines
    if text.startswith("["):
        try:
            arr=json.loads(text)
            if isinstance(arr, list):
                lines=[json.dumps(x, ensure_ascii=False) for x in arr]
            else:
                lines=[text]
        except:
            lines=text.splitlines()
    else:
        lines=[l for l in text.splitlines() if l.strip()]

    parsed=[]
    for idx, line in enumerate(lines):
        line=line.strip()
        if not line:
            continue
        # Remove trailing commas?
        try:
            obj=json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"line {idx+1} not valid JSON: {e}\n{line[:500]}")
        if not isinstance(obj, dict):
            raise ValueError(f"line {idx+1} not an object: {line[:200]}")
        if "messages" not in obj:
            raise ValueError(f"line {idx+1} missing 'messages': {line[:500]}")
        msgs=obj["messages"]
        if not isinstance(msgs, list) or not msgs:
            raise ValueError(f"line {idx+1} 'messages' must be non-empty list")
        for m in msgs:
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                raise ValueError(f"line {idx+1} message missing role/content: {m}")
            if m["role"] not in ("system","user","assistant"):
                # allow but warn
                pass
            if not isinstance(m["content"], str) or not m["content"].strip():
                raise ValueError(f"line {idx+1} message content empty: {m}")
        parsed.append(obj)
    if not parsed:
        raise ValueError("no valid JSONL lines found")
    return parsed

def call_llm(endpoint, api_key, model, system_prompt, user_prompt, temperature, max_tokens, top_p, reasoning_effort):
    import requests
    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # Groq uses openai-compatible chat completions
    payload={
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stream": False,
    }
    # reasoning_effort is not standard openai, but groq's gpt-oss may use it; include if present
    # Some models use reasoning_effort as extra param; we add as is if not None
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    resp=requests.post(endpoint, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"LLM error {resp.status_code}: {resp.text[:1000]}")
    data=resp.json()
    # openai format: choices[0].message.content
    try:
        content=data["choices"][0]["message"]["content"]
        usage=data.get("usage", {})
    except Exception as e:
        raise RuntimeError(f"unexpected LLM response shape: {json.dumps(data)[:2000]}")
    return content, usage

def main(argv=None):
    parser=argparse.ArgumentParser(description="Generate IR scenarios via LLM")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="config path")
    parser.add_argument("--domains", help="override domains.jsonl path")
    parser.add_argument("--cases", help="override cases.jsonl path")
    parser.add_argument("--turns", help="override turns.jsonl path")
    parser.add_argument("--output", "-o", help="override output dir")
    parser.add_argument("--compact", action="store_true", help="each domain into single jsonl file")
    parser.add_argument("--llm", action="store_true", help="actually call the LLM (default is to write the payload for review)")
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs (default is idempotent, skip existing)")
    parser.add_argument("--validate", action="store_true", help="validate final assistant message (IR via ir-ast-sanity, NL via markdown links) and write {domain}-results.jsonl + usage.json")
    parser.add_argument("--dry-run", action="store_true", help="render prompts without calling LLM (shows token table, no files)")
    parser.add_argument("--limit", type=int, help="limit number of domain/case/turn combos for testing")
    parser.add_argument("domain_filter", nargs="?", default=None, help="filter domain id (e.g., literature) or 'all'")
    parser.add_argument("extra_files", nargs="*", help="optional positional domains/cases/turns files (alternative to --domains/--cases/--turns)")
    args=parser.parse_args(argv)

    load_dotenv()
    config_path=Path(args.config)
    if not config_path.is_file():
        config_path=ROOT / args.config
    config=load_config(config_path)

    # Resolve inject paths (allow CLI overrides) – also support positional file args
    # If user passed file paths as extra_files, use them as domains/cases/turns in order
    extra = getattr(args, "extra_files", []) or []
    # Also handle case where domain_filter looks like a file path (legacy positional file args)
    if args.domain_filter and (args.domain_filter.endswith(".jsonl") or args.domain_filter.endswith(".json") or "/" in args.domain_filter):
        # Check if it's a file path that exists
        cand = Path(args.domain_filter)
        if cand.exists() or (ROOT / cand).exists():
            extra = [args.domain_filter] + extra
            args.domain_filter = None
    inject=config.get("inject", {})
    # Override with extra_files in order: domains, cases, turns
    if len(extra) >= 1 and not args.domains:
        args.domains = extra[0]
    if len(extra) >= 2 and not args.cases:
        args.cases = extra[1]
    if len(extra) >= 3 and not args.turns:
        args.turns = extra[2]
    # If still extra left, treat first as domain filter if it doesn't look like file
    if len(extra) > 3 and not args.domain_filter:
        # Remaining extra could be domain filter
        for e in extra[3:]:
            if not e.endswith(".jsonl") and not e.endswith(".json") and "/" not in e:
                args.domain_filter = e
                break

    # Generic inject handling: load all inject entries (domain, case, turn, instruct, etc.)
    # CLI overrides for domains/cases/turns still supported, but also handle any new inject keys
    inject_cfg = config.get("inject", {})
    # CLI overrides take precedence for the three main ones
    if args.domains:
        inject_cfg["domain"] = args.domains
    if args.cases:
        inject_cfg["case"] = args.cases
    if args.turns:
        # also handle --turns
        inject_cfg["turn"] = args.turns
    # Also allow --domains etc. to be passed via extra_files already handled above

    def resolve(p: Path) -> Path:
        if p.is_absolute():
            return p
        c=ROOT / p
        if c.exists():
            return c
        return p

    # Load all inject datasets generically
    inject_data = {}
    inject_paths = {}
    for key, rel_path in inject_cfg.items():
        p = Path(rel_path)
        p = resolve(p)
        inject_paths[key] = p
        if not p.is_file():
            print(f"warning: inject '{key}' not found at {p} – skipping", file=sys.stderr)
            continue
        try:
            inject_data[key] = load_jsonl(p)
        except Exception as e:
            print(f"warning: failed to load {key} at {p}: {e}", file=sys.stderr)
            continue
        print(f"inject {key}: {len(inject_data[key])} items from {p}")

    if not inject_data:
        print("error: no inject data loaded", file=sys.stderr)
        return 1

    # Ensure we have at least domain, case, turn if expected, but be generic
    # For backward compat, keep domains/cases/turns variables if present
    domains = inject_data.get("domain", [])
    cases = inject_data.get("case", [])
    turns = inject_data.get("turn", [])

    output_dir=Path(args.output) if args.output else Path(config.get("output", "tmp/ir/generated"))
    if not output_dir.is_absolute():
        output_dir=ROOT / output_dir

    prompts_cfg=config.get("prompts", [])
    system_template, user_template = load_prompts(prompts_cfg)
    prompts, scenarios, missing, extra = ensure_prompts_have_scenarios()
    if missing and not args.dry_run:
        print(f"warning: prompts without scenarios: {', '.join(missing)}", file=sys.stderr)
    if extra:
        print(f"note: scenarios without prompts: {', '.join(extra)}", file=sys.stderr)

    endpoint=config.get("endpoint", "https://api.groq.com/openai/v1/chat/completions")
    api_key=None
    if args.llm and not args.dry_run:
        api_key=config.get("api_key") or os.environ.get("GROQ_AI_API_KEY") or os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("error: GROQ_AI_API_KEY not set (checked .env and config) – required for --llm", file=sys.stderr)
            return 1
        if isinstance(api_key, str) and "{{" in api_key:
            api_key=resolve_env_template(api_key)
            if "{{" in api_key:
                print(f"error: api_key still templated: {api_key}", file=sys.stderr)
                return 1

    model=config.get("model", "openai/gpt-oss-120b")
    temperature=config.get("temperature", 0.2)
    max_tokens=config.get("max_tokens", 2048)
    top_p=config.get("top_p", 1)
    reasoning_effort=config.get("reasoning_effort")

    # The configured inject map is the source of truth. Do not silently auto-discover
    # additional JSONL files from generate/iterations/*.jsonl; that can pull in stale or
    # legacy fixtures like cases.old.jsonl that are not intentionally configured.
    # For backward compat, keep domains/cases/turns variables if present, else use generic
    # Build ordered list of inject keys to preserve domain/case/turn order, plus any extra like instruct
    inject_keys = list(inject_data.keys())
    # Ensure domain/case/turn are first if present for stable ordering and filtering
    ordered_keys = []
    for k in ["domain", "case", "turn"]:
        if k in inject_keys:
            ordered_keys.append(k)
            inject_keys.remove(k)
    ordered_keys.extend(sorted(inject_keys))  # remaining like instruct sorted
    inject_keys = ordered_keys

    # Extract lists for each key
    inject_lists = [inject_data[k] for k in inject_keys]

    # domain_filter only applies to domain
    if args.domain_filter and args.domain_filter != "all" and "domain" in inject_data:
        filt=args.domain_filter.lower()
        orig_len=len(inject_data["domain"])
        inject_data["domain"]=[d for d in inject_data["domain"] if filt in json.dumps(d).lower() or filt == d.get("id","").lower()]
        if not inject_data["domain"]:
            print(f"no domains match filter '{args.domain_filter}'", file=sys.stderr)
            return 1
        # Rebuild lists after filter
        inject_lists = [inject_data[k] for k in inject_keys]
        print(f"filtered domain: {orig_len} -> {len(inject_data['domain'])}")

    # Also keep legacy variables for compatibility
    domains=inject_data.get("domain", [])
    cases=inject_data.get("case", [])
    turns=inject_data.get("turn", [])

    total=1
    for lst in inject_lists:
        total*=len(lst)
    # Human-readable breakdown
    breakdown=", ".join(f"{k}: {len(inject_data[k])}" for k in inject_keys)
    print(f"{breakdown} => {total} prompts")
    if args.limit:
        print(f"limit {args.limit} for testing")

    # For compact mode, we will open one file handle per domain
    compact_handles={}
    # For per-case mode, track which case files have been initialized (to handle append vs truncate)
    case_seen=set()
    # Ensure output dir exists (never delete existing outputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Snapshot of existing files for idempotent check (before this run)
    existing_files = set(p.resolve() for p in output_dir.rglob("*.jsonl")) if output_dir.exists() else set()

    # Track stats
    success=0
    failed=0
    skipped=0
    dry_rows=[]
    # Usage tracking for --llm
    usage_per_domain = {}
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "requests": 0}

    # Prepare iterator - generic cartesian product of all injects
    combos=list(itertools.product(*inject_lists))
    if args.limit:
        combos=combos[:args.limit]
    # Map combos back to dict
    # Keep legacy combos as tuple of (domain, case, turn) for backward compat when those keys exist
    # For generic, we will use dict

    for idx, combo in enumerate(combos, start=1):
        inject_dict=dict(zip(inject_keys, combo))
        domain=inject_dict.get("domain", {})
        case=inject_dict.get("case", {})
        turn=inject_dict.get("turn", {})
        # For backward compat, also support instruct as separate
        domain_id=domain.get("id") if isinstance(domain, dict) else f"domain_{idx}"
        if not domain_id:
            domain_id=inject_dict.get("domain", {}).get("id") or f"domain_{idx}" if isinstance(inject_dict.get("domain"), dict) else f"domain_{idx}"
        case_id=case.get("id") if isinstance(case, dict) else f"case_{idx}"
        turn_id=turn.get("id") if isinstance(turn, dict) else f"turn_{idx}"
        # Fallback for case where domain/case/turn are not dicts
        if isinstance(domain, dict):
            domain_id=domain.get("id") or domain.get("domain") or f"domain_{idx}"
        else:
            domain_id=f"domain_{idx}"
        if isinstance(case, dict):
            case_id=case.get("id") or f"case_{idx}"
        else:
            case_id=f"case_{idx}"
        if isinstance(turn, dict):
            turn_id=turn.get("id") or f"turn_{idx}"
        else:
            turn_id=f"turn_{idx}"

        # Render prompts with all injects
        try:
            system_prompt=render_template(system_template, inject_dict)
            user_prompt=render_template(user_template, inject_dict)
        except Exception as e:
            print(f"[{idx}/{len(combos)}] render error {domain_id}/{case_id}/{turn_id}: {e}", file=sys.stderr)
            failed+=1
            continue

        if args.dry_run:
            # collect stats for table with token estimates – now per-case (turns concatenated)
            if args.compact:
                safe_domain=re.sub(r"[^a-z0-9_-]+","-", domain_id.lower()).strip("-") or "unknown"
                out_preview=str(output_dir / f"{safe_domain}.jsonl")
            else:
                safe_domain=re.sub(r"[^a-z0-9_-]+","-", domain_id.lower()).strip("-") or "unknown"
                safe_case=re.sub(r"[^a-z0-9_-]+","-", case_id.lower()).strip("-")
                out_preview=str(output_dir / safe_domain / f"{safe_case}.jsonl")
            try:
                out_preview=str(Path(out_preview).relative_to(ROOT))
            except:
                pass
            sys_tok=estimate_tokens(system_prompt)
            user_tok=estimate_tokens(user_prompt)
            dry_rows.append((idx, domain_id, case_id, turn_id, sys_tok, user_tok, sys_tok+user_tok, out_preview, "OK"))
            success+=1
            continue

        if not args.llm:
            # Payload review mode: write the actual request payload that would be sent to the LLM
            payload_obj = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
            if reasoning_effort:
                payload_obj["reasoning_effort"] = reasoning_effort
            # Also include meta for traceability – all inject ids
            meta = {}
            for k in inject_keys:
                v = inject_dict.get(k, {})
                if isinstance(v, dict):
                    meta[k] = v.get("id") or v.get("name") or str(v)[:50]
                else:
                    meta[k] = str(v)[:50]
            payload_obj["_meta"] = meta
            # Idempotent: skip if output already exists unless --force (check pre-existing files only)
            if args.compact:
                safe_domain=re.sub(r"[^a-z0-9_-]+","-", domain_id.lower()).strip("-") or "unknown"
                out_path=output_dir / f"{safe_domain}.jsonl"
                # Check if file existed before this run
                existed_before = out_path.resolve() in existing_files
                if existed_before and not args.force:
                    print(f"[{idx}/{len(combos)}] {domain_id}/{case_id}/{turn_id} -> {out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path} exists")
                    skipped+=1
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)
                key_compact=safe_domain
                mode="a" if key_compact in case_seen else "w"
                case_seen.add(key_compact)
                with out_path.open(mode, encoding="utf-8") as out:
                    out.write(json.dumps(payload_obj, ensure_ascii=False, indent=2) + "\n")
                if len(combos) <= 20 or idx <= 3 or idx == len(combos):
                    print(f"[{idx}/{len(combos)}] {domain_id}/{case_id}/{turn_id} -> {out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path} (payload{' +append' if mode=='a' else ''})")
            else:
                safe_domain=re.sub(r"[^a-z0-9_-]+","-", domain_id.lower()).strip("-") or "unknown"
                safe_case=re.sub(r"[^a-z0-9_-]+","-", case_id.lower()).strip("-")
                out_dir=output_dir / safe_domain
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path=out_dir / f"{safe_case}.jsonl"
                key=(safe_domain, safe_case)
                mode="a" if key in case_seen else "w"
                case_seen.add(key)
                with out_path.open(mode, encoding="utf-8") as out:
                    out.write(json.dumps(payload_obj, ensure_ascii=False, indent=2) + "\n")
                if len(combos) <= 20 or idx <= 3 or idx == len(combos):
                    print(f"[{idx}/{len(combos)}] {domain_id}/{case_id}/{turn_id} -> {out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path} (payload{' +append' if mode=='a' else ''})")
            success+=1
            continue

        # Call LLM
        label=f"{domain_id}/{case_id}/{turn_id}"
        print(f"[{idx}/{len(combos)}] {label} ...", end=" ", flush=True)
        try:
            response_text, usage = call_llm(endpoint, api_key, model, system_prompt, user_prompt, temperature, max_tokens, top_p, reasoning_effort)
            # Record usage per domain
            safe_domain_u = re.sub(r"[^a-z0-9_-]+","-", domain_id.lower()).strip("-") or "unknown"
            if usage:
                # Normalize usage keys
                p_tok = int(usage.get("prompt_tokens") or usage.get("promptTokenCount") or 0)
                c_tok = int(usage.get("completion_tokens") or usage.get("completionTokenCount") or 0)
                t_tok = int(usage.get("total_tokens") or usage.get("totalTokenCount") or p_tok + c_tok)
                # Update per-domain and totals
                if safe_domain_u not in usage_per_domain:
                    usage_per_domain[safe_domain_u] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "requests": 0}
                usage_per_domain[safe_domain_u]["prompt_tokens"] += p_tok
                usage_per_domain[safe_domain_u]["completion_tokens"] += c_tok
                usage_per_domain[safe_domain_u]["total_tokens"] += t_tok
                usage_per_domain[safe_domain_u]["requests"] += 1
                total_usage["prompt_tokens"] += p_tok
                total_usage["completion_tokens"] += c_tok
                total_usage["total_tokens"] += t_tok
                total_usage["requests"] += 1
                # Write per-domain usages JSONL (pretty-printed, one JSON per line, but pretty)
                usage_path = output_dir / f"{safe_domain_u}-usages.jsonl"
                # Idempotent for usages: append unless --force and file existed before
                existed_u = usage_path.resolve() in existing_files
                if existed_u and not args.force:
                    # Don't append if already exists from previous run and not force – but for usages we should still append current run's usage?
                    # For idempotent, we skip writing if file exists and not force, but for usages we want to accumulate per run
                    # So we check if this is first write for this domain in this run
                    pass
                # Determine mode for usages file
                u_key = f"usage_{safe_domain_u}"
                u_mode = "a" if u_key in case_seen else "w"
                # Use a separate set for usages to avoid conflict with case_seen for data files
                # Reuse case_seen for usages as well
                if u_key not in case_seen:
                    # If file existed before and not force, we already skipped above, so this is first write in this run
                    pass
                case_seen.add(u_key)
                # Prepare usage record
                usage_record = {
                    "domain": domain_id,
                    "case": case_id,
                    "turn": turn_id,
                    "model": model,
                    "usage": {"prompt_tokens": p_tok, "completion_tokens": c_tok, "total_tokens": t_tok},
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
                # Also include any extra usage fields
                for k, v in usage.items():
                    if k not in ("prompt_tokens", "completion_tokens", "total_tokens", "promptTokenCount", "completionTokenCount", "totalTokenCount"):
                        usage_record["usage"][k] = v
                output_dir.mkdir(parents=True, exist_ok=True)
                with usage_path.open(u_mode, encoding="utf-8") as uf:
                    uf.write(json.dumps(usage_record, ensure_ascii=False, indent=2) + "\n")
        except Exception as e:
            print(f"LLM error: {e}", file=sys.stderr)
            failed+=1
            time.sleep(1)
            continue

        # Validate
        try:
            parsed=validate_jsonl_response(response_text)
        except Exception as e:
            print(f"validate failed: {e}", file=sys.stderr)
            # Save raw for debugging
            err_dir=output_dir / "_errors"
            err_dir.mkdir(parents=True, exist_ok=True)
            safe_domain=re.sub(r"[^a-z0-9_-]+","-", domain_id.lower()).strip("-")
            safe_case=re.sub(r"[^a-z0-9_-]+","-", case_id.lower()).strip("-")
            safe_turn=re.sub(r"[^a-z0-9_-]+","-", turn_id.lower()).strip("-")
            err_path=err_dir / f"{safe_domain}__{safe_case}__{safe_turn}.txt"
            err_path.write_text(response_text, encoding="utf-8")
            print(f"  -> saved raw to {err_path}")
            failed+=1
            continue

        # Write structured output (idempotent, never delete)
        if args.compact:
            safe_domain=re.sub(r"[^a-z0-9_-]+","-", domain_id.lower()).strip("-") or "unknown"
            out_path=output_dir / f"{safe_domain}.jsonl"
            existed_before = out_path.resolve() in existing_files
            if existed_before and not args.force:
                print(f"OK ({len(parsed)} lines) -> {out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path} exists")
                skipped+=1
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            key_compact=safe_domain
            mode="a" if key_compact in case_seen else "w"
            case_seen.add(key_compact)
            with out_path.open(mode, encoding="utf-8") as out:
                for obj in parsed:
                    out.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
            print(f"OK ({len(parsed)} lines) -> {out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path} ({'append' if mode=='a' else 'new'})")
        else:
            safe_domain=re.sub(r"[^a-z0-9_-]+","-", domain_id.lower()).strip("-") or "unknown"
            safe_case=re.sub(r"[^a-z0-9_-]+","-", case_id.lower()).strip("-")
            out_dir=output_dir / safe_domain
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path=out_dir / f"{safe_case}.jsonl"
            key=(safe_domain, safe_case)
            existed_before = out_path.resolve() in existing_files
            if existed_before and not args.force:
                if key not in case_seen:
                    print(f"OK ({len(parsed)} lines) -> {out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path} exists")
                    skipped+=1
                    case_seen.add(key)
                    continue
                else:
                    skipped+=1
                    continue
            mode="a" if key in case_seen else "w"
            case_seen.add(key)
            with out_path.open(mode, encoding="utf-8") as out:
                for obj in parsed:
                    out.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
            print(f"OK ({len(parsed)} lines) -> {out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path} ({'append' if mode=='a' else 'new'})")

        success+=1
        # --validate: check final assistant message (uses turn's result field)
        if args.validate and not args.dry_run:
            # Only validate if we have an LLM response (or payload that contains a generated scenario)
            # For payload mode (not --llm), the output is the request payload, not the generated IR, so skip
            # For --llm, validate each parsed object's final assistant message
            if args.llm:
                for obj in parsed:
                    msgs = obj.get("messages", [])
                    last_assistant = None
                    for m in reversed(msgs):
                        if m.get("role") == "assistant":
                            last_assistant = m.get("content", "")
                            break
                    obj_id = obj.get("id") or f"{domain_id}/{case_id}/{turn_id}"
                    # Determine expected result from turn
                    expected = "ir"
                    if isinstance(turn, dict):
                        expected = turn.get("result") or turn.get("expected") or "ir"
                    expected = str(expected).lower()
                    if "nl" in expected and "ir" not in expected:
                        expected = "nl"
                    else:
                        expected = "ir"
                    success_flag, err = validate_final_message(last_assistant or "", expected)
                    result_rec = {"id": obj_id, "success": success_flag}
                    if not success_flag:
                        result_rec["error"] = err
                    # Write to per-domain results file (pretty-printed, idempotent)
                    safe_domain_r = re.sub(r"[^a-z0-9_-]+","-", domain_id.lower()).strip("-") or "unknown"
                    results_path = output_dir / f"{safe_domain_r}-results.jsonl"
                    existed_r = results_path.resolve() in existing_files
                    if existed_r and not args.force:
                        # Idempotent: skip if already exists from previous run
                        # But if this is first time in this run for this domain, we would have skipped the main output, so also skip results
                        # Check if we already skipped main output for this domain/case
                        # For simplicity, skip writing results if main output was skipped
                        # We can detect by checking if case_seen already contains the domain/case
                        # For now, just append if not existed before, otherwise skip
                        # To keep it simple, we will not write results if main was skipped
                        # Since we already continued for main output skip, we won't reach here for those
                        pass
                    # Determine mode for results file
                    r_key = f"results_{safe_domain_r}"
                    r_mode = "a" if r_key in case_seen else "w"
                    # Also respect idempotent for results file itself
                    if r_mode == "w" and existed_r and not args.force:
                        # File existed before and not force, skip
                        continue
                    case_seen.add(r_key)
                    if not hasattr(main, "_results_seen"):
                        main._results_seen = set()
                    main._results_seen.add(r_key)
                    with results_path.open(r_mode, encoding="utf-8") as rf:
                        rf.write(json.dumps(result_rec, ensure_ascii=False, indent=2) + "\n")
        # be nice to rate limits
        time.sleep(0.2)

    if args.dry_run:
        # Pretty table for dry-run
        print(f"\nDry-run stats")
        print(f"  Config:     {config_path}")
        print(f"  Model:      {model}  (endpoint: {endpoint})")
        print(f"  Output:     {output_dir} ({'compact: per-domain' if args.compact else 'per-case (turns concatenated)'})")
        # Count actual prompts loaded (including auto-discovered)
        actual_prompt_count = len(all_prompt_paths) if 'all_prompt_paths' in locals() else len(prompts_cfg)
        print(f"  Prompts:    {actual_prompt_count} files → system:{len(system_template)} chars, user:{len(user_template)} chars")
        # Generic inject stats
        inject_summary = ", ".join(f"{k}: {len(inject_data[k])}" for k in inject_keys)
        print(f"  Injects:    {inject_summary} => {total} total, {len(combos)} with filter/limit")
        if missing:
            print(f"  Warning:    prompts without scenarios: {', '.join(missing)}")
        print()
        # Table header with token estimates
        hdr = ["#", "Domain", "Case", "Turn", "SysTok", "UserTok", "Total", "Output", "Status"]
        rows = dry_rows[:20]
        # Totals
        total_sys = sum(r[4] for r in dry_rows)
        total_user = sum(r[5] for r in dry_rows)
        total_all = sum(r[6] for r in dry_rows)
        # Estimate output tokens (max_tokens per request)
        est_out_per = max_tokens
        total_out_est = len(dry_rows) * est_out_per
        table = [hdr]
        for r in rows:
            idx, d, c, t, stok, utok, tot, outp, st = r
            d_short = (d[:14] + "…") if len(d) > 15 else d
            c_short = (c[:18] + "…") if len(c) > 19 else c
            t_short = (t[:10] + "…") if len(t) > 11 else t
            out_short = outp
            if len(out_short) > 32:
                out_short = "…" + out_short[-29:]
            table.append([str(idx), d_short, c_short, t_short, str(stok), str(utok), str(tot), out_short, st])
        widths = [max(len(str(row[i])) for row in table) for i in range(len(hdr))]
        def fmt_row(row):
            return " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row))
        print(fmt_row(table[0]))
        print("-+-".join("-" * w for w in widths))
        for row in table[1:]:
            print(fmt_row(row))
        if len(dry_rows) > 20:
            print(f"... and {len(dry_rows)-20} more rows not shown (use --limit to narrow)")
        print()
        print(f"  Token sub-totals (shown):  sys {sum(r[4] for r in rows)}  user {sum(r[5] for r in rows)}  total {sum(r[6] for r in rows)}")
        print(f"  Token totals (all {len(dry_rows)}): sys {total_sys}  user {total_user}  input {total_all}  + est. output ~{total_out_est} (max_tokens {max_tokens} × {len(dry_rows)})  ⇒ ~{total_all + total_out_est} total")
        print(f"  Avg per prompt: input ~{total_all//max(1,len(dry_rows))}  output ~{est_out_per}")
        print()

    # Write usage summary if LLM was used
    if args.llm and not args.dry_run and total_usage["requests"] > 0:
        usage_path = output_dir / "usage.json"
        summary = {
            "model": model,
            "total": total_usage,
            "per_domain": usage_per_domain,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "output_dir": str(output_dir),
            "compact": bool(args.compact),
        }
        # Pretty-printed
        with usage_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"usage: {total_usage['prompt_tokens']} prompt + {total_usage['completion_tokens']} completion = {total_usage['total_tokens']} total tokens over {total_usage['requests']} requests")
        print(f"  per-domain usages -> {output_dir}/*-usages.jsonl")
        print(f"  summary -> {usage_path}")

    print(f"done: {success} succeeded, {failed} failed, {skipped} skipped, total {len(combos)}")
    # Close compact handles if any
    for h in compact_handles.values():
        try: h.close()
        except: pass
    return 0 if failed==0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
