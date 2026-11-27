"""qtrain bench — model-backend benchmark harness (SPEC §14c).

Re-runnable, configurable evaluation against any OpenAI-compatible
endpoint (our vLLM tier on nano0 by default, but --url works for cloud
endpoints too). Per (model, question, repeat): wall latency, streamed
tok/s (time-to-first-token and generation rate measured honestly from
the SSE stream), and a rule check that decides pass/fail per question.
Output: a markdown table (paste anywhere), --json/--csv for machines.

Defaults answer the questions that keep coming up: identity ("Who are
you?"), persona state ("How are you feeling?"), plus instruction
follow-through, JSON compliance, and a tiny world-model probe. Every
dimension is overridable via --config:

  {"targets": [{"name": "lora", "url": "http://nano0.lan:8000",
                "models": ["kandi"]}],
   "questions": [{"id": "identity", "prompt": "Who are you?",
                  "check": "contains:kandi"}],
   "repeats": 3, "max_tokens": 64, "temperature": 0.0, "seed": 7}

check grammar (R13 — the bench IS the oracle harness, docs/ORACLES.md):
  contains:SUBSTR | regex:PATTERN | one_of:a,b | json | none
  not:<check>              — negation (composition)
  rate:<check>/<lo>-<hi>   — fraction of REPEATS passing <check> must
                             land in [lo,hi] (behavioural dials: bleed
                             floors, self-name coverage, refusals)
rate: rows carry the measured fraction in "observed"; verdicts are
computed over the whole repeat group, not the first sample only.
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

# deployment endpoints live in the corpora suite; this is last-resort only
DEFAULT_URLS = ["http://localhost:8000"]

# Mechanical fallback: persona-neutral capability probes. The task suite
# (identity/feeling/voice + endpoint urls) lives in corpora/bench/*.json
# per the SPEC §4b boundary — machinery ships no domain data.
DEFAULT_QUESTIONS = [
    {"id": "instruction",
     "prompt": "Reply with exactly one word: is 7 a prime number?",
     "check": "regex:^\\W*yes\\W*$"},
    {"id": "json",
     "prompt": "Respond with JSON only, keys mood and topic "
               "(short strings), about rain.",
     "check": "json"},
    {"id": "world",
     "prompt": "A coin is under a cup; I move the cup to the table. "
               "Where is the coin? One short sentence.",
     "check": "contains:cup"},
]


def _suite_field(key: str):
    """Corpora bench suites (all bench/*.json, name order) merged for one
    field. Best-effort: absent/broken corpora silently falls through to
    the mechanical defaults above."""
    try:
        from .config import corpora_dir
        d = corpora_dir() / "bench"
    except Exception:                              # noqa: BLE001
        return None
    if not d.is_dir():
        return None
    merged, found = [], False
    for f in sorted(d.glob("*.json")):
        try:
            v = json.loads(f.read_text()).get(key)
        except (ValueError, OSError):
            continue
        if v:
            merged.extend(v)
            found = True
    return merged if found else None


def suite_questions() -> list[dict] | None:
    return _suite_field("questions")


def suite_urls() -> list[str] | None:
    return _suite_field("urls")


def _req(url: str, body: dict | None = None, timeout: float = 120,
         stream: bool = False):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data,
                               headers={"Content-Type": "application/json"},
                               method="POST" if body else "GET")
    if stream:
        # caller owns the socket (SSE iteration needs it open)
        return urllib.request.urlopen(r, timeout=timeout)
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read())


def discover(urls: list[str]) -> list[dict]:
    """One target per (url, model) — the engine IS the truth: whatever
    /v1/models answers is benchmarkable. Offline endpoints are skipped."""
    targets = []
    for url in urls:
        base = url.rstrip("/")
        try:
            data = _req(f"{base}/v1/models", timeout=6)["data"]
        except Exception as e:
            print(f"bench: {base} skipped ({_short(e)})", file=sys.stderr)
            continue
        for m in data:
            mid = m["id"]
            targets.append({"name": base.split("//")[1].split(":")[0],
                            "url": base, "model": mid})
    return targets


def _short(e: Exception) -> str:
    return f"{type(e).__name__}: {str(e)[:80]}"


def chat_stream(url: str, model: str, prompt: str, *, max_tokens: int,
                temperature: float, seed: int | None,
                timeout: float) -> tuple[str, float, float, float]:
    """Returns (text, wall_s, ttft_s, tok_per_s) using the SSE stream."""
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature,
            "stream": True}
    if seed is not None:
        body["seed"] = seed
    t0 = time.perf_counter()
    parts: list[str] = []
    ttft = None
    n_chunks = 0
    resp = _req(f"{url}/v1/chat/completions", body, timeout=timeout,
                stream=True)
    try:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except ValueError:
                continue
            delta = (d.get("choices") or [{}])[0].get("delta") or {}
            piece = delta.get("content") or ""
            if piece and ttft is None:
                ttft = time.perf_counter() - t0
            if piece:
                n_chunks += 1
            parts.append(piece)
    finally:
        resp.close()
    wall = time.perf_counter() - t0
    # tok/s is only honest when there were enough chunks to divide;
    # a 1-token answer makes the decode window ~0 and fakes 3000+ tok/s
    tps = (n_chunks / (wall - ttft)
           if ttft and wall > ttft and n_chunks >= 4 else 0.0)
    return "".join(parts), wall, (ttft or wall), tps


def _primitive(check: str, text: str) -> str:
    """Single-sample oracle primitives (R13). Composed by score()."""
    if text.startswith("ERROR:") and check != "none":
        # transport failures must NEVER be vacuous passes: a not: check
        # is trivially true against an error string (caught red-handed
        # by the first dead-tunnel bench — two "3/3" from refusals)
        return "FAIL"
    if check.startswith("not:"):
        inner = _primitive(check[4:], text)
        return {"PASS": "FAIL", "FAIL": "PASS"}.get(inner, inner)
    if check.startswith("contains:"):
        return "PASS" if check[9:].lower() in text.lower() else "FAIL"
    if check.startswith("regex:"):
        try:
            return ("PASS" if re.match(check[6:], text.strip(), re.I | re.S)
                    else "FAIL")
        except re.error:
            return "?"
    if check.startswith("one_of:"):
        low = text.lower()
        return ("PASS" if any(w in low for w in check[7:].split(","))
                else "FAIL")
    if check == "json":
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return "FAIL"
        try:
            json.loads(m.group(0))
            return "PASS"
        except ValueError:
            return "FAIL"
    if check.startswith("json_keys:"):
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return "FAIL"
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            return "FAIL"
        keys = check[10:].split(",")
        return ("PASS" if all(k.strip() in obj for k in keys) else "FAIL")
    if check.startswith("not_contains:"):
        return "FAIL" if check[13:].lower() in text.lower() else "PASS"
    if check.startswith("max_len:"):
        return "PASS" if len(text.strip()) <= int(check[8:]) else "FAIL"
    if check == "nonempty":
        return "PASS" if text.strip() else "FAIL"
    return "—"


def score(check: str, text_or_texts) -> str:
    """Verdict for a question. Plain checks grade the (first) sample;
    rate:<inner>/<lo>-<hi> grades the FRACTION of repeated samples
    passing <inner> — the behavioural-dial primitive (ORACLES.md R13)."""
    texts = ([text_or_texts] if isinstance(text_or_texts, str)
             else list(text_or_texts))
    if not texts:
        return "—"
    first = texts[0]
    if check.startswith("rate:"):
        spec = check[5:]
        inner, _, bound = spec.rpartition("/")
        try:
            lo, _, hi = bound.partition("-")
            lo, hi = float(lo), float(hi)
        except ValueError:
            return "?"
        hits = sum(1 for t in texts if _primitive(inner, t) == "PASS")
        frac = hits / len(texts)
        return "PASS" if lo <= frac <= hi else "FAIL"
    return _primitive(check, first)


def rate_observed(check: str, texts) -> float | None:
    """The measured fraction for rate: questions (debug/plotting)."""
    if not check.startswith("rate:") or not texts:
        return None
    inner, _, _ = check[5:].rpartition("/")
    return round(sum(1 for t in texts if _primitive(inner, t) == "PASS")
                 / len(texts), 3)


def score_all(check: str, texts) -> list[str]:
    """Per-sample verdicts (for rate: groups, the INNER check)."""
    inner = (check[5:].rpartition("/")[0] if check.startswith("rate:")
             else check)
    return [_primitive(inner, t) for t in texts]


def normalize_url(base: str) -> str:
    """ONE accepted shape for every caller, whichever the user typed.
    The cross-tool trap (bench wanted the BARE base, synth wanted a
    /v1 suffix — the wrong shape 404s or worse, silently) dies here:
    bare base, /v1, or the full /v1/chat/completions endpoint all
    normalize to the bare base chat_stream expects."""
    u = base.strip().rstrip("/")
    for suffix in ("/chat/completions", "/v1"):
        if u.endswith(suffix):
            u = u[: -len(suffix)].rstrip("/")
    return u


def _inner_regexes(check: str):
    """The regex pattern a (possibly composed) check contains, if any.
    Alternation is REGEX syntax, not a check separator — never split
    on '|' (the canonical regex:READ|MATCH|BIND|RETURN is ONE pattern)."""
    c = check
    for _ in range(2):                      # not: / rate: wrappers
        if c.startswith("not:"):
            c = c[4:]
        elif c.startswith("rate:"):
            c = c[5:].rpartition("/")[0]
    while c.startswith("not:"):
        c = c[4:]
    return [c[6:]] if c.startswith("regex:") else []


def lint_suite(questions) -> list[str]:
    """Loader lint (the bench v2 incidents, mechanized): an impossible
    rate: range grades silently forever (the /33-100 fraction-scale
    bug), and a bad regex returns '?' per sample instead of naming
    itself. Both are suite-authoring errors, not data — catch them at
    LOAD, naming the question. Regex ANCHORING is documented law
    (re.match: prefix-shaped golds need .*); lint cannot know a
    question's gold, so authoring stays human — this catches what is
    provably broken."""
    errors = []
    for q in questions:
        check = q.get("check", "none")
        qid = q.get("id", "?")
        if check.startswith("rate:"):
            _, _, bound = check.rpartition("/")
            lo_s, _, hi_s = bound.partition("-")
            try:
                lo, hi = float(lo_s), float(hi_s)
                if not (0.0 <= lo <= hi <= 1.0):
                    errors.append(f"{qid}: rate bounds {bound!r} are "
                                  "FRACTION scale (0.0-1.0) — an "
                                  "impossible range grades FAIL forever")
            except ValueError:
                errors.append(f"{qid}: rate bounds {bound!r} unparseable")
        for pat in _inner_regexes(check):
            try:
                re.compile(pat)
            except re.error as e:
                errors.append(f"{qid}: regex {pat!r}: {e}")
    return errors


def preflight(targets) -> None:
    """Fail fast, fail LOUD: a dead tunnel used to burn a whole bench
    run and return a table of ERROR-shaped 'verdicts' (one such run
    reported proud '3/3' rows built from connection refusals). One
    /v1/models call per distinct URL before any question loop; a dead
    endpoint is a TRANSPORT finding, never a model result."""
    for base in dict.fromkeys(t["url"] for t in targets):
        try:
            _req(f"{base}/v1/models", timeout=8)
        except Exception as e:
            raise SystemExit(
                f"bench preflight: no engine answers at {base} "
                f"({_short(e)}) — tunnel down or pod replaced? "
                f"Fix the endpoint; a dead socket is not a measurement.")


def run(config: dict, sink: list | None = None) -> list[dict]:
    urls = [normalize_url(u) for u in
            (config.get("urls") or suite_urls() or DEFAULT_URLS)]
    targets = config.get("targets") or discover(urls)
    targets = [{**t, "url": normalize_url(t["url"])} for t in targets]
    questions = config.get("questions") or suite_questions() \
        or DEFAULT_QUESTIONS
    problems = lint_suite(questions)
    if problems:
        raise ValueError("bench suite lint: " + "; ".join(problems))
    if config.get("preflight", True):
        preflight(targets)
    repeats = int(config.get("repeats", 3))
    max_tokens = int(config.get("max_tokens", 64))
    temperature = float(config.get("temperature", 0.0))
    seed = config.get("seed", 7)
    timeout = float(config.get("timeout", 120))

    rows = []
    for t in targets:
        for q in questions:
            lat, tps, ttfts, texts = [], [], [], []
            for _ in range(repeats):
                try:
                    text, wall, ttft, rate = chat_stream(
                        t["url"], t["model"], q["prompt"],
                        max_tokens=max_tokens, temperature=temperature,
                        seed=seed, timeout=timeout)
                except Exception as e:
                    text, wall, ttft, rate = f"ERROR: {_short(e)}", 0, 0, 0
                lat.append(wall); tps.append(rate); ttfts.append(ttft)
                texts.append(text.strip())
            verdicts = score_all(q.get("check", "none"), texts)
            verdict = score(q.get("check", "none"), texts)
            rows.append({
                "host": t["name"], "model": t["model"], "q": q["id"],
                "pass": verdict, "passes": sum(v == "PASS"
                                               for v in verdicts),
                "repeats": repeats,
                "observed": rate_observed(q.get("check", "none"), texts),
                "tok_s_med": round(statistics.median(tps), 1),
                "ttft_med": round(statistics.median(ttfts), 2),
                "wall_med": round(statistics.median(lat), 2),
                "answer": texts[0][:120],
            })
            if sink is not None:
                # FULL text, every sample: the 60-char table cell and
                # 120-char row field once cost a wrong diagnosis (a
                # truncation hid that the "bleed" answer was complete).
                sink.append({"host": t["name"], "model": t["model"],
                             "q": q["id"], "check": q.get("check", "none"),
                             "verdict": verdict,
                             "passes": sum(v == "PASS" for v in verdicts),
                             "answers": texts})
            print(f"  {t['model']:>18} / {q['id']:<10} "
                  f"{verdicts.count('PASS')}/{repeats} "
                  f"{statistics.median(tps):5.1f} tok/s", file=sys.stderr)
    return rows


def table(rows: list[dict]) -> str:
    head = ("| host | model | question | pass | tok/s | TTFT s | "
            "answer |")
    sep = "|---|---|---|---|---|---|---|"
    lines = [head, sep]
    for r in rows:
        lines.append(
            f"| {r['host']} | {r['model']} | {r['q']} | "
            f"{r['passes']}/{r['repeats']} | {r['tok_s_med']} | "
            f"{r['ttft_med']} | {r['answer'][:60].replace('|', '!')} |")
    # per-model rollup
    models = sorted({(r["model"], r["host"]) for r in rows})
    lines.append("")
    lines.append("| model | identity | checks | tok/s (med) |")
    lines.append("|---|---|---|---|")
    for m, h in models:
        rs = [r for r in rows if r["model"] == m]
        ident = next((r["pass"] for r in rs if r["q"] == "identity"), "—")
        tot = sum(1 for r in rs if r["pass"] != "—")
        ok = sum(1 for r in rs if r["pass"] == "PASS")
        rates = [r["tok_s_med"] for r in rs if r["tok_s_med"] > 0]
        avg = round(statistics.median(rates), 1) if rates else 0
        lines.append(f"| {m} | {ident} | {ok}/{tot} | {avg} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="qtrain bench")
    ap.add_argument("--config", help="json config (targets/questions/params)")
    ap.add_argument("--url", action="append", help="endpoint(s), repeatable")
    ap.add_argument("--model", action="append",
                    help="pin model ids (with --url); repeatable")
    ap.add_argument("--question", action="append",
                    help='"id|prompt|check" override, repeatable '
                         "(check: contains:X regex:P one_of:a,b json none)")
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--json", dest="json_out", help="write full rows json")
    ap.add_argument("--answers", dest="answers_out",
                    help="write FULL per-sample answers as jsonl "
                         "(the table truncates; forensics needs the rest)")
    ap.add_argument("--no-preflight", dest="preflight", action="store_false",
                    help="skip the live-engine probe (offline replay only)")
    ap.add_argument("--csv", dest="csv_out", help="write rows csv")
    ap.add_argument("--record", action="store_true",
                    help="persist to the eval-history ledger (SPEC 13am)")
    ap.add_argument("--run", default="",
                    help="artifact run_id to key the recording on "
                         "(implies --record)")
    args = ap.parse_args(argv)

    config: dict = {}
    if args.config:
        config = json.loads(open(args.config).read())
    if args.url:
        if args.model:
            config["targets"] = [{"name": u.split("//")[1].split(":")[0],
                                  "url": u.rstrip("/"), "model": m}
                                 for u in args.url for m in args.model]
        else:
            config["urls"] = args.url
    if args.question:
        qs = []
        for spec in args.question:
            bits = spec.split("|")
            if len(bits) < 3:
                print(f"bad --question {spec!r} (need id|prompt|check)",
                      file=sys.stderr)
                return 2
            qs.append({"id": bits[0], "prompt": bits[1],
                       "check": bits[2]})
        config["questions"] = qs
    for key, val in (("repeats", args.repeats),
                     ("max_tokens", args.max_tokens),
                     ("temperature", args.temperature)):
        if val is not None:
            config[key] = val
    config["preflight"] = args.preflight

    print("bench: running...", file=sys.stderr)
    sink: list = [] if args.answers_out else None
    try:
        rows = run(config, sink=sink)
    except ValueError as e:              # suite lint: authoring error,
        print(f"bench: {e}", file=sys.stderr)   # clean exit, no traceback
        return 2
    if args.answers_out:
        with open(args.answers_out, "w") as f:
            for rec in sink:
                f.write(json.dumps(rec) + "\n")
    print(table(rows))
    if args.record or args.run:
        from .config import ROOT as _ROOT
        from .ledger import Ledger
        led = Ledger(_ROOT / "state" / "ledger.db")
        for (m, h) in sorted({(r["model"], r["host"]) for r in rows}):
            led.record_eval(m, [r for r in rows if r["model"] == m
                                and r["host"] == h],
                            run_id=args.run, host=h,
                            target=config.get("target", "chat"))
        print(f"bench: recorded eval history for "
              f"{len({r['model'] for r in rows})} model(s)", file=sys.stderr)
    if args.json_out:
        open(args.json_out, "w").write(json.dumps(rows, indent=1))
    if args.csv_out:
        with open(args.csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
