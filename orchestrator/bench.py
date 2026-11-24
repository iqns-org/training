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

check grammar: contains:SUBSTR | regex:PATTERN | one_of:a,b | json
| none (report only).
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


def score(check: str, text: str) -> str:
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
    return "—"


def run(config: dict) -> list[dict]:
    urls = config.get("urls") or suite_urls() or DEFAULT_URLS
    targets = config.get("targets") or discover(urls)
    questions = config.get("questions") or suite_questions() \
        or DEFAULT_QUESTIONS
    repeats = int(config.get("repeats", 3))
    max_tokens = int(config.get("max_tokens", 64))
    temperature = float(config.get("temperature", 0.0))
    seed = config.get("seed", 7)
    timeout = float(config.get("timeout", 120))

    rows = []
    for t in targets:
        for q in questions:
            lat, tps, ttfts, texts, verdicts = [], [], [], [], []
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
                verdicts.append(score(q.get("check", "none"), text))
            rows.append({
                "host": t["name"], "model": t["model"], "q": q["id"],
                "pass": verdicts[0], "passes": sum(v == "PASS"
                                                   for v in verdicts),
                "repeats": repeats,
                "tok_s_med": round(statistics.median(tps), 1),
                "ttft_med": round(statistics.median(ttfts), 2),
                "wall_med": round(statistics.median(lat), 2),
                "answer": texts[0][:120],
            })
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
    ap.add_argument("--csv", dest="csv_out", help="write rows csv")
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

    print("bench: running...", file=sys.stderr)
    rows = run(config)
    print(table(rows))
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
