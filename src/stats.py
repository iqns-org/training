#!/usr/bin/env python3
import json, os, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expand import resolve_domain, expand_row

CHATML = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
    "{{ m['content'] }}<|im_end|>\n{% endfor %}"
)


def axolotl_base_model():
    pat = re.compile(r"^\s*base_model\s*:\s*(\S+)", re.M)
    for cfg in sorted(Path(__file__).resolve().parents[1].glob("axolotl/*.y*ml")) + sorted(
        Path(__file__).resolve().parents[1].glob("*.y*ml")
    ):
        try:
            m = pat.search(cfg.read_text())
        except OSError:
            continue
        if m:
            return m.group(1)
    return None


class _HFTok:
    name = "hf"

    def __init__(self, model):
        from transformers import AutoTokenizer
        self._t = AutoTokenizer.from_pretrained(model)
        self.name = f"hf:{model}"
        try:
            self._t.chat_template = CHATML
        except Exception:
            pass

    def count(self, text):
        return len(self._t.encode(text, add_special_tokens=False))


class _TTTok:
    def __init__(self, enc="cl100k_base"):
        import tiktoken
        self._e = tiktoken.get_encoding(enc)
        self.name = f"tiktoken:{enc}"

    def count(self, text):
        return len(self._e.encode(text))


class _EstTok:
    name = "estimate:utf8/4"

    def count(self, text):
        return max(1, (len(text.encode("utf-8")) + 3) // 4)


def load_counter(offline=False):
    try:
        model = axolotl_base_model()
        if model:
            try:
                import transformers  # noqa: F401
                return _HFTok(model)
            except Exception:
                pass
    except Exception:
        pass
    try:
        return _TTTok("cl100k_base")
    except Exception:
        return _EstTok()


def row_text(tok, row):
    msgs = row.get("messages") or []
    text = "".join(
        f"<|im_start|>{m.get('role','user')}\n{m.get('content','')}<|im_end|>\n"
        for m in msgs
        if m.get("role") and m.get("content")
    )
    return tok.count(text)


def main_stats(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        raise SystemExit("usage: stats {domain}")
    domain = resolve_domain(argv[0])
    tok = load_counter()
    files = sorted(domain.rglob("*.jsonl"))
    if not files:
        raise SystemExit(f"no jsonl in {domain}")

    rows = []
    for src in files:
        data = [json.loads(l) for l in src.open() if l.strip()]
        before = sum(row_text(tok, r) for r in data)
        expanded = [e for r in data for e in expand_row(r)]
        after = sum(row_text(tok, r) for r in expanded)
        rows.append((str(src.relative_to(domain)), len(data), len(expanded), before, after))

    print(f"# tokenizer: {tok.name}\n")
    w = max(len("file"), max(len(r[0]) for r in rows))
    print(f"{'file':<{w}}  {'rows':>9}  {'tokens':>15}  {'x':>5}")
    print(f"{'-' * w}  {'-' * 9}  {'-' * 15}  {'-' * 5}")
    for name, rb, ra, tb, ta in rows:
        x = f"{ta / tb:.2f}" if tb else "-"
        print(f"{name:<{w}}  {rb:>4}->{ra:<4}  {tb:>6}->{ta:<7}  {x:>5}")
    trb = sum(r[1] for r in rows)
    tra = sum(r[2] for r in rows)
    ttb = sum(r[3] for r in rows)
    tta = sum(r[4] for r in rows)
    x = f"{tta / ttb:.2f}" if ttb else "-"
    print(f"{'-' * w}  {'-' * 9}  {'-' * 15}  {'-' * 5}")
    print(f"{'total':<{w}}  {trb:>4}->{tra:<4}  {ttb:>6}->{tta:<7}  {x:>5}")


if __name__ == "__main__":
    main_stats()
