"""Env loading shared by CLI and (later) mcp-trainer."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path | None = None) -> None:
    """Populate os.environ from .env (first match wins, values are not
    overwritten if already set — process env beats file env)."""
    candidates = [path or ROOT / ".env", Path.cwd() / ".env"]
    for file in candidates:
        if not file.is_file():
            continue
        for line in file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value
        break


def require_env(name: str) -> str:
    load_dotenv()
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set (export it or put it in {ROOT / '.env'})")
    return value


def corpora_dir() -> Path:
    """The task-data checkout (SPEC §4b boundary: training repo = machinery,
    corpora = data + domain config: datasets, recipes, bench suites,
    templates, tools). CORPORA_DIR may be relative — resolved against ROOT
    so CWD never changes the answer (the pod sets it absolute, .env sets
    it ../corpora). Does NOT load_dotenv (entry points do; doing it here
    would leak deployment env into mid-process lane decisions — a real
    bug the test suite caught). Falls back to the sibling checkout, then
    the legacy in-repo datasets/ location, whichever exists."""
    d = os.environ.get("CORPORA_DIR", "").strip()
    if d:
        p = Path(d)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        if p.is_dir():
            return p
    for cand in (ROOT.parent / "corpora", ROOT):
        if (cand / "datasets").is_dir():
            return cand
    return ROOT


def corpora_git_sha() -> str:
    """Short git SHA of the corpora checkout — dataset+recipe provenance
    per run (SPEC 14c: any UI row is reproducible against exact task
    state). '' when corpora isn't a git repo (test fixtures, legacy)."""
    import subprocess
    try:
        r = subprocess.run(["git", "-C", str(corpora_dir()), "rev-parse",
                            "--short", "HEAD"], capture_output=True,
                           text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:                              # noqa: BLE001
        return ""


def resolve_recipe(recipe: str) -> Path:
    """Recipe path resolution across the boundary: absolute as-is; else
    corpora-relative (recipes/<domain>/x.yaml or bare <domain>/x.yaml),
    then legacy in-repo axolotl/. Returns the first that exists."""
    p = Path(recipe)
    if p.is_absolute():
        return p
    c = corpora_dir()
    for cand in (c / "recipes" / recipe, c / recipe, ROOT / recipe):
        if cand.is_file():
            return cand
    return p  # not found: caller reports the original path honestly
