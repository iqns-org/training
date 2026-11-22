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
