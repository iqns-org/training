"""Pre-signed URL transfer helpers (SPEC §5/§8) — P1 lane.

The pod never holds S3 credentials: the orchestrator mints pre-signed GET
(job payload + dataset) and PUT (artifacts + manifest) URLs at submit time.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import httpx


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def download(url: str, dest: Path, sha256: str | None = None,
             attempts: int = 3, backoff: float = 5.0) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for i in range(attempts):
        try:
            with httpx.stream("GET", url, timeout=600, follow_redirects=True) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with tmp.open("wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
                tmp.replace(dest)
            if sha256 and sha256_file(dest) != sha256:
                raise ValueError(f"sha256 mismatch for {dest.name}")
            return dest
        except Exception as e:                      # noqa: BLE001 — retry lane
            last = e
            time.sleep(backoff * (i + 1))
    raise RuntimeError(f"download failed after {attempts} attempts: {last}")


def upload(path: Path, url: str, attempts: int = 3, backoff: float = 5.0) -> None:
    last: Exception | None = None
    for i in range(attempts):
        try:
            r = httpx.put(url, content=path.read_bytes(), timeout=600)
            r.raise_for_status()
            return
        except Exception as e:                      # noqa: BLE001
            last = e
            time.sleep(backoff * (i + 1))
    raise RuntimeError(f"upload of {path.name} failed after {attempts}: {last}")
