"""RustFS/S3 pre-signed URL lane (SPEC §5/§8, P1).

The pod never holds S3 credentials: the orchestrator mints pre-signed GETs
(job payload + dataset) and PUTs (artifacts + manifest). Path-style against
any SigV4-compatible endpoint (rustfs). Stdlib-only signing.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import os
import urllib.parse
from pathlib import Path

from .config import load_dotenv

SERVICE = "s3"
ALGO = "AWS4-HMAC-SHA256"


class StorageError(RuntimeError):
    pass


def _cfg() -> dict:
    load_dotenv()
    endpoint = os.environ.get("S3_ENDPOINT", "http://pi00.lan:30900").rstrip("/")
    key = os.environ.get("S3_ACCESS_KEY", "")
    secret = os.environ.get("S3_SECRET_KEY", "")
    if not key or not secret:
        raise StorageError(
            "S3_ACCESS_KEY/S3_SECRET_KEY unset — for local dev these come from "
            "the s3-credentials secret in quadra/infra (SOPS), never committed.")
    return {"endpoint": endpoint, "key": key, "secret": secret,
            "region": os.environ.get("S3_REGION", "us-east-1"),
            "bucket": os.environ.get("S3_BUCKET", "quadra-ml")}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def presign(method: str, object_key: str, expires: int = 48 * 3600,
            extra_headers: dict[str, str] | None = None) -> str:
    """Pre-signed URL for GET/PUT of bucket/object_key (path-style)."""
    c = _cfg()
    now = dt.datetime.now(dt.UTC)
    date = now.strftime("%Y%m%dT%H%M%SZ")
    day = now.strftime("%Y%m%d")
    scope = f"{c['region']}/{SERVICE}/aws4_request"
    host = urllib.parse.urlsplit(c["endpoint"]).netloc
    path = f"/{c['bucket']}/{object_key.lstrip('/')}"

    headers = {"host": host}
    headers.update({k.lower(): v for k, v in (extra_headers or {}).items()})
    signed = ";".join(sorted(headers))
    canon_headers = "".join(f"{k}:{headers[k].strip()}\n" for k in sorted(headers))
    payload_hash = "UNSIGNED-PAYLOAD"

    query = {
        "X-Amz-Algorithm": ALGO,
        "X-Amz-Credential": f"{c['key']}/{scope}",
        "X-Amz-Date": date,
        "X-Amz-Expires": str(expires),
        "X-Amz-SignedHeaders": signed,
    }
    canon_query = "&".join(f"{urllib.parse.quote(k, safe='')}="
                           f"{urllib.parse.quote(v, safe='')}"
                           for k, v in sorted(query.items()))
    canon_req = "\n".join([method.upper(), urllib.parse.quote(path, safe="/"),
                           canon_query, canon_headers, signed, payload_hash])
    to_sign = "\n".join([ALGO, date, scope,
                         hashlib.sha256(canon_req.encode()).hexdigest()])
    k_date = _sign(("AWS4" + c["secret"]).encode(), day)
    k_region = _sign(k_date, c["region"])
    k_service = _sign(k_region, SERVICE)
    k_signing = _sign(k_service, "aws4_request")
    sig = hmac.new(k_signing, to_sign.encode(), hashlib.sha256).hexdigest()
    return (f"{c['endpoint']}{urllib.parse.quote(path, safe='/')}"
            f"?{canon_query}&X-Amz-Signature={sig}")


def mint_job_urls(run_id: str, dataset_sha: str | None = None,
                  hours: int = 48) -> dict[str, str]:
    """The standard credential bundle handed to a pod (SPEC §5)."""
    base = f"runs/{run_id}"
    return {
        "payload_get": presign("GET", f"{base}/job.json", hours * 3600),
        "dataset_get": (presign("GET", dataset_sha, hours * 3600)
                        if dataset_sha else ""),
        "artifacts_put": presign("PUT", f"{base}/artifacts.tar.zst", hours * 3600),
        "manifest_put": presign("PUT", f"{base}/manifest.json", hours * 3600),
        "log_put": presign("PUT", f"{base}/run.log", hours * 3600),
    }
