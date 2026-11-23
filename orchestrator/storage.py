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

SERVICE = "s3"
ALGO = "AWS4-HMAC-SHA256"


class StorageError(RuntimeError):
    pass


def _cfg(public: bool = False) -> dict:
    # env contract: caller loads .env (cli main / tests set os.environ).
    endpoint = os.environ.get("S3_ENDPOINT", "http://pi00.lan:30900").rstrip("/")
    if public:
        endpoint = os.environ.get("S3_PUBLIC_ENDPOINT", endpoint).rstrip("/")
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
            extra_headers: dict[str, str] | None = None,
            public: bool = False) -> str:
    """Pre-signed URL for GET/PUT of bucket/object_key (path-style).
    public=True signs against S3_PUBLIC_ENDPOINT (URLs handed to pods)."""
    c = _cfg(public=public)
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


def mint_job_urls(run_id: str, dataset_key: str | None = None,
                  hours: int = 48) -> dict[str, str]:
    """The standard credential bundle handed to a pod (SPEC §5). All URLs
    are PUBLIC-endpoint-signed: only the pod consumes them."""
    base = f"runs/{run_id}"
    return {
        "payload_get": presign("GET", f"{base}/payload.json", hours * 3600,
                               public=True),
        "dataset_get": (presign("GET", dataset_key, hours * 3600, public=True)
                        if dataset_key else ""),
        "artifacts_put": presign("PUT", f"{base}/artifacts.tar.gz",
                                 hours * 3600, public=True),
        "manifest_put": presign("PUT", f"{base}/manifest.json", hours * 3600,
                                public=True),
        "log_put": presign("PUT", f"{base}/run.log", hours * 3600, public=True),
    }


# -- orchestrator-side object ops (LAN endpoint, presign + httpx) -----------

import httpx  # noqa: E402  (kept below to minimize diff noise)


def put_object(key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    url = presign("PUT", key, 600, extra_headers={"content-type": content_type})
    r = httpx.put(url, content=data, headers={"content-type": content_type},
                  timeout=300)
    r.raise_for_status()


def get_object(key: str) -> bytes | None:
    """None when the object does not exist yet."""
    url = presign("GET", key, 600)
    r = httpx.get(url, timeout=120)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.content


def ensure_bucket() -> None:
    """Create the bucket if missing (header-signed PUT to bucket root)."""
    c = _cfg()
    now = dt.datetime.now(dt.UTC)
    date, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    scope = f"{c['region']}/{SERVICE}/aws4_request"
    host = urllib.parse.urlsplit(c["endpoint"]).netloc
    path = f"/{c['bucket']}/"
    payload_hash = hashlib.sha256(b"").hexdigest()
    canon_headers = f"host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{date}\n"
    canon_req = "\n".join(["PUT", path, "", canon_headers,
                           "host;x-amz-content-sha256;x-amz-date", payload_hash])
    to_sign = "\n".join([ALGO, date, scope,
                         hashlib.sha256(canon_req.encode()).hexdigest()])
    k = _sign(_sign(_sign(_sign(b"AWS4" + c["secret"].encode(), day),
                          c["region"]), SERVICE), "aws4_request")
    sig = hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()
    r = httpx.put(f"{c['endpoint']}{path}", content=b"", timeout=30,
                  headers={
                      "Host": host,
                      "x-amz-date": date,
                      "x-amz-content-sha256": payload_hash,
                      "Authorization": (f"{ALGO} Credential={c['key']}/{scope}, "
                                        f"SignedHeaders=host;x-amz-content-sha256;"
                                        f"x-amz-date, Signature={sig}")})
    if r.status_code not in (200, 409):
        raise StorageError(f"ensure_bucket: HTTP {r.status_code} {r.text[:200]}")
