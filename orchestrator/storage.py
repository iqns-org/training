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
import shlex
import subprocess
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
    scope = f"{day}/{c['region']}/{SERVICE}/aws4_request"
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
                  hours: int = 48,
                  holdout_key: str | None = None) -> dict[str, str]:
    """The standard credential bundle handed to a pod (SPEC §5). All URLs
    are PUBLIC-endpoint-signed: only the pod consumes them."""
    base = f"runs/{run_id}"
    return {
        "payload_get": presign("GET", f"{base}/payload.json", hours * 3600,
                               public=True),
        "dataset_get": (presign("GET", dataset_key, hours * 3600, public=True)
                        if dataset_key else ""),
        "holdout_get": (presign("GET", holdout_key, hours * 3600,
                                public=True) if holdout_key else ""),
        "artifacts_put": presign("PUT", f"{base}/artifacts.tar.gz",
                                 hours * 3600, public=True),
        "manifest_put": presign("PUT", f"{base}/manifest.json", hours * 3600,
                                public=True),
        "status_put": presign("PUT", f"{base}/status.json", hours * 3600,
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


def s3_request(method: str, endpoint: str, region: str, access_key: str,
               secret_key: str, path: str, body: bytes = b"",
               timeout: float = 300.0,
               query: dict | None = None) -> httpx.Response:
    """Header-signed SigV4 request (path begins with '/'; RunPod's S3 API
    supports NO presign, so the orchestrator side signs headers).
    `query` params (SPEC 13ao lane tooling: ListObjectsV2 paging) ride
    the canonical request string — an unsigned query = SignatureDoesNotMatch."""
    now = dt.datetime.now(dt.UTC)
    date, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    scope = f"{day}/{region}/{SERVICE}/aws4_request"   # 5-part credential
    host = urllib.parse.urlsplit(endpoint).netloc
    payload_hash = hashlib.sha256(body).hexdigest()
    canon_uri = urllib.parse.quote(path, safe="/")
    qstr = "&".join(f"{urllib.parse.quote(k, safe='')}="
                    f"{urllib.parse.quote(str(v), safe='')}"
                    for k, v in sorted((query or {}).items()))
    signed = "host;x-amz-content-sha256;x-amz-date"
    canon_headers = (f"host:{host}\nx-amz-content-sha256:{payload_hash}\n"
                     f"x-amz-date:{date}\n")
    canon_req = "\n".join([method.upper(), canon_uri, qstr, canon_headers,
                           signed, payload_hash])
    to_sign = "\n".join([ALGO, date, scope,
                         hashlib.sha256(canon_req.encode()).hexdigest()])
    k = _sign(_sign(_sign(_sign(b"AWS4" + secret_key.encode(), day), region),
                    SERVICE), "aws4_request")
    sig = hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()
    url = f"{endpoint}{canon_uri}" + (f"?{qstr}" if qstr else "")
    return httpx.request(method, url, content=body,
                         timeout=timeout,
                         headers={"Host": host, "x-amz-date": date,
                                  "x-amz-content-sha256": payload_hash,
                                  "Authorization":
                                  f"{ALGO} Credential={access_key}/{scope}, "
                                  f"SignedHeaders={signed}, Signature={sig}"})


# -- host lane (local provider): workdir files on cluster machines -------
# Inside the mcp-trainer pod the same tree is mounted, so reads/writes are
# plain filesystem ops; from any box WITHOUT the mount (workstation) they
# ride ssh cat. LOCAL_WORKDIR = the real host path (docker -v source);
# LOCAL_LANE_DIR = orchestrator's view of it (/app inside the pod).

def _lane_dir() -> str | None:
    d = os.environ.get("LOCAL_LANE_DIR") or os.environ.get(
        "LOCAL_WORKDIR", "/srv/mcps/trainer")
    return d if Path(d).is_dir() else None


def _default_host() -> str:
    return os.environ.get("LOCAL_SSH_HOST", "me@pi00.lan")


def host_put(key: str, data: bytes) -> None:
    direct = _lane_dir()
    if direct:
        p = Path(direct) / key
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(p)
        return
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=10",
                        "-o", "BatchMode=yes", _default_host(),
                        f"mkdir -p $(dirname {shlex.quote(key_path(key))}) && "
                        f"cat > {shlex.quote(key_path(key))}"],
                       input=data, capture_output=True, timeout=300)
    if r.returncode != 0:
        raise StorageError(f"host_put {key}: {r.stderr[:200]!r}")


def key_path(key: str) -> str:
    wd = os.environ.get("LOCAL_WORKDIR", "/srv/mcps/trainer")
    return f"{wd.rstrip('/')}/{key.lstrip('/')}"


def host_get(key: str) -> bytes | None:
    direct = _lane_dir()
    if direct:
        p = Path(direct) / key
        return p.read_bytes() if p.is_file() else None
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=10",
                        "-o", "BatchMode=yes", _default_host(),
                        f"cat {shlex.quote(key_path(key))}"],
                       capture_output=True, timeout=300)
    return r.stdout if r.returncode == 0 else None


def ensure_bucket() -> None:
    c = _cfg()
    r = s3_request("PUT", c["endpoint"], c["region"], c["key"], c["secret"],
                   f"/{c['bucket']}/")
    if r.status_code not in (200, 409):
        raise StorageError(f"ensure_bucket: HTTP {r.status_code} {r.text[:200]}")


# -- RunPod network-volume lane ----------------------------------------------
# Pods mount the volume (zero credentials in-pod, no public route needed);
# the orchestrator reads/writes the same bytes through the per-DC S3 API.

def vol_cfg() -> dict | None:
    """RunPod S3-API config from RUNPOD_S3_* / RUNPOD_VOLUME_ID; None when
    unset = volume lane off (presign lane stays the portable default)."""
    vid = os.environ.get("RUNPOD_VOLUME_ID", "")
    ak = os.environ.get("RUNPOD_S3_ACCESS_KEY", "")
    if not (vid and ak):
        return None
    return {"bucket": vid, "key": ak,
            "secret": os.environ.get("RUNPOD_S3_SECRET_KEY", ""),
            "endpoint": os.environ.get("RUNPOD_S3_ENDPOINT", "").rstrip("/"),
            "region": os.environ.get("RUNPOD_S3_REGION", "EU-RO-1"),
            "dc": os.environ.get("RUNPOD_VOLUME_DC",
                                 os.environ.get("RUNPOD_S3_REGION", "EU-RO-1")),
            "mount": os.environ.get("RUNPOD_VOLUME_MOUNT", "/data")}


def vol_put(key: str, data: bytes) -> None:
    c = vol_cfg()
    r = s3_request("PUT", c["endpoint"], c["region"], c["key"], c["secret"],
                   f"/{c['bucket']}/{key}", data)
    if r.status_code != 200:
        raise StorageError(f"vol_put {key}: HTTP {r.status_code} {r.text[:200]}")


def vol_del(key: str) -> bool:
    """DELETE one volume-object (SPEC 13ao lane hygiene). Returns True on
    success. Retention policy is deliberate: only objects proven junk
    (no-op base-clone merges, superseded failed-stage payloads) get here
    — the lane is the family's evidence locker, not a scratch disk."""
    c = vol_cfg()
    if not c:
        return False
    r = s3_request("DELETE", c["endpoint"], c["region"], c["key"],
                   c["secret"], f"/{c['bucket']}/{key}")
    return r.status_code in (204, 200)


def vol_get(key: str) -> bytes | None:
    c = vol_cfg()
    r = s3_request("GET", c["endpoint"], c["region"], c["key"], c["secret"],
                   f"/{c['bucket']}/{key}")
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        raise StorageError(f"vol_get {key}: HTTP {r.status_code} {r.text[:200]}")
    return r.content
