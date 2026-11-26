"""Provenance verification (SPEC 13ak): expected -> staged -> attested.

Three witnesses for every byte that entered a run:

  expected : what the LEDGER recorded at submit (dataset sha/lines, exact
             image digest) — set by merge_spec since 13ak
  staged   : what the payload.json in the lane handed the pod
  attested : what the AGENT hashed in-container after its verify-or-raise
             gates (manifest.attestation, agent >= 0.3.0)

Any two agreeing is comfortable; all three is proof. A disagreement names
the leg that lied: expected!=staged means we STAGED the wrong bytes
(local expansion/checkout bug); staged!=attested means the REMOTE leg
corrupted or substituted (lane/mount/pod bug). Pre-13ak runs report
``warn`` rows ("ledger record none", "attestation none") — the answers
are still one command away via the staged leg, which has always existed
in the payload (de3c7d post-mortem: the payload carried dc9eb6c2 all
along; nobody could ask it).
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from . import storage
from .config import ROOT, corpora_dir
from .dispatch import _sha, corpora_git_sha
from .ledger import Ledger
from .models import TERMINAL

LEDGER_DB = ROOT / "state" / "ledger.db"


def lane_get(key: str, provider: str = "runpod") -> bytes | None:
    """Lane accessor mirroring dispatch's: host/local, volume, or presign."""
    if provider in ("local", "host"):
        return storage.host_get(key)
    try:
        vol = storage.vol_cfg()
    except Exception:
        vol = None
    return storage.vol_get(key) if vol else storage.get_object(key)


def _check(name: str, expected, actual, status: str, note: str = "") -> dict:
    out = {"check": name, "expected": expected, "actual": actual,
           "status": status}
    if note:
        out["note"] = note
    return out


def verify_run(run_id: str | None = None, *, provider: str | None = None,
               live: bool = False, recompute: bool = False,
               led: Ledger | None = None) -> dict:
    """Cross-check ledger vs lane payload vs agent attestation for one run
    (default: most recent). Exit-code-friendly: result['ok'] is False iff
    any check FAILS; warns flag pre-provenance runs / corpus drift."""
    led = led or Ledger(LEDGER_DB)
    if run_id is None:
        rows = led.list()
        if not rows:
            return {"run_id": None, "ok": False,
                    "error": "ledger empty — nothing to verify"}
        row = rows[-1]
        run_id = row["run_id"]
    else:
        row = led.get(run_id)
        if row is None:
            return {"run_id": run_id, "ok": False,
                    "error": f"no such run in ledger: {run_id}"}
    spec = json.loads(row["spec_json"])
    prov = provider or row.get("provider") or "runpod"

    checks: list[dict] = []
    payload = manifest = None
    try:
        raw = lane_get(f"runs/{run_id}/payload.json", prov)
        payload = json.loads(raw) if raw else None
    except Exception as e:                        # noqa: BLE001
        checks.append(_check("payload readable", "json", f"error: {e}",
                             "fail"))
    try:
        raw = lane_get(f"runs/{run_id}/manifest.json", prov)
        manifest = json.loads(raw) if raw else None
    except Exception:                             # noqa: BLE001
        manifest = None

    # -- dataset: expected (ledger) vs staged (payload) vs attested -------
    staged_ds = (payload or {}).get("dataset") or {}
    staged_sha = staged_ds.get("sha256", "")
    exp_sha = spec.get("dataset_sha", "")
    if exp_sha and staged_sha:
        checks.append(_check("dataset ledger==staged", exp_sha[:16],
                             staged_sha[:16],
                             "pass" if exp_sha == staged_sha else "fail"))
    elif staged_sha:
        checks.append(_check("dataset ledger record", "(sha at submit)",
                             "none — pre-13ak run", "warn",
                             note=f"staged was {staged_sha[:16]}"))
    att = (manifest or {}).get("attestation") or {}
    att_ds = att.get("dataset") or {}
    if staged_sha:
        if att_ds:
            ok = att_ds.get("sha256") == staged_sha
            checks.append(_check("dataset staged==attested", staged_sha[:16],
                                 att_ds.get("sha256", "")[:16],
                                 "pass" if ok else "fail"))
            exp_lines = spec.get("dataset_lines")
            if exp_lines is not None and att_ds.get("lines") is not None:
                checks.append(_check("dataset lines", exp_lines,
                                     att_ds["lines"],
                                     "pass" if exp_lines == att_ds["lines"]
                                     else "fail"))
        else:
            checks.append(_check("agent attestation", "agent>=0.3.0",
                                 "none — pre-attestation image", "warn"))

    # -- stage inputs (merge/awq/...): staged contract vs ledger spec ------
    spec_inputs = {i.get("sha256", ""): i.get("name")
                   for i in spec.get("inputs", [])}
    for i in (payload or {}).get("inputs", []):
        sha = i.get("sha256", "")
        if not sha:
            continue
        if spec_inputs:
            checks.append(_check(f"input {i.get('name')}", "in ledger spec",
                                 "recorded" if sha in spec_inputs
                                 else "MISSING",
                                 "pass" if sha in spec_inputs else "fail"))

    # -- holdout (SPEC 13am): the eval leg of a train run -----------------
    exp_hold = spec.get("holdout_sha", "")
    staged_hold = next((i.get("sha256", "") for i in
                        (payload or {}).get("inputs", [])
                        if "holdout" in str(i.get("name", ""))), "")
    if exp_hold or staged_hold:
        checks.append(_check("holdout ledger==staged", exp_hold[:16] or "?",
                             staged_hold[:16] or "none",
                             "pass" if exp_hold and exp_hold == staged_hold
                             else "fail"))
        held = next((i for i in (att.get("inputs") or [])
                     if "holdout" in str(i.get("name", ""))), None)
        if held:
            checks.append(_check("holdout staged==attested",
                                 staged_hold[:16],
                                 held.get("sha256", "")[:16],
                                 "pass" if held.get("sha256") == staged_hold
                                 else "fail"))

    # -- config: staged bytes vs what the pod hashed ----------------------
    if payload and att.get("config_sha") and payload.get("config_yaml"):
        want = hashlib.sha256(payload["config_yaml"].encode()).hexdigest()
        checks.append(_check("config staged==attested", want[:16],
                             att["config_sha"][:16],
                             "pass" if want == att["config_sha"] else "fail"))

    # -- image: recorded vs launched/attested -----------------------------
    exp_img = spec.get("image", "")
    got_img = (manifest or {}).get("image_tag", "")
    if exp_img and got_img:
        checks.append(_check("image ledger==manifest", exp_img[-20:],
                             got_img[-20:],
                             "pass" if exp_img == got_img else "fail"))
    elif exp_img:
        checks.append(_check("image attested", exp_img[-20:],
                             "no manifest yet" if row["state"]
                             not in TERMINAL else "manifest/image missing",
                             "warn"))

    # -- drift: is the local tree still what the run trained on? ----------
    if spec.get("corpora"):
        cur = corpora_git_sha()
        checks.append(_check("corpora drift", spec["corpora"][:12],
                             (cur or "?")[:12],
                             "pass" if cur == spec["corpora"] else "warn",
                             note="local checkout moved since run"
                             if cur != spec["corpora"] else ""))
    if spec.get("recipe_sha") and spec.get("recipe"):
        rp = Path(spec["recipe"])
        if not rp.is_absolute():
            rp = corpora_dir() / str(spec["recipe"])
        if rp.is_file():
            cur = _sha(rp)[:12]
            checks.append(_check("recipe drift", spec["recipe_sha"], cur,
                                 "pass" if cur == spec["recipe_sha"]
                                 else "warn"))

    if recompute and spec.get("domain"):
        from .dispatch import prepare_dataset
        key, _p, sha, lines = prepare_dataset(spec["domain"])
        checks.append(_check("recompute vs staged", staged_sha[:16] or "?",
                             sha[:16],
                             "pass" if sha == staged_sha else "fail",
                             note=f"lines now {lines}"))

    # -- live: ask the pod itself (running jobs only) ---------------------
    if live and row["state"] not in TERMINAL and row.get("base_url"):
        token = hashlib.sha256(run_id.encode()).hexdigest()[:24]
        try:
            req = urllib.request.Request(
                row["base_url"].rstrip("/") + "/v1/sanity",
                headers={"Authorization": f"Bearer {token}"})
            got = json.loads(urllib.request.urlopen(req, timeout=15).read())
            checks.append(_check("live agent image", exp_img or "?",
                                 got.get("image_tag", "") or "?",
                                 "pass" if (not exp_img
                                             or got.get("image_tag")
                                             == exp_img) else "fail"))
            livesha = ((got.get("attestation") or {}).get("dataset")
                       or {}).get("sha256", "")
            if livesha and staged_sha:
                checks.append(_check("live staged==attested",
                                     staged_sha[:16], livesha[:16],
                                     "pass" if livesha == staged_sha
                                     else "fail"))
        except Exception as e:                    # noqa: BLE001
            checks.append(_check("live sanity", "reachable", f"error: {e}",
                                 "fail"))

    ok = all(c["status"] != "fail" for c in checks)
    return {"run_id": run_id, "kind": row["kind"], "state": row["state"],
            "agent_version": (manifest or {}).get("agent_version", ""),
            "ok": ok, "checks": checks}
