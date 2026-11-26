"""Lane hygiene (SPEC 13ao, entry (ar)) — delete only PROVEN junk,
verify-after-delete, evidence forever.

The volume lane carries the account-wide RunPod quota, so housekeeping
exists — but its whole design is that a deletion needs a PROOF, and two
proof shapes are implemented here and nothing else:

  orphan .part   only a LIVE pod writes ``*.part`` objects (finalize's
                 _vol_write stages through them), so a .part under a run
                 the ledger says is TERMINAL is a corpse — byte-for-byte
                 unresumable. Unknown run ids are NOT proof of death:
                 absence of a row keeps the object.
  manifest-sha   finalize hashed every artifact and wrote manifest.json
                 LAST, so the tiny manifest is the authoritative index:
                 a run whose manifest records an artifact sha256 in the
                 caller's bad-sha set (e.g. the no-op base-clone blob
                 2cdf190a, 13ao) has a proven-junk artifacts.tar.gz —
                 no 1.3 GB download needed to prove it.

manifest.json itself is NEVER a candidate: after the junk bytes die,
the manifest is the evidence of what the junk was. Every other object
lives forever. And a delete is not done until HEAD answers 404 —
unverified removals are reported as failures, never as freed bytes.
"""

from __future__ import annotations

import json

from . import storage
from .budget import BILLED
from .ledger import Ledger


def _run_of(key: str) -> str | None:
    parts = key.split("/")
    if key.startswith("runs/") and len(parts) > 2:
        return parts[1]
    return None


def classify(keys, manifests, terminal_ids, bad_shas) -> list[dict]:
    """Pure proof classifier (no I/O — the suite pins the laws here).

    keys: [(key, size)]; manifests: {run_id: parsed manifest dict or None};
    terminal_ids: runs the LEDGER explicitly says are terminal;
    bad_shas: artifact sha256 values proven junk. -> plan rows."""
    plan = []
    for key, size in keys:
        rid = _run_of(key)
        if rid is None or key.endswith("/manifest.json"):
            continue                     # out of scope / evidence law
        if key.endswith(".part"):
            if rid in terminal_ids:
                plan.append({"key": key, "size": size,
                             "reason": f"orphan .part under terminal "
                                       f"run {rid}"})
            continue
        if key.endswith("/artifacts.tar.gz") and bad_shas:
            man = manifests.get(rid)
            for name, meta in (man or {}).get("artifacts", {}).items():
                if meta.get("sha256") in bad_shas:
                    plan.append({"key": key, "size": size,
                                 "reason": f"manifest records proven-bad "
                                           f"sha {meta['sha256'][:12]} "
                                           f"({name})"})
                    break
    return plan


def sweep(bad_shas=(), apply: bool = False, led: Ledger | None = None) -> dict:
    """List the lane, classify, and (only with apply=True) delete +
    VERIFY each removal. Dry-run is the default — the plan is printed
    for a human before bytes die."""
    led = led or Ledger()
    bad_shas = set(bad_shas)
    keys = storage.vol_list("runs/")
    terminal = {r["run_id"] for r in led.list() if r["state"] not in BILLED}
    want_manifests = {rid for k, _ in keys if (rid := _run_of(k))
                      and k.endswith("/artifacts.tar.gz")} if bad_shas else set()
    manifests: dict = {}
    for rid in want_manifests:
        raw = storage.vol_get(f"runs/{rid}/manifest.json")
        try:
            manifests[rid] = json.loads(raw) if raw else None
        except ValueError:
            manifests[rid] = None        # unreadable manifest = not proof
    plan = classify(keys, manifests, terminal, bad_shas)
    out = {"apply": apply, "plan": plan, "deleted": [],
           "unverified": [], "freed_bytes": 0}
    if not apply:
        return out
    for row in plan:
        if storage.vol_del(row["key"]) and storage.vol_head(row["key"]) == 404:
            out["deleted"].append(row["key"])
            out["freed_bytes"] += row["size"]
        else:
            # verify-after-delete law: still there (or delete refused) —
            # report the failure, never count it as freed
            out["unverified"].append(row["key"])
    return out
