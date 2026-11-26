"""Lane hygiene law tests (SPEC (ar)): deletion needs PROOF, deletes
verify, and the manifest is evidence — never a candidate."""
import json

from orchestrator import hygiene, storage


def _rows(*specs):
    return [{"run_id": rid, "state": st} for rid, st in specs]


TERMINAL = "done"          # any state outside budget.BILLED
OPEN = "running"


def test_classify_orphan_part_terminal_run_is_junk():
    keys = [("runs/dead1/artifacts.tar.gz.part", 900),
            ("runs/dead1/manifest.json", 10)]
    plan = hygiene.classify(keys, {}, {"dead1"}, set())
    assert [p["key"] for p in plan] == ["runs/dead1/artifacts.tar.gz.part"]


def test_classify_part_of_live_or_unknown_run_is_NOT_proof():
    keys = [("runs/live1/artifacts.tar.gz.part", 900),
            ("runs/ghost/artifacts.tar.gz.part", 900)]
    # live1 is OPEN (not in terminal_ids) and ghost has no ledger row at
    # all — absence of evidence is not proof of death; nothing dies here.
    plan = hygiene.classify(keys, {}, set(), set())
    assert plan == []


def test_classify_manifest_never_a_candidate():
    bad = {"a" * 64}
    keys = [("runs/r1/manifest.json", 10), ("runs/r1/artifacts.tar.gz", 500)]
    manifests = {"r1": {"artifacts": {
        "model.safetensors": {"size": 1, "sha256": "a" * 64}}}}
    plan = hygiene.classify(keys, manifests, {"r1"}, bad)
    assert [p["key"] for p in plan] == ["runs/r1/artifacts.tar.gz"]


def test_classify_bad_sha_needs_the_manifest_record():
    # no sha match -> keep; no manifest -> keep (unreadable != proof)
    keys = [("runs/r1/artifacts.tar.gz", 500)]
    ok = {"artifacts": {"model.safetensors": {"sha256": "b" * 64}}}
    assert hygiene.classify(keys, {"r1": ok}, {"r1"}, {"a" * 64}) == []
    assert hygiene.classify(keys, {}, {"r1"}, {"a" * 64}) == []
    hit = {"artifacts": {"model.safetensors": {"sha256": "a" * 64},
                         "adapter_model.safetensors": {"sha256": "c" * 64}}}
    plan = hygiene.classify(keys, {"r1": hit}, {"r1"}, {"a" * 64})
    assert len(plan) == 1 and "manifest records proven-bad" in plan[0]["reason"]


def test_classify_out_of_scope_prefixes_survive():
    keys = [("datasets/whatever.part", 1), ("toplevel.part", 1),
            ("runs/nested", 1)]
    assert hygiene.classify(keys, {"n": None}, {"n"}, {"z" * 64}) == []


def _wire(monkeypatch, keys, manifests, rows, head_after=404):
    deleted = []

    class FakeLedger:
        def list(self, *a, **k):
            return rows

    monkeypatch.setattr(storage, "vol_list", lambda prefix: list(keys))
    monkeypatch.setattr(storage, "vol_get",
                        lambda k: (json.dumps(manifests[k.split('/')[1]])
                                   if k in [f"runs/{r}/manifest.json"
                                            for r in manifests] else None))
    monkeypatch.setattr(storage, "vol_del",
                        lambda k: (deleted.append(k) or True))
    monkeypatch.setattr(storage, "vol_head", lambda k: head_after)
    return deleted, FakeLedger


def test_sweep_dry_run_touches_nothing(monkeypatch):
    deleted, FakeLedger = _wire(
        monkeypatch, [("runs/dead1/artifacts.tar.gz.part", 900)], {},
        _rows(("dead1", TERMINAL)))
    out = hygiene.sweep(apply=False, led=FakeLedger())
    assert deleted == [] and out["apply"] is False
    assert len(out["plan"]) == 1


def test_sweep_apply_verifies_and_counts_freed(monkeypatch):
    deleted, FakeLedger = _wire(
        monkeypatch, [("runs/dead1/artifacts.tar.gz.part", 900)], {},
        _rows(("dead1", TERMINAL)))
    out = hygiene.sweep(apply=True, led=FakeLedger())
    assert deleted == ["runs/dead1/artifacts.tar.gz.part"]
    assert out["deleted"] == deleted and out["freed_bytes"] == 900
    assert out["unverified"] == []


def test_unverified_delete_is_never_a_success(monkeypatch):
    # verify-after-delete law: HEAD still answers 200 after DELETE ->
    # reported as failure, bytes NOT counted as freed
    deleted, FakeLedger = _wire(
        monkeypatch, [("runs/dead1/artifacts.tar.gz.part", 900)], {},
        _rows(("dead1", TERMINAL)), head_after=200)
    out = hygiene.sweep(apply=True, led=FakeLedger())
    assert out["deleted"] == [] and out["freed_bytes"] == 0
    assert out["unverified"] == ["runs/dead1/artifacts.tar.gz.part"]


def test_sweep_bad_sha_reads_manifest_from_lane(monkeypatch):
    sha = "e" * 64
    deleted, FakeLedger = _wire(
        monkeypatch, [("runs/r1/artifacts.tar.gz", 1_300_000_000),
                      ("runs/r1/manifest.json", 40)],
        {"r1": {"artifacts": {"model.safetensors": {"sha256": sha}}}},
        _rows(("r1", TERMINAL)))
    out = hygiene.sweep(bad_shas=[sha], apply=True, led=FakeLedger())
    assert out["deleted"] == ["runs/r1/artifacts.tar.gz"]
    assert out["freed_bytes"] == 1_300_000_000
    # the manifest itself was never a delete candidate
    assert "runs/r1/manifest.json" not in deleted
