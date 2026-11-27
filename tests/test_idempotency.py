"""SPEC 14 idempotency: reap / wave-launch backoff / relaunch / janitor."""
import hashlib
import json
import time

import pytest

from orchestrator import dispatch, models
from orchestrator.ledger import Ledger
from orchestrator.models import Instance, Offer, ProviderError


def _inst(pid, name, created_ago=3600, status="running"):
    return Instance(provider="runpod", id=pid, status=status, name=name,
                    gpu_type="RTX A4500", price_hr=0.25,
                    raw={"createdAt": time.time() - created_ago})


class StubProv:
    def __init__(self, instances):
        self.instances = instances
        self.terminated = []

    def list_instances(self):
        return list(self.instances)

    def terminate(self, pid):
        self.terminated.append(pid)
        return True


@pytest.fixture()
def led_root(tmp_path, monkeypatch):
    (tmp_path / "state").mkdir(exist_ok=True)
    monkeypatch.setattr(dispatch, "ROOT", tmp_path)
    return tmp_path


def test_reap_classifies_by_claim(led_root, monkeypatch):
    monkeypatch.delenv("QTRAIN_REAP_UNCLAIMED", raising=False)
    led = Ledger(led_root / "state" / "ledger.db")
    rid_done = led.create("stage", {"kind": "onnx"}, provider="runpod")
    led.update(rid_done, state="done", instance_id="pod-done")
    rid_run = led.create("train", {}, provider="runpod")
    led.update(rid_run, state="running", instance_id="pod-live")
    pods = [_inst("pod-done", f"trainer-{rid_done[-8:]}"),           # leak
            _inst("pod-live", f"trainer-{rid_run[-8:]}"),            # live
            _inst("pod-new", "trainer-abcd1234"),                    # unclaimed old
            _inst("pod-baby", "trainer-abcd5678", created_ago=60),   # grace
            _inst("pod-vllm", "user-vllm-stuff")]                    # foreign
    prov = StubProv(pods)
    monkeypatch.setattr(dispatch, "get_provider", lambda p: prov)
    out = dispatch.reap("runpod", grace=900)
    killed = {k["id"] for k in out["killed"]}
    kept = {k["id"] for k in out["kept"]}
    assert killed == {"pod-done"}                 # terminal claim only
    assert {"pod-live", "pod-baby"} <= kept       # open run + fresh launch
    assert "pod-vllm" not in killed | kept        # never even considered
    reason = next(k["why"] for k in out["kept"] if k["id"] == "pod-new")
    assert "another controller" in reason         # default: keep, don't kill


def test_reap_stale_suffix_claim_and_unclaimed_opt_in(led_root, monkeypatch):
    led = Ledger(led_root / "state" / "ledger.db")
    rid = led.create("train", {}, provider="runpod")  # never got instance
    led.update(rid, state="failed")                   # terminal, no claim
    pods = [_inst("pod-x", f"trainer-{rid[-8:]}"),    # stale suffix match
            _inst("pod-y", "trainer-zzzzzzzz")]       # unknown run id
    prov = StubProv(pods)
    monkeypatch.setattr(dispatch, "get_provider", lambda p: prov)
    monkeypatch.delenv("QTRAIN_REAP_UNCLAIMED", raising=False)
    out = dispatch.reap("runpod", grace=900)
    assert {k["id"] for k in out["killed"]} == {"pod-x"}   # suffix rule works
    monkeypatch.setenv("QTRAIN_REAP_UNCLAIMED", "1")
    prov2 = StubProv(pods)
    monkeypatch.setattr(dispatch, "get_provider", lambda p: prov2)
    out2 = dispatch.reap("runpod", grace=900)
    assert {k["id"] for k in out2["killed"]} == {"pod-x", "pod-y"}


def test_reap_dry_kills_nothing(led_root, monkeypatch):
    led = Ledger(led_root / "state" / "ledger.db")
    rid = led.create("train", {}, provider="runpod")
    led.update(rid, state="done", instance_id="pod-1")
    prov = StubProv([_inst("pod-1", f"trainer-{rid[-8:]}")])
    monkeypatch.setattr(dispatch, "get_provider", lambda p: prov)
    out = dispatch.reap("runpod", dry=True)
    assert out["killed"][0]["dry"] is True
    assert prov.terminated == []


def test_wave_launch_backs_off_then_wins():
    calls = {"n": 0}

    class Prov:
        def launch(self, **kw):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ProviderError("HTTP 400 no instances")
            return Instance(provider="runpod", id="pod-ok", status="booting")

    offers = [Offer(provider="runpod", gpu_id="A5000", vram_gb=24,
                    price_hr=0.69, secure=True)]
    slept = []
    inst, offer = dispatch._wave_launch(
        Prov(), lambda: offers, name="n", image="i", env={}, lk={},
        run_id="r" * 24, kind="awq", max_wait=3600,
        backoff0=1, sleep=slept.append)
    assert inst.id == "pod-ok" and offer.gpu_id == "A5000"
    assert slept == [1, 2]                       # exponential, not busy-loop


def test_wave_launch_deadline_fastfail():
    class Prov:
        def launch(self, **kw):
            raise ProviderError("HTTP 400 no instances")

    offers = [Offer(provider="runpod", gpu_id="A5000", vram_gb=24,
                    price_hr=0.69, secure=True)]
    with pytest.raises(ProviderError) as e:
        dispatch._wave_launch(
            Prov(), lambda: offers, name="n", image="i", env={}, lk={},
            run_id="r" * 24, kind="awq", max_wait=0)
    assert "no capacity across offers" in str(e.value)


def test_relaunch_pending_stage_run(led_root, monkeypatch):
    rid = "20261124-relaun1"
    led = Ledger(led_root / "state" / "ledger.db")
    led.create("stage", {"kind": "onnx", "from_run": "src"},
               provider="runpod", run_id=rid)   # pending, no instance
    payload = {"run_id": rid, "command": "python /work/inputs/export_onnx.py",
               "inputs": [], "outputs_dir": "/work/outputs",
               "gate": {"required_files": ["model.onnx"]}, "urls": {},
               "volume": {"mount": "/data", "prefix": f"runs/{rid}"}}
    tar = b"fake-tar-bytes"
    manifest = {"status": "succeeded", "artifacts": {"model.onnx": {}},
                "artifacts_tar_sha256": hashlib.sha256(tar).hexdigest()}
    lane = {f"runs/{rid}/payload.json": json.dumps(payload).encode(),
            f"runs/{rid}/manifest.json": json.dumps(manifest).encode(),
            f"runs/{rid}/artifacts.tar.gz": tar}
    monkeypatch.setattr(dispatch.storage, "vol_cfg",
                        lambda: {"bucket": "v1", "mount": "/data",
                                 "dc": "EU-RO-1", "key": "k", "secret": "s",
                                 "endpoint": "e", "region": "EU-RO-1"})
    monkeypatch.setattr(dispatch.storage, "vol_get", lambda k: lane.get(k))
    monkeypatch.setattr(dispatch.storage, "vol_put",
                        lambda k, d, ct="": None)

    terminated = []

    class Prov:
        def catalog(self, gpus=None, tier=None):
            return [Offer(provider="runpod", gpu_id="A4500", vram_gb=24,
                           price_hr=0.25, secure=True)]

        def launch(self, **kw):
            assert kw["env"]["JOB_PAYLOAD_PATH"].endswith("payload.json")
            return Instance(provider="runpod", id="pod-relaunch",
                            status="booting", base_url="http://x")

        def poll(self, pid):
            return Instance(provider="runpod", id=pid, status="running")

        def terminate(self, pid):
            terminated.append(pid)
            return True
    monkeypatch.setattr(dispatch, "get_provider", lambda p: Prov())

    res = dispatch.relaunch(rid, poll=1, max_runtime=30)
    assert res.get("status") == "succeeded", res
    assert terminated == ["pod-relaunch"]
    row = led.get(rid)
    assert row["state"] == "done" and row["instance_id"] == "pod-relaunch"
    assert row["price_hr"] == 0.25 and row["est_cost"] > 0


def test_resume_deadline_rides_the_row(tmp_path, monkeypatch):
    """DEADLINE RIDES THE ROW (485a22 law): a re-attach must NOT restart
    the clock and must use the budget RECORDED AT SUBMIT, not resume()'s
    4 h default. One row discriminates both: created 2 h ago with a
    recorded 10-min budget. Old code (started=now, default budget) would
    keep watching and poll the provider; the fake provider would raise
    LOUDLY. New code fires the deadline on the first check, before any
    poll — the looping supervisor's 90-min zombie pod becomes impossible."""
    import time as _t
    from orchestrator.ledger import Ledger
    (tmp_path / "state").mkdir(exist_ok=True)
    monkeypatch.setattr(dispatch, "ROOT", tmp_path)
    led = Ledger(tmp_path / "state" / "ledger.db")
    rid = led.create("train", {"domain": "queer", "max_runtime": 600},
                     provider="runpod")
    led.update(rid, state="provisioning", instance_id="pod-old")
    with led.db:
        led.db.execute("UPDATE runs SET created_at=? WHERE run_id=?",
                       (_t.time() - 7200, rid))

    class LoudProv:
        def poll(self, inst):
            raise AssertionError("must never poll: deadline must fire first")

        def terminate(self, inst):
            return True
    monkeypatch.setattr(dispatch, "get_provider", lambda p: LoudProv())
    res = dispatch.resume(rid, poll=1)
    assert res["status"] == "error"
    assert "max runtime exceeded without manifest" in res["error"]
    assert led.get(rid)["state"] == "failed"      # terminal, not zombie


def test_relaunch_refuses_wrong_shapes(led_root):
    led = Ledger(led_root / "state" / "ledger.db")
    rid = led.create("train", {}, provider="runpod")
    led.update(rid, state="running", instance_id="pod-live")
    assert "already has an instance" in dispatch.relaunch(rid)["error"]
    rid2 = led.create("train", {}, provider="runpod")
    led.update(rid2, state="done")
    assert "not pending" in dispatch.relaunch(rid2)["error"]


def test_resume_routes_pending_ghost_to_relaunch(led_root, monkeypatch):
    calls = {"relaunch": []}
    monkeypatch.setattr(dispatch, "relaunch",
                        lambda rid, **kw: calls["relaunch"].append(rid)
                        or {"status": "routed"})
    led = Ledger(led_root / "state" / "ledger.db")
    rid = led.create("stage", {"kind": "int8"}, provider="runpod")
    # state stays pending, no instance: exactly the ghost shape
    out = dispatch.resume(rid)
    assert out == {"status": "routed"}
    assert calls["relaunch"] == [rid]


def test_pod_name_is_human_and_contract_safe(tmp_path):
    # kind-first, internal slug from base_model, trailing run-suffix
    rec = tmp_path / "Qwen3.5-0.8B-qlora.yaml"
    rec.write_text("base_model: Qwen/Qwen3.5-0.8B\n")
    n = dispatch._pod_name("train", "20261124-191736-425e6c",
                           dispatch._pod_hint("queer", str(rec)))
    assert n == "train-queer-qwen3.5-0.8b-6-425e6c"
    assert n.endswith("6-425e6c") and len(n) <= 63
    assert models.pod_is_ours(n)
    # legacy trainer- names stay reapable (pods may live across the change)
    assert models.pod_is_ours("trainer-train.queer.qwen3508b-6-425e6c")
    assert models.pod_is_ours("trainer-1-b9b130")
    # foreign names are NOT ours
    for f in ("user-vllm-stuff", "training-run-job", "qdrant-0"):
        assert not models.pod_is_ours(f)
    # fallback: recipe without base_model -> filename stem minus qlora
    rec2 = tmp_path / "SmolLM2-360M-qlora.yaml"
    rec2.write_text("nope: 1\n")
    h = dispatch._pod_hint("queer", str(rec2))
    assert "smollm2-360m" in h
    # stages tag the parent run; no hint must not double-dash
    s = dispatch._pod_name("awq", "20261124-200000-abc123",
                           dispatch._pod_hint(parent="20261123-162929-20e8db"))
    assert s == "awq-for20e8db-0-abc123"
    assert dispatch._pod_name("merge", "r-x-99887766") == "merge-99887766"


def test_reap_matches_new_and_legacy_names(led_root, monkeypatch):
    """name[-8:] suffix parsing kills BOTH kind-first hint names and
    legacy trainer- names — the contract token always sits at the tail."""
    monkeypatch.delenv("QTRAIN_REAP_UNCLAIMED", raising=False)
    led = Ledger(led_root / "state" / "ledger.db")
    rid = led.create("train", {}, provider="runpod")
    led.update(rid, state="failed")
    new = f"train-queer-qwen3.5-0.8b-{rid[-8:]}"
    legacy = f"trainer-{rid[-8:]}"
    prov = StubProv([_inst("pod-a", new), _inst("pod-b", legacy),
                     _inst("pod-c", "user-vllm-stuff")])
    monkeypatch.setattr(dispatch, "get_provider", lambda p: prov)
    out = dispatch.reap("runpod", grace=900)
    assert {k["id"] for k in out["killed"]} == {"pod-a", "pod-b"}


def test_net_blip_classification():
    assert dispatch._net_blip(OSError(-3, "Temporary failure in name "
                                          "resolution"))
    assert dispatch._net_blip(RuntimeError("[Errno -3] Temporary failure"))
    assert not dispatch._net_blip(RuntimeError("budget cap $1 crossed"))
    assert not dispatch._net_blip(RuntimeError("max runtime exceeded"))
    assert not dispatch._net_blip(RuntimeError("instance x terminated "
                                               "without manifest"))


def test_watch_lost_keeps_pod(led_root, monkeypatch):
    """DNS death mid-watch must NOT terminate the running pod: row stays
    OPEN with an unwatched note, status=unwatched, watch_lost emitted."""
    from orchestrator import notify
    journal = led_root / "notifications.jsonl"
    monkeypatch.setattr(notify, "JOURNAL", journal)

    class DeadProv:
        def terminate(self, i):
            raise AssertionError(f"terminate({i}) must not be called on blip")

        def poll(self, i):
            raise OSError(-3, "Temporary failure in name resolution")

        def list_instances(self):
            return []

    monkeypatch.setattr(dispatch, "get_provider", lambda name: DeadProv())
    # fast path: the watch raises the blip (first lane read behind DNS)
    monkeypatch.setattr(dispatch, "_watch",
                        lambda *a, **k: (_ for _ in ()).throw(
                            OSError(-3, "Temporary failure in name "
                                        "resolution")))
    led = Ledger(led_root / "state" / "ledger.db")
    rid = led.create("stage", {"kind": "onnx", "from_run": "r-parent"},
                     provider="runpod")
    led.update(rid, state="provisioning", instance_id="pod-live")
    out = dispatch.resume(rid)          # re-attaches then hits thrown blip
    assert out["status"] == "unwatched"
    assert out["pod"] == "pod-live"
    row = led.get(rid)
    assert row["state"] == "provisioning"        # still OPEN, not failed
    assert "watch lost" in (row["error"] or "")
    assert out.get("terminated") is None
    assert json.loads(journal.read_text().strip().splitlines()[-1])[
        "event"] == "watch_lost"


def test_terminated_flag_is_honest(led_root, monkeypatch):
    """terminate() returning False (kill refused / blip during confirm)
    must surface as terminated:false — 2026-11-24 leak: a blanket true
    hid two billing pods for an hour."""
    class BadProv:
        def terminate(self, i):
            return False

        def poll(self, i):
            raise ProviderError("gone")

        def list_instances(self):
            return []

    monkeypatch.setattr(dispatch, "get_provider", lambda name: BadProv())
    monkeypatch.setattr(dispatch, "_watch",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("budget cap $1 crossed")))
    led = Ledger(led_root / "state" / "ledger.db")
    rid = led.create("train", {}, provider="runpod")
    led.update(rid, state="running", instance_id="pod-x")
    out = dispatch.resume(rid)
    assert out["status"] == "error"
    assert out["terminated"] is False


def test_stage_base_auto_resolves(led_root, monkeypatch, tmp_path):
    """merge without --base inherits base_model from the from-run's
    recorded recipe (provenance), error only when truly unknowable."""
    rec = tmp_path / "r.yaml"
    rec.write_text("base_model: Qwen/Qwen3.5-0.8B\n")
    monkeypatch.setattr(dispatch, "resolve_recipe", lambda r: rec)
    led = Ledger(led_root / "state" / "ledger.db")
    rid = led.create("train", {"recipe": "queer/x.yaml"},
                     provider="runpod")
    assert dispatch._recipe_base_from_row(rid) == "Qwen/Qwen3.5-0.8B"
    # no recipe recorded -> empty -> caller returns the actionable error
    rid2 = led.create("train", {}, provider="runpod")
    assert dispatch._recipe_base_from_row(rid2) == ""


def test_resume_suffix_resolves_unknown_id_errors(tmp_path, monkeypatch):
    """resume(run_id): suffix ids resolve like the rest of the CLI
    (verify/register/doctor all take suffixes) — 'no open runs' to a
    suffix of a LIVE run was a no-op wearing success (f831b7). And an
    unknown id must ERROR, not note: reconcile exits non-zero on it."""
    monkeypatch.setattr(dispatch, "ROOT", tmp_path)
    (tmp_path / "state").mkdir(parents=True)
    led = Ledger(tmp_path / "state" / "ledger.db")
    rid = led.create("train", {}, provider="mock")
    led.update(rid, state="running", instance_id="pod-x")
    watched = []

    def fake_watch(run_id, *a, **k):
        watched.append(run_id)
    monkeypatch.setattr(dispatch, "_watch", fake_watch)
    monkeypatch.setattr(dispatch, "get_provider", lambda p: object())

    res = dispatch.resume(run_id=rid[-8:])
    assert res["run_id"] == rid and watched == [rid]

    res2 = dispatch.resume(run_id="zzzz9999")
    assert "no OPEN run matching" in (res2.get("error") or ""), res2
    assert not watched or len(watched) == 1       # nothing new attached


def test_watch_gone_pod_fails_fast_not_max_runtime(tmp_path, monkeypatch):
    # f831b7: killed pod, definitive absence, yet the watch looped
    # calling it a "blip" until max_runtime (~an hour of zombie watch,
    # row pretending to be alive). Definite-not-found on two
    # consecutive cycles must run the death path: grace -> manifest
    # recheck -> run lost.
    import time as _t
    led = Ledger(tmp_path / "ledger.db")
    rid = led.create("train", {}, provider="runpod")
    led.update(rid, state="running", instance_id="pod-gone")
    monkeypatch.setattr(dispatch.time, "sleep", lambda s: None)

    class GoneProv:
        calls = 0
        def poll(self, iid):
            GoneProv.calls += 1
            raise ProviderError("pod pod-gone not found (404)")

    result = {"run_id": rid, "kind": "train"}
    with pytest.raises(ProviderError, match="run lost"):
        dispatch._watch(rid, lambda k, **kw: None, result, led, 1,
                        started=_t.time(), max_runtime=3600,
                        prov=GoneProv(), inst_id="pod-gone",
                        price_hr=1.0,
                        put_run=lambda k, d, ct="": None)
    assert GoneProv.calls >= 2          # convergence, not single-shot


def test_watch_keeps_blipped_pod_alive_for_reap(tmp_path, monkeypatch):
    # the other half of the law: a genuine NET blip is NOT death — the
    # pod may be alive and paying (2026-11-24 leak lesson: blipped
    # watchers must not bury live pods). The loop must actually RUN a
    # blip cycle, then exit at the deadline: fake clock advances so the
    # first iteration executes with the blip prov.
    import time as _t
    led = Ledger(tmp_path / "ledger.db")
    rid = led.create("train", {}, provider="runpod")
    led.update(rid, state="running", instance_id="pod-blip")
    monkeypatch.setattr(dispatch.time, "sleep", lambda s: None)
    t0 = _t.time()
    calls = {"n": 0}

    def fake_time():
        calls["n"] += 1
        # iterations 1-2 inside the deadline (blip exercised), then past
        return t0 if calls["n"] <= 2 else t0 + 9999
    monkeypatch.setattr(dispatch.time, "time", fake_time)

    class BlipProv:
        n = 0
        def poll(self, iid):
            BlipProv.n += 1
            raise ProviderError("Network is unreachable")

    result = {"run_id": rid, "kind": "train"}
    with pytest.raises(ProviderError) as ei:
        dispatch._watch(rid, lambda k, **kw: None, result, led, 1,
                        started=t0, max_runtime=60,
                        prov=BlipProv(), inst_id="pod-blip",
                        price_hr=1.0, put_run=lambda k, d, ct="": None)
    assert "max runtime" in str(ei.value)   # NOT "run lost"
    assert BlipProv.n >= 1                  # the blip WAS exercised
