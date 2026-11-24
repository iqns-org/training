"""SPEC 14 idempotency: reap / wave-launch backoff / relaunch / janitor."""
import hashlib
import json
import time

import pytest

from orchestrator import dispatch
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


def test_pod_name_is_human_and_contract_safe():
    # trainer- prefix + trailing run-suffix are the reap contract; the
    # middle is purely for eyes on provider consoles / kubectl
    n = dispatch._pod_name("train", "20261124-190341-b9b130",
                           dispatch._pod_hint("queer",
                                              "x/Qwen3.5-0.8B-qlora.yaml"))
    assert n.startswith(dispatch.POD_PREFIX) and n.endswith("1-b9b130")
    assert "queer" in n and "qwen" in n and "qlora" not in n
    assert len(n) <= 63   # k8s-safe
    s = dispatch._pod_name("awq", "20261124-200000-abc123",
                           dispatch._pod_hint(parent="20261123-162929-20e8db"))
    assert s.endswith("0-abc123") and "for20e8db" in s   # own tail, parent tag
    # plain fallback (old payloads without meta)
    assert dispatch._pod_name("merge", "r-x-99887766") == \
        dispatch.POD_PREFIX + "merge-99887766"


def test_reap_matches_hinted_and_legacy_names(led_root, monkeypatch):
    """name[-8:] suffix parsing kills BOTH new hint-bearing stale pods and
    legacy plain names — the contract token always sits at the tail."""
    monkeypatch.delenv("QTRAIN_REAP_UNCLAIMED", raising=False)
    led = Ledger(led_root / "state" / "ledger.db")
    rid = led.create("train", {}, provider="runpod")
    led.update(rid, state="failed")
    hinted = f"trainer-train.queer-qwen3508b-{rid[-8:]}"
    legacy = f"trainer-{rid[-8:]}"
    prov = StubProv([_inst("pod-a", hinted), _inst("pod-b", legacy)])
    monkeypatch.setattr(dispatch, "get_provider", lambda p: prov)
    out = dispatch.reap("runpod", grace=900)
    assert {k["id"] for k in out["killed"]} == {"pod-a", "pod-b"}
