"""Deadman watchdog (SPEC 13an): ping flow, knob wiring, stamp truth."""
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import pytest

from orchestrator import dispatch

ROOT = Path(__file__).resolve().parents[1]


def _led(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch, "ROOT", tmp_path)
    led = dispatch.Ledger(tmp_path / "state" / "ledger.db")
    return led


# ------------------------------------------------------------------ ping
def test_watch_writes_ping(tmp_path, monkeypatch):
    """Every _watch cycle rewrites runs/<id>/ping BEFORE checking the
    manifest — the pod compares CONTENT (never clocks), so tokens must
    change while the watcher lives."""
    led = _led(tmp_path, monkeypatch)
    rid = led.create("train", {"domain": "queer"}, provider="runpod")
    tar = b"fake-tar"
    manifest = {"status": "succeeded", "artifacts": {},
                "artifacts_tar_sha256": hashlib.sha256(tar).hexdigest()}
    puts = {}

    def fake_get(key):
        if key.endswith("manifest.json"):
            return json.dumps(manifest).encode()
        if key.endswith("artifacts.tar.gz"):
            return tar
        return None

    def fake_put(key, data, ct=""):
        puts[key] = data
        return key

    result = {"instance": {"id": "i1", "gpu": "g", "price_hr": 0.3}}
    dispatch._watch(rid, fake_get, result, led, poll=1,
                    started=time.time(), max_runtime=30, put_run=fake_put)
    assert rid and f"runs/{rid}/ping" in puts
    # token must be a changing string, not mtime reuse
    assert puts[f"runs/{rid}/ping"].decode()


def test_watch_ping_blip_never_kills_the_loop(tmp_path, monkeypatch):
    """A raising ping write (lane blip) must NOT crash the watch — the
    pod's grace window is the safety, and the next successful cycle
    re-arms it. Retry IS the protocol."""
    led = _led(tmp_path, monkeypatch)
    rid = led.create("train", {"domain": "queer"}, provider="runpod")
    tar = b"fake-tar"
    manifest = {"status": "succeeded", "artifacts": {},
                "artifacts_tar_sha256": hashlib.sha256(tar).hexdigest()}
    calls = {"n": 0}

    def fake_get(key):
        calls["n"] += 1
        if calls["n"] < 3:
            if key.endswith("manifest.json"):
                return None
        if key.endswith("manifest.json"):
            return json.dumps(manifest).encode()
        if key.endswith("artifacts.tar.gz"):
            return tar
        return None

    def angry_put(key, data, ct=""):
        raise OSError("volume write failed")

    result = {"instance": {"id": "i1", "gpu": "g", "price_hr": 0.3}}
    out = dispatch._watch(rid, fake_get, result, led, poll=1,
                          started=time.time(), max_runtime=30,
                          put_run=angry_put)
    assert out["status"] == "succeeded"


# ------------------------------------------------------------------ knob
def test_deadman_knob_flows_to_pod_env():
    """All three pod-create sites (train/stage/relaunch) carry the grace
    explicitly so pod forensics always sees the value in effect."""
    src = (ROOT / "orchestrator" / "dispatch.py").read_text()
    assert src.count('"DEADMAN_GRACE_S"') == 3
    assert src.count("QTRAIN_DEADMAN_GRACE_S") >= 3


# ------------------------------------------------------------------ logic
def _dm(tmp_path):
    from trainer_agent import deadman as mod
    return mod


def test_step_protocol(tmp_path):
    dm = _dm(tmp_path)
    ping = tmp_path / "ping"
    w = dm.Deadman(ping, grace=100, boot_grace=30, interval=15)
    # absent file: never armed, never fires (boot handshake)
    assert w.step(now=0) is False
    assert w.step(now=10 ** 9) is False        # un-armed == cannot fire
    # first READ arms the handshake but NOT the kill clock (r2 law: a
    # resurrected container finds the previous life's stale ping).
    ping.write_bytes(b"t1")
    assert w.step(now=1000) is False           # handshake read
    assert w.step(now=1029) is False           # boot clock, not yet
    assert w.step(now=1030) is True            # BOOT grace: never changed
    # a live CHANGE switches to the normal grace countdown
    w = dm.Deadman(ping, grace=100, boot_grace=30, interval=15)
    ping.write_bytes(b"")                      # reset for fresh watch
    w.step(now=2000)                           # handshake
    ping.write_bytes(b"t2")                    # watcher alive: CHANGE
    assert w.step(now=2010) is False
    assert w.step(now=2109) is False           # rides the GRACE clock
    assert w.step(now=2110) is True            # 100s after the change
    # file vanishing mid-flight (lane hiccup) must NOT fake liveness nor
    # kill a healthy run: last_change stays, decision rides grace only
    ping.unlink()
    assert w.step(now=2115) is True            # still past last change
    # but a fresh ping (content change) resets fully
    ping.write_bytes(b"t3")
    assert w.step(now=2120) is False


def test_boot_grace_env(tmp_path, monkeypatch):
    dm = _dm(tmp_path)
    monkeypatch.delenv("DEADMAN_BOOT_GRACE_S", raising=False)
    assert dm.boot_grace_from_env() == 180.0
    monkeypatch.setenv("DEADMAN_BOOT_GRACE_S", "60")
    assert dm.boot_grace_from_env() == 60.0
    monkeypatch.setenv("DEADMAN_BOOT_GRACE_S", "junk")
    assert dm.boot_grace_from_env() == 180.0   # garbage never disables


def test_grace_env_default(tmp_path, monkeypatch):
    dm = _dm(tmp_path)
    monkeypatch.delenv("DEADMAN_GRACE_S", raising=False)
    assert dm.grace_from_env() == 600.0
    monkeypatch.setenv("DEADMAN_GRACE_S", "120")
    assert dm.grace_from_env() == 120.0
    monkeypatch.setenv("DEADMAN_GRACE_S", "not-a-number")
    assert dm.grace_from_env() == 600.0        # garbage never disables


# ------------------------------------------------------------------ stamp
def test_deadman_stamp_survives_completed_gate(tmp_path):
    """A fire between checkpoint saves that happened to find complete
    artifacts is STILL a deadman — a truncated round can never masquerade
    as a clean one."""
    from trainer_agent import finalize as fin
    m = fin.build_manifest("R9", tmp_path,
                           {"state": "cancelled", "deadman": True,
                            "step": 3}, "succeeded")
    assert m["training"]["deadman"] is True
    assert m["training"]["state"] == "cancelled"
    m2 = fin.build_manifest("R9", tmp_path, {"state": "succeeded"},
                            "succeeded")
    assert "deadman" not in m2["training"]


# ------------------------------------------------------------ pod-kill
class _Resp:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------- arming
def test_arm_deadman_wires_the_watchdog(monkeypatch, tmp_path):
    """REGRESSION (2026-11-27): the r2 pod-stop refactor swallowed the
    `Deadman(...).start(fire)` line into _pod_stop_best_effort's body —
    unreachable, so the watchdog NEVER ARMED on any pod while every test
    kept passing (they proved the class, never the wiring). Law this pins:
    _arm_deadman on a volume-lane run MUST construct+start a Deadman on
    the ping file; presign-lane runs stay deliberately UNARMED (no mount,
    SPEC 13an documented hole — flip only with a lane change)."""
    from trainer_agent import app as app_mod
    from trainer_agent import deadman as dm_mod

    armed = []

    class SpyDeadman:
        def __init__(self, path, *, log=None, **kw):
            armed.append(("ctor", str(path)))

        def start(self, on_fire):
            armed.append(("start", on_fire))

        def stop(self):
            armed.append(("stop", None))

    class FakeRunner:
        def _write_state(self, **f):
            armed.append(("state", f))

        def cancel(self):
            armed.append(("cancel", None))

    monkeypatch.setattr(dm_mod, "Deadman", SpyDeadman)

    # volume lane: must arm on <mount>/<prefix>/ping
    app_mod._arm_deadman(FakeRunner(),
                         {"volume": {"mount": str(tmp_path),
                                     "prefix": "runs/R1"}})
    assert ("ctor", str(tmp_path / "runs/R1/ping")) in armed
    assert any(k == "start" for k, _ in armed)

    # presign lane: deliberately unarmed — no mount to watch
    armed.clear()
    app_mod._arm_deadman(FakeRunner(), {"volume": None, "urls": {}})
    assert armed == []


def test_pod_kill_env_contract(monkeypatch):
    """r2 law: fire must ASK THE PLATFORM to stop the pod before exit —
    provider restarts exited containers, so bare os._exit = zombie loop.
    Env contract: QTRAIN_POD_STOP_URL (+ METHOD/TOKEN optional);
    absent env = old behavior (no regression), failures swallowed."""
    import urllib.request
    from trainer_agent import app as app_mod

    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append((req.get_method(), req.full_url,
                      req.headers.get("Authorization")))
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    monkeypatch.delenv("QTRAIN_POD_STOP_URL", raising=False)
    assert app_mod._pod_stop_best_effort() is False     # absent = no call
    assert calls == []

    monkeypatch.setenv("QTRAIN_POD_STOP_URL",
                       "https://api.runpod.io/v2/pods/abc/stop")
    monkeypatch.setenv("QTRAIN_POD_STOP_TOKEN", "tok")
    assert app_mod._pod_stop_best_effort() is True
    m, u, auth = calls[-1]
    assert m == "POST" and u.endswith("/stop") and auth == "Bearer tok"

    def angry(req, timeout=None):
        raise OSError("net down")

    monkeypatch.setattr(urllib.request, "urlopen", angry)
    assert app_mod._pod_stop_best_effort() is False      # best-effort




def test_pod_stop_token_single_source(monkeypatch):
    # mint (dispatch) and check (route) are LITERALLY one function:
    # drift silently disarms the deadman kill — the exact r2 failure
    from mcp_trainer import server
    from orchestrator import dispatch
    assert server.pod_stop_token is dispatch.pod_stop_token
    monkeypatch.delenv("MCP_TRAINER_TOKEN", raising=False)
    assert dispatch.pod_stop_token("r1") == ""     # no secret, no token
    monkeypatch.setenv("MCP_TRAINER_TOKEN", "sekret")
    assert len(dispatch.pod_stop_token("r1")) == 32
    assert dispatch.pod_stop_token("r1") != dispatch.pod_stop_token("r2")


def test_pod_stop_env_gate(monkeypatch):
    # INERT by default: without BOTH base URL and secret, dispatch
    # injects nothing — launches behave exactly as pre-wiring
    from orchestrator import dispatch
    monkeypatch.delenv("QTRAIN_POD_STOP_URL_BASE", raising=False)
    monkeypatch.setenv("MCP_TRAINER_TOKEN", "sekret")
    assert dispatch._pod_stop_env("r1") == {}
    monkeypatch.delenv("MCP_TRAINER_TOKEN", raising=False)
    monkeypatch.setenv("QTRAIN_POD_STOP_URL_BASE", "https://x.test/mcp")
    assert dispatch._pod_stop_env("r1") == {}
    monkeypatch.setenv("MCP_TRAINER_TOKEN", "sekret")
    env = dispatch._pod_stop_env("20261127-150613-d2886a")
    assert env["QTRAIN_POD_STOP_URL"] == \
        "https://x.test/mcp/pod-stop/20261127-150613-d2886a"
    assert env["QTRAIN_POD_STOP_TOKEN"] == dispatch.pod_stop_token(
        "20261127-150613-d2886a")


def test_pod_stop_route_contract(tmp_path, monkeypatch):
    # handler contract (direct invocation — the ASGI client adds nothing
    # the handler owns): wrong/absent bearer 401; short id 400; good
    # bearer kills ONLY the pod whose name ENDS run_id[-8:] (14b tail —
    # prefix and mid-name decoys stay alive); terminal row WITH a live
    # pod is STILL killed (that is the leak case the wire exists for);
    # no live pod -> 200 empty (idempotent); no secret -> 401 fail-closed
    import asyncio
    import json
    from orchestrator.ledger import Ledger
    from orchestrator.models import Instance
    from mcp_trainer import server
    from orchestrator import dispatch

    monkeypatch.setenv("MCP_TRAINER_TOKEN", "sekret")
    monkeypatch.setattr(server, "ROOT", tmp_path)
    led = Ledger(tmp_path / "state" / "ledger.db")
    rid = led.create("train", {"domain": "queer"}, provider="mock")
    led.update(rid, state="provisioning", instance_id="pod-abc")
    good = dispatch.pod_stop_token(rid)

    class Req:
        def __init__(self, auth="", run=None):
            self.path_params = {"run_id": run or rid}
            self.headers = {"authorization": auth}

    assert asyncio.run(server._pod_stop(
        Req("Bearer nope"))).status_code == 401
    assert asyncio.run(server._pod_stop(Req(""))).status_code == 401
    assert asyncio.run(server._pod_stop(
        Req(f"Bearer {good}", run="short"))).status_code == 400

    calls = []

    class FakeProv:
        def list_instances(self):
            return [
                Instance(provider="mock", id="pod-abc", status="running",
                         name=f"train-hint-{rid[-8:]}", gpu_type="",
                         price_hr=0.1, raw={}),
                Instance(provider="mock", id="pod-decoy1", status="running",
                         name=f"trainer-{rid[:8]}-z", gpu_type="",
                         price_hr=0.1, raw={}),
                Instance(provider="mock", id="pod-decoy2", status="running",
                         name=f"x-{rid[-8:]}-more", gpu_type="",
                         price_hr=0.1, raw={})]

        def terminate(self, iid):
            calls.append(iid)
            return True

    monkeypatch.setattr(server, "get_provider", lambda p: FakeProv())
    r = asyncio.run(server._pod_stop(Req(f"Bearer {good}")))
    assert r.status_code == 200
    assert calls == ["pod-abc"] and json.loads(r.body)["killed"] == ["pod-abc"]
    # leak case: row harvested (done) but pod still live -> kill anyway
    led.update(rid, state="done", artifact_uri="runs/x/artifacts.tar.gz")
    calls.clear()
    r = asyncio.run(server._pod_stop(Req(f"Bearer {good}")))
    assert r.status_code == 200 and calls == ["pod-abc"]
    # idempotent: nothing live -> 200, empty, no error
    class Empty(FakeProv):
        def list_instances(self):
            return []
    monkeypatch.setattr(server, "get_provider", lambda p: Empty())
    r = asyncio.run(server._pod_stop(Req(f"Bearer {good}")))
    assert r.status_code == 200 and json.loads(r.body)["killed"] == []
    # fail-closed: secret removed -> even a once-valid token 401s
    monkeypatch.delenv("MCP_TRAINER_TOKEN")
    assert asyncio.run(server._pod_stop(
        Req(f"Bearer {good}"))).status_code == 401


def test_gate_pod_stop_prefix_only(tmp_path, monkeypatch):
    # BearerGate exempts /pod-stop/* (the handler is the authorizer);
    # every other path still demands the global bearer — a prefix,
    # never a hole
    import asyncio
    from mcp_trainer.auth import BearerGate

    reached = []

    async def inner(scope, receive, send):
        reached.append(scope["path"])
        await send({"type": "http.response.start", "status": 200,
                    "headers": []})
        await send({"type": "http.response.body", "body": b"inner"})

    gate = BearerGate(inner, token="sekret")

    async def hit(path, auth=""):
        out = {}

        async def send(msg):
            if msg["type"] == "http.response.start":
                out["status"] = msg["status"]
        await gate({"type": "http", "path": path,
                    "headers": [(b"authorization", auth.encode())] if auth
                    else []}, None, send)
        return out.get("status")

    assert asyncio.run(hit("/pod-stop/whatever", "Bearer junk")) == 200
    assert reached[-1] == "/pod-stop/whatever"
    assert asyncio.run(hit("/mcp", "Bearer junk")) == 401
    assert asyncio.run(hit("/podstop-ish", "")) == 401   # NOT the prefix
    assert asyncio.run(hit("/healthz")) == 200
