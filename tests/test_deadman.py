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
