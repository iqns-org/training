import os

import pytest

from orchestrator.config import load_dotenv
from orchestrator.models import ProviderError
from orchestrator.providers.runpod import RunpodProvider, normalize_pod


def test_normalize_pod_running():
    pod = {
        "id": "abc123", "name": "trainer-run-1", "status": "RUNNING",
        "gpu": {"id": "NVIDIA RTX A5000", "memory": 24, "price": 0.16},
        "ports": [{"type": "http", "ip": "1.2.3.4", "port": 40001}],
    }
    inst = normalize_pod(pod)
    assert (inst.id, inst.status, inst.ready) == ("abc123", "running", True)
    assert inst.base_url == "http://1.2.3.4:40001"
    assert inst.gpu_type == "NVIDIA RTX A5000"
    assert inst.price_hr == 0.16


def test_normalize_pod_unknown_status_and_no_ports():
    inst = normalize_pod({"id": "x", "status": "WEIRD_STATE"})
    assert inst.status == "weird_state"
    assert inst.base_url == "" and not inst.ready


def test_normalize_ports_spec_echo_strings():
    # live RunPod v2 shape: ports is a list of strings, no mapping info
    inst = normalize_pod({"id": "w", "status": "RUNNING",
                          "ports": ["8000/http", "22/tcp"], "cost": 0.13})
    assert inst.status == "running"
    assert inst.base_url == ""
    assert inst.price_hr == 0.13


def test_normalize_ports_dict_mapping():
    inst = normalize_pod({"id": "w", "status": "RUNNING",
                          "ports": {"8000/http": {"ip": "1.2.3.4", "port": 42}}})
    assert inst.base_url == "http://1.2.3.4:42"


def test_launch_mounts_and_dc(monkeypatch):
    captured = {}

    def fake_req(self, method, path, **kw):
        captured[method + path] = kw.get("json")
        return {"id": "p1", "status": "RUNNING"}

    monkeypatch.setattr(RunpodProvider, "_req", fake_req)
    p = RunpodProvider(api_key="x")
    inst = p.launch("n", "img", "NVIDIA A40",
                    env={"JOB_PAYLOAD_PATH": "/data/runs/r/payload.json"},
                    volume_id="vol1", volume_path="/data",
                    data_center="EU-RO-1")
    body = captured["POST/v2/pods"]
    assert body["mounts"] == {"network": [{"volumeId": "vol1", "path": "/data"}]}
    assert body["dataCenterIds"] == ["EU-RO-1"]
    assert inst.id == "p1"
    p.launch("n", "img", "NVIDIA A40")
    assert "mounts" not in captured["POST/v2/pods"]
    assert "dataCenterIds" not in captured["POST/v2/pods"]


@pytest.mark.skipif(not os.environ.get("RUNPOD_LIVE_TESTS"),
                    reason="set RUNPOD_LIVE_TESTS=1 to hit the live API")
def test_live_catalog_and_pods():
    load_dotenv()
    p = RunpodProvider()
    offers = p.catalog(["RTX A5000", "A40"])
    assert offers and offers == sorted(offers, key=lambda o: o.price_hr)
    assert all(o.provider == "runpod" for o in offers)
    assert isinstance(p.list_instances(), list)


def test_poll_definite_absence_not_ghost():
    """Provider contract the watcher consumes (f831b7): RunPod's
    terminate() already documents '404 anywhere = gone' — poll() must
    say the same thing DEFINITELY. A deleted pod answers as a raise
    ('not found'), never as the empty-status ghost that a watch loop
    must keep re-checking forever. NET blips stay blips."""
    # (ProviderError + RunpodProvider already imported at module top;
    # the class is RunpodProvider — lowercase 'pod' — caught by the
    # test failing against disk, not memory)

    class Stub(RunpodProvider):
        def __init__(self, mode):
            self.mode = mode

        def _req(self, method, path, body=None, **kw):
            if self.mode == "none":             # 204/empty body -> None
                return None
            if self.mode == "empty":            # 200 with no payload
                return {}
            if self.mode == "found":            # RunPod v2 GET /v2/pods/{id}
                return {"id": "x", "status": "RUNNING"}   # BARE pod object
            raise ProviderError(self.mode)      # _req RAISES on HTTP>=400

    assert Stub("found").poll("x").status == "running"   # normalize_pod lowercases
    for mode, sigil in (("none", "not found"),
                        ("empty", "not found"),
                        ("HTTP 404: gone", "not found")):
        with pytest.raises(ProviderError, match="not found"):
            Stub(mode).poll("x")
    with pytest.raises(ProviderError, match="Temporary failure"):
        Stub("Temporary failure in name resolution").poll("x")
