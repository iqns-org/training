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


@pytest.mark.skipif(not os.environ.get("RUNPOD_LIVE_TESTS"),
                    reason="set RUNPOD_LIVE_TESTS=1 to hit the live API")
def test_live_catalog_and_pods():
    load_dotenv()
    p = RunpodProvider()
    offers = p.catalog(["RTX A5000", "A40"])
    assert offers and offers == sorted(offers, key=lambda o: o.price_hr)
    assert all(o.provider == "runpod" for o in offers)
    assert isinstance(p.list_instances(), list)
