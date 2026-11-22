from orchestrator.models import ProviderError
from orchestrator.providers.mock import MockProvider

import pytest


def test_mock_flow():
    p = MockProvider()
    inst = p.launch("job1", "img", "NVIDIA RTX A5000")
    assert inst.status == "creating"
    assert p.poll(inst.id).status == "creating"
    running = p.poll(inst.id)
    assert running.status == "running" and running.base_url

    assert len(p.catalog(["a5000"])) == 1
    assert all("4090" not in o.gpu_id for o in p.catalog(["a5000"]))

    assert p.terminate(inst.id) is True
    with pytest.raises(ProviderError):
        p.poll(inst.id)
