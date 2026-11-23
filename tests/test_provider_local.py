import json

import pytest

from orchestrator import storage
from orchestrator.models import ProviderError
from orchestrator.providers.local import LocalProvider


@pytest.fixture()
def prov(monkeypatch):
    p = LocalProvider(ssh_host="me@pi00.lan")
    calls = []

    def fake_kctl(remote_args, stdin=None, timeout=120):
        calls.append((remote_args, stdin))
        if remote_args[:2] == ["apply", "-f"]:
            return "job.batch/trainer-x created\n"
        if "job" in remote_args and "trainer-x" in remote_args:
            return json.dumps({"status": {}})
        if "pods" in remote_args:
            return "Running"
        return ""
    monkeypatch.setattr(p, "_kctl", fake_kctl)
    return p, calls


def test_launch_builds_job_manifest(prov):
    p, calls = prov
    inst = p.launch("trainer-x", "ghcr.io/iqns-org/postproc:p-v0.1.0",
                    "local-cpu-me@pi00.lan",
                    env={"JOB_PAYLOAD_PATH": "/data/runs/r/payload.json"},
                    args="echo hi")
    assert inst.status == "provisioning"
    _, stdin = calls[0]
    text = stdin.decode()
    assert "kind: Job" in text and "trainer-x" in text
    assert "value: /data/runs/r/payload.json" in text
    assert "sh" in text and "echo hi" in text
    assert "/srv/mcps/trainer" in text          # lane hostPath
    assert "nvidia.com/gpu" not in text         # cpu offer, no gpu block


def test_poll_maps_pod_phase(prov):
    p, _ = prov
    inst = p.poll("me@pi00.lan/trainer-x")
    assert inst.status == "running"


def test_poll_gone_is_terminated(monkeypatch):
    p = LocalProvider(ssh_host="me@pi00.lan")

    def boom(*a, **k):
        raise ProviderError("not found")
    monkeypatch.setattr(p, "_kctl", boom)
    assert p.poll("me@pi00.lan/nope").status == "terminated"


def test_catalog_cpu_only(monkeypatch):
    monkeypatch.delenv("LOCAL_GPU_OFFER", raising=False)
    p = LocalProvider(ssh_host="me@pi00.lan")
    offers = p.catalog()
    assert len(offers) == 1 and offers[0].price_hr == 0.0
    assert p.catalog(["nano"]) == []


def test_host_lane_direct_fs(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_LANE_DIR", str(tmp_path))
    storage.host_put("runs/r1/manifest.json", b"{}")
    assert (tmp_path / "runs/r1/manifest.json").read_bytes() == b"{}"
    assert not list((tmp_path / "runs/r1").glob("*.part"))
    assert storage.host_get("runs/r1/missing.json") is None
    assert storage.host_get("runs/r1/manifest.json") == b"{}"
