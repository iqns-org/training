import time

import pytest
from fastapi.testclient import TestClient

from trainer_agent import app as app_mod
from trainer_agent import runner as runner_mod


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "WORKDIR", tmp_path)
    monkeypatch.setenv("AGENT_TOKEN", "sekrit")
    app_mod._active = None
    return TestClient(app_mod._app)


def _headers():
    return {"Authorization": "Bearer sekrit"}


def test_healthz_open_api_gated(client):
    assert client.get("/healthz").status_code == 200
    assert client.get("/v1/info").status_code == 401
    assert client.get("/v1/info", headers=_headers()).status_code == 200


def test_job_lifecycle(client):
    r = client.post("/v1/jobs", headers=_headers(),
                    json={"run_id": "r1", "command": "sh -c 'echo hi; exit 0'"})
    assert r.status_code == 200

    # second job while first may still run → 409 (or finishes fast; then state ok)
    deadline = time.time() + 10
    state = {}
    while time.time() < deadline:
        state = client.get("/v1/jobs/r1", headers=_headers()).json()
        if state.get("state") in {"succeeded", "failed"}:
            break
        time.sleep(0.1)
    assert state.get("state") == "succeeded", state
    logs = client.get("/v1/jobs/r1/logs", headers=_headers()).json()["logs"]
    assert "hi" in logs


def test_conflict_after_launch(client):
    client.post("/v1/jobs", headers=_headers(),
                json={"run_id": "a", "command": "sh -c 'sleep 5'"})
    r = client.post("/v1/jobs", headers=_headers(),
                    json={"run_id": "b", "command": "echo nope"})
    assert r.status_code == 409
    c = client.post("/v1/jobs/a/cancel", headers=_headers())
    assert c.json()["cancelling"] is True


def test_unknown_job_404(client):
    assert client.get("/v1/jobs/nope", headers=_headers()).status_code == 404
