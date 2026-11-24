import asyncio

from mcp_trainer import server


def _tool_names():
    return {t.name for t in asyncio.run(server.mcp.list_tools())}


def test_tools_registered():
    expected = {"gpu_catalog", "list_recipes", "submit_training",
                "job_status", "job_logs", "list_runs", "list_pods",
                "terminate_pod", "doctor", "refresh_corpora", "reconcile",
                "submit_stage"}
    assert expected <= _tool_names()


def test_list_recipes_finds_repo(tmp_path, monkeypatch):
    # SPEC §4b: recipes live in corpora; CORPORA_DIR drives resolution
    monkeypatch.setenv("CORPORA_DIR", str(tmp_path))
    monkeypatch.setattr(server, "ROOT", tmp_path)
    r = (tmp_path / "recipes" / "queer")
    r.mkdir(parents=True)
    (r / "SmolLM2-360M-qlora.yaml").write_text("base_model: x\n")
    got = {t.name: t for t in asyncio.run(server.mcp.list_tools())}[
        "list_recipes"].fn()
    assert got["recipes"] == [{"domain": "queer",
                               "model": "SmolLM2-360M-qlora",
                               "recipe": "queer/SmolLM2-360M-qlora.yaml"}]


def test_bearer_gate(monkeypatch):
    from starlette.testclient import TestClient

    from mcp_trainer.auth import BearerGate

    async def ok_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200,
                    "headers": []})
        await send({"type": "http.response.body", "body": b"inner"})

    gate = BearerGate(ok_app, token="sekret")
    client = TestClient(gate)
    assert client.get("/healthz").status_code == 200        # open always
    assert client.get("/mcp").status_code == 401            # no auth
    assert client.get("/mcp", headers={
        "Authorization": "Bearer nope"}).status_code == 401
    r = client.get("/mcp", headers={"Authorization": "Bearer sekret"})
    assert r.status_code == 200 and r.text == "inner"


def test_gate_open_mode_needs_explicit_flag(monkeypatch):
    from starlette.testclient import TestClient

    from mcp_trainer.auth import BearerGate

    async def ok_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200,
                    "headers": []})
        await send({"type": "http.response.body", "body": b"inner"})

    monkeypatch.delenv("MCP_TRAINER_TOKEN", raising=False)
    monkeypatch.delenv("MCP_TRAINER_ALLOW_OPEN", raising=False)
    client = TestClient(BearerGate(ok_app, token=""))
    r = client.get("/mcp")                      # refuse-all default
    assert r.status_code == 503
    assert client.get("/healthz").status_code == 200
    monkeypatch.setenv("MCP_TRAINER_ALLOW_OPEN", "1")
    open_client = TestClient(BearerGate(ok_app, token=""))  # explicit opt-in
    assert open_client.get("/mcp").status_code == 200
