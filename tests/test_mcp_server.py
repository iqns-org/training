import asyncio
import json

import yaml

from mcp_trainer import server


def _tool_names():
    return {t.name for t in asyncio.run(server.mcp.list_tools())}


def test_tools_registered():
    expected = {"gpu_catalog", "list_recipes", "submit_training",
                "job_status", "job_logs", "list_runs", "list_pods",
                "terminate_pod", "doctor", "refresh_corpora", "reconcile",
                "submit_stage", "synth_derive", "synth_abstain",
                "synth_graph", "synth_audit", "synth_manifest",
                "synth_bundle", "synth_status", "synth_sample",
                "synth_mix", "synth_grade", "synth_gaps", "synth_gbnf",
                "synth_merge", "lane_hygiene"}
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


TTL = ('@PREFIX my: <http://example.org/entity/> .\n'
       '@PREFIX ai: <http://example.org/property/> .\n'
       'my:site1 ai:inspectedBy my:inspector7 .\n'
       'my:inspector7 ai:yearsExp 12 .\n'
       'my:site1 ai:hasStatus my:Active .\n')


def test_laws_surface_and_pair_parity(tmp_path, monkeypatch):
    """MCP must CONTAIN the lore, not just link it: the laws resource is
    readable, cites the spine, and the pair-mode signature + docstring
    laws are pinned — drift here once let a deadman sit unarmed while
    every test stayed green (r2 lesson generalized by 86d5e9a)."""
    resources = {str(r.uri): r for r in asyncio.run(server.mcp.list_resources())}
    laws_uri = next((u for u in resources if u.endswith("laws")), None)
    assert laws_uri is not None, "trainer://laws missing from the surface"
    text = asyncio.run(server.mcp.read_resource(laws_uri))
    if isinstance(text, str):
        body = text
    elif hasattr(text, "contents"):       # fastmcp ResourceResult
        body = "".join(getattr(c, "content", "") or getattr(c, "text", "")
                       for c in text.contents)
    else:
        items = list(text)
        body = getattr(items[0], "text", "") if items else ""
    for law in ("DEADMAN", "HOLDOUT LAW", "micro_batch_size", "tools",
                "CO-TENANT", "whitelist", "scrub", "QUANT & SERVING"):
        assert law in body, f"law text missing: {law}"
    # pair mode on the surface: parameter + its docstring laws
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    st = tools["submit_training"]
    import inspect
    sig = inspect.signature(st.fn)
    assert "pair" in sig.parameters
    doc = st.description or ""
    assert "CO-TENANT" in doc and "micro_batch_size 1" in doc
    assert "FFT" in doc  # the don't-pair-heavy-legs rule travels too

    # plumbing: pair resolves and reaches dispatch.submit as a Path
    seen = {}

    def fake_submit(domain, recipe, **kw):
        seen.update(kw, domain=domain, recipe=recipe)
        kw["on_run_id"]("test-pair-run")
        return {"run_id": "test-pair-run"}
    monkeypatch.setattr(server.dispatch, "submit", fake_submit)
    ra = tmp_path / "a.yaml"
    ra.write_text("base_model: X/Y\n")
    rb = tmp_path / "b.yaml"
    rb.write_text("base_model: X/Z\n")
    out = st.fn(domain="opencode", recipe=str(ra), pair=str(rb), mini=True)
    assert out["run_id"] == "test-pair-run"
    assert seen["pair"] == rb


def test_mcp_reports_and_guard_codes(tmp_path):
    """recipe_report is the guidance surface: resolved knobs, family,
    refusal codes + bypass names — an agent gets the lore WITHOUT being
    locked into it (priors arrive labeled as priors)."""
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    assert "recipe_report" in tools
    r = tmp_path / "q.yaml"
    r.write_text(yaml.safe_dump({"base_model": "Qwen/Qwen3.5-0.8B",
                                 "adapter": "qlora", "lora_r": 32,
                                 "datasets": [{"path": "x"}]}))
    rep = tools["recipe_report"].fn(recipe=str(r),
                                    overrides={"micro_batch_size": 2})
    assert rep["family"] == "qwen3.5"
    # both burned classes fire: batch AND the missing target list
    assert rep["bypassable"] == ["fla_batch", "vlm_lora"]
    assert rep["resolved"]["lora_r"] == 32
    # the guide EXPLICITLY labels r=32 a prior, not a law
    assert "PRIOR" in rep["knob_guide"]["lora_r"]
    assert "burned" in rep["knob_guide"]["micro_batch_size"].lower()
    # a recipe with nothing wrong: no refusals, empty bypass list
    ok = tmp_path / "ok.yaml"
    ok.write_text(yaml.safe_dump({"base_model": "Qwen/Qwen3.5-0.8B",
                                  "adapter": "lora", "micro_batch_size": 1,
                                  "lora_target_modules": ["q_proj"],
                                  "datasets": [{"path": "x"}]}))
    rep2 = tools["recipe_report"].fn(recipe=str(ok))
    assert rep2["refusals"] == []


def test_laws_text_labels_priors_as_priors():
    """The 2026-11-27 correction: 'r=32 by law' was FALSE prose (nothing
    enforced it) — the laws surface must now carry the guidance-vs-law
    distinction itself, keyword-pinned."""
    resources = {str(r.uri): r for r in asyncio.run(server.mcp.list_resources())}
    laws_uri = next(u for u in resources if u.endswith("laws"))
    text = asyncio.run(server.mcp.read_resource(laws_uri))
    body = "".join(getattr(c, "content", "") or getattr(c, "text", "")
                   for c in text.contents)
    assert "GUIDANCE vs LAW" in body
    assert "PRIORS, not ceilings" in body
    assert "Silent forcing is a bug" in body
    # and nothing claims r=32 is enforced
    assert "r=32 ALL-LAYER BY LAW" not in body.upper()


def test_pipeline_report_surface(tmp_path):
    """The stage-side audit must be READABLE from the surface: every GPU
    stage declares min_vram (no silent 10), CPU kinds declare 0, ests
    name their basis, and the CLI/MCP see the SAME object (one function
    object law)."""
    from orchestrator import dispatch
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    assert "pipeline_report" in tools
    rep = tools["pipeline_report"].fn()
    assert rep is dispatch.pipeline_report() or rep == dispatch.pipeline_report()
    st = rep["stages"]
    assert st["onnx"]["min_vram"] == 0 and st["onnx"]["min_vram_declared"]
    assert st["int8"]["min_vram"] == 0
    for gpu_kind in ("merge", "ablate", "holdout", "awq"):
        assert st[gpu_kind]["min_vram_declared"], gpu_kind
    assert st["awq"]["min_vram"] == 16 and st["holdout"]["min_vram"] == 12
    assert st["merge"]["est"]["basis"] in ("static-prior", "observed-median")
    assert rep["holdout"]["modulo"] == 50 and rep["holdout"]["law_version"] == 2
    assert "86d5e9a" in rep["deadman"]["law"]
    assert rep["assumption_doctrine"]
    # kind filter works
    only = tools["pipeline_report"].fn(kind="onnx")
    assert list(only["stages"]) == ["onnx"]


def test_laws_cover_pipeline_defaults():
    resources = {str(r.uri): r for r in asyncio.run(server.mcp.list_resources())}
    laws_uri = next(u for u in resources if u.endswith("laws"))
    text = asyncio.run(server.mcp.read_resource(laws_uri))
    body = "".join(getattr(c, "content", "") or getattr(c, "text", "")
                   for c in text.contents)
    for law in ("PIPELINE DEFAULTS", "min_vram", "SECONDS", "HOLDOUT modulo",
                "pipeline_report"):
        assert law in body, f"pipeline law missing: {law}"


def test_synth_tools_chain(tmp_path, monkeypatch):
    """graph → manifest → bundle entirely through the MCP surface."""
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    ttl = tmp_path / "register.ttl"
    ttl.write_text(TTL)
    out = tmp_path / "derived"

    stats = tools["synth_graph"].fn(ttl_path=str(ttl), domain="construction",
                                    out=str(out))
    assert stats["templates"] >= 1 and stats["preflight_fail"] == []

    man = tools["synth_manifest"].fn(out=str(out))
    assert man["corpus_id"].startswith("derived@")

    synth = server._synth()
    monkeypatch.setattr(synth, "DATA_ROOT", tmp_path)
    bun = tools["synth_bundle"].fn(domain="cognition", out=str(out))
    assert bun["rows"] > 0
    dest = tmp_path / "datasets" / "cognition"
    tags = {p.stem for p in dest.glob("*.jsonl")}
    assert {"graph-qa", "abstain"} <= tags


def test_synth_graph_rejects_missing_path():
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    r = tools["synth_graph"].fn(ttl_path="/nonexistent/x.ttl")
    assert "error" in r


def test_synth_derive_background_and_status(tmp_path):
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    r = tools["synth_derive"].fn(limit=2, per_code=1, background=True,
                                 out=str(tmp_path / "d"),
                                 tmp=str(tmp_path / "t"))
    assert r["state"] == "running" and r["job"]
    st = tools["synth_status"].fn(job="")
    assert r["job"] in st["jobs"]


def test_synth_closeout_tools(tmp_path):
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    r = tools["synth_gbnf"].fn(out=str(tmp_path / "ir.gbnf"))
    assert r["undefined_refs"] == [] and (tmp_path / "ir.gbnf").is_file()
    bench = tmp_path / "b.json"
    bench.write_text(json.dumps({"rows": [
        {"model": "m", "question": "ir-repair", "passed": False}]}))
    g = tools["synth_gaps"].fn(bench_path=str(bench))
    assert g["failing"] == 1 and g["gaps"][0]["commands"]
    mg = tools["synth_merge"].fn(base="Qwen/Qwen3.5-0.8B",
                                 adapters="/models/a,/models/b",
                                 method="dare_ties", density=0.3)
    assert mg["merge_method"] == "dare_ties" and len(mg["models"]) == 3


