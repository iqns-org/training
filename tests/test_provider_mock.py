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


def test_mini_submit_estimates_smoke_not_train(tmp_path, monkeypatch):
    """THE ESTIMATE MUST MATCH THE WORK PROFILE (f44f27, 2026-11-27):
    mini was gate-estimated as a 60min train ($2.25) so a $0.50-balance
    account refused a ~$0.25 validation at the gate. The 'smoke' profile
    (10min -> $0.38 at the ceiling) was dead weight — selected by
    nothing. Same law as cb059d, opposite direction: the estimate is a
    claim about the WORK, and a wrong claim either kills runs at the cap
    or starves the lane at the gate."""
    import os
    import subprocess
    from orchestrator import dispatch
    monkeypatch.setattr(dispatch, "ROOT", tmp_path)
    monkeypatch.setattr(dispatch, "corpora_dir", lambda: tmp_path)
    (tmp_path / "tmp").mkdir()
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets/queer.jsonl").write_text(
        "\n".join(json.dumps({"messages": [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": f"a{i}"}]}) for i in range(60)))
    rec = tmp_path / "rec.yaml"
    rec.write_text(yaml.safe_dump({
        "base_model": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "datasets": [{"path": "./datasets/queer.jsonl", "type": "chat_template"}],
        "sequence_len": 1024, "num_epochs": 1}))
    sha = subprocess.run(["sha256sum", str(tmp_path / "datasets/queer.jsonl")],
                         capture_output=True, text=True).stdout.split()[0][:16]
    monkeypatch.setattr(dispatch.storage, "host_put",
                        lambda k, d: (tmp_path / "lane" / k).write_bytes(d)
                        or (tmp_path / "lane" / k).exists())
    monkeypatch.setattr(dispatch.storage, "key_path",
                        lambda k: str(tmp_path / "lane" / k))
    monkeypatch.setattr(dispatch.storage, "host_get",
                        lambda k: (tmp_path / "lane" / k).read_bytes())

    def fake_run(cmd, **kw):
        m = re.search(r"cat > (\S+)", " ".join(cmd))
        if m:
            src = cmd[0][-1] if isinstance(cmd[0], list) else cmd[-1]
            Path(m.group(1)).write_bytes(src if isinstance(src, bytes)
                                         else f"sha256 {sha}\n".encode())
        return NS(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)

    res = dispatch.submit("queer", rec, mini=True, provider="mock")
    assert not res.get("error"), res
    # smoke profile: 1.50/hr ceiling x 10min x 1.5 margin = $0.38
    assert res["estimate"] == 0.38, res["estimate"]

    # and a FULL train of the same recipe is still estimated as train
    res2 = dispatch.submit("queer", rec, provider="mock")
    assert not res2.get("error"), res2
    assert res2["estimate"] == 2.25, res2["estimate"]
