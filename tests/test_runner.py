import json
import time

import pytest

from trainer_agent import runner as runner_mod
from trainer_agent.runner import Runner


@pytest.fixture()
def work(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "WORKDIR", tmp_path)
    return tmp_path


def test_scrape_repr_dict_line(work):
    r = Runner("r1", "true")
    r._scrape("{'loss': '0.67', 'grad_norm': '0.9', "
              "'learning_rate': '0.0002', 'epoch': '1'}\n")
    st = r.read_state()
    assert st["loss"] == 0.67 and st["epoch"] == 1.0 and st["lr"] == 0.0002


def test_scrape_tqdm_without_loss(work):
    r = Runner("r2", "true")
    r._scrape("  25%|██        | 1423/5700 [1:12:03<3:36:09,  2.7it/s]")
    st = r.read_state()
    assert st["step"] == 1423 and st["total"] == 5700 and "loss" not in st


def test_heartbeat_writes_status_to_volume(work):
    r = Runner("r3", "true")
    r._write_state(state="running", step=10, total=100)
    vol = {"mount": str(work / "data"), "prefix": "runs/r3"}
    r.start_heartbeat(volume=vol, interval=0.05)
    status = None
    deadline = time.time() + 5
    while time.time() < deadline and status is None:
        p = work / "data/runs/r3/status.json"
        if p.is_file():
            status = json.loads(p.read_text())
        else:
            time.sleep(0.05)
    assert status and status["step"] == 10 and "updated_at" in status
    assert not list((work / "data/runs/r3").glob("*.part"))
