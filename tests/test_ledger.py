import pytest

from orchestrator.ledger import Ledger


@pytest.fixture()
def led(tmp_path):
    return Ledger(tmp_path / "ledger.db")


def test_create_get_update(led):
    rid = led.create("train", {"dataset": "queer"}, provider="runpod")
    row = led.get(rid)
    assert row and row["state"] == "pending" and row["provider"] == "runpod"

    led.update(rid, state="provisioning", instance_id="abc123")
    row = led.get(rid)
    assert row["state"] == "provisioning" and row["instance_id"] == "abc123"

    events = led.db.execute(
        "SELECT state FROM events WHERE run_id=? ORDER BY id", (rid,)).fetchall()
    assert [e["state"] for e in events] == ["pending", "provisioning"]


def test_list_and_live_instances(led):
    r1 = led.create("train", {}, provider="runpod")
    r2 = led.create("train", {}, provider="runpod")
    led.update(r1, instance_id="pod-a")
    led.update(r2, instance_id="pod-b")
    led.update(r2, state="done")

    assert len(led.list()) == 2
    assert len(led.list(include_terminal=False)) == 1
    assert led.live_instances("runpod") == {"pod-a"}
