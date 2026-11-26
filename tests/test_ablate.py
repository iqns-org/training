"""R15 ablate tooling: direction math, hard projection, soft steering-LoRA
spec — pure python, exercised end-to-end with synthetic activations."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ab():
    spec = importlib.util.spec_from_file_location("ablate",
                                                  ROOT / "postproc" / "ablate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mean_direction_and_norm(ab):
    pos = [[1.0, 0.0, 0.0], [1.0, 0.1, 0.0]]
    neg = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    v = ab.mean_direction(pos, neg)
    assert abs(ab.norm(v) - 1.0) < 1e-9
    assert v[0] > 0.9


def test_pca_direction_aborts_on_identical(ab):
    same = [[1.0, 2.0], [3.0, 4.0]]
    with pytest.raises(ValueError):
        ab.mean_direction(same, same)          # zero direction


def test_projection_removes_component(ab):
    v = [1.0, 0.0]
    W = [[3.0, 2.0], [-1.0, 5.0]]
    out = ab.project_matrix(W, v)
    assert abs(out[0][0]) < 1e-9 and abs(out[1][0]) < 1e-9   # v-comp gone
    assert out[0][1] == 2.0 and out[1][1] == 5.0             # rest intact


def test_residual_stays_invariant(ab):
    v = [0.6, 0.8]
    W = [[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]]
    out = ab.project_matrix(W, v)
    for row in out:
        assert abs(ab.dot(row, v)) < 1e-9      # orthogonal to v


def test_steering_lora_spec(ab):
    spec = ab.lora_from_direction([2.0, 0.0], strength=0.5)
    assert spec["rank"] == 1
    eff = spec["B"][0][0] * spec["A"][0][0]     # B@A == strength * v[0]
    assert abs(eff - 0.5 * (2.0 / 2.0 * 2.0)) < 1e-6 or abs(eff - 0.5) < 1e-6


def test_cli_end_to_end(tmp_path, ab):
    pairs = {"layer0": {
        "pos": [[2.0, 0.0], [2.0, 0.2]],
        "neg": [[0.0, 0.0], [0.0, 0.0]]}}
    pf = tmp_path / "pairs.json"
    pf.write_text(json.dumps(pairs))
    out = tmp_path / "v.json"
    assert ab.main(["direction", "--pairs", str(pf), "--out", str(out)]) == 0
    v = json.loads(out.read_text())["layer0"]
    assert v[0] > 0.99 and abs(ab.norm(v) - 1.0) < 1e-9   # mean-pos [2,.1]
    Wf = tmp_path / "W.json"
    Wf.write_text(json.dumps([[1.0, 1.0], [1.0, 1.0]]))
    out2 = tmp_path / "Wp.json"
    assert ab.main(["project", "--weights", str(Wf), "--direction", str(out),
                    "--layer", "layer0", "--out", str(out2)]) == 0
    assert all(abs(sum(r[i] * v[i] for i in range(2))) < 1e-9
               for r in json.loads(out2.read_text()))
    out3 = tmp_path / "steer.json"
    assert ab.main(["spec", "--direction", str(out), "--layer", "layer0",
                    "--model", "kandi", "--out", str(out3)]) == 0
    spec = json.loads(out3.read_text())
    assert spec["mode"] == "soft" and spec["layer0"]["rank"] == 1
