"""orchestrator/budget.py — normalized budgeting, three gates (SPEC 13q)."""
import json

import pytest
from orchestrator import budget


class Offer:
    """Stand-in for models.GpuOffer (duck-typed: fits() reads the fields)."""
    def __init__(self, gpu_id, price_hr, vram_gb, secure=True):
        self.gpu_id, self.price_hr = gpu_id, price_hr
        self.vram_gb, self.secure = vram_gb, secure


def test_estimate_is_rate_x_mins_x_margin():
    # awq est 25 min at $0.60/hr, margin 1.5 -> 0.6*25/60*1.5 = 0.375
    assert budget.estimate(0.60, "awq") == pytest.approx(0.375)
    # unknown kind falls back to 30-min estimate
    assert budget.estimate(1.00, "mystery") == pytest.approx(0.75)


def test_margin_and_totals_are_env_overridable():
    import importlib
    import os
    os.environ["QTRAIN_BUDGET_MARGIN"] = "2.0"
    os.environ["QTRAIN_BUDGET"] = "9.0"
    os.environ["QTRAIN_BUDGET_OBS"] = "/nonexistent/obs.json"  # pin static
    importlib.reload(budget)
    try:
        assert budget.MARGIN == 2.0
        assert budget.TOTAL == 9.0
        assert budget.estimate(0.60, "awq") == pytest.approx(0.60 * 25 / 60 * 2.0)
    finally:
        del os.environ["QTRAIN_BUDGET_MARGIN"]
        del os.environ["QTRAIN_BUDGET"]
        del os.environ["QTRAIN_BUDGET_OBS"]
        importlib.reload(budget)


def test_fits_prefers_cheap_secure_vram_and_respects_ceiling():
    pool = [Offer("B200", 7.99, 180), Offer("A5000", 0.69, 24),
            Offer("RTX2000", 0.24, 6), Offer("A100", 2.09, 40)]
    ok, best = budget.fits(pool, min_vram=10, ceiling_hr=0.80)
    # only A5000 is both >=10GB and <=$0.80
    assert [o.gpu_id for o in ok] == ["A5000"]
    # best = cheapest vram-fit REGARDLESS of ceiling (RTX2000 has 6GB,
    # out of pool) — this is the price quoted in the "raise --max-price"
    # error so the operator sees what waiting/overriding would buy
    assert best == 0.69


def test_fits_sorted_ascending_by_price():
    pool = [Offer("big", 0.70, 40), Offer("small", 0.30, 12)]
    ok, _ = budget.fits(pool, 10, 0.80)
    assert [o.gpu_id for o in ok] == ["small", "big"]


def test_fits_empty_when_nothing_within_ceiling():
    pool = [Offer("B200", 7.99, 180)]
    ok, best = budget.fits(pool, 10, 0.80)
    assert ok == [] and best == 7.99    # surfaced to caller as a wait/raise


def test_aggregate_gate_blocks_over_budget():
    # empty opens: this run's est alone is far under the default $5
    assert budget.aggregate_gate([], 0.375) is None
    # exact sum of open ests + this one crosses the cap
    why = budget.aggregate_gate([0.375] * 20, 0.375)
    assert why and "aggregate budget" in why


def test_balance_gate_keeps_reserve():
    assert budget.balance_gate(9.00, 0.375) is None   # plenty above reserve
    assert budget.balance_gate(None, 0.375) is None   # provider won't say
    why = budget.balance_gate(1.50, 0.375)            # < RESERVE(2) + est
    assert why and "balance" in why


def test_observed_medians_replace_guesses(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, "OBS_FILE", tmp_path / "obs.json")
    assert budget.est_minutes("awq") == 25          # static until data
    budget.record_observed("awq", 9.0)
    assert budget.est_minutes("awq") == 25          # needs >=2 samples
    budget.record_observed("awq", 21.0)
    assert budget.est_minutes("awq") == 15.0        # median of the tail
    # train is deliberately excluded: recipe-driven, polluted by minis
    budget.record_observed("train", 3.0)
    assert budget.est_minutes("train") == 60
    # outliers (boot ghosts, >12h hangs) never enter the tail
    budget.record_observed("awq", 0.01)
    budget.record_observed("awq", 5000)
    assert budget.est_minutes("awq") == 15.0


def test_preflight_reports_without_launching(tmp_path, monkeypatch):
    from orchestrator import dispatch
    from orchestrator.models import Offer

    def offers():
        return [Offer(provider="runpod", gpu_id="RTX B200", vram_gb=180,
                      price_hr=7.99, secure=True),
                Offer(provider="runpod", gpu_id="A5000", vram_gb=24,
                      price_hr=0.69, secure=True),
                Offer(provider="runpod", gpu_id="RTX2000", vram_gb=6,
                      price_hr=0.24, secure=True)]

    class Stub:
        def catalog(self, gpus=None, tier=None):
            return offers()
        def balance(self):
            return 9.0

    monkeypatch.setattr(dispatch, "get_provider", lambda name: Stub())
    monkeypatch.setattr(dispatch.storage, "vol_cfg", lambda: None)
    monkeypatch.setattr(dispatch, "ROOT", tmp_path)
    out = dispatch.preflight("awq")
    assert out["candidates"] == [{"gpu": "A5000", "price_hr": 0.69,
                                  "vram_gb": 24}]     # B200 ceiling-culled
    assert out["estimate"] == round(0.69 * 25 / 60 * budget.MARGIN, 2)
    assert out["decision"].startswith("launch A5000")
    assert out["balance"] == 9.0
    # read-only: the (fresh) ledger holds nothing — no run was registered
    from orchestrator.ledger import Ledger
    assert Ledger(tmp_path / "state" / "ledger.db").list() == []


def test_preflight_cpu_kinds_free_the_floor(tmp_path, monkeypatch):
    """The floor was EXCLUDING cheap cards for CPU-only stages: onnx/int8
    declare min_vram 0, so the 6GB RTX2000 is now their best (cheapest)
    fit — an assumption-audit finding becoming a candidate list."""
    from orchestrator import dispatch
    from orchestrator.models import Offer

    class Stub:
        def catalog(self, gpus=None, tier=None):
            return [Offer(provider="runpod", gpu_id="A5000", vram_gb=24,
                          price_hr=0.69, secure=True),
                    Offer(provider="runpod", gpu_id="RTX2000", vram_gb=6,
                          price_hr=0.24, secure=True)]

        def balance(self):
            return 9.0

    monkeypatch.setattr(dispatch, "get_provider", lambda n: Stub())
    monkeypatch.setattr(dispatch.storage, "vol_cfg", lambda: None)
    monkeypatch.setattr(dispatch, "ROOT", tmp_path)
    out = dispatch.preflight("onnx")
    assert out["candidates"][0]["gpu"] == "RTX2000"   # was floor-blocked
    assert out["assumed"] == []                        # onnx DECLARES now


def test_preflight_kill_units_match_launch(tmp_path, monkeypatch):
    """425e6c UNITS LAW: max_runtime is SECONDS. Preflight's quoted kill
    threshold must equal the launch path's price*runtime/3600 — the /60
    variant (60x) is exactly how 425e6c's $33-for-$0.50 bug was born."""
    from orchestrator import dispatch
    from orchestrator.models import Offer

    class Stub:
        def catalog(self, gpus=None, tier=None):
            return [Offer(provider="runpod", gpu_id="A5000", vram_gb=24,
                          price_hr=0.50, secure=True)]

        def balance(self):
            return 50.0

    monkeypatch.setattr(dispatch, "get_provider", lambda n: Stub())
    monkeypatch.setattr(dispatch.storage, "vol_cfg", lambda: None)
    monkeypatch.setattr(dispatch, "ROOT", tmp_path)
    out = dispatch.preflight("train", max_runtime=3600)
    # 1 hour on a $0.50/hr card = $0.50 — NOT $30 (the /60 lie)
    assert out["kill_threshold"] == 0.5
    assert out["assumed"] and "QTRAIN_MAX_PRICE_HR" in out["assumed"][0]


def test_est_basis_names_its_guess(tmp_path, monkeypatch):
    """An estimate must say whether it is measurement or folklore:
    observed-median / static-prior / FALLBACK-30 for unknown kinds."""
    monkeypatch.setattr(budget, "OBS_FILE", tmp_path / "obs.json")
    assert budget.est_basis("train")["basis"] == "static-prior"
    assert budget.est_basis("awq")["basis"] == "static-prior"
    budget.record_observed("awq", 9.0)
    budget.record_observed("awq", 21.0)   # >=2 trusted samples graduate it
    assert budget.est_basis("awq")["basis"] == "observed-median"
    fb = budget.est_basis("no-such-kind-xyz")
    assert fb["basis"].startswith("FALLBACK-30")
    assert "declare" in fb["basis"]      # tells you the fix


def test_calibration_contamination_auto_retires(tmp_path, monkeypatch):
    # THE LAW (cb059d as code): when a kind's work profile changes
    # (EST_VERSION bump), its old obs samples retire from estimation —
    # kept in the file as evidence, never trusted again. No folklore
    # "remember to purge" step survives contact with this.
    monkeypatch.setattr(budget, "OBS_FILE", tmp_path / "obs.json")
    budget.record_observed("merge", 11.0)
    budget.record_observed("merge", 12.0)      # no-op-era durations
    assert budget.est_minutes("merge") == 11.5
    monkeypatch.setattr(budget, "EST_VERSION", budget.EST_VERSION + 1)
    assert budget.est_minutes("merge") == budget.EST_MINUTES["merge"] == 45
    raw = json.loads((tmp_path / "obs.json").read_text())
    assert len(raw["merge"]) == 2              # evidence preserved, trust gone
    budget.record_observed("merge", 50.0)      # fresh profile's samples
    # ONE trusted sample is not a calibration yet (median needs >=2):
    assert budget.est_minutes("merge") == 45   # static still answers
    budget.record_observed("merge", 46.0)
    assert budget.est_minutes("merge") == 48.0  # trusted median takes over


def test_legacy_bare_float_samples_are_untrusted(tmp_path, monkeypatch):
    # pre-stamp file format: unknown work profile = untrusted by law
    obs = tmp_path / "obs.json"
    obs.write_text(json.dumps({"awq": [8.0, 9.0, 10.0]}))
    monkeypatch.setattr(budget, "OBS_FILE", obs)
    assert budget.obs_minutes("awq") == []
    assert budget.est_minutes("awq") == budget.EST_MINUTES["awq"]
