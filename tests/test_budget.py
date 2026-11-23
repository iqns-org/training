"""orchestrator/budget.py — normalized budgeting, three gates (SPEC 13q)."""
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
    importlib.reload(budget)
    try:
        assert budget.MARGIN == 2.0
        assert budget.TOTAL == 9.0
        assert budget.estimate(0.60, "awq") == pytest.approx(0.60 * 25 / 60 * 2.0)
    finally:
        del os.environ["QTRAIN_BUDGET_MARGIN"]
        del os.environ["QTRAIN_BUDGET"]
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
    # 1 open run + this, est 0.375 each -> projected 0.75 well under default
    assert budget.aggregate_gate(1, 0.375) is None
    # many opens blow the default $5 concurrent-estimate cap
    why = budget.aggregate_gate(20, 0.375)
    assert why and "aggregate budget" in why


def test_balance_gate_keeps_reserve():
    assert budget.balance_gate(9.00, 0.375) is None   # plenty above reserve
    assert budget.balance_gate(None, 0.375) is None   # provider won't say
    why = budget.balance_gate(1.50, 0.375)            # < RESERVE(2) + est
    assert why and "balance" in why
