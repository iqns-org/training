"""Normalized budgeting (SPEC §13q) — one vocabulary, three gates.

Hard vs soft, stated honestly:
  HARD floor   provider prepaid balance — you cannot spend past what the
               account holds (RunPod refuses). Checked pre-launch with a
               reserve, best-effort (scoped keys may not expose it).
  SOFT kill    existing _watch threshold: wallclock x rate crosses
               max_cost -> terminate + failed. Reactive backstop.
  PREDICTIVE   THE normalized piece: every job class knows its estimated
               minutes; estimate = rate x est_minutes x MARGIN. The kill
               threshold defaults to the estimate (never a magic number),
               and the aggregate gate refuses to start a job whose
               estimate would put all concurrent open estimates over
               QTRAIN_BUDGET.

Right-sizing (also here, not in the ladder): the RATE CEILING is the
statement "a 360M job must not burn a $7.99/hr B200 even when it's the
only thing in stock — waiting is cheaper than overbuying". Margin of
error = MARGIN (default 1.5x estimate), nothing more.
"""

from __future__ import annotations

import os

MARGIN = float(os.environ.get("QTRAIN_BUDGET_MARGIN", "1.5"))
TOTAL = float(os.environ.get("QTRAIN_BUDGET", "5.0"))       # concurrent est.
RESERVE = float(os.environ.get("QTRAIN_RESERVE", "2.0"))    # balance floor

EST_MINUTES = {"merge": 12, "onnx": 20, "int8": 8, "awq": 25,
               "train": 60, "smoke": 10}

# states that can actually BURN money (ledger row exists, pod may be up).
# 'verified' is deliberately absent: it marks DRY runs — staged, never
# launched, $0 — which must not haunt the aggregate gate.
BILLED = ("pending", "provisioning", "running", "uploading")


def est_minutes(kind: str) -> int:
    return EST_MINUTES.get(kind, 30)


def estimate(price_hr: float, kind: str) -> float:
    """Predicted all-in dollars with bounded margin."""
    return price_hr * est_minutes(kind) / 60 * MARGIN


def fits(offers, min_vram: int, ceiling_hr: float):
    """Right GPU = cheapest offer that FITS: vram floor + rate ceiling.
    Returns (fit_offers, best_price_seen) — fit sorted by price."""
    ok = sorted((o for o in offers
                 if o.vram_gb >= min_vram and o.price_hr <= ceiling_hr),
                key=lambda o: o.price_hr)
    pool = [o for o in offers if o.vram_gb >= min_vram]
    best = min((o.price_hr for o in pool), default=None)
    return ok, best


def aggregate_gate(open_count: int, est: float, total: float | None = None) -> str | None:
    """None = allowed; else the refusal reason."""
    cap = total if total is not None else TOTAL
    projected = est * (open_count + 1)
    if projected > cap:
        return (f"aggregate budget: {open_count} open run(s) x est "
                f"${est:.2f} -> projected ${projected:.2f} > QTRAIN_BUDGET "
                f"${cap:.2f}; raise the env or finish/wait out open runs")
    return None


def balance_gate(balance: float | None, est: float) -> str | None:
    if balance is None:
        return None                        # provider won't tell us; prepaid
    if balance - RESERVE < est:            # hard floor is still the account
        return (f"balance ${balance:.2f} minus reserve ${RESERVE:.2f} < "
                f"estimate ${est:.2f}")
    return None
