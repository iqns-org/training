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

import json
import os
import statistics
from pathlib import Path

from .config import ROOT

MARGIN = float(os.environ.get("QTRAIN_BUDGET_MARGIN", "1.5"))
TOTAL = float(os.environ.get("QTRAIN_BUDGET", "5.0"))       # concurrent est.
RESERVE = float(os.environ.get("QTRAIN_RESERVE", "2.0"))    # balance floor

# merge: 12 was calibrated against the NO-OP era (c7cde2 forensics: the
# peft merge moved zero bytes and finished in 9 min). A real merge
# downloads base, applies ~186 fp32 deltas, saves+uploads ~1.5GB: 45.
EST_MINUTES = {"merge": 45, "onnx": 20, "int8": 8, "awq": 25,
               "train": 60, "smoke": 10}

# states that can actually BURN money (ledger row exists, pod may be up).
# 'verified' is deliberately absent: it marks DRY runs — staged, never
# launched, $0 — which must not haunt the aggregate gate.
BILLED = ("pending", "provisioning", "running", "uploading")

# self-calibration: observed wall-clock per kind, kept as a rolling tail of
# the last samples; the median replaces the static EST_MINUTES guess once
# real data exists (observations from live runs beat folklore). Stage
# kinds only: train duration is a property of the user's recipe, not a
# job-class default, and must not be polluted by mini validations.
OBS_FILE = Path(os.environ.get("QTRAIN_BUDGET_OBS",
                               str(ROOT / "state" / "budget_obs.json")))
OBS_KINDS = ("merge", "onnx", "int8", "awq")


def _obs():
    try:
        return json.loads(OBS_FILE.read_text())
    except (OSError, ValueError):
        return {}


def record_observed(kind: str, minutes: float) -> None:
    if kind not in OBS_KINDS or not (0.2 <= minutes <= 720):
        return                                   # outliers: boot ghosts
    data = _obs()
    tail = (data.get(kind) or [])[-9:] + [round(minutes, 2)]
    data[kind] = tail
    try:
        OBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = OBS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, OBS_FILE)
    except OSError:
        pass


def est_minutes(kind: str):
    obs = _obs().get(kind) or []
    if len(obs) >= 2:
        return statistics.median(obs)
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


def aggregate_gate(open_ests, est: float, total: float | None = None) -> str | None:
    """None = allowed; else the refusal reason. open_ests = the stored
    est_cost of every already-BILLED run (exact sum, not a count proxy)."""
    cap = total if total is not None else TOTAL
    projected = sum(open_ests) + est
    if projected > cap:
        return (f"aggregate budget: {len(open_ests)} open run(s) at "
                f"${sum(open_ests):.2f} + this ${est:.2f} -> projected "
                f"${projected:.2f} > QTRAIN_BUDGET ${cap:.2f}; raise the "
                "env or wait out open runs")
    return None


def balance_gate(balance: float | None, est: float) -> str | None:
    if balance is None:
        return None                        # provider won't tell us; prepaid
    if balance - RESERVE < est:            # hard floor is still the account
        return (f"balance ${balance:.2f} minus reserve ${RESERVE:.2f} < "
                f"estimate ${est:.2f}")
    return None
