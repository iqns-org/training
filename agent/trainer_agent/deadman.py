"""Deadman watchdog (SPEC 13an): pod-side cost guard against a dead watcher.

The failure mode this kills: the orchestrator (or its local watcher) dies
while training pays. The row stays OPEN by design — that is the
idempotent-reattach half — but nothing bounded the BILLING, because the
enforcer of max_runtime/max_price was the dead process itself. The
watcher-killed-by-`timeout 590` incident (dd0ae0/29ce66 era) is exactly
this hole: the run survived only because a human reattached, not because
the design protected the wallet.

Contract:
  * Orchestrator (_watch): every poll cycle rewrites `runs/<id>/ping`
    with a fresh token (content = timestamp; change = liveness).
  * Pod (here): reads that file beside its own status.json mirror.
    Liveness = CONTENT CHANGE, never wall clocks — host/pod clock skew
    and NFS writer-clock mtimes are exactly the class of silent lie this
    refuses to be built on.
  * Arms only after the FIRST successful read (boot handshake), but the
    KILL CLOCK starts only on the first content CHANGE (r2 law): a
    resurrected container finds the previous life's stale ping, so a
    never-changing first read fires after DEADMAN_BOOT_GRACE_S (default
    180s), not after the full grace. An orchestrator that dies pre-boot
    costs the estimate (recipe-capped), not a false fire mid-round.
  * Grace (DEADMAN_GRACE_S env, default 600s) of UNCHANGED content ->
    fire: SIGTERM the trainer group, stamp state.json `deadman: true`,
    let the runner's own completion path finalize (gate -> pack ->
    manifest LAST, exactly like a normal terminal), wait for the manifest
    to land on the lane, then POD-KILL (QTRAIN_POD_STOP_URL env —
    the provider restarts exited containers, so bare os._exit = zombie
    billing loop) and exit the agent.
  * Resume needs no new machinery: the row was left OPEN; the next boot
    `resume` reattaches (ping flows again -> watchdog un-fires on fresh
    content), or the pod is already gone and reconcile finds the manifest
    on the lane and closes the row honestly with partial artifacts.
  * Presign-lane pods (no mount) are NOT armed in v1 — the fleet default
    is the volume lane, and a second channel for a legacy lane is
    half-tested code waiting to fire wrong.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

DEFAULT_GRACE_S = 600.0
DEFAULT_BOOT_GRACE_S = 180.0      # r2 law: a resurrected container reads
CHECK_INTERVAL_S = 15.0           # the PREVIOUS life's stale ping — never
                                  # trust a first READ, only a CHANGE

def grace_from_env(default: float = DEFAULT_GRACE_S) -> float:
    try:
        v = os.environ.get("DEADMAN_GRACE_S", "")
        return float(v) if v else default
    except ValueError:
        return default


def boot_grace_from_env(default: float = DEFAULT_BOOT_GRACE_S) -> float:
    try:
        v = os.environ.get("DEADMAN_BOOT_GRACE_S", "")
        return float(v) if v else default
    except ValueError:
        return default


def should_fire(last_change: float | None, now: float, grace: float) -> bool:
    """Pure decision: None == never armed (no handshake yet)."""
    return last_change is not None and (now - last_change) >= grace


class Deadman:
    """Watchdog thread over the lane ping file. `path` is the dir that
    mirrors `runs/<id>/` on the pod mount (same dir as status.json).

    `on_fire` runs in the watchdog thread and must: stop training, let
    finalize publish, then terminate the process (os._exit). It is
    guarded so a raising handler never leaves the pod billing silently —
    but also never kills the pod before a save attempt."""

    def __init__(self, path: Path, *, grace: float | None = None,
                 boot_grace: float | None = None,
                 interval: float = CHECK_INTERVAL_S,
                 log=lambda *a: None):
        self.path = Path(path)
        self.grace = grace_from_env() if grace is None else float(grace)
        self.boot_grace = (boot_grace_from_env() if boot_grace is None
                           else float(boot_grace))
        self.interval = float(interval)
        self.log = log
        self.first_seen: float | None = None     # handshake time
        self.last_change: float | None = None    # our clock, last CHANGE
        self._last_content: bytes | None = None
        self.fired = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def step(self, now: float) -> bool:
        """One watchdog beat; return True when the applicable grace is
        exhausted. BOOT LAW (r2): the first READ does NOT start the kill
        clock — a resurrected container finds the previous life's stale
        ping, and trusting a read turned the fail-safe into a 600s-per-
        attempt zombie loop. Only a content CHANGE starts the grace
        countdown; a first-seen ping that never changes fires after the
        shorter boot_grace. A MISSING file after handshake stays NO NEWS
        (silence is death in both directions)."""
        try:
            content = self.path.read_bytes()
        except OSError:
            content = None
        if content is not None and content != self._last_content:
            if self.first_seen is None:            # handshake read
                self.first_seen = now
                self._last_content = content
                self.log(f"deadman armed (ping seen; boot "
                         f"{self.boot_grace:.0f}s -> grace "
                         f"{self.grace:.0f}s)")
                return False                       # read alone: NO clock
            self.last_change = now                 # CHANGE: the real clock
            self._last_content = content
        if self.first_seen is None:
            return False                           # never armed, never fires
        if self.last_change is None:
            return (now - self.first_seen) >= self.boot_grace
        return should_fire(self.last_change, now, self.grace)

    def start(self, on_fire):
        def loop():
            while not self._stop.wait(self.interval):
                if self.step(time.time()):
                    if self.fired:
                        continue
                    self.fired = True
                    self.log("DEADMAN: orchestrator ping unchanged for "
                             f"{self.grace:.0f}s — saving artifacts and "
                             "exiting the pod")
                    try:
                        on_fire()
                    except Exception as e:                      # noqa: BLE001
                        self.log(f"deadman fire handler failed: {e!r}")
                    return
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
