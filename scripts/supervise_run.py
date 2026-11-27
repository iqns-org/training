#!/usr/bin/env python3
"""Orchestrator-side keep-alive + fail-safe exit for volume-lane runs.

The pod-side deadman (agent/trainer_agent/deadman.py) assumes container
exit == billing stop. RunPod v2's runtime RESTARTS the container
(no restartPolicy exists in the CreatePodRequest schema — verified
against the live OpenAPI), so a silently-dead watcher turns the
fail-safe into a zombie loop: attempt from zero -> stale ping ->
fire -> restart -> ... billing forever. r2 (20261126-221343-d30d5b)
died this way five times and drained the account to $0.21.

This script is the failsafe for the failsafe. Run it alongside every
volume-lane submit (same shell env, e.g. setsid nohup):

  * feeds the deadman ping every --poll seconds (un-fires the watchdog)
  * on a FRESH manifest (finished_at after --since) -> `qtrain
    reconcile` harvests the row and terminates the pod honestly
  * at --hard-stop-after seconds -> terminate, no matter what state

Usage:
  scripts/supervise_run.py RUN_ID POD_ID --since EPOCH \
      [--poll 30] [--hard-stop-after 9000]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("pod_id")
    ap.add_argument("--since", type=float, required=True,
                    help="epoch of the attempt we supervise; a manifest "
                         "finished_at newer than this means real success")
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--hard-stop-after", type=int, default=9000,
                    help="seconds from NOW (not from --since) before the "
                         "pod is terminated unconditionally")
    a = ap.parse_args()

    from orchestrator.config import load_dotenv
    load_dotenv()
    from orchestrator import storage

    log = lambda *x: print(f"[{time.strftime('%H:%M:%S')}]", *x, flush=True)
    hard_deadline = time.time() + a.hard_stop_after

    while time.time() < hard_deadline:
        try:
            storage.vol_put(f"runs/{a.run_id}/ping",
                            f"{time.time():.6f}".encode())
        except Exception as e:                                  # noqa: BLE001
            log("ping fail:", str(e)[:90])
        try:
            raw = storage.vol_get(f"runs/{a.run_id}/manifest.json")
            man = json.loads(raw) if raw else {}
        except Exception as e:                                  # noqa: BLE001
            log("manifest read fail:", str(e)[:90])
            man = {}
        if man.get("finished_at", 0) > a.since:
            log("fresh manifest (status:", man.get("status"),
                ") — harvesting via reconcile")
            subprocess.run([sys.executable, "-m", "orchestrator.cli",
                            "reconcile", a.run_id, "--poll", "10"])
            return 0
        time.sleep(a.poll)

    log("HARD STOP — terminating pod to stop billing")
    subprocess.run([sys.executable, "-m", "orchestrator.cli",
                    "terminate", a.pod_id])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
