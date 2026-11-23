"""Job runner: one subprocess per pod, state file + log file on disk."""

from __future__ import annotations

import ast
import json
import os
import re
import shlex
import signal
import subprocess
import threading
import time
from pathlib import Path

WORKDIR = Path(os.environ.get("AGENT_WORKDIR", "/work"))

PROGRESS_RE = re.compile(
    r"(?:(?P<step>\d+)\s*/\s*(?P<total>\d+)).*?"
    r"(?:loss['\":= ]+(?P<loss>[\d.]+))?", re.I)
EPOCH_RE = re.compile(r"epoch['\":= ]+(?P<epoch>[\d.]+)", re.I)


class Runner:
    """Owns exactly one active job (one pod = one job in v1).
    `post_hook(state: dict)` runs after the terminal state is written —
    this is where finalization (gate/pack/PUT) lives."""

    def __init__(self, run_id: str, command: str, env: dict | None = None,
                 post_hook=None, outputs_dir: Path | None = None):
        self.run_id = run_id
        self.command = command
        self.env = env or {}
        self.post_hook = post_hook
        self.outputs_dir = outputs_dir or (WORKDIR / "outputs" / run_id)
        self.dir = WORKDIR / "runs" / run_id
        self.log_path = self.dir / "run.log"
        self.state_path = self.dir / "state.json"
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._heartbeat: threading.Thread | None = None

    # -- state -------------------------------------------------------------
    def _write_state(self, **fields):
        state = self.read_state()
        state.update(fields)
        state.setdefault("run_id", self.run_id)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=1))

    def read_state(self) -> dict:
        if self.state_path.is_file():
            try:
                return json.loads(self.state_path.read_text())
            except json.JSONDecodeError:
                pass
        return {}

    # -- lifecycle -----------------------------------------------------------
    def start(self):
        if self._proc is not None or (
                self.read_state().get("state") in {"running", "uploading"}):
            raise RuntimeError(f"job {self.run_id} already active")
        self.dir.mkdir(parents=True, exist_ok=True)
        self._write_state(state="running", started_at=time.time(),
                          command=self.command)
        env = {**os.environ, **self.env}
        self._proc = subprocess.Popen(
            shlex.split(self.command), cwd=self.dir.parent.parent, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True, text=True, bufsize=1)
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self):
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        with self.log_path.open("a") as log:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                self._scrape(line)
        rc = proc.wait()
        if rc == 0:
            self._write_state(state="succeeded", exit_code=0,
                              finished_at=time.time())
        elif self.read_state().get("state") == "cancelling":
            self._write_state(state="cancelled", exit_code=rc,
                              finished_at=time.time())
        else:
            self._write_state(state="failed", exit_code=rc,
                              finished_at=time.time())
        if self.post_hook is not None:
            try:
                self.post_hook(self.read_state())
            except Exception as e:                      # noqa: BLE001
                self._write_state(finalize_error=str(e)[:500])

    def _scrape(self, line: str):
        """Best-effort progress extraction from axolotl/tqdm output."""
        if "{" in line and "loss" in line:          # log dict line
            body = line[line.index("{"):line.rindex("}") + 1]
            try:
                payload = ast.literal_eval(body)    # repr-dict, single quotes
            except (ValueError, SyntaxError):
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    return
            if isinstance(payload, dict) and "loss" in payload:
                def num(v):
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return None
                self._write_state(
                    **{k: v for k, v in
                       {"step": num(payload.get("current_steps")),
                        "total": num(payload.get("total_steps")),
                        "loss": num(payload.get("loss")),
                        "epoch": num(payload.get("epoch")),
                        "lr": num(payload.get("learning_rate"))}.items()
                       if v is not None})
            return
        m = PROGRESS_RE.search(line)
        if m and m.group("step"):                   # tqdm bar: step/total even
            fields = {"step": int(m.group("step"))}  # without loss on the same
            if m.group("total"):                     # line (loss arrives on the
                fields["total"] = int(m.group("total"))
            if m.group("loss"):
                fields["loss"] = float(m.group("loss"))
            if e := EPOCH_RE.search(line):
                fields["epoch"] = float(e.group("epoch"))
            self._write_state(**fields)

    def logs(self, tail: int = 100) -> list[str]:
        if not self.log_path.is_file():
            return []
        lines = self.log_path.read_text(errors="replace").splitlines()
        return lines[-tail:]

    def cancel(self) -> bool:
        if self._proc is None or self._proc.poll() is not None:
            return False
        self._write_state(state="cancelling")
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        return True

    # -- heartbeat -----------------------------------------------------------
    def start_heartbeat(self, volume: dict | None = None,
                        urls: dict | None = None, interval: float = 30.0):
        """Mirror state.json onto the lane every `interval`s so the
        orchestrator can watch live progress with zero inbound connections.
        Best-effort: terminal manifest is the authoritative signal."""
        import shutil

        def publish():
            self._write_state(updated_at=time.time())          # refresh mtime
            if volume:
                dest_dir = Path(volume["mount"]) / volume["prefix"]
                dest_dir.mkdir(parents=True, exist_ok=True)
                tmp = dest_dir / "status.json.part"
                shutil.copyfile(self.state_path, tmp)
                os.replace(tmp, dest_dir / "status.json")
            elif (urls or {}).get("status_put"):
                from .transfer import upload
                upload(self.state_path, urls["status_put"])

        def loop():
            while True:
                try:
                    publish()
                except Exception:                               # noqa: BLE001
                    pass
                time.sleep(interval)

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        self._heartbeat = t

    @property
    def active(self) -> bool:
        return self._proc is not None and self._proc.poll() is None
