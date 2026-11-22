"""Provider interface shared by GPU.ai, RunPod, and future backends.

Every method takes plain data and returns JSON-serializable dicts so the
CLI can print them and job files can store them. Methods that talk to a
cloud must raise :class:`GpuError` (never raw CalledProcessError) with an
actionable message.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Provider(ABC):
    """Abstract GPU compute provider."""

    name = "base"

    # -- provision / creation -------------------------------------------
    @abstractmethod
    def provision(self, app: str, name: str, ssh_key_id: Optional[str] = None,
                  **kwargs) -> Dict[str, Any]:
        """Create (or reuse) an instance. Returns ``{"instance": {...}}``."""

    # -- inspection / monitoring -----------------------------------------
    @abstractmethod
    def list_instances(self) -> List[Dict[str, Any]]:
        """All instances visible to this account."""

    @abstractmethod
    def status(self, instance_id: str) -> Dict[str, Any]:
        """Single instance record (must include ``id`` and ``status``)."""

    def wait_ssh(self, instance_id: str, max_wait: int = 1800,
                 poll: int = 15) -> Dict[str, str]:
        """Block until SSH metadata is available.

        Returns ``{"user": ..., "host": ..., "port": ..., "ssh_command": ...}``.
        Providers may override; the default polls :meth:`status`.
        """
        import time

        from .common import parse_ssh_connection

        elapsed = 0
        last: Dict[str, Any] = {}
        while True:
            last = self.status(instance_id)
            conn = parse_ssh_connection(last)
            if conn["host"] and conn["port"] and conn["user"]:
                return conn
            if elapsed >= max_wait:
                raise self._timeout(instance_id, max_wait, last)
            time.sleep(poll)
            elapsed += poll

    def _timeout(self, instance_id, max_wait, last):
        from .common import GpuError

        return GpuError(
            f"instance {instance_id} exposed no SSH metadata within {max_wait}s "
            f"(last status: {last.get('status', 'unknown')})"
        )

    # -- software installer -----------------------------------------------
    @abstractmethod
    def run_remote(self, instance_id: str, script: str,
                   env: Optional[Dict[str, str]] = None,
                   timeout: Optional[int] = None) -> Dict[str, Any]:
        """Run a shell script on the instance. Returns ``{"rc": int, ...}``.

        ``timeout`` (seconds, None = no limit) must be honored: a wedged
        SSH session raises GpuError instead of hanging forever."""

    # -- data upload / download --------------------------------------------
    @abstractmethod
    def upload(self, instance_id: str, src: str, dst: str) -> Dict[str, Any]:
        """Copy a local path to the instance."""

    @abstractmethod
    def download(self, instance_id: str, remote: str, local: str,
                 exclude: Optional[List[str]] = None) -> Dict[str, Any]:
        """Copy a remote path locally."""

    # -- job scheduling / monitoring ----------------------------------------
    @abstractmethod
    def run_job(self, instance_id: str, command: str,
                env: Optional[Dict[str, str]] = None,
                timeout: int = 86400) -> Dict[str, Any]:
        """Run a long command, stream it, return ``{"rc": int}``.

        ``timeout`` is passed through to the remote execution (see
        :meth:`run_remote`); expiry raises GpuError."""

    # -- termination ----------------------------------------------------------
    @abstractmethod
    def terminate(self, instance_id: str, wait_timeout: str = "5m") -> Dict[str, Any]:
        """Delete the instance. Returns ``{"id": ..., "deleted": bool}``."""

    # -- keys -------------------------------------------------------------------
    def keys_create(self, name: str, public_key_file: str) -> Dict[str, Any]:
        raise NotImplementedError(f"{self.name} does not implement keys_create")

    def keys_list(self) -> List[Dict[str, Any]]:
        raise NotImplementedError(f"{self.name} does not implement keys_list")
