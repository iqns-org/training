"""Bearer gate + unauthenticated /healthz (chassis invariant: nothing
unauthenticated reaches the tools). Token from MCP_TRAINER_TOKEN; unset =
open mode (local dev only) with a loud startup warning.

Exception: the /pod-stop/ prefix passes the gate and authenticates
PER-RUN inside the route handler (deadman kill capability —
sha256(run_id|MCP_TRAINER_TOKEN) minted by dispatch; the pod must never
hold the global token). The handler is fail-closed: unset
MCP_TRAINER_TOKEN means every pod-stop call 401s, so the pass-through
cannot open a hole."""

from __future__ import annotations

import hmac
import os


class BearerGate:
    """Secure default: no token configured = deny everything (except
    /healthz). Open mode ONLY with MCP_TRAINER_ALLOW_OPEN=1 (local dev)."""

    def __init__(self, app, token: str | None = None):
        self.app = app
        self.token = token if token is not None else os.environ.get(
            "MCP_TRAINER_TOKEN", "")
        self.open = not self.token and os.environ.get(
            "MCP_TRAINER_ALLOW_OPEN") == "1"

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        if path == "/healthz":
            return await self._reply(send, 200, b'{"ok":true}')
        if path.startswith("/pod-stop/"):
            # Per-run auth lives in the route handler (fail-closed): the
            # pod carries sha256(run_id|secret)[:32], never the global.
            return await self.app(scope, receive, send)
        if self.token:
            hdrs = dict(scope.get("headers") or [])
            got = hdrs.get(b"authorization", b"").decode()
            want = f"Bearer {self.token}"
            if not hmac.compare_digest(got, want):
                return await self._reply(send, 401, b'{"error":"unauthorized"}')
        elif not self.open:
            return await self._reply(
                send, 503, b'{"error":"auth not configured '
                           b'(MCP_TRAINER_TOKEN missing; refuse-all mode)"}')
        return await self.app(scope, receive, send)

    async def _reply(self, send, code: int, body: bytes) -> None:
        await send({"type": "http.response.start", "status": code,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})
