"""MCP client — talks to a Model Context Protocol server.

Supports three transports:

``stdio``
    Spawns a subprocess and speaks newline-delimited JSON-RPC over its
    stdin/stdout. This is the transport used by ``gpt6-mcp-gateway mcp``.
``http``
    Streamable HTTP: one POST per JSON-RPC message.
``sse``
    Server-Sent Events: POST to the message endpoint, read the reply stream.

The client is an async context manager::

    async with MCPClient(url="python -m my_server", transport="stdio") as client:
        tools = await client.list_tools()
        result = await client.call_tool("read_file", {"path": "README.md"})
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
from dataclasses import dataclass, field
from typing import Any

import httpx

from .logging_utils import log_debug, log_info, redact

SUPPORTED_TRANSPORTS = ("stdio", "http", "sse")

_JSONRPC_VERSION = "2.0"
_MCP_PROTOCOL_VERSION = "2024-11-05"


def split_command(command: str) -> list[str]:
    """Split a stdio command line into argv.

    ``shlex.split(posix=False)`` keeps the surrounding quotes on Windows, so
    ``"C:\\Python\\python.exe" -m server`` would try to execute a file literally
    named ``"C:\\Python\\python.exe"``. POSIX mode strips the quotes correctly
    for both platforms, but it also eats backslashes, so Windows paths are
    handled by a small dedicated scanner instead.
    """
    if os.name != "nt":
        return shlex.split(command, posix=True)

    argv: list[str] = []
    current: list[str] = []
    quote: str | None = None
    started = False

    for char in command:
        if quote:
            if char == quote:
                quote = None
            else:
                current.append(char)
            continue
        if char in ("'", '"'):
            quote = char
            started = True
            continue
        if char.isspace():
            if started:
                argv.append("".join(current))
                current = []
                started = False
            continue
        current.append(char)
        started = True

    if started:
        argv.append("".join(current))
    return argv


class MCPClientError(RuntimeError):
    """Raised for any MCP transport or protocol failure."""


@dataclass
class MCPTool:
    """A tool advertised by the MCP server."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> MCPTool:
        return cls(
            name=str(raw.get("name") or ""),
            description=str(raw.get("description") or ""),
            input_schema=raw.get("inputSchema") or raw.get("input_schema") or {},
        )


@dataclass
class MCPCallResult:
    """The outcome of one ``tools/call``."""

    tool: str
    content: str = ""
    is_error: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


def _flatten_content(blocks: Any) -> str:
    """Flatten an MCP content array into plain text.

    MCP returns ``[{"type": "text", "text": "..."}]``. Non-text blocks are
    rendered as a short placeholder so the model still sees that something
    was returned instead of silently losing it.
    """
    if blocks is None:
        return ""
    if isinstance(blocks, str):
        return blocks
    if isinstance(blocks, dict):
        blocks = [blocks]
    if not isinstance(blocks, list):
        return str(blocks)

    parts: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        kind = block.get("type")
        if kind == "text" or "text" in block:
            parts.append(str(block.get("text") or ""))
        elif kind == "resource":
            resource = block.get("resource") or {}
            parts.append(str(resource.get("text") or resource.get("uri") or ""))
        elif kind == "image":
            parts.append(f"[image {block.get('mimeType') or 'unknown'}]")
        else:
            parts.append(json.dumps(block, ensure_ascii=False))
    return "\n".join(p for p in parts if p)


class MCPClient:
    """Async MCP client for stdio, http and sse transports."""

    def __init__(
        self,
        *,
        url: str,
        transport: str = "stdio",
        auth_token: str = "",
        timeout: float = 60.0,
        env: dict[str, str] | None = None,
        transport_impl: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        normalised = (transport or "").strip().lower()
        if normalised not in SUPPORTED_TRANSPORTS:
            raise MCPClientError(
                f"Unsupported MCP transport {redact(normalised)!r}. "
                f"Supported: {', '.join(SUPPORTED_TRANSPORTS)}"
            )
        if not (url or "").strip():
            raise MCPClientError("MCP server url/command is required")

        self.url = url.strip()
        self.transport = normalised
        self.auth_token = (auth_token or "").strip()
        self.timeout = float(timeout)
        self.env = env
        self._transport_impl = transport_impl

        self._process: asyncio.subprocess.Process | None = None
        self._http: httpx.AsyncClient | None = None
        self._next_id = 0
        self._lock = asyncio.Lock()
        self._initialised = False
        self._server_info: dict[str, Any] = {}

    # -- lifecycle --------------------------------------------------------
    async def __aenter__(self) -> MCPClient:
        await self.connect()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def connect(self) -> None:
        """Open the transport and perform the MCP ``initialize`` handshake."""
        if self.transport == "stdio":
            await self._spawn()
        else:
            headers = {"Accept": "application/json, text/event-stream"}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            self._http = httpx.AsyncClient(
                timeout=self.timeout,
                headers=headers,
                transport=self._transport_impl,
            )

        try:
            result = await self._request(
                "initialize",
                {
                    "protocolVersion": _MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "gpt6-sol-mcp-gateway", "version": "0.1.0"},
                },
            )
            self._server_info = result or {}
            await self._notify("notifications/initialized", {})
            self._initialised = True
            log_info(
                "mcp connected transport=%s server=%s",
                self.transport,
                redact(str(self._server_info.get("serverInfo", {}).get("name", "?"))),
            )
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        if self._process is not None:
            process = self._process
            self._process = None
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
        self._initialised = False

    # -- stdio plumbing ---------------------------------------------------
    async def _spawn(self) -> None:
        argv = split_command(self.url)
        if not argv:
            raise MCPClientError("Empty stdio command")

        env = dict(os.environ)
        if self.env:
            env.update(self.env)

        try:
            self._process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except (OSError, ValueError) as exc:
            raise MCPClientError(
                f"Failed to start MCP server {redact(argv[0])!r}: {type(exc).__name__}"
            ) from exc

        log_debug("mcp stdio spawned pid=%s", self._process.pid)

    async def _stdio_send(self, message: dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise MCPClientError("MCP stdio process is not running")
        line = json.dumps(message, ensure_ascii=False) + "\n"
        self._process.stdin.write(line.encode("utf-8"))
        await self._process.stdin.drain()

    async def _stdio_read(self) -> dict[str, Any]:
        if self._process is None or self._process.stdout is None:
            raise MCPClientError("MCP stdio process is not running")
        while True:
            try:
                raw = await asyncio.wait_for(self._process.stdout.readline(), timeout=self.timeout)
            except asyncio.TimeoutError as exc:
                raise MCPClientError(f"MCP stdio read timed out after {self.timeout}s") from exc
            if not raw:
                stderr = b""
                if self._process.stderr is not None:
                    try:
                        stderr = await asyncio.wait_for(self._process.stderr.read(), timeout=2)
                    except asyncio.TimeoutError:
                        stderr = b""
                detail = redact(stderr.decode("utf-8", "replace")[:300])
                raise MCPClientError(
                    f"MCP stdio server closed the stream (exit={self._process.returncode})"
                    + (f": {detail}" if detail else "")
                )
            text = raw.decode("utf-8", "replace").strip()
            if not text:
                continue
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                # Servers may print banners on stdout; skip non-JSON lines.
                log_debug("mcp stdio skipped non-JSON line: %s", redact(text[:120]))
                continue
            if isinstance(decoded, dict):
                return decoded
            log_debug("mcp stdio skipped non-object JSON line")
            continue

    # -- http / sse plumbing ----------------------------------------------
    async def _http_post(self, message: dict[str, Any]) -> dict[str, Any]:
        if self._http is None:
            raise MCPClientError("MCP http client is not connected")
        try:
            response = await self._http.post(self.url, json=message)
        except httpx.HTTPError as exc:
            raise MCPClientError(f"MCP http transport error: {type(exc).__name__}") from exc
        if response.status_code >= 400:
            raise MCPClientError(
                f"MCP http returned HTTP {response.status_code}: {redact(response.text[:300])}"
            )
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            return self._parse_sse(response.text)
        try:
            decoded = response.json()
        except ValueError as exc:
            raise MCPClientError("MCP http returned a non-JSON body") from exc
        if not isinstance(decoded, dict):
            raise MCPClientError("MCP http returned a non-object JSON body")
        return decoded

    @staticmethod
    def _parse_sse(body: str) -> dict[str, Any]:
        """Return the first JSON-RPC payload found in an SSE body."""
        for line in body.splitlines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict) and ("result" in decoded or "error" in decoded):
                return decoded
        raise MCPClientError("MCP sse stream contained no JSON-RPC response")

    # -- JSON-RPC ---------------------------------------------------------
    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    async def _request(self, method: str, params: dict[str, Any]) -> Any:
        """Send a request and return its ``result``, raising on ``error``."""
        async with self._lock:
            message = {
                "jsonrpc": _JSONRPC_VERSION,
                "id": self._new_id(),
                "method": method,
                "params": params,
            }
            if self.transport == "stdio":
                await self._stdio_send(message)
                reply = await self._stdio_read()
            else:
                reply = await self._http_post(message)

        if reply.get("error"):
            error = reply["error"] or {}
            raise MCPClientError(
                f"MCP error {error.get('code')}: {redact(str(error.get('message')))}"
            )
        return reply.get("result")

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        """Send a notification (no id, no reply expected)."""
        message = {"jsonrpc": _JSONRPC_VERSION, "method": method, "params": params}
        if self.transport == "stdio":
            await self._stdio_send(message)
        elif self._http is not None:
            try:
                await self._http.post(self.url, json=message)
            except httpx.HTTPError:
                # Notifications are best-effort; a failure must not break setup.
                log_debug("mcp notification %s failed", method)

    # -- public API -------------------------------------------------------
    async def list_tools(self) -> list[MCPTool]:
        """Return every tool the server advertises."""
        result = await self._request("tools/list", {})
        raw_tools = (result or {}).get("tools") or []
        tools = [MCPTool.from_raw(t) for t in raw_tools if isinstance(t, dict) and t.get("name")]
        log_debug("mcp tools/list returned %d tools", len(tools))
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> MCPCallResult:
        """Invoke *name* with *arguments* and flatten the content blocks."""
        if not name:
            raise MCPClientError("Tool name is required")
        result = await self._request("tools/call", {"name": name, "arguments": arguments or {}})
        payload = result or {}
        return MCPCallResult(
            tool=name,
            content=_flatten_content(payload.get("content")),
            is_error=bool(payload.get("isError")),
            raw=payload,
        )

    @property
    def server_info(self) -> dict[str, Any]:
        return dict(self._server_info)

    @property
    def initialised(self) -> bool:
        return self._initialised


def default_stdio_command() -> str:
    """Command used when ``MCP_SERVER_URL`` is unset for the stdio transport."""
    return f"{shlex.quote(sys.executable)} -m gpt6_sol_mcp.gateway.server"
