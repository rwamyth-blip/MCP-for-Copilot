"""Gateway facade — one object that owns the whole stack.

:class:`Gateway` wires the provider, the MCP client, the router, the approval
layer and the orchestrator together, and manages the MCP connection lifetime.

Typical use::

    async with Gateway() as gateway:
        result = await gateway.chat([{"role": "user", "content": "List the files"}])
        print(result.content)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..approval import ApprovalLayer, Approver
from ..config import Settings, get_settings
from ..debug_marathon import DebugMarathon
from ..logging_utils import log_info, log_warning, redact
from ..mcp_client import MCPClient, MCPClientError, default_stdio_command
from ..orchestrator import LLMMCPOrchestrator, OrchestratorResult
from ..provider import LLMProvider
from ..tool_router import ToolRouter


@dataclass
class GatewayResult:
    """A chat completion plus the tool audit trail."""

    content: str = ""
    model: str = ""
    rounds: int = 0
    used_tools: list[str] = field(default_factory=list)
    invocations: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @classmethod
    def from_orchestrator(cls, result: OrchestratorResult) -> GatewayResult:
        return cls(
            content=result.content,
            model=result.model,
            rounds=result.rounds,
            used_tools=result.used_tools,
            invocations=[
                {
                    "tool": i.tool,
                    "arguments": i.arguments,
                    "allowed": i.allowed,
                    "approved": i.approved,
                    "executed": i.executed,
                    "is_error": i.is_error,
                    "reason": i.reason,
                }
                for i in result.invocations
            ],
            usage=result.usage,
            error=result.error,
        )


class Gateway:
    """Owns the provider + MCP client + orchestrator for one process."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        provider: LLMProvider | None = None,
        approver: Approver | None = None,
        connect_mcp: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = provider or LLMProvider()
        self.debug_marathon = DebugMarathon(self.provider)
        self._approver = approver
        self._connect_mcp = connect_mcp

        self.client: MCPClient | None = None
        self.router = ToolRouter(
            allowed_tools=self.settings.mcp_allowed_tools or None,
            require_approval=self.settings.mcp_require_approval,
        )
        self.approval = ApprovalLayer(approver)
        self.orchestrator = LLMMCPOrchestrator(
            provider=self.provider,
            client=None,
            router=self.router,
            approval=self.approval,
            settings=self.settings,
        )
        self._connected = False

    # -- lifecycle --------------------------------------------------------
    async def __aenter__(self) -> Gateway:
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.stop()

    async def start(self) -> None:
        """Connect to the MCP server, if one is configured."""
        if self._connected or not self._connect_mcp:
            return
        if not self.settings.mcp_configured and self.settings.mcp_transport != "stdio":
            log_warning("MCP_SERVER_URL is not set; running without tools")
            self._connected = True
            return

        command = self.settings.mcp_server_url or default_stdio_command()
        self.client = MCPClient(
            url=command,
            transport=self.settings.mcp_transport,
            auth_token=self.settings.mcp_auth_token,
            timeout=float(self.settings.mcp_timeout_seconds),
        )
        try:
            await self.client.connect()
        except MCPClientError as exc:
            log_warning("MCP connection failed: %s", redact(str(exc)))
            self.client = None
        self.orchestrator.client = self.client
        self._connected = True

    async def stop(self) -> None:
        await self.debug_marathon.stop()
        if self.client is not None:
            await self.client.close()
            self.client = None
        self.orchestrator.client = None
        self._connected = False

    # -- public API -------------------------------------------------------
    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        model_id: str | None = None,
        max_rounds: int | None = None,
    ) -> GatewayResult:
        """Run one chat turn through the full tool loop."""
        if not self._connected:
            await self.start()
        result = await self.orchestrator.run(
            messages, system=system, model_id=model_id, max_rounds=max_rounds
        )
        return GatewayResult.from_orchestrator(result)

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return the allowed tools as OpenAI function schemas."""
        if not self._connected:
            await self.start()
        if self.client is None:
            return []
        try:
            tools = await self.client.list_tools()
        except MCPClientError as exc:
            log_warning("could not list tools: %s", redact(str(exc)))
            return []
        self.router.register_all(tools)
        return LLMProvider.to_openai_tools(self.router.available())

    def status(self) -> dict[str, Any]:
        """Non-secret status snapshot, safe to return over HTTP."""
        return {
            "llm": self.provider.describe(),
            "mcp": {
                "transport": self.settings.mcp_transport,
                "configured": self.settings.mcp_configured,
                "connected": self.client is not None,
                "server_info": self.client.server_info if self.client else {},
                "allowed_tools": list(self.router.allowed_tools),
                "require_approval": self.settings.mcp_require_approval,
                "approver_configured": self.approval.configured,
            },
        }

    def log_summary(self) -> None:
        info = self.status()
        log_info(
            "gateway ready model=%s mcp_connected=%s tools=%d",
            info["llm"]["model_id"],
            info["mcp"]["connected"],
            len(info["mcp"]["allowed_tools"]),
        )
