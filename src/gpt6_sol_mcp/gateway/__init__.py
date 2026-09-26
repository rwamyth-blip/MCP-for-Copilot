"""Gateway layer — the two deployable entry points.

* :mod:`gpt6_sol_mcp.gateway.app` — a FastAPI app exposing an
  OpenAI-compatible ``/v1/chat/completions`` endpoint.
* :mod:`gpt6_sol_mcp.gateway.server` — a stdio MCP server exposing the gateway
  itself as MCP tools.

:class:`Gateway` is the shared facade both entry points build on.
"""

from .facade import Gateway, GatewayResult

__all__ = ["Gateway", "GatewayResult"]
