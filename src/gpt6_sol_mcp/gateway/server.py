"""stdio MCP server — exposes the gateway itself as MCP tools.

Run it directly::

    python -m gpt6_sol_mcp.gateway.server

or let a client spawn it::

    gpt6-mcp-gateway mcp

Tools
-----
``gpt6_chat``
    Send a prompt to GPT-6 Sol and return the answer.
``gpt6_models``
    List the models this gateway knows about.
``gpt6_status``
    Report the non-secret gateway configuration.
``gpt6_tools``
    List the MCP tools the gateway is allowed to call.
``gpt6_debug_marathon``
    Queue prioritized debug tasks through GPT-6 Luna, Sol, and Astra.
``gpt6_debug_marathon_status``
    Get a queued debug marathon's progress and results.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from ..config import get_settings
from ..logging_utils import log_error, log_info, redact
from ..provider import KNOWN_MODELS, MODEL_ALIASES
from .facade import Gateway

SERVER_NAME = "gpt6-sol-mcp-gateway"

_CHAT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            "description": "The user message to send to the model.",
        },
        "system": {
            "type": "string",
            "description": "Optional system instruction.",
        },
        "model": {
            "type": "string",
            "description": "Model id or alias. Defaults to the configured model.",
        },
    },
    "required": ["prompt"],
    "additionalProperties": False,
}

_EMPTY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}

_DEBUG_MARATHON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "minItems": 1,
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "question": {"type": "string"},
                    "priority": {"type": "integer", "minimum": 1, "maximum": 5},
                    "difficulty": {"type": "integer", "minimum": 1, "maximum": 5},
                    "complexity": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tasks"],
    "additionalProperties": False,
}

_DEBUG_MARATHON_STATUS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"job_id": {"type": "string"}},
    "required": ["job_id"],
    "additionalProperties": False,
}


def build_server(gateway: Gateway | None = None) -> Server:
    """Create the MCP server. *gateway* is injectable for tests."""
    state: dict[str, Gateway | None] = {"gateway": gateway}

    async def _gateway() -> Gateway:
        current = state["gateway"]
        if current is None:
            current = Gateway(settings=get_settings())
            await current.start()
            state["gateway"] = current
        return current

    async def list_tools(_ctx: Any, _params: Any = None) -> ListToolsResult:
        return ListToolsResult(
            tools=[
                Tool(
                    name="gpt6_chat",
                    description=(
                        "Ask GPT-6 Sol a question. The model may call MCP tools "
                        "through the gateway to answer."
                    ),
                    input_schema=_CHAT_SCHEMA,
                ),
                Tool(
                    name="gpt6_models",
                    description="List the models this gateway knows about, with aliases.",
                    input_schema=_EMPTY_SCHEMA,
                ),
                Tool(
                    name="gpt6_status",
                    description="Report the gateway configuration (no secrets).",
                    input_schema=_EMPTY_SCHEMA,
                ),
                Tool(
                    name="gpt6_tools",
                    description="List the MCP tools the gateway is allowed to call.",
                    input_schema=_EMPTY_SCHEMA,
                ),
                Tool(
                    name="gpt6_debug_marathon",
                    description=(
                        "Queue 1-10 debug tasks. Tasks are ordered by priority, "
                        "difficulty, then complexity (highest first). Each task "
                        "runs through GPT-6 Luna, then Sol, then Astra. Returns a job_id."
                    ),
                    input_schema=_DEBUG_MARATHON_SCHEMA,
                ),
                Tool(
                    name="gpt6_debug_marathon_status",
                    description="Get progress and results for a debug marathon job_id.",
                    input_schema=_DEBUG_MARATHON_STATUS_SCHEMA,
                ),
            ]
        )

    async def call_tool(_ctx: Any, params: Any) -> CallToolResult:
        name = str(getattr(params, "name", "") or "")
        args = dict(getattr(params, "arguments", None) or {})
        blocks: list[Any] = list(await _dispatch(name, args))
        return CallToolResult(content=blocks)

    async def _dispatch(name: str, args: dict[str, Any]) -> list[TextContent]:
        try:
            if name == "gpt6_chat":
                prompt = str(args.get("prompt") or "").strip()
                if not prompt:
                    return [TextContent(type="text", text="Error: 'prompt' is required")]
                current = await _gateway()
                result = await current.chat(
                    [{"role": "user", "content": prompt}],
                    system=args.get("system"),
                    model_id=args.get("model"),
                )
                payload = {
                    "content": result.content,
                    "model": result.model,
                    "rounds": result.rounds,
                    "used_tools": result.used_tools,
                }
                return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

            if name == "gpt6_models":
                payload = {
                    model_id: {
                        "label": spec["label"],
                        "tier": spec["tier"],
                        "context_window": spec["context_window"],
                        "max_output": spec["max_output"],
                        "aliases": sorted(
                            a for a, target in MODEL_ALIASES.items() if target == model_id
                        ),
                    }
                    for model_id, spec in sorted(KNOWN_MODELS.items())
                }
                return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

            if name == "gpt6_status":
                current = await _gateway()
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(current.status(), ensure_ascii=False),
                    )
                ]

            if name == "gpt6_tools":
                current = await _gateway()
                tools = await current.list_tools()
                return [
                    TextContent(
                        type="text",
                        text=json.dumps([t["function"]["name"] for t in tools], ensure_ascii=False),
                    )
                ]

            if name == "gpt6_debug_marathon":
                current = await _gateway()
                submitted_job = current.debug_marathon.submit(args.get("tasks") or [])
                return [TextContent(type="text", text=json.dumps(submitted_job, ensure_ascii=False))]

            if name == "gpt6_debug_marathon_status":
                current = await _gateway()
                status_payload = current.debug_marathon.get(str(args.get("job_id") or ""))
                if status_payload is None:
                    return [TextContent(type="text", text="Error: debug marathon job not found")]
                return [TextContent(type="text", text=json.dumps(status_payload, ensure_ascii=False))]

            return [TextContent(type="text", text=f"Error: unknown tool {name!r}")]
        except Exception as exc:
            log_error("mcp tool %s failed: %s", redact(name), redact(str(exc)))
            return [TextContent(type="text", text=f"Error: {exc}")]

    return Server(
        SERVER_NAME,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


async def _run() -> None:
    server = build_server()
    log_info("starting stdio MCP server %s", SERVER_NAME)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    """Console entry point for ``python -m gpt6_sol_mcp.gateway.server``."""
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_run())


if __name__ == "__main__":
    main()
