"""Command-line interface for the gateway.

Subcommands
-----------
``serve``
    Run the FastAPI gateway with uvicorn.
``mcp``
    Run the stdio MCP server.
``chat``
    Send one prompt and print the answer.
``models``
    List the known models.
``status``
    Print the non-secret configuration.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from .config import get_settings
from .logging_utils import log_error, redact
from .provider import KNOWN_MODELS, MODEL_ALIASES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpt6-mcp-gateway",
        description="GPT-6 Sol MCP Gateway — OpenAI-compatible gateway with MCP tools.",
    )
    parser.add_argument("--version", action="version", version="gpt6-sol-mcp-gateway 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the FastAPI gateway (uvicorn).")
    serve.add_argument("--host", default=None, help="Bind host (default: GATEWAY_HOST).")
    serve.add_argument("--port", type=int, default=None, help="Bind port (default: GATEWAY_PORT).")
    serve.add_argument("--reload", action="store_true", help="Auto-reload on code changes.")

    sub.add_parser("mcp", help="Run the stdio MCP server.")

    chat = sub.add_parser("chat", help="Send one prompt and print the answer.")
    chat.add_argument("prompt", help="The prompt to send.")
    chat.add_argument("--system", default=None, help="Optional system instruction.")
    chat.add_argument("--model", default=None, help="Model id or alias.")
    chat.add_argument("--json", action="store_true", help="Print the full JSON result.")

    sub.add_parser("models", help="List the known models.")
    sub.add_parser("status", help="Print the non-secret configuration.")

    return parser


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is not installed. Install the server extras:\n"
            "  pip install 'gpt6-sol-mcp-gateway[server]'",
            file=sys.stderr,
        )
        return 1

    settings = get_settings()
    host = args.host or settings.gateway_host
    port = args.port or settings.gateway_port
    print(f"Starting gateway on http://{host}:{port}")
    uvicorn.run(
        "gpt6_sol_mcp.gateway.app:app",
        host=host,
        port=port,
        reload=bool(args.reload),
    )
    return 0


def _cmd_mcp(_args: argparse.Namespace) -> int:
    from .gateway.server import main as server_main

    server_main()
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    from .gateway.facade import Gateway

    async def _run() -> int:
        async with Gateway() as gateway:
            result = await gateway.chat(
                [{"role": "user", "content": args.prompt}],
                system=args.system,
                model_id=args.model,
            )
        if args.json:
            print(
                json.dumps(
                    {
                        "content": result.content,
                        "model": result.model,
                        "rounds": result.rounds,
                        "used_tools": result.used_tools,
                        "usage": result.usage,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(result.content)
        return 0

    try:
        return asyncio.run(_run())
    except Exception as exc:
        log_error("chat failed: %s", redact(str(exc)))
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _cmd_models(_args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {
        model_id: {
            "label": spec["label"],
            "tier": spec["tier"],
            "context_window": spec["context_window"],
            "max_output": spec["max_output"],
            "aliases": sorted(a for a, t in MODEL_ALIASES.items() if t == model_id),
        }
        for model_id, spec in sorted(KNOWN_MODELS.items())
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_status(_args: argparse.Namespace) -> int:
    print(json.dumps(get_settings().describe(), ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``gpt6-mcp-gateway`` console script."""
    args = _build_parser().parse_args(argv)
    handlers = {
        "serve": _cmd_serve,
        "mcp": _cmd_mcp,
        "chat": _cmd_chat,
        "models": _cmd_models,
        "status": _cmd_status,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
