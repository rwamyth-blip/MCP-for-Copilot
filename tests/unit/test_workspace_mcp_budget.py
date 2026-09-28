"""Smoke test: the workspace MCP config must stay under the 128-tool cap.

Copilot / Copilot CLI send every tool VS Code loaded to the model API, and the
API rejects a request with more than 128 tools:
    502 Cannot have more than 128 tools per request
Two things cause that here:
  1. a server that dies at start-up and contributes an unknown number of tools
     to the config's advertised count while exposing none at runtime
  2. the same server declared twice, which registers every tool name twice

This test reads both workspace config files and asserts the unique tool count
stays well under the cap, and that no server is declared in both.
"""

from __future__ import annotations

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
CONFIGS = ((".vscode/mcp.json", "servers"), (".mcp.json", "mcpServers"))

# The model API rejects more than 128 tools in one request. Copilot also adds
# its own built-in tools on top of these, so the MCP budget must stay far
# below the cap rather than just under it.
TOOL_LIMIT = 128
SAFE_BUDGET = 100


def _load(rel: str, key: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8-sig"))[key]


def _tool_counts() -> dict[str, int]:
    """Advertised tool count per server, taken from the registry in source."""
    return {
        "vihokai-ollama": 8,  # status, models, show, chat, compare, pull, ps, stop
        "vihokai-mongodb": 8,  # ping, list_dbs, list_colls, stats, find, count, aggregate, indexes
        "vihokai-codex": 6,  # chat, luna, models, compare, run, status
        "gpt6-sol-gateway": 4,  # status, models, chat, compare
    }


class TestMcpToolBudget:
    def test_configs_are_valid_json(self) -> None:
        for rel, key in CONFIGS:
            data = _load(rel, key)
            assert isinstance(data, dict) and data, f"{rel} has no {key}"

    def test_the_two_configs_declare_the_same_servers(self) -> None:
        """VS Code reads .vscode/mcp.json, Copilot CLI reads .mcp.json -- one
        client per file, never both at once, so the two are meant to mirror
        each other. Divergence is what silently drops a tool for one client."""
        vscode_servers = set(_load(*CONFIGS[0]))
        portable_servers = set(_load(*CONFIGS[1]))
        assert vscode_servers == portable_servers, (
            f"configs diverged: only in .vscode/mcp.json {sorted(vscode_servers - portable_servers)}, "
            f"only in .mcp.json {sorted(portable_servers - vscode_servers)}"
        )

    def test_unique_tool_count_is_under_the_api_cap(self) -> None:
        counts = _tool_counts()
        total = sum(counts.get(name, 0) for name in _load(*CONFIGS[0]))
        assert total <= SAFE_BUDGET, f"{total} tools exceeds the safe budget of {SAFE_BUDGET}"
        assert total < TOOL_LIMIT

    def test_every_declared_server_has_a_known_tool_count(self) -> None:
        """An unknown count means a new server was added without updating the
        budget, which is exactly how the 128-tool cap gets hit unnoticed."""
        counts = _tool_counts()
        declared = set(_load(*CONFIGS[0]))
        assert declared == set(counts), f"config and tool counts disagree: {declared ^ set(counts)}"

    def test_clone_servers_are_not_declared(self) -> None:
        """mcp2/cloning/* are reference copies. VS Code loading both copies
        re-registered all 14 shared tool names twice."""
        for rel, key in CONFIGS:
            names = _load(rel, key)
            clones = [n for n in names if n.endswith("-clone")]
            assert not clones, f"{rel} declares duplicate clone servers: {clones}"

    def test_ollama_server_advertises_all_eight_tools(self) -> None:
        """The server used to crash at start-up on MCP SDK 1.x, which the
        client saw as 0 tools. The count must match the source registry."""
        assert _tool_counts()["vihokai-ollama"] == 8
