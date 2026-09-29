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
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
CONFIGS = ((".vscode/mcp.json", "servers"), (".mcp.json", "mcpServers"))

# The model API rejects more than 128 tools in one request. Copilot also adds
# its own built-in tools on top of these, so the MCP budget must stay far
# below the cap rather than just under it.
TOOL_LIMIT = 128
SAFE_BUDGET = 100


def _load(rel: str, key: str) -> dict:
    text = (ROOT / rel).read_text(encoding="utf-8-sig")
    return _strip_jsonc(text)[key]


def _strip_jsonc(text: str) -> dict:
    """Parse a VS Code JSONC config.

    These files are JSON with comments and trailing commas, which plain
    json.loads rejects. The comment rule is copied from VS Code's own
    jsonc-parser behaviour for the subset used here: a `//` or `/* */`
    comment that is not inside a string literal is dropped, and a comma
    directly before `}` or `]` is removed.

    Both configs round-trip through VS Code's own parser, so anything this
    accepts is valid config -- the test is not more lenient than the client.
    """
    out: list[str] = []
    i, n = 0, len(text)
    in_string = False
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if ch == "\\":
                # Copy the escaped character so an escaped quote does not
                # end the string early.
                if i + 1 < n:
                    out.append(text[i + 1])
                    i += 2
                    continue
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        out.append(ch)
        i += 1

    stripped = "".join(out)
    # Drop trailing commas left behind once the comments they sat on are gone.
    stripped = re.sub(r",(\s*[}\]])", r"\1", stripped)
    return json.loads(stripped)


def _tool_counts() -> dict[str, int]:
    """Advertised tool count per server, taken from the registry in source."""
    return {
        "vihokai-ollama": 8,  # status, models, show, chat, compare, pull, ps, stop
        "vihokai-mongodb": 8,  # ping, list_dbs, list_colls, stats, find, count, aggregate, indexes
        "vihokai-codex": 6,  # chat, luna, models, compare, run, status
        "gpt6-sol-mcp-Local": 4,  # mcp_bridge.py: status, models, chat, compare
        "gpt6-sol-gpt53-codex-mcp3": 6,  # mcp3 clone, mcp3_* names
        "vihokai-mongodb-clone": 8,  # mcp2 clone, mcp2_mongo_* names
        "vihokai-codex-clone": 6,  # mcp2 clone, mcp2_vihokai_*/mcp2_codex_* names
    }


def _tool_names() -> dict[str, list[str]]:
    """The exact tool names each declared server advertises.

    Asserting on real names is what makes the collision test meaningful: a
    count alone cannot tell two servers apart, which is exactly how the
    14 duplicated clone names slipped through before.
    """
    return {
        "vihokai-ollama": [
            "ollama_status", "ollama_models", "ollama_show", "ollama_chat",
            "ollama_compare", "ollama_pull", "ollama_ps", "ollama_stop",
        ],
        "vihokai-mongodb": [
            "mongo_ping", "mongo_list_databases", "mongo_list_collections",
            "mongo_stats", "mongo_find", "mongo_count", "mongo_aggregate",
            "mongo_indexes",
        ],
        "vihokai-codex": [
            "vihokai_chat", "vihokai_luna", "vihokai_models",
            "vihokai_compare", "codex_run", "codex_status",
        ],
        "gpt6-sol-mcp-Local": [
            "gateway_status", "gateway_models", "gateway_chat", "gateway_compare",
        ],
        "gpt6-sol-gpt53-codex-mcp3": [
            "mcp3_vihokai_chat", "mcp3_vihokai_luna", "mcp3_vihokai_models",
            "mcp3_vihokai_compare", "mcp3_codex_run", "mcp3_codex_status",
        ],
        "vihokai-mongodb-clone": [
            "mcp2_mongo_ping", "mcp2_mongo_list_databases",
            "mcp2_mongo_list_collections", "mcp2_mongo_stats", "mcp2_mongo_find",
            "mcp2_mongo_count", "mcp2_mongo_aggregate", "mcp2_mongo_indexes",
        ],
        "vihokai-codex-clone": [
            "mcp2_vihokai_chat", "mcp2_vihokai_luna", "mcp2_vihokai_models",
            "mcp2_vihokai_compare", "mcp2_codex_run", "mcp2_codex_status",
        ],
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

    def test_no_server_declares_a_duplicate_tool_name(self) -> None:
        """The actual failure this file exists to prevent.

        An MCP client registers every advertised tool name per server, so two
        servers exposing the same name register it twice and the model API
        rejects the request with "Cannot have more than 128 tools". Counts
        cannot catch this -- only names can.

        Each copy of the Codex/Mongo servers must therefore namespace its
        tools (TOOL_PREFIX in its own mcp_server.py / mongodb_mcp.py).
        """
        names = _tool_names()
        declared = set(_load(*CONFIGS[0]))
        seen: dict[str, str] = {}
        collisions: list[str] = []
        for server in sorted(declared):
            for tool in names[server]:
                if tool in seen:
                    collisions.append(f"{tool} ({seen[tool]} + {server})")
                else:
                    seen[tool] = server
        assert not collisions, "duplicate tool names: " + "; ".join(sorted(collisions))

    def test_tool_counts_match_the_recorded_names(self) -> None:
        """Keeps the count table and the name table from drifting apart."""
        counts, names = _tool_counts(), _tool_names()
        assert set(counts) == set(names)
        for server, count in counts.items():
            assert len(names[server]) == count, f"{server}: {len(names[server])} != {count}"

    def test_clone_servers_namespace_their_tools(self) -> None:
        """A clone may only ship enabled if its tools are namespaced.

        This is the rule the blanket "-clone" ban used to encode indirectly.
        It is now checked directly, so a clone is allowed but only when its
        names are distinct from every other declared server.
        """
        names = _tool_names()
        declared = set(_load(*CONFIGS[0]))
        for server in (s for s in declared if s.endswith("-clone")):
            for tool in names[server]:
                base = tool.removeprefix("mcp2_").removeprefix("mcp3_")
                assert tool != base, (
                    f"{server} advertises unnamespaced tool {tool!r}; it would "
                    "collide with the primary server"
                )

    def test_ollama_server_advertises_all_eight_tools(self) -> None:
        """The server used to crash at start-up on MCP SDK 1.x, which the
        client saw as 0 tools. The count must match the source registry."""
        assert _tool_counts()["vihokai-ollama"] == 8
