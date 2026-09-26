# MCP Server

The gateway can also run *as* an MCP server, so other MCP clients (Claude Desktop, Cursor, VS Code,
another agent) can use GPT6-SOL as a tool.

## Start it

```bash
gpt6-mcp-gateway mcp
```

This speaks MCP over **stdio**. The process reads JSON-RPC from stdin and writes to stdout, so it must
be launched by an MCP client rather than run interactively.

## Exposed tools

| Tool | Arguments | Returns |
| --- | --- | --- |
| `gpt6_chat` | `prompt` (required), `model`, `system` | The model's answer as text |
| `gpt6_models` | — | JSON object of model id → metadata |
| `gpt6_status` | — | JSON object of configuration booleans |
| `gpt6_tools` | — | JSON array of tools advertised by the connected MCP server |

## Client configuration

=== "Claude Desktop"

    Add to `claude_desktop_config.json`:

    ```json
    {
      "mcpServers": {
        "gpt6-sol": {
          "command": "gpt6-mcp-gateway",
          "args": ["mcp"],
          "env": {"OPENAI_API_KEY": "sk-..."}
        }
      }
    }
    ```

=== "VS Code"

    Add to `.vscode/mcp.json`:

    ```json
    {
      "servers": {
        "gpt6-sol": {
          "type": "stdio",
          "command": "gpt6-mcp-gateway",
          "args": ["mcp"],
          "env": {"OPENAI_API_KEY": "sk-..."}
        }
      }
    }
    ```

=== "Cursor"

    Add to `~/.cursor/mcp.json`:

    ```json
    {
      "mcpServers": {
        "gpt6-sol": {
          "command": "gpt6-mcp-gateway",
          "args": ["mcp"],
          "env": {"OPENAI_API_KEY": "sk-..."}
        }
      }
    }
    ```

## Programmatic use

```python
import asyncio
from gpt6_sol_mcp.gateway.server import build_server

server = build_server()
asyncio.run(server.run_stdio_async())
```

## Verifying the server

The test suite includes an end-to-end test that spawns this server as a real subprocess and performs a
full MCP handshake over stdio:

```bash
pytest tests/e2e -v
```

## Notes

- The server uses the `mcp` 2.x callback API (`on_list_tools` / `on_call_tool` passed to the `Server`
  constructor). The older decorator API (`@server.list_tools()`) was removed in `mcp` 2.0.
- `gpt6_chat` requires a non-empty `prompt`; an empty value returns an error result rather than
  raising, so the client sees a normal tool error.
- Tool results are returned as MCP `TextContent` blocks. `gpt6_models`, `gpt6_status`, and
  `gpt6_tools` return JSON-encoded strings.
