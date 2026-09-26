# Configuration

All settings are read from environment variables, with a `.env` file in the current working directory
as a fallback. **Real environment variables always win** — `.env` values are applied with
`os.environ.setdefault`, so they never overwrite an already-set variable.

## LLM provider

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` | Only `openai` is implemented |
| `LLM_MODEL_ID` | `gpt-6-sol` | Model id or alias |
| `LLM_API_KEY` | *(empty)* | **Required.** Provider API key |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL, no trailing `/chat/completions` |
| `LLM_TIMEOUT_SECONDS` | `120` | Per-request timeout |
| `LLM_REASONING_EFFORT` | *(empty)* | `none`, `low`, `medium`, `high`, `xhigh`, `max` |

!!! warning "`LLM_REASONING_EFFORT` and tools"
    The GPT-6 family rejects `reasoning_effort` together with function tools on
    `/v1/chat/completions`. When the request contains `tools`, the gateway forces
    `reasoning_effort = "none"`. The field must be **present and equal to `"none"`** — omitting it is
    also rejected by the upstream API.

## MCP server

| Variable | Default | Description |
| --- | --- | --- |
| `MCP_SERVER_URL` | *(empty)* | stdio: a command to spawn. http/sse: a URL |
| `MCP_TRANSPORT` | `stdio` | `stdio`, `http`, or `sse` |
| `MCP_AUTH_TOKEN` | *(empty)* | Bearer token for http/sse transports |
| `MCP_REQUIRE_APPROVAL` | `true` | Risky tools need an explicit approval decision |
| `MCP_ALLOWED_TOOLS` | *(empty)* | Comma-separated allowlist. Empty = read-only defaults |
| `MCP_MAX_TOOL_ROUNDS` | `5` | Cap on model → tool → model rounds per turn |
| `MCP_TIMEOUT_SECONDS` | `60` | Per-tool-call timeout |

### `MCP_ALLOWED_TOOLS` semantics

| Value | Meaning |
| --- | --- |
| unset / empty | Use the built-in read-only default set (12 tools) |
| `read_file,list_directory` | Allow exactly those two |
| `,` (only separators) | Allow nothing |

The distinction between "unset" and "empty tuple" is preserved in code: `None` means defaults, an
empty tuple means deny everything.

## Gateway server

| Variable | Default | Description |
| --- | --- | --- |
| `GATEWAY_HOST` | `0.0.0.0` | Bind address |
| `GATEWAY_PORT` | `8000` | Bind port |
| `GATEWAY_API_KEY` | *(empty)* | When set, callers must send `Authorization: Bearer <key>` |
| `GATEWAY_CORS_ORIGINS` | `*` | Comma-separated origins |

## Model catalogue

| Model id | Tier | Context | Max output | Input / Output per Mtok |
| --- | --- | --- | --- | --- |
| `gpt-6-astra` | flagship | — | — | $10 / $50 |
| `gpt-6-sol` | balanced | 1.05M | 128K | $2 / $10 |
| `gpt-6-luna` | efficient | — | — | $0.10 / $0.50 |
| `gpt-5.6-sol` | previous gen | — | — | $4 / $20 |
| `gpt-5.6-luna` | previous gen | — | — | $0.20 / $1.20 (cached $0.02) |

### Aliases

| Alias | Resolves to |
| --- | --- |
| `gpt-6`, `gpt6`, `sol` | `gpt-6-sol` |
| `luna` | `gpt-6-luna` |
| `astra` | `gpt-6-astra` |
| `gpt-5.6` | `gpt-5.6-sol` |

## Inspecting the effective configuration

```bash
gpt6-mcp-gateway status
```

```python
from gpt6_sol_mcp import get_settings

print(get_settings().describe())
```

`describe()` and `status()` report **booleans** for credential presence. They never print key values.

## Reloading settings

`get_settings()` is `lru_cache`d. In tests, call `reset_settings_cache()` after mutating the
environment:

```python
from gpt6_sol_mcp import get_settings, reset_settings_cache

os.environ["LLM_MODEL_ID"] = "gpt-6-luna"
reset_settings_cache()
assert get_settings().llm_model_id == "gpt-6-luna"
```
