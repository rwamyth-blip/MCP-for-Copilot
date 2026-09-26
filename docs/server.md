# Server Usage

The gateway ships an OpenAI-compatible HTTP API, so any OpenAI SDK client works against it unchanged.

## Start the server

```bash
gpt6-mcp-gateway serve --host 0.0.0.0 --port 8080
```

Or programmatically:

```python
import uvicorn
from gpt6_sol_mcp.gateway.app import create_app

app = create_app()
uvicorn.run(app, host="0.0.0.0", port=8080)
```

## Routes

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness probe |
| `GET` | `/v1/models` | OpenAI-shaped model list |
| `GET` | `/v1/status` | Configuration summary (booleans only) |
| `POST` | `/v1/chat/completions` | Chat completions, streaming and non-streaming |

### `GET /health`

```json
{"status": "ok", "service": "gpt6-sol-mcp-gateway"}
```

### `GET /v1/models`

```json
{
  "object": "list",
  "data": [
    {"id": "gpt-6-astra", "object": "model", "owned_by": "openai"},
    {"id": "gpt-6-sol", "object": "model", "owned_by": "openai"},
    {"id": "gpt-6-luna", "object": "model", "owned_by": "openai"},
    {"id": "gpt-5.6-sol", "object": "model", "owned_by": "openai"},
    {"id": "gpt-5.6-luna", "object": "model", "owned_by": "openai"}
  ]
}
```

### `POST /v1/chat/completions`

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "gpt-6-sol",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": false
      }'
```

Streaming:

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "gpt-6-sol",
        "messages": [{"role": "user", "content": "Count to five."}],
        "stream": true
      }'
```

## Status codes

| Code | When |
| --- | --- |
| `200` | Success |
| `400` | `messages` missing, empty, or not a list; unknown model |
| `502` | The upstream provider failed (see `GatewayResult.error`) |

A provider failure is reported as `502` rather than `500` because the gateway itself is healthy — the
upstream is not.

## Docker

### Build

```bash
docker build -t gpt6-sol-mcp-gateway:latest .
```

The image is multi-stage: a builder stage compiles the wheel, and a slim runtime stage installs it as a
non-root user (uid `10001`) with a `HEALTHCHECK` against `/health`.

### Run

```bash
docker run --rm -p 8080:8080 \
  -e LLM_API_KEY="sk-..." \
  -e LLM_MODEL_ID="gpt-6-sol" \
  gpt6-sol-mcp-gateway:latest
```

### Compose

```bash
docker compose up --build
```

`docker-compose.yml` defines two services:

| Service | Role |
| --- | --- |
| `gateway` | The FastAPI server on port 8080 |
| `mcp` | A stdio MCP server the gateway connects to |

Both declare healthchecks, and `gateway` waits for `mcp` to become healthy before starting.

## Reverse proxy notes

- Streaming uses `text/event-stream`. Disable response buffering in nginx
  (`proxy_buffering off;`) or the stream will arrive in one burst.
- Set generous read timeouts. A tool-calling round trip can take tens of seconds.
- The gateway does not terminate TLS. Put it behind a proxy that does.

## Production checklist

- [ ] `LLM_API_KEY` injected from a secret manager, never baked into the image
- [ ] `MCP_ALLOWED_TOOLS` restricted to the minimum set the workload needs
- [ ] An approver configured, or risky tools removed from the allowlist entirely
- [ ] `MCP_MAX_TOOL_ROUNDS` set low enough to bound cost
- [ ] Logs shipped to a collector that respects the built-in redaction
- [ ] `/health` wired to your orchestrator's liveness probe
