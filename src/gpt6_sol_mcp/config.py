"""Configuration — env-driven, no hardcoded secrets.

Every value has a safe default so the package imports cleanly with an empty
environment. A credential is never given a default: an unset key stays empty
and the layer that needs it fails loudly instead of silently using a literal.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from the environment or a ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- LLM provider -----------------------------------------------------
    llm_provider: str = Field(
        default="openai",
        description="Provider adapter to use. Only 'openai' is implemented.",
    )
    llm_model_id: str = Field(
        default="gpt-6-sol",
        description="Default model id or alias (e.g. 'gpt-6-sol', 'sol').",
    )
    llm_api_key: str = Field(
        default="",
        description="Provider API key. Never commit this.",
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI-compatible base URL (no trailing /chat/completions).",
    )
    llm_timeout_seconds: int = Field(default=120, ge=1)
    llm_reasoning_effort: str = Field(
        default="",
        description=(
            "Reasoning effort forwarded to the model. Empty means 'do not send "
            "the field'. Ignored (forced to 'none') whenever tools are present."
        ),
    )

    # -- MCP server -------------------------------------------------------
    mcp_server_url: str = Field(
        default="",
        description="MCP server URL for http/sse, or a command for stdio.",
    )
    mcp_transport: str = Field(
        default="stdio",
        description="One of: stdio, http, sse.",
    )
    mcp_auth_token: str = Field(default="", description="Bearer token for the MCP server.")
    mcp_require_approval: bool = Field(
        default=True,
        description="When True, risky tools need an explicit approval decision.",
    )
    mcp_allowed_tools_raw: str = Field(
        default="",
        description=(
            "Comma-separated tool allowlist. Empty means the read-only default "
            "set in tool_router.DEFAULT_ALLOWED_TOOLS is used."
        ),
    )
    mcp_max_tool_rounds: int = Field(
        default=5,
        ge=0,
        description="Cap on model -> tool -> model rounds per turn.",
    )
    mcp_timeout_seconds: int = Field(default=60, ge=1)

    # -- Gateway server ---------------------------------------------------
    gateway_host: str = Field(default="0.0.0.0")
    gateway_port: int = Field(default=8000, ge=1, le=65535)
    gateway_api_key: str = Field(
        default="",
        description=(
            "When set, callers must send 'Authorization: Bearer <key>'. "
            "When empty, any key is accepted (development only)."
        ),
    )
    gateway_cors_origins_raw: str = Field(
        default="*",
        description="Comma-separated allowed CORS origins.",
    )

    # -- validators -------------------------------------------------------
    @field_validator("mcp_transport")
    @classmethod
    def _check_transport(cls, value: str) -> str:
        allowed = ("stdio", "http", "sse")
        normalised = (value or "").strip().lower()
        if normalised not in allowed:
            raise ValueError(f"mcp_transport must be one of {', '.join(allowed)}; got {value!r}")
        return normalised

    # -- derived ----------------------------------------------------------
    @property
    def mcp_allowed_tools(self) -> tuple[str, ...]:
        """Parsed allowlist. Empty tuple means 'use the default set'."""
        return tuple(t.strip() for t in self.mcp_allowed_tools_raw.split(",") if t.strip())

    @property
    def gateway_cors_origins(self) -> list[str]:
        return [o.strip() for o in self.gateway_cors_origins_raw.split(",") if o.strip()]

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key.strip() and self.llm_model_id.strip())

    @property
    def mcp_configured(self) -> bool:
        return bool(self.mcp_server_url.strip())

    def describe(self) -> dict[str, object]:
        """Non-secret summary, safe to log or return over HTTP."""
        return {
            "llm_provider": self.llm_provider,
            "llm_model_id": self.llm_model_id,
            "llm_base_url": self.llm_base_url,
            "llm_api_key_present": bool(self.llm_api_key),
            "llm_reasoning_effort": self.llm_reasoning_effort or None,
            "mcp_transport": self.mcp_transport,
            "mcp_server_url": self.mcp_server_url,
            "mcp_configured": self.mcp_configured,
            "mcp_require_approval": self.mcp_require_approval,
            "mcp_max_tool_rounds": self.mcp_max_tool_rounds,
            "gateway_api_key_required": bool(self.gateway_api_key),
        }


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader used when pydantic-settings is not enough.

    ``setdefault`` semantics: a real environment variable always wins over the
    file, so CI secrets are never shadowed by a stale local ``.env``.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    _load_dotenv(Path.cwd() / ".env")
    return Settings()


def reset_settings_cache() -> None:
    """Clear the settings cache. Intended for tests."""
    get_settings.cache_clear()
