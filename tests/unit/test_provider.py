"""Unit tests for the LLM provider adapter."""

from __future__ import annotations

from typing import ClassVar

import httpx
import pytest

from gpt6_sol_mcp.provider import (
    KNOWN_MODELS,
    MODEL_ALIASES,
    LLMProvider,
    LLMProviderError,
    ProviderResponse,
    ToolCall,
    aliases_for,
    resolve_model_id,
)

from ..conftest import make_completion, make_tool_call, mock_transport


class TestResolveModelId:
    @pytest.mark.parametrize(
        ("alias", "expected"),
        [
            ("gpt-6", "gpt-6-sol"),
            ("gpt6", "gpt-6-sol"),
            ("sol", "gpt-6-sol"),
            ("luna", "gpt-6-luna"),
            ("astra", "gpt-6-astra"),
            ("gpt-5.6", "gpt-5.6-sol"),
        ],
    )
    def test_aliases_resolve(self, alias: str, expected: str) -> None:
        assert resolve_model_id(alias) == expected

    def test_canonical_id_passes_through(self) -> None:
        assert resolve_model_id("gpt-6-luna") == "gpt-6-luna"

    def test_case_insensitive(self) -> None:
        assert resolve_model_id("SOL") == "gpt-6-sol"

    def test_unknown_id_raises(self) -> None:
        with pytest.raises(LLMProviderError, match="Unknown model id"):
            resolve_model_id("gpt-9-imaginary")

    def test_empty_id_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from gpt6_sol_mcp.config import Settings, reset_settings_cache

        reset_settings_cache()
        monkeypatch.setattr(
            "gpt6_sol_mcp.provider.get_settings",
            lambda: Settings(llm_model_id=""),
        )
        with pytest.raises(LLMProviderError, match="No model configured"):
            resolve_model_id(None)

    def test_aliases_for_returns_all(self) -> None:
        assert aliases_for("gpt-6-sol") == ["gpt-6", "gpt6", "sol"]


class TestModelCatalog:
    def test_every_alias_targets_a_known_model(self) -> None:
        for alias, target in MODEL_ALIASES.items():
            assert target in KNOWN_MODELS, f"{alias} -> {target} is not in KNOWN_MODELS"

    def test_gpt6_sol_metadata(self) -> None:
        spec = KNOWN_MODELS["gpt-6-sol"]
        assert spec["context_window"] == 1_050_000
        assert spec["max_output"] == 128_000
        assert "none" in spec["reasoning_effort"]

    def test_gpt56_luna_pricing_is_not_gpt6_luna_pricing(self) -> None:
        # A copy-paste bug here would silently misreport cost.
        assert KNOWN_MODELS["gpt-5.6-luna"]["input_price_per_mtok"] == 0.2
        assert KNOWN_MODELS["gpt-6-luna"]["input_price_per_mtok"] == 0.1


class TestToolCallParsing:
    def test_parses_json_string_arguments(self) -> None:
        call = ToolCall.from_openai(make_tool_call("read_file", {"path": "a.txt"}))
        assert call.name == "read_file"
        assert call.arguments == {"path": "a.txt"}

    def test_parses_dict_arguments(self) -> None:
        call = ToolCall.from_openai({"id": "c1", "function": {"name": "x", "arguments": {"a": 1}}})
        assert call.arguments == {"a": 1}

    def test_malformed_json_is_marked_not_raised(self) -> None:
        call = ToolCall.from_openai(
            {"id": "c1", "function": {"name": "x", "arguments": "{not json"}}
        )
        assert "__malformed_arguments__" in call.arguments

    def test_non_object_json_is_marked(self) -> None:
        call = ToolCall.from_openai({"id": "c1", "function": {"name": "x", "arguments": "[1,2]"}})
        assert "__malformed_arguments__" in call.arguments

    def test_empty_arguments_becomes_empty_dict(self) -> None:
        call = ToolCall.from_openai({"id": "c1", "function": {"name": "x"}})
        assert call.arguments == {}


class TestProviderResponse:
    def test_wants_tools_true_with_calls(self) -> None:
        response = ProviderResponse(tool_calls=[ToolCall(id="1", name="x")])
        assert response.wants_tools is True

    def test_wants_tools_false_without_calls(self) -> None:
        assert ProviderResponse(content="hi").wants_tools is False


class TestLLMProviderInit:
    def test_rejects_unsupported_provider(self, settings) -> None:
        with pytest.raises(LLMProviderError, match="Unsupported LLM_PROVIDER"):
            LLMProvider(provider="anthropic", api_key="k", model_id="gpt-6-sol")

    def test_configured_requires_key(self, settings) -> None:
        assert LLMProvider(api_key="sk-x", model_id="gpt-6-sol").configured is True
        assert LLMProvider(api_key="", model_id="gpt-6-sol").configured is False

    def test_describe_never_leaks_the_key(self) -> None:
        provider = LLMProvider(api_key="sk-super-secret", model_id="gpt-6-sol")
        described = provider.describe()
        assert described["api_key_present"] is True
        assert "sk-super-secret" not in str(described)

    def test_base_url_trailing_slash_is_stripped(self) -> None:
        provider = LLMProvider(
            api_key="k", base_url="https://api.example.test/v1/", model_id="gpt-6-sol"
        )
        assert provider.base_url == "https://api.example.test/v1"


class TestComplete:
    async def test_missing_key_raises(self) -> None:
        provider = LLMProvider(api_key="", model_id="gpt-6-sol")
        with pytest.raises(LLMProviderError, match="LLM_API_KEY is not configured"):
            await provider.complete([{"role": "user", "content": "hi"}])

    async def test_plain_completion(self) -> None:
        transport = mock_transport(make_completion(content="Hello there"))
        provider = LLMProvider(api_key="sk-test", model_id="gpt-6-sol", transport=transport)
        result = await provider.complete([{"role": "user", "content": "hi"}])
        assert result.content == "Hello there"
        assert result.wants_tools is False
        assert result.usage["total_tokens"] == 15

    async def test_tool_calls_are_parsed(self) -> None:
        payload = make_completion(
            content="",
            tool_calls=[make_tool_call("read_file", {"path": "a.txt"})],
            finish_reason="tool_calls",
        )
        provider = LLMProvider(
            api_key="sk-test", model_id="gpt-6-sol", transport=mock_transport(payload)
        )
        result = await provider.complete([{"role": "user", "content": "read a.txt"}])
        assert result.wants_tools is True
        assert result.tool_calls[0].name == "read_file"

    async def test_reasoning_effort_forced_to_none_with_tools(self) -> None:
        captured: list[dict] = []
        transport = mock_transport(make_completion(), capture=captured)
        provider = LLMProvider(api_key="sk-test", model_id="gpt-6-sol", transport=transport)
        await provider.complete(
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "x", "parameters": {}}}],
            reasoning_effort="high",
        )
        assert captured[0]["reasoning_effort"] == "none"

    async def test_reasoning_effort_forwarded_without_tools(self) -> None:
        captured: list[dict] = []
        transport = mock_transport(make_completion(), capture=captured)
        provider = LLMProvider(api_key="sk-test", model_id="gpt-6-sol", transport=transport)
        await provider.complete([{"role": "user", "content": "hi"}], reasoning_effort="high")
        assert captured[0]["reasoning_effort"] == "high"

    async def test_reasoning_effort_omitted_when_unset(self) -> None:
        captured: list[dict] = []
        transport = mock_transport(make_completion(), capture=captured)
        provider = LLMProvider(api_key="sk-test", model_id="gpt-6-sol", transport=transport)
        await provider.complete([{"role": "user", "content": "hi"}])
        assert "reasoning_effort" not in captured[0]

    async def test_unsupported_effort_raises(self) -> None:
        provider = LLMProvider(
            api_key="sk-test", model_id="gpt-6-sol", transport=mock_transport(make_completion())
        )
        with pytest.raises(LLMProviderError, match="not supported by"):
            await provider.complete([{"role": "user", "content": "hi"}], reasoning_effort="turbo")

    async def test_http_error_is_redacted(self) -> None:
        transport = mock_transport(
            {"error": {"message": "bad key sk-abcdef1234567890abcdef"}}, status_code=401
        )
        provider = LLMProvider(api_key="sk-test", model_id="gpt-6-sol", transport=transport)
        with pytest.raises(LLMProviderError) as excinfo:
            await provider.complete([{"role": "user", "content": "hi"}])
        assert "HTTP 401" in str(excinfo.value)
        assert "sk-abcdef1234567890abcdef" not in str(excinfo.value)

    async def test_no_choices_raises(self) -> None:
        provider = LLMProvider(
            api_key="sk-test",
            model_id="gpt-6-sol",
            transport=mock_transport({"choices": []}),
        )
        with pytest.raises(LLMProviderError, match="no choices"):
            await provider.complete([{"role": "user", "content": "hi"}])

    async def test_timeout_raises_provider_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        provider = LLMProvider(
            api_key="sk-test",
            model_id="gpt-6-sol",
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(LLMProviderError, match="timed out"):
            await provider.complete([{"role": "user", "content": "hi"}])

    async def test_max_tokens_uses_max_completion_tokens(self) -> None:
        captured: list[dict] = []
        provider = LLMProvider(
            api_key="sk-test",
            model_id="gpt-6-sol",
            transport=mock_transport(make_completion(), capture=captured),
        )
        await provider.complete([{"role": "user", "content": "hi"}], max_tokens=64)
        assert captured[0]["max_completion_tokens"] == 64


class TestToOpenAITools:
    def test_converts_mcp_tools(self, fake_tools) -> None:
        converted = LLMProvider.to_openai_tools(fake_tools)
        assert len(converted) == 2
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "read_file"
        assert converted[0]["function"]["parameters"]["required"] == ["path"]

    def test_skips_nameless_tools(self) -> None:
        class Nameless:
            name = ""
            description = "x"
            input_schema: ClassVar[dict] = {}

        assert LLMProvider.to_openai_tools([Nameless()]) == []

    def test_missing_schema_defaults_to_empty_object(self) -> None:
        class Bare:
            name = "bare"
            description = ""
            input_schema = None

        converted = LLMProvider.to_openai_tools([Bare()])
        assert converted[0]["function"]["parameters"] == {"type": "object", "properties": {}}
