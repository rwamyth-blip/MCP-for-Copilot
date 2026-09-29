"""Scrub provider credentials from the test environment.

``tests/conftest.py`` isolates ``LLM_*`` / ``MCP_*`` / ``GATEWAY_*`` from the
developer's shell, but ``OPENAI_API_KEY`` matches none of those prefixes. A
developer who exported it -- a service wrapper, a Startup-folder shortcut, or
``setx`` -- had the real credential reach ``Settings``, which falls back to it.
The suite then failed while printing that credential into the log:

    assert settings.llm_api_key == ""
    E   AssertionError: assert 'sk-proj-...' == ''

Loading this as a plugin (``-p`` on the command line, or the ``pytest11``
entry point once installed) is used instead of editing conftest.py, which is
memory-mapped by another process on this host and cannot be rewritten.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

# Names the provider reads directly. Settings.llm_api_key falls back to the
# first non-empty one, so leaving any of them set makes the "no key" assertions
# read a live credential.
CREDENTIAL_ENV_VARS = (
    "OPENAI_API_KEY",
    "LLM_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY",
    "GROQ_API_KEY",
    "CODEX_API_KEY",
    "VIHOKAI_API_KEY",
)


@pytest.fixture(autouse=True)
def _scrub_provider_credentials(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    yield
