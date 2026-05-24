from __future__ import annotations

from pathlib import Path

import pytest

from co_scientist.config import deep_merge, load_config


def write_yaml(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_deep_merge_preserves_nested_defaults() -> None:
    merged = deep_merge(
        {"runtime": {"max_ideas": 20, "worker_concurrency": 4}},
        {"runtime": {"max_ideas": 5}},
    )

    assert merged == {"runtime": {"max_ideas": 5, "worker_concurrency": 4}}


def test_load_config_applies_default_local_session_and_cli_order(tmp_path: Path) -> None:
    default_path = write_yaml(
        tmp_path / "default.yaml",
        """
runtime:
  max_ideas: 20
  max_matches_per_idea: 5
  worker_concurrency: 4
  request_timeout_seconds: 60
search:
  max_results: 5
llm:
  default_provider: openai
  providers:
    openai:
      chat_model: gpt-4o-mini
      embedding_model: text-embedding-3-small
      temperature: 0.4
  agents:
    generation:
      provider: openai
""",
    )
    local_path = write_yaml(
        tmp_path / "local.yaml",
        """
runtime:
  max_ideas: 10
llm:
  providers:
    openai:
      temperature: 0.2
""",
    )
    session_path = write_yaml(
        tmp_path / "session.yaml",
        """
runtime:
  max_matches_per_idea: 3
llm:
  agents:
    generation:
      chat_model: gpt-4o
""",
    )

    config = load_config(
        default_path=default_path,
        local_path=local_path,
        session_path=session_path,
        cli_overrides={"runtime.max_ideas": 7},
    )

    assert config.runtime.max_ideas == 7
    assert config.runtime.max_matches_per_idea == 3
    assert config.runtime.worker_concurrency == 4
    provider = config.llm.provider_for_agent("generation")
    assert provider.chat_model == "gpt-4o"
    assert provider.temperature == 0.2


def test_provider_resolves_environment_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_API_KEY", "secret")
    monkeypatch.setenv("TEST_BASE_URL", "https://example.test/v1")
    default_path = write_yaml(
        tmp_path / "default.yaml",
        """
runtime:
  max_ideas: 20
  max_matches_per_idea: 5
  worker_concurrency: 4
  request_timeout_seconds: 60
search:
  max_results: 5
llm:
  default_provider: test
  providers:
    test:
      api_key_env: TEST_API_KEY
      base_url_env: TEST_BASE_URL
      base_url: https://fallback.test/v1
      chat_model: test-chat
      embedding_model: test-embed
""",
    )

    config = load_config(default_path=default_path, local_path=tmp_path / "missing.yaml")
    provider = config.llm.provider_for_agent(None)

    assert provider.resolved_api_key == "secret"
    assert provider.resolved_base_url == "https://example.test/v1"


def test_unknown_agent_provider_raises(tmp_path: Path) -> None:
    default_path = write_yaml(
        tmp_path / "default.yaml",
        """
runtime:
  max_ideas: 20
  max_matches_per_idea: 5
  worker_concurrency: 4
  request_timeout_seconds: 60
search:
  max_results: 5
llm:
  default_provider: openai
  providers:
    openai:
      chat_model: gpt-4o-mini
  agents:
    ranking:
      provider: missing
""",
    )

    config = load_config(default_path=default_path, local_path=tmp_path / "missing.yaml")

    with pytest.raises(KeyError):
        config.llm.provider_for_agent("ranking")
