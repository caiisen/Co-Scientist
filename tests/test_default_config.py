from __future__ import annotations

from co_scientist.config import load_config


def test_default_config_agent_routing_follows_default_provider(tmp_path) -> None:
    local_path = tmp_path / "local.yaml"
    local_path.write_text(
        """
llm:
  default_provider: deepseek
  providers:
    deepseek:
      chat_model: deepseek-v4-pro
""",
        encoding="utf-8",
    )

    config = load_config(local_path=local_path)

    assert config.llm.provider_for_agent("generation").chat_model == "deepseek-v4-pro"
    assert config.llm.provider_for_agent("reflection").chat_model == "deepseek-v4-pro"
