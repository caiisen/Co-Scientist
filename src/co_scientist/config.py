from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "config" / "local.yaml"


class RuntimeConfig(BaseModel):
    initial_ideas: int = Field(default=5, gt=0)
    max_ideas: int = Field(gt=0)
    max_matches_per_idea: int = Field(gt=0)
    worker_concurrency: int = Field(gt=0)
    request_timeout_seconds: int = Field(gt=0)
    elo_stagnation_threshold: float = Field(default=5.0, ge=0)
    elo_stagnation_window: int = Field(default=4, ge=2)


class SearchConfig(BaseModel):
    tavily_enabled: bool = True
    pubmed_enabled: bool = True
    semantic_scholar_enabled: bool = True
    arxiv_enabled: bool = True
    max_results: int = Field(default=5, gt=0)
    cache_ttl_seconds: int = Field(default=604800, ge=0)
    failed_cache_ttl_seconds: int = Field(default=300, ge=0)
    tavily_api_key_env: str = "TAVILY_API_KEY"
    ncbi_api_key_env: str = "NCBI_API_KEY"
    ncbi_email_env: str = "NCBI_EMAIL"
    semantic_scholar_api_key_env: str = "SEMANTIC_SCHOLAR_API_KEY"
    tavily_max_results: int | None = Field(default=None, gt=0)
    pubmed_max_results: int | None = Field(default=None, gt=0)
    semantic_scholar_max_results: int | None = Field(default=None, gt=0)
    arxiv_max_results: int | None = Field(default=None, gt=0)
    openalex_enabled: bool = True
    private_corpus_enabled: bool = False
    private_corpus_paths: list[str] = Field(default_factory=list)
    private_corpus_max_results: int = Field(default=3, gt=0)
    private_corpus_chunk_chars: int = Field(default=1600, gt=0)
    private_corpus_chunk_overlap: int = Field(default=200, ge=0)

    @field_validator("private_corpus_chunk_overlap")
    @classmethod
    def require_overlap_smaller_than_chunk(cls, value: int, info) -> int:
        chunk_chars = info.data.get("private_corpus_chunk_chars")
        if chunk_chars is not None and value >= chunk_chars:
            raise ValueError("private_corpus_chunk_overlap must be smaller than chunk size")
        return value


class ObservabilityConfig(BaseModel):
    metrics_enabled: bool = True
    runs_dir: str = "runs"


class ProviderConfig(BaseModel):
    api_key_env: str | None = None
    api_key: str | None = None
    base_url_env: str | None = None
    base_url: str | None = None
    chat_model: str
    embedding_model: str | None = None
    temperature: float = Field(default=0.4, ge=0, le=2)
    max_tokens: int | None = Field(default=4096, gt=0)

    @property
    def resolved_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        return None

    @property
    def resolved_base_url(self) -> str | None:
        if self.base_url_env and os.getenv(self.base_url_env):
            return os.getenv(self.base_url_env)
        return self.base_url


class AgentModelConfig(BaseModel):
    provider: str | None = None
    chat_model: str | None = None
    embedding_model: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)


class LLMConfig(BaseModel):
    default_provider: str
    providers: dict[str, ProviderConfig]
    agents: dict[str, AgentModelConfig] = Field(default_factory=dict)

    @field_validator("providers")
    @classmethod
    def require_provider_names(cls, value: dict[str, ProviderConfig]) -> dict[str, ProviderConfig]:
        if not value:
            raise ValueError("at least one LLM provider must be configured")
        return value

    def provider_for_agent(self, agent: str | None) -> ProviderConfig:
        provider_name = self.default_provider
        agent_override = self.agents.get(agent or "")
        if agent_override and agent_override.provider:
            provider_name = agent_override.provider
        try:
            provider = self.providers[provider_name]
        except KeyError as exc:
            raise KeyError(f"unknown LLM provider '{provider_name}' for agent '{agent}'") from exc

        if not agent_override:
            return provider

        data = provider.model_dump()
        for field_name in ("chat_model", "embedding_model", "temperature", "max_tokens"):
            override = getattr(agent_override, field_name)
            if override is not None:
                data[field_name] = override
        return ProviderConfig(**data)


class AppConfig(BaseModel):
    runtime: RuntimeConfig
    search: SearchConfig
    llm: LLMConfig
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)


def load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"configuration file must contain a mapping: {path}")
    return loaded


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def set_nested_value(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = data
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        next_value = current.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"cannot set nested override through non-mapping key '{part}'")
        current = next_value
    current[parts[-1]] = value


def load_config(
    *,
    default_path: Path = DEFAULT_CONFIG_PATH,
    local_path: Path = LOCAL_CONFIG_PATH,
    session_path: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> AppConfig:
    raw = load_yaml_file(default_path)
    raw = deep_merge(raw, load_yaml_file(local_path))
    if session_path:
        raw = deep_merge(raw, load_yaml_file(session_path))
    for key, value in (cli_overrides or {}).items():
        if value is not None:
            set_nested_value(raw, key, value)
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"invalid configuration: {exc}") from exc
