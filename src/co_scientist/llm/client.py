from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from co_scientist.config import LLMConfig, ProviderConfig

ChatMessage = dict[str, str]
RETRYABLE_OPENAI_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)


@dataclass(frozen=True)
class LLMCallMetadata:
    model: str
    latency_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class LLMChatResult:
    text: str
    metadata: LLMCallMetadata


@dataclass(frozen=True)
class LLMEmbeddingResult:
    vectors: list[list[float]]
    metadata: LLMCallMetadata


class LLMClient:
    def __init__(
        self,
        provider: ProviderConfig,
        *,
        async_client: AsyncOpenAI | None = None,
    ) -> None:
        self.provider = provider
        _validate_ascii_secret(
            provider.resolved_api_key,
            provider.api_key_env or "provider.api_key",
        )
        _validate_ascii_secret(
            provider.resolved_base_url,
            provider.base_url_env or "provider.base_url",
        )
        self._client = async_client or AsyncOpenAI(
            api_key=provider.resolved_api_key or "missing-api-key",
            base_url=provider.resolved_base_url,
        )
        self.last_call: LLMCallMetadata | None = None

    @retry(
        retry=retry_if_exception_type(RETRYABLE_OPENAI_ERRORS),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        return (await self.chat_with_metadata(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )).text

    @retry(
        retry=retry_if_exception_type(RETRYABLE_OPENAI_ERRORS),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def chat_with_metadata(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMChatResult:
        selected_model = model or self.provider.chat_model
        started = time.monotonic()
        response = await self._client.chat.completions.create(
            model=selected_model,
            messages=messages,
            temperature=self.provider.temperature if temperature is None else temperature,
            max_tokens=self.provider.max_tokens if max_tokens is None else max_tokens,
            **kwargs,
        )
        metadata = self._metadata_from_response(selected_model, started, response)
        self.last_call = metadata
        content = response.choices[0].message.content
        return LLMChatResult(text=content or "", metadata=metadata)

    @retry(
        retry=retry_if_exception_type(RETRYABLE_OPENAI_ERRORS),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        return (await self.embed_with_metadata(texts, model=model, **kwargs)).vectors

    @retry(
        retry=retry_if_exception_type(RETRYABLE_OPENAI_ERRORS),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def embed_with_metadata(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMEmbeddingResult:
        selected_model = model or self.provider.embedding_model
        if not selected_model:
            raise ValueError("no embedding model configured for this provider")
        started = time.monotonic()
        response = await self._client.embeddings.create(
            model=selected_model,
            input=texts,
            **kwargs,
        )
        metadata = self._metadata_from_response(selected_model, started, response)
        self.last_call = metadata
        return LLMEmbeddingResult(
            vectors=[item.embedding for item in response.data],
            metadata=metadata,
        )

    def _metadata_from_response(
        self,
        model: str,
        started: float,
        response: Any,
    ) -> LLMCallMetadata:
        usage = getattr(response, "usage", None)
        return LLMCallMetadata(
            model=model,
            latency_seconds=time.monotonic() - started,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
        )


class LLMRouter:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._clients: dict[tuple[str | None, str, str | None, str | None], LLMClient] = {}

    def client_for(self, agent: str | None = None) -> LLMClient:
        provider = self.config.provider_for_agent(agent)
        key = (
            agent,
            provider.chat_model,
            provider.embedding_model,
            provider.resolved_base_url,
        )
        if key not in self._clients:
            self._clients[key] = LLMClient(provider)
        return self._clients[key]


def _validate_ascii_secret(value: str | None, name: str) -> None:
    if value is None:
        return
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(
            f"{name} contains non-ASCII characters. Re-enter it with plain ASCII quotes; "
            "smart quotes such as “...” are not valid in API keys or base URLs."
        ) from exc
