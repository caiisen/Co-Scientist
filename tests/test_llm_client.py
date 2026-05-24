from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError

from co_scientist.config import AgentModelConfig, LLMConfig, ProviderConfig
from co_scientist.llm.client import LLMClient, LLMRouter


class FakeChatCompletions:
    def __init__(self) -> None:
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=f"model={kwargs['model']}"))
            ],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7),
        )


class FlakyChatCompletions:
    def __init__(self) -> None:
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise APIConnectionError(request=httpx.Request("POST", "https://example.test/v1"))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=None,
        )


class BuggyChatCompletions:
    def __init__(self) -> None:
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        raise RuntimeError("bug")


class FakeEmbeddings:
    async def create(self, **kwargs):
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[1.0, 0.0]),
                SimpleNamespace(embedding=[0.0, 1.0]),
            ],
            usage=None,
        )


class FakeAsyncOpenAI:
    def __init__(self, chat_completions=None) -> None:
        self.chat = SimpleNamespace(completions=chat_completions or FakeChatCompletions())
        self.embeddings = FakeEmbeddings()


@pytest.mark.asyncio
async def test_chat_returns_content_and_metadata() -> None:
    client = LLMClient(
        ProviderConfig(chat_model="test-chat", embedding_model="test-embed"),
        async_client=FakeAsyncOpenAI(),
    )

    content = await client.chat([{"role": "user", "content": "hello"}])

    assert content == "model=test-chat"
    assert client.last_call is not None
    assert client.last_call.model == "test-chat"
    assert client.last_call.total_tokens == 7


@pytest.mark.asyncio
async def test_embed_returns_vectors() -> None:
    client = LLMClient(
        ProviderConfig(chat_model="test-chat", embedding_model="test-embed"),
        async_client=FakeAsyncOpenAI(),
    )

    vectors = await client.embed(["a", "b"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert client.last_call is not None
    assert client.last_call.model == "test-embed"


@pytest.mark.asyncio
async def test_chat_retries_transient_errors() -> None:
    flaky = FlakyChatCompletions()
    client = LLMClient(
        ProviderConfig(chat_model="test-chat", embedding_model="test-embed"),
        async_client=FakeAsyncOpenAI(chat_completions=flaky),
    )

    assert await client.chat([{"role": "user", "content": "hello"}]) == "ok"
    assert flaky.calls == 2
    assert client.last_call is not None
    assert client.last_call.total_tokens is None


@pytest.mark.asyncio
async def test_chat_does_not_retry_non_openai_transient_errors() -> None:
    buggy = BuggyChatCompletions()
    client = LLMClient(
        ProviderConfig(chat_model="test-chat", embedding_model="test-embed"),
        async_client=FakeAsyncOpenAI(chat_completions=buggy),
    )

    with pytest.raises(RuntimeError, match="bug"):
        await client.chat([{"role": "user", "content": "hello"}])

    assert buggy.calls == 1


def test_router_applies_agent_override() -> None:
    config = LLMConfig(
        default_provider="cheap",
        providers={
            "cheap": ProviderConfig(chat_model="cheap-chat", embedding_model="cheap-embed"),
            "strong": ProviderConfig(chat_model="strong-chat", embedding_model="strong-embed"),
        },
        agents={
            "ranking": AgentModelConfig(provider="strong", temperature=0.1),
            "generation": AgentModelConfig(chat_model="creative-chat"),
        },
    )

    router = LLMRouter(config)

    assert router.client_for("ranking").provider.chat_model == "strong-chat"
    assert router.client_for("ranking").provider.temperature == 0.1
    assert router.client_for("generation").provider.chat_model == "creative-chat"
    assert router.client_for("reflection").provider.chat_model == "cheap-chat"
