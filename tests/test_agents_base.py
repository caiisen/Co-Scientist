from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from co_scientist.agents.base import AgentContext
from co_scientist.agents.echo import EchoAgent
from co_scientist.agents.results import AgentResultKind
from co_scientist.config import (
    AgentModelConfig,
    AppConfig,
    LLMConfig,
    ProviderConfig,
    RuntimeConfig,
    SearchConfig,
)
from co_scientist.llm.client import LLMCallMetadata, LLMChatResult, LLMRouter
from co_scientist.memory import SQLiteStore, Task
from co_scientist.supervisor.metrics import MetricsSink
from co_scientist.utils.prompts import PromptTemplateStore


def make_config() -> AppConfig:
    return AppConfig(
        runtime=RuntimeConfig(
            max_ideas=5,
            max_matches_per_idea=2,
            worker_concurrency=1,
            request_timeout_seconds=30,
        ),
        search=SearchConfig(max_results=3),
        llm=LLMConfig(
            default_provider="cheap",
            providers={
                "cheap": ProviderConfig(chat_model="cheap-chat", embedding_model="cheap-embed"),
                "strong": ProviderConfig(chat_model="strong-chat", embedding_model="strong-embed"),
            },
            agents={
                "echo": AgentModelConfig(provider="strong"),
            },
        ),
    )


@pytest.mark.asyncio
async def test_echo_agent_executes_with_context(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "echo.txt").write_text("Echo: {text}", encoding="utf-8")

    async with SQLiteStore(tmp_path / "agent.sqlite") as store:
        session = await store.create_session("agent test")
        ctx = AgentContext(
            store=store,
            llm_router=LLMRouter(make_config().llm),
            config=make_config(),
            session_id=session.id,
            current_feedback="prefer testable mechanisms",
            prompt_store=PromptTemplateStore(prompt_dir),
        )
        task = Task(
            session_id=session.id,
            agent="echo",
            action="render",
            payload_json={"template": "echo", "text": "hello"},
        )

        result = await EchoAgent().execute(task, ctx)

    assert result.kind == AgentResultKind.NOOP
    assert result.ok
    assert result.payload == {"prompt": "Echo: hello", "agent": "echo"}
    assert result.raw_text == "Echo: hello"


@pytest.mark.asyncio
async def test_agent_build_messages_and_llm_routing(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "echo.txt").write_text("Echo: {text}", encoding="utf-8")
    config = make_config()

    async with SQLiteStore(tmp_path / "agent_routing.sqlite") as store:
        session = await store.create_session("agent routing")
        agent = EchoAgent()
        ctx = AgentContext(
            store=store,
            llm_router=LLMRouter(config.llm),
            config=config,
            session_id=session.id,
            current_feedback="review novelty carefully",
            prompt_store=PromptTemplateStore(prompt_dir),
        )

        messages = agent.build_messages(ctx, user_prompt="compare ideas")

        assert ctx.llm_for("echo").provider.chat_model == "strong-chat"
        assert messages == [
            {
                "role": "system",
                "content": (
                    agent.system_prompt
                    + "\n\n## Meta-review feedback for this run:\nreview novelty carefully"
                ),
            },
            {"role": "user", "content": "compare ideas"},
        ]

        messages_without_feedback = agent.build_messages(
            ctx,
            user_prompt="compare ideas",
            include_feedback=False,
        )
        assert messages_without_feedback == [
            {"role": "system", "content": agent.system_prompt},
            {"role": "user", "content": "compare ideas"},
        ]


class ConcurrentMetadataClient:
    last_call = None

    async def chat_with_metadata(self, messages, **kwargs):
        content = messages[-1]["content"]
        if content == "slow":
            await asyncio.sleep(0.02)
            metadata = LLMCallMetadata(model="slow-model", latency_seconds=0.02, total_tokens=11)
        else:
            metadata = LLMCallMetadata(model="fast-model", latency_seconds=0.001, total_tokens=3)
        self.last_call = metadata
        return LLMChatResult(text=content, metadata=metadata)


class ConcurrentRouter:
    def __init__(self) -> None:
        self.client = ConcurrentMetadataClient()

    def client_for(self, agent=None):
        return self.client


@pytest.mark.asyncio
async def test_agent_chat_metrics_use_call_local_metadata(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "agent_metrics.sqlite") as store:
        session = await store.create_session("agent metrics")
        agent = EchoAgent()
        ctx = AgentContext(
            store=store,
            llm_router=ConcurrentRouter(),
            config=make_config(),
            session_id=session.id,
            metrics_sink=MetricsSink(runs_dir=tmp_path / "runs"),
        )

        await asyncio.gather(
            agent.chat(ctx, [{"role": "user", "content": "slow"}]),
            agent.chat(ctx, [{"role": "user", "content": "fast"}]),
        )

    events = [
        line
        for line in (tmp_path / "runs" / session.id / "metrics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    assert any(
        '"model": "slow-model"' in event and '"total_tokens": 11' in event
        for event in events
    )
    assert any(
        '"model": "fast-model"' in event and '"total_tokens": 3' in event
        for event in events
    )
