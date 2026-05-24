from __future__ import annotations

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
from co_scientist.llm.client import LLMRouter
from co_scientist.memory import SQLiteStore, Task
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
            {"role": "system", "content": agent.system_prompt},
            {
                "role": "system",
                "content": "Meta-review feedback for this run:\nreview novelty carefully",
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
