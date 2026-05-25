from __future__ import annotations

from pathlib import Path

import pytest

from co_scientist.agents.base import AgentContext
from co_scientist.config import AppConfig, LLMConfig, ProviderConfig, RuntimeConfig, SearchConfig
from co_scientist.memory import SQLiteStore
from co_scientist.supervisor.planner import create_research_plan


class PlannerClient:
    async def chat(self, messages, **kwargs):
        return """
        {
          "preferences": ["novel"],
          "attributes": ["mechanistic"],
          "constraints": ["safe"],
          "idea_attributes": ["testable"]
        }
        """


class PlannerRouter:
    def client_for(self, agent=None):
        return PlannerClient()


def make_config() -> AppConfig:
    return AppConfig(
        runtime=RuntimeConfig(
            max_ideas=5,
            max_matches_per_idea=1,
            worker_concurrency=1,
            request_timeout_seconds=30,
        ),
        search=SearchConfig(max_results=1),
        llm=LLMConfig(
            default_provider="test",
            providers={"test": ProviderConfig(chat_model="test-chat")},
        ),
    )


@pytest.mark.asyncio
async def test_create_research_plan_persists_json_plan(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "planner.sqlite") as store:
        session = await store.create_session("goal")
        ctx = AgentContext(
            store=store,
            llm_router=PlannerRouter(),
            config=make_config(),
            session_id=session.id,
        )

        plan = await create_research_plan("goal", ctx)
        loaded = await store.get_research_plan(session.id)

    assert plan.preferences == ["novel"]
    assert loaded == plan
