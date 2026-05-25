from __future__ import annotations

from pathlib import Path

import pytest

from co_scientist.agents.base import Agent
from co_scientist.agents.proximity import ProximityAgent
from co_scientist.agents.ranking import RankingAgent
from co_scientist.agents.results import AgentResult, AgentResultKind
from co_scientist.config import AppConfig, LLMConfig, ProviderConfig, RuntimeConfig, SearchConfig
from co_scientist.llm.client import LLMRouter
from co_scientist.memory import SQLiteStore, Task
from co_scientist.supervisor import Supervisor


class Phase5Client:
    def __init__(self) -> None:
        self.chat_calls = 0

    async def chat(self, messages, **kwargs):
        self.chat_calls += 1
        if self.chat_calls == 1:
            return (
                '{"preferences":["specific"],"attributes":[],'
                '"constraints":[],"idea_attributes":[]}'
            )
        return "Ranking rationale.\nbetter idea: 1"

    async def embed(self, texts, **kwargs):
        vectors = []
        for text in texts:
            if "H1" in text:
                vectors.append([1.0, 0.0])
            elif "H2" in text:
                vectors.append([0.9, 0.1])
            else:
                vectors.append([0.0, 1.0])
        return vectors


class StaticRouter(LLMRouter):
    def __init__(self, client: Phase5Client) -> None:
        self.client = client

    def client_for(self, agent=None):
        return self.client


class StubGenerationAgent(Agent):
    name = "generation"

    async def execute(self, task: Task, ctx) -> AgentResult:
        return AgentResult(
            kind=AgentResultKind.HYPOTHESIS_CREATED,
            payload={
                "hypotheses": [
                    {"content": "H1 content", "summary": "H1", "source_strategy": "test"},
                    {"content": "H2 content", "summary": "H2", "source_strategy": "test"},
                    {"content": "H3 content", "summary": "H3", "source_strategy": "test"},
                ]
            },
        )


class StubReflectionAgent(Agent):
    name = "reflection"

    async def execute(self, task: Task, ctx) -> AgentResult:
        assert task.target_id is not None
        return AgentResult(
            kind=AgentResultKind.REVIEW_COMPLETED,
            payload={
                "hypothesis_id": task.target_id,
                "type": "full",
                "score": 7,
                "content": f"Review for {task.target_id}",
            },
        )


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
            providers={
                "test": ProviderConfig(
                    chat_model="test-chat",
                    embedding_model="test-embed",
                )
            },
        ),
    )


@pytest.mark.asyncio
async def test_supervisor_runs_phase5_tournament_to_match_target(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "supervisor_phase5.sqlite") as store:
        supervisor = Supervisor(
            store=store,
            config=make_config(),
            llm_router=StaticRouter(Phase5Client()),
            agents={
                "generation": StubGenerationAgent(),
                "reflection": StubReflectionAgent(),
                "proximity": ProximityAgent(),
                "ranking": RankingAgent(),
            },
        )

        session_id = await supervisor.start("goal")
        markdown = await supervisor.export_markdown(session_id)

        assert await store.count_hypotheses(session_id) == 3
        assert await store.count_reviews(session_id) == 3
        assert await store.count_matches(session_id) == 3
        assert len(await store.proximity_edges_for_session(session_id)) == 3
        async with store.db.execute(
            """
            SELECT COUNT(*) AS count
            FROM tasks
            WHERE agent = 'proximity' AND action = 'update_proximity_graph'
            """
        ) as cursor:
            proximity_task_count = (await cursor.fetchone())["count"]
        assert proximity_task_count == 1
        async with store.db.execute(
            """
            SELECT target_id, result_json
            FROM tasks
            WHERE agent = 'ranking' AND action = 'run_tournament_match'
            ORDER BY id
            LIMIT 1
            """
        ) as cursor:
            ranking_task = await cursor.fetchone()
        assert ranking_task is not None
        assert ranking_task["target_id"] is None
        assert '"match_id"' in ranking_task["result_json"]
        assert '"winner_id"' in ranking_task["result_json"]
        assert "## Tournament Summary" in markdown
        assert "Matches: 3" in markdown


@pytest.mark.asyncio
async def test_supervisor_redacts_direct_api_keys_in_session_config(tmp_path: Path) -> None:
    config = make_config()
    provider = config.llm.providers["test"]
    config.llm.providers["test"] = provider.model_copy(update={"api_key": "plain-secret"})
    async with SQLiteStore(tmp_path / "supervisor_redacted.sqlite") as store:
        supervisor = Supervisor(
            store=store,
            config=config,
            llm_router=StaticRouter(Phase5Client()),
            agents={
                "generation": StubGenerationAgent(),
                "reflection": StubReflectionAgent(),
                "proximity": ProximityAgent(),
                "ranking": RankingAgent(),
            },
        )

        session_id = await supervisor.start("goal")
        session = await store.get_session(session_id)

    assert session is not None
    assert session.config_json["llm"]["providers"]["test"]["api_key"] == "<redacted>"


@pytest.mark.asyncio
async def test_supervisor_verbose_reports_phase_task_input_and_output(tmp_path: Path) -> None:
    events: list[str] = []
    async with SQLiteStore(tmp_path / "supervisor_verbose.sqlite") as store:
        supervisor = Supervisor(
            store=store,
            config=make_config(),
            llm_router=StaticRouter(Phase5Client()),
            agents={
                "generation": StubGenerationAgent(),
                "reflection": StubReflectionAgent(),
                "proximity": ProximityAgent(),
                "ranking": RankingAgent(),
            },
            verbose=True,
            event_sink=events.append,
        )

        await supervisor.start("goal")

    joined = "\n".join(events)
    assert "[phase:start]" in joined
    assert "[phase:planner:output]" in joined
    assert "[task:start]" in joined
    assert "input={}" in joined
    assert "[task:output]" in joined
    assert "hypothesis_created" in joined
    assert "[queue] enqueued ranking.run_tournament_match" in joined
    assert "[phase:done]" in joined
