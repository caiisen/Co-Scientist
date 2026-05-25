from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from co_scientist.agents.base import AgentContext
from co_scientist.agents.generation import GenerationAgent
from co_scientist.agents.reflection import ReflectionAgent
from co_scientist.agents.results import AgentResultKind
from co_scientist.config import (
    AppConfig,
    LLMConfig,
    ProviderConfig,
    RuntimeConfig,
    SearchConfig,
)
from co_scientist.llm.client import LLMRouter
from co_scientist.memory import Hypothesis, ResearchPlan, SQLiteStore, Task
from co_scientist.tools.models import Citation, SearchDocument, ToolResult


class SequenceClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.messages: list[list[dict[str, str]]] = []

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        self.messages.append(messages)
        if not self.responses:
            raise AssertionError("unexpected chat call")
        return self.responses.pop(0)


class StaticRouter(LLMRouter):
    def __init__(self, client: SequenceClient) -> None:
        self.client = client

    def client_for(self, agent: str | None = None) -> SequenceClient:
        return self.client


def make_config() -> AppConfig:
    return AppConfig(
        runtime=RuntimeConfig(
            max_ideas=5,
            max_matches_per_idea=1,
            worker_concurrency=1,
            request_timeout_seconds=30,
        ),
        search=SearchConfig(
            max_results=1,
            semantic_scholar_enabled=False,
            tavily_enabled=False,
            arxiv_enabled=False,
        ),
        llm=LLMConfig(
            default_provider="test",
            providers={"test": ProviderConfig(chat_model="test-chat")},
        ),
    )


async def fake_literature_search(*args, **kwargs) -> ToolResult:
    assert "Propose a hypothesis" not in args[0]
    return ToolResult.from_documents(
        source="literature",
        documents=[
            SearchDocument(
                source="pubmed",
                title="Evidence paper",
                abstract_or_snippet="Evidence text.",
                citation=Citation(source="pubmed", title="Evidence paper", pmid="1"),
            )
        ],
    )


@pytest.mark.asyncio
async def test_generation_agent_creates_five_hypotheses(tmp_path: Path) -> None:
    client = SequenceClient(
        [
            "Reasoning\nHYPOTHESIS\nLiterature hypothesis 1",
            "Turn 1 discussion",
            "Turn 2 discussion",
            "Turn 3\nHYPOTHESIS\nDebate hypothesis",
            "Assumptions\nHYPOTHESIS\nAssumption hypothesis",
            "Expansion\nHYPOTHESIS\nExpansion hypothesis",
            "Reasoning\nHYPOTHESIS\nLiterature hypothesis 2",
        ]
    )
    config = make_config()
    async with SQLiteStore(tmp_path / "generation.sqlite") as store:
        session = await store.create_session("goal")
        await store.save_research_plan(
            ResearchPlan(session_id=session.id, goal=session.goal, preferences=["novel"])
        )
        ctx = AgentContext(
            store=store,
            llm_router=StaticRouter(client),
            config=config,
            session_id=session.id,
        )

        result = await GenerationAgent(literature_search=fake_literature_search).execute(
            Task(session_id=session.id, agent="generation", action="create_initial_hypotheses"),
            ctx,
        )

    assert result.kind == AgentResultKind.HYPOTHESIS_CREATED
    assert result.ok
    assert [item["source_strategy"] for item in result.payload["hypotheses"]] == [
        "literature_review",
        "scientific_debate",
        "iterative_assumptions",
        "research_expansion",
        "literature_review",
    ]
    assert result.payload["hypotheses"][0]["query_variant"] == "summary"
    assert result.payload["hypotheses"][4]["query_variant"] == "goal"
    assert all("citations" in item for item in result.payload["hypotheses"])
    assert len(result.citations) == 4
    assert len(client.messages) == 7


@pytest.mark.asyncio
async def test_generation_agent_skips_single_parse_failure(tmp_path: Path) -> None:
    client = SequenceClient(
        [
            "No final marker here.",
            "Turn 1 discussion",
            "Turn 2 discussion",
            "Turn 3\nHYPOTHESIS\nDebate hypothesis",
            "Assumptions\nHYPOTHESIS\nAssumption hypothesis",
            "Expansion\nHYPOTHESIS\nExpansion hypothesis",
            "Reasoning\nHYPOTHESIS\nLiterature hypothesis 2",
        ]
    )
    config = make_config()
    async with SQLiteStore(tmp_path / "generation_partial.sqlite") as store:
        session = await store.create_session("goal")
        await store.save_research_plan(ResearchPlan(session_id=session.id, goal=session.goal))
        ctx = AgentContext(
            store=store,
            llm_router=StaticRouter(client),
            config=config,
            session_id=session.id,
        )

        result = await GenerationAgent(literature_search=fake_literature_search).execute(
            Task(session_id=session.id, agent="generation", action="create_initial_hypotheses"),
            ctx,
        )

    assert result.ok
    assert len(result.payload["hypotheses"]) == 4
    assert result.payload["errors"][0]["strategy"] == "literature_review"
    assert result.payload["errors"][0]["query_variant"] == "summary"


@pytest.mark.asyncio
async def test_generation_agent_runs_initial_strategies_concurrently(tmp_path: Path) -> None:
    async def slow_literature_search(*args, **kwargs) -> ToolResult:
        await asyncio.sleep(0.05)
        return ToolResult(source="literature")

    class EchoClient:
        def __init__(self) -> None:
            self.messages: list[list[dict[str, str]]] = []

        async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
            self.messages.append(messages)
            await asyncio.sleep(0.05)
            prompt = messages[-1]["content"]
            if "You may now end with a final HYPOTHESIS." in prompt:
                return "HYPOTHESIS\nDebate hypothesis"
            if "Continue the discussion." in prompt:
                return "Discussion"
            return "HYPOTHESIS\nGenerated hypothesis"

    config = make_config()
    client = EchoClient()
    started = asyncio.Event()
    release = asyncio.Event()
    active_searches = 0
    max_active_searches = 0

    async def gated_literature_search(*args, **kwargs) -> ToolResult:
        nonlocal active_searches, max_active_searches
        active_searches += 1
        max_active_searches = max(max_active_searches, active_searches)
        started.set()
        await release.wait()
        active_searches -= 1
        return await slow_literature_search(*args, **kwargs)

    async with SQLiteStore(tmp_path / "generation_concurrent.sqlite") as store:
        session = await store.create_session("goal")
        await store.save_research_plan(ResearchPlan(session_id=session.id, goal=session.goal))
        ctx = AgentContext(
            store=store,
            llm_router=StaticRouter(client),
            config=config,
            session_id=session.id,
        )

        execution = asyncio.create_task(
            GenerationAgent(literature_search=gated_literature_search).execute(
                Task(
                    session_id=session.id,
                    agent="generation",
                    action="create_initial_hypotheses",
                ),
                ctx,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=0.1)
        await asyncio.sleep(0)
        release.set()
        result = await asyncio.wait_for(execution, timeout=1)

    assert result.ok
    assert len(result.payload["hypotheses"]) == 5
    assert max_active_searches > 1


@pytest.mark.asyncio
async def test_agents_infer_non_biomed_search_domain(tmp_path: Path) -> None:
    domains: list[str] = []

    async def recording_literature_search(*args, **kwargs) -> ToolResult:
        domains.append(kwargs["domain"])
        return ToolResult.from_documents(
            source="literature",
            documents=[
                SearchDocument(
                    source="arxiv",
                    title="Evidence paper",
                    citation=Citation(
                        source="arxiv",
                        title="Evidence paper",
                        arxiv_id="2601.00001",
                    ),
                )
            ],
        )

    config = make_config()
    client = SequenceClient(
        [
            "HYPOTHESIS\nGenerated",
            "turn one",
            "turn two",
            "HYPOTHESIS\nDebate",
            "HYPOTHESIS\nAssumption",
            "HYPOTHESIS\nExpansion",
            "HYPOTHESIS\nGenerated goal",
            "Review\nOverall score: 7/10",
        ]
    )
    async with SQLiteStore(tmp_path / "domain.sqlite") as store:
        session = await store.create_session("goal")
        await store.save_research_plan(
            ResearchPlan(
                session_id=session.id,
                goal="Develop a computer science algorithm for theorem proving",
                attributes=["computer science"],
            )
        )
        hypothesis = await store.add_hypothesis(
            Hypothesis(session_id=session.id, content="Hypothesis body", summary="Hypothesis")
        )
        assert hypothesis.id is not None
        ctx = AgentContext(
            store=store,
            llm_router=StaticRouter(client),
            config=config,
            session_id=session.id,
        )

        await GenerationAgent(literature_search=recording_literature_search).execute(
            Task(session_id=session.id, agent="generation", action="create_initial_hypotheses"),
            ctx,
        )
        await ReflectionAgent(literature_search=recording_literature_search).execute(
            Task(
                session_id=session.id,
                agent="reflection",
                action="full_review",
                target_id=hypothesis.id,
            ),
            ctx,
        )

    assert set(domains) == {"cs"}


@pytest.mark.asyncio
async def test_reflection_agent_returns_full_review(tmp_path: Path) -> None:
    client = SequenceClient(["Grounded review.\nOverall score: 7.5/10"])
    config = make_config()
    async with SQLiteStore(tmp_path / "reflection.sqlite") as store:
        session = await store.create_session("goal")
        await store.save_research_plan(
            ResearchPlan(session_id=session.id, goal=session.goal, preferences=["testable"])
        )
        hypothesis = await store.add_hypothesis(
            Hypothesis(session_id=session.id, content="Hypothesis body", summary="Hypothesis")
        )
        assert hypothesis.id is not None
        ctx = AgentContext(
            store=store,
            llm_router=StaticRouter(client),
            config=config,
            session_id=session.id,
        )

        result = await ReflectionAgent(literature_search=fake_literature_search).execute(
            Task(
                session_id=session.id,
                agent="reflection",
                action="full_review",
                target_id=hypothesis.id,
            ),
            ctx,
        )

    assert result.kind == AgentResultKind.REVIEW_COMPLETED
    assert result.ok
    assert result.payload["score"] == 7.5
    assert result.payload["type"] == "full"
    assert result.citations[0].pmid == "1"


@pytest.mark.asyncio
async def test_reflection_agent_keeps_review_without_parseable_score(tmp_path: Path) -> None:
    client = SequenceClient(["Grounded review without a final numeric score."])
    config = make_config()
    async with SQLiteStore(tmp_path / "reflection_no_score.sqlite") as store:
        session = await store.create_session("goal")
        await store.save_research_plan(ResearchPlan(session_id=session.id, goal=session.goal))
        hypothesis = await store.add_hypothesis(
            Hypothesis(session_id=session.id, content="Hypothesis body", summary="Hypothesis")
        )
        assert hypothesis.id is not None
        ctx = AgentContext(
            store=store,
            llm_router=StaticRouter(client),
            config=config,
            session_id=session.id,
        )

        result = await ReflectionAgent(literature_search=fake_literature_search).execute(
            Task(
                session_id=session.id,
                agent="reflection",
                action="full_review",
                target_id=hypothesis.id,
            ),
            ctx,
        )

    assert result.ok
    assert result.payload["score"] is None
