from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from openai import APIConnectionError

from co_scientist.agents.base import AgentContext
from co_scientist.agents.proximity import ProximityAgent, cosine_similarity, lexical_embedding
from co_scientist.agents.ranking import RankingAgent, choose_pair
from co_scientist.agents.results import AgentResultKind
from co_scientist.config import AppConfig, LLMConfig, ProviderConfig, RuntimeConfig, SearchConfig
from co_scientist.llm.client import LLMRouter
from co_scientist.memory import Hypothesis, ResearchPlan, Review, SQLiteStore, Task


class Phase5Client:
    def __init__(self, chat_responses: list[str] | None = None) -> None:
        self.chat_responses = chat_responses or []
        self.messages: list[list[dict[str, str]]] = []

    async def chat(self, messages, **kwargs):
        self.messages.append(messages)
        if not self.chat_responses:
            raise AssertionError("unexpected chat call")
        return self.chat_responses.pop(0)

    async def embed(self, texts, **kwargs):
        return [_vector_for_text(text) for text in texts]


class FailingEmbedClient(Phase5Client):
    async def embed(self, texts, **kwargs):
        raise APIConnectionError(
            message="embedding endpoint unavailable",
            request=httpx.Request("POST", "https://example.test/embeddings"),
        )


class MissingEmbeddingModelClient(Phase5Client):
    async def embed(self, texts, **kwargs):
        raise ValueError("no embedding model configured for this provider")


class BuggyEmbedClient(Phase5Client):
    async def embed(self, texts, **kwargs):
        raise TypeError("programming bug")


class StaticRouter(LLMRouter):
    def __init__(self, client: Phase5Client) -> None:
        self.client = client

    def client_for(self, agent=None):
        return self.client


def make_config(*, max_matches_per_idea: int = 1) -> AppConfig:
    return AppConfig(
        runtime=RuntimeConfig(
            max_ideas=5,
            max_matches_per_idea=max_matches_per_idea,
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


def _vector_for_text(text: str) -> list[float]:
    if "A" in text:
        return [1.0, 0.0]
    if "B" in text:
        return [0.9, 0.1]
    return [0.0, 1.0]


def test_cosine_similarity() -> None:
    assert cosine_similarity([1, 0], [1, 0]) == 1
    assert round(cosine_similarity([1, 0], [0, 1]), 6) == 0


def test_lexical_embedding_is_deterministic() -> None:
    assert lexical_embedding("same text") == lexical_embedding("same text")
    assert len(lexical_embedding("same text")) == 128


@pytest.mark.asyncio
async def test_proximity_agent_updates_incremental_edges(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "proximity.sqlite") as store:
        session = await store.create_session("goal")
        first = await store.add_hypothesis(
            Hypothesis(session_id=session.id, content="A content", summary="A")
        )
        second = await store.add_hypothesis(
            Hypothesis(session_id=session.id, content="B content", summary="B")
        )
        assert first.id is not None
        assert second.id is not None
        ctx = AgentContext(
            store=store,
            llm_router=StaticRouter(Phase5Client()),
            config=make_config(),
            session_id=session.id,
        )

        result = await ProximityAgent().execute(
            Task(
                session_id=session.id,
                agent="proximity",
                action="update_proximity_graph",
                target_id=second.id,
            ),
            ctx,
        )
        edges = await store.proximity_edges_for_session(session.id)

    assert result.kind == AgentResultKind.PROXIMITY_UPDATED
    assert list(edges) == [(first.id, second.id)]
    assert edges[(first.id, second.id)] > 0.99


@pytest.mark.asyncio
async def test_proximity_agent_falls_back_when_embedding_provider_fails(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "proximity_fallback.sqlite") as store:
        session = await store.create_session("goal")
        first = await store.add_hypothesis(
            Hypothesis(session_id=session.id, content="A content", summary="A kinase")
        )
        second = await store.add_hypothesis(
            Hypothesis(session_id=session.id, content="B content", summary="B kinase")
        )
        assert first.id is not None
        assert second.id is not None
        ctx = AgentContext(
            store=store,
            llm_router=StaticRouter(FailingEmbedClient()),
            config=make_config(),
            session_id=session.id,
        )

        result = await ProximityAgent().execute(
            Task(
                session_id=session.id,
                agent="proximity",
                action="update_proximity_graph",
                target_id=second.id,
            ),
            ctx,
        )
        edges = await store.proximity_edges_for_session(session.id)

    assert result.ok
    assert result.payload["embedding_source"] == "lexical_fallback"
    assert "embedding endpoint unavailable" in result.payload["embedding_error"]
    assert list(edges) == [(first.id, second.id)]


@pytest.mark.asyncio
async def test_proximity_agent_falls_back_without_embedding_model(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "proximity_no_embedding_model.sqlite") as store:
        session = await store.create_session("goal")
        first = await store.add_hypothesis(
            Hypothesis(session_id=session.id, content="A content", summary="A kinase")
        )
        second = await store.add_hypothesis(
            Hypothesis(session_id=session.id, content="B content", summary="B kinase")
        )
        assert first.id is not None
        assert second.id is not None
        ctx = AgentContext(
            store=store,
            llm_router=StaticRouter(MissingEmbeddingModelClient()),
            config=make_config(),
            session_id=session.id,
        )

        result = await ProximityAgent().execute(
            Task(
                session_id=session.id,
                agent="proximity",
                action="update_proximity_graph",
            ),
            ctx,
        )
        edges = await store.proximity_edges_for_session(session.id)

    assert result.ok
    assert result.payload["embedding_source"] == "lexical_fallback"
    assert result.payload["embedding_error"] == "no embedding model configured for this provider"
    assert list(edges) == [(first.id, second.id)]


@pytest.mark.asyncio
async def test_proximity_agent_does_not_fallback_on_programming_errors(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "proximity_bug.sqlite") as store:
        session = await store.create_session("goal")
        await store.add_hypothesis(
            Hypothesis(session_id=session.id, content="A content", summary="A kinase")
        )
        ctx = AgentContext(
            store=store,
            llm_router=StaticRouter(BuggyEmbedClient()),
            config=make_config(),
            session_id=session.id,
        )

        with pytest.raises(TypeError, match="programming bug"):
            await ProximityAgent().execute(
                Task(
                    session_id=session.id,
                    agent="proximity",
                    action="update_proximity_graph",
                ),
                ctx,
            )


@pytest.mark.asyncio
async def test_ranking_agent_selects_weighted_pair_and_returns_decision(tmp_path: Path) -> None:
    client = Phase5Client(["Rationale.\nbetter idea: 2"])
    async with SQLiteStore(tmp_path / "ranking.sqlite") as store:
        session = await store.create_session("goal")
        await store.save_research_plan(
            ResearchPlan(session_id=session.id, goal=session.goal, preferences=["testable"])
        )
        first = await _reviewed_hypothesis(store, session.id, "A", elo=1300)
        second = await _reviewed_hypothesis(store, session.id, "B", elo=1290)
        third = await _reviewed_hypothesis(store, session.id, "C", elo=1100)
        assert first.id is not None and second.id is not None and third.id is not None
        await store.upsert_proximity_edge(
            session_id=session.id,
            hypo_a_id=first.id,
            hypo_b_id=second.id,
            similarity=0.95,
        )
        await store.upsert_proximity_edge(
            session_id=session.id,
            hypo_a_id=first.id,
            hypo_b_id=third.id,
            similarity=0.10,
        )
        ctx = AgentContext(
            store=store,
            llm_router=StaticRouter(client),
            config=make_config(),
            session_id=session.id,
        )

        pair = await choose_pair(ctx, [first, second, third])
        result = await RankingAgent().execute(
            Task(session_id=session.id, agent="ranking", action="run_tournament_match"),
            ctx,
        )

    assert pair is not None
    assert [pair.first.id, pair.second.id] == [first.id, second.id]
    assert result.kind == AgentResultKind.RANKING_DECISION
    assert result.payload["winner_id"] == second.id
    assert result.payload["mode"] == "debate"
    assert "structured discussion" in client.messages[0][-1]["content"]


async def _reviewed_hypothesis(
    store: SQLiteStore,
    session_id: str,
    summary: str,
    *,
    elo: int = 1200,
) -> Hypothesis:
    hypothesis = await store.add_hypothesis(
        Hypothesis(session_id=session_id, content=f"{summary} content", summary=summary, elo=elo)
    )
    assert hypothesis.id is not None
    await store.add_review(
        Review(
            session_id=session_id,
            hypothesis_id=hypothesis.id,
            type="full",
            score=7,
            content=f"{summary} review",
        )
    )
    return hypothesis
