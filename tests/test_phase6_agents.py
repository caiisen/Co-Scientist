from __future__ import annotations

from pathlib import Path

import pytest

from co_scientist.agents.base import AgentContext
from co_scientist.agents.evolution import EvolutionAgent
from co_scientist.agents.metareview import MetaReviewAgent
from co_scientist.agents.results import AgentResultKind
from co_scientist.config import AppConfig, LLMConfig, ProviderConfig, RuntimeConfig, SearchConfig
from co_scientist.llm.client import LLMRouter
from co_scientist.memory import (
    Hypothesis,
    Match,
    ResearchPlan,
    Review,
    SQLiteStore,
    SystemFeedback,
    Task,
)
from co_scientist.tools.models import Citation, SearchDocument, ToolResult


class SequenceClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.messages: list[list[dict[str, str]]] = []

    async def chat(self, messages, **kwargs):
        self.messages.append(messages)
        if not self.responses:
            raise AssertionError("unexpected chat call")
        return self.responses.pop(0)


class StaticRouter(LLMRouter):
    def __init__(self, client: SequenceClient) -> None:
        self.client = client

    def client_for(self, agent=None):
        return self.client


def make_config(*, max_ideas: int = 8) -> AppConfig:
    return AppConfig(
        runtime=RuntimeConfig(
            max_ideas=max_ideas,
            max_matches_per_idea=10,
            worker_concurrency=1,
            request_timeout_seconds=30,
        ),
        search=SearchConfig(max_results=1),
        llm=LLMConfig(
            default_provider="test",
            providers={"test": ProviderConfig(chat_model="test-chat")},
        ),
    )


async def fake_literature_search(*args, **kwargs) -> ToolResult:
    return ToolResult.from_documents(
        source="literature",
        documents=[
            SearchDocument(
                source="pubmed",
                title="Grounding paper",
                citation=Citation(source="pubmed", title="Grounding paper", pmid="1"),
            )
        ],
    )


@pytest.mark.asyncio
async def test_evolution_agent_respects_capacity_and_preserves_parent_ids(
    tmp_path: Path,
) -> None:
    client = SequenceClient(
        [
            "Rationale\nHYPOTHESIS\nGrounded child",
            "Rationale\nHYPOTHESIS\nFeasible child",
            "Rationale\nHYPOTHESIS\nInspired child",
        ]
    )
    async with SQLiteStore(tmp_path / "evolution.sqlite") as store:
        session = await store.create_session("goal")
        await store.save_research_plan(
            ResearchPlan(session_id=session.id, goal=session.goal, preferences=["testable"])
        )
        parents = []
        for index, elo in enumerate([1500, 1400, 1300, 1200, 1100], start=1):
            parents.append(
                await store.add_hypothesis(
                    Hypothesis(
                        session_id=session.id,
                        content=f"Parent {index}",
                        summary=f"P{index}",
                        elo=elo,
                    )
                )
            )
        ctx = AgentContext(
            store=store,
            llm_router=StaticRouter(client),
            config=make_config(max_ideas=8),
            session_id=session.id,
        )

        result = await EvolutionAgent(literature_search=fake_literature_search).execute(
            Task(session_id=session.id, agent="evolution", action="evolve_top_hypotheses"),
            ctx,
        )

    assert result.kind == AgentResultKind.HYPOTHESIS_CREATED
    assert [item["source_strategy"] for item in result.payload["hypotheses"]] == [
        "evolution:grounding_enhancement",
        "evolution:feasibility_refinement",
        "evolution:inspiration_from_existing",
    ]
    assert [item["parent_ids"] for item in result.payload["hypotheses"]] == [
        [parents[0].id],
        [parents[1].id],
        [parents[2].id],
    ]
    assert len(client.messages) == 3
    assert result.citations[0].pmid == "1"
    assert all(
        item["meta_review_round"] == 0 for item in result.payload["hypotheses"]
    )


@pytest.mark.asyncio
async def test_evolution_records_current_meta_review_round_in_payload(
    tmp_path: Path,
) -> None:
    client = SequenceClient(
        [
            "Rationale\nHYPOTHESIS\nGrounded child",
            "Rationale\nHYPOTHESIS\nFeasible child",
        ]
    )
    async with SQLiteStore(tmp_path / "evolution_round.sqlite") as store:
        session = await store.create_session("goal")
        await store.save_research_plan(
            ResearchPlan(session_id=session.id, goal=session.goal, preferences=["novel"])
        )
        for index, elo in enumerate([1500, 1400], start=1):
            await store.add_hypothesis(
                Hypothesis(
                    session_id=session.id,
                    content=f"Parent {index}",
                    summary=f"P{index}",
                    elo=elo,
                )
            )
        for round_number in (1, 2, 3):
            await store.add_feedback(
                SystemFeedback(
                    session_id=session.id,
                    round=round_number,
                    content=f"Feedback round {round_number}",
                )
            )
        ctx = AgentContext(
            store=store,
            llm_router=StaticRouter(client),
            config=make_config(max_ideas=4),
            session_id=session.id,
        )

        result = await EvolutionAgent(literature_search=fake_literature_search).execute(
            Task(session_id=session.id, agent="evolution", action="evolve_top_hypotheses"),
            ctx,
        )

        first_stored = await store.list_session_hypotheses(session.id)

    assert result.ok
    assert all(item["meta_review_round"] == 3 for item in result.payload["hypotheses"])
    evolved = [h for h in first_stored if (h.source_strategy or "").startswith("evolution:")]
    assert evolved == []
    assert result.payload["hypotheses"][0]["meta_review_round"] == 3


@pytest.mark.asyncio
async def test_evolution_skips_combination_when_only_one_parent(tmp_path: Path) -> None:
    client = SequenceClient(
        [
            "Rationale\nHYPOTHESIS\nGrounded child",
            "Rationale\nHYPOTHESIS\nFeasible child",
            "Rationale\nHYPOTHESIS\nInspired child",
            "Rationale\nHYPOTHESIS\nSimplified child",
            "Rationale\nHYPOTHESIS\nOut-of-box child",
        ]
    )
    async with SQLiteStore(tmp_path / "evolution_skip.sqlite") as store:
        session = await store.create_session("goal")
        await store.save_research_plan(
            ResearchPlan(session_id=session.id, goal=session.goal)
        )
        await store.add_hypothesis(
            Hypothesis(session_id=session.id, content="Sole parent", summary="P1", elo=1500)
        )
        ctx = AgentContext(
            store=store,
            llm_router=StaticRouter(client),
            config=make_config(max_ideas=10),
            session_id=session.id,
        )

        result = await EvolutionAgent(literature_search=fake_literature_search).execute(
            Task(session_id=session.id, agent="evolution", action="evolve_top_hypotheses"),
            ctx,
        )

    strategies = [item["source_strategy"] for item in result.payload["hypotheses"]]
    assert "evolution:combination" not in strategies
    assert len(strategies) == 5


@pytest.mark.asyncio
async def test_metareview_agent_generates_feedback_and_overview(tmp_path: Path) -> None:
    client = SequenceClient(["System feedback", "Final overview"])
    async with SQLiteStore(tmp_path / "metareview.sqlite") as store:
        session = await store.create_session("goal")
        await store.save_research_plan(
            ResearchPlan(session_id=session.id, goal=session.goal, preferences=["novel"])
        )
        first = await store.add_hypothesis(
            Hypothesis(session_id=session.id, content="A content", summary="A", elo=1300)
        )
        second = await store.add_hypothesis(
            Hypothesis(session_id=session.id, content="B content", summary="B", elo=1200)
        )
        assert first.id is not None and second.id is not None
        await store.add_review(
            Review(
                session_id=session.id,
                hypothesis_id=first.id,
                type="full",
                score=8,
                content="Needs better controls.",
            )
        )
        await store.add_match_and_update_elo(
            Match(
                session_id=session.id,
                hypo_a_id=first.id,
                hypo_b_id=second.id,
                winner_id=first.id,
                transcript="A has stronger evidence.",
            )
        )
        ctx = AgentContext(
            store=store,
            llm_router=StaticRouter(client),
            config=make_config(),
            session_id=session.id,
        )
        agent = MetaReviewAgent()

        feedback = await agent.execute(
            Task(session_id=session.id, agent="metareview", action="generate_system_feedback"),
            ctx,
        )
        overview = await agent.execute(
            Task(session_id=session.id, agent="metareview", action="generate_final_overview"),
            ctx,
        )

    assert feedback.kind == AgentResultKind.FEEDBACK_GENERATED
    assert feedback.payload == {"round": 1, "content": "System feedback"}
    assert overview.kind == AgentResultKind.OVERVIEW_GENERATED
    assert overview.payload["content"] == "Final overview"
    assert overview.payload["top_hypothesis_ids"][0] == first.id
    feedback_prompt = client.messages[0][-1]["content"]
    assert "Needs better controls." in feedback_prompt
    assert "A has stronger evidence." in feedback_prompt
    assert "Previous compressed feedback" in feedback_prompt
    assert "Recent tournament matches" in feedback_prompt
    assert "Additional instructions:" not in feedback_prompt
    assert "id=" not in feedback_prompt
