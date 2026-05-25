from __future__ import annotations

from pathlib import Path

import pytest

from co_scientist.agents.base import Agent
from co_scientist.agents.results import AgentResult, AgentResultKind
from co_scientist.config import AppConfig, LLMConfig, ProviderConfig, RuntimeConfig, SearchConfig
from co_scientist.llm.client import LLMRouter
from co_scientist.memory import (
    Hypothesis,
    Match,
    ResearchOverview,
    ResearchPlan,
    Review,
    SQLiteStore,
    SystemFeedback,
    Task,
    TaskPriority,
)
from co_scientist.supervisor import Supervisor
from co_scientist.supervisor.task_queue import TaskQueue


class StubClient:
    async def chat(self, messages, **kwargs):
        return "{}"

    async def embed(self, texts, **kwargs):
        return [[1.0, 0.0] for _ in texts]


class StaticRouter(LLMRouter):
    def __init__(self) -> None:
        self.client = StubClient()

    def client_for(self, agent=None):
        return self.client


class StubAgent(Agent):
    name = "stub"

    async def execute(self, task: Task, ctx) -> AgentResult:
        return AgentResult(kind=AgentResultKind.NOOP)


def make_config(
    *,
    max_ideas: int = 10,
    max_matches_per_idea: int = 10,
    elo_stagnation_threshold: float = 5.0,
    elo_stagnation_window: int = 4,
) -> AppConfig:
    return AppConfig(
        runtime=RuntimeConfig(
            max_ideas=max_ideas,
            max_matches_per_idea=max_matches_per_idea,
            worker_concurrency=1,
            request_timeout_seconds=30,
            elo_stagnation_threshold=elo_stagnation_threshold,
            elo_stagnation_window=elo_stagnation_window,
        ),
        search=SearchConfig(max_results=1),
        llm=LLMConfig(
            default_provider="test",
            providers={"test": ProviderConfig(chat_model="test-chat")},
        ),
    )


@pytest.mark.asyncio
async def test_supervisor_stale_elo_enqueues_feedback_before_evolution(
    tmp_path: Path,
) -> None:
    async with SQLiteStore(tmp_path / "stale.sqlite") as store:
        session = await store.create_session("goal")
        await store.save_research_plan(ResearchPlan(session_id=session.id, goal=session.goal))
        hypotheses = await _reviewed_hypotheses(store, session.id, count=5)
        for _ in range(3):
            await store.add_elo_checkpoint(session.id, top_k=5)
        supervisor = Supervisor(
            store=store,
            config=make_config(max_ideas=10),
            llm_router=StaticRouter(),
            agents={
                "ranking": StubAgent(name="ranking"),
                "evolution": StubAgent(name="evolution"),
                "metareview": StubAgent(name="metareview"),
            },
        )
        queue = TaskQueue(store, session.id)
        task = await queue.enqueue(
            Task(session_id=session.id, agent="ranking", action="run_tournament_match")
        )
        result = AgentResult(
            kind=AgentResultKind.RANKING_DECISION,
            payload={
                "hypo_a_id": hypotheses[0].id,
                "hypo_b_id": hypotheses[1].id,
                "winner_id": hypotheses[0].id,
                "transcript": "Close match.",
            },
        )

        await supervisor._handle_result(queue, task, result)
        pending = await store.pending_tasks(session.id)

    assert any(
        task.agent == "metareview" and task.action == "generate_system_feedback"
        for task in pending
    )
    assert not any(
        task.agent == "evolution" and task.action == "evolve_top_hypotheses"
        for task in pending
    )
    assert not any(
        task.agent == "ranking" and task.action == "run_tournament_match"
        for task in pending
    )
    assert pending[0].priority == int(TaskPriority.META_REVIEW)


@pytest.mark.asyncio
async def test_supervisor_feedback_updates_context_and_enqueues_evolution(
    tmp_path: Path,
) -> None:
    async with SQLiteStore(tmp_path / "feedback.sqlite") as store:
        session = await store.create_session("goal")
        await store.save_research_plan(ResearchPlan(session_id=session.id, goal=session.goal))
        await _reviewed_hypotheses(store, session.id, count=5)
        supervisor = Supervisor(
            store=store,
            config=make_config(max_ideas=10),
            llm_router=StaticRouter(),
            agents={
                "evolution": StubAgent(name="evolution"),
                "metareview": StubAgent(name="metareview"),
            },
        )
        ctx = supervisor._context(session.id, http_session=None)
        queue = TaskQueue(store, session.id)
        task = await queue.enqueue(
            Task(session_id=session.id, agent="metareview", action="generate_system_feedback")
        )
        result = AgentResult(
            kind=AgentResultKind.FEEDBACK_GENERATED,
            payload={"round": 1, "content": "Prefer simpler assays."},
        )

        await supervisor._handle_result(queue, task, result, ctx)
        pending = await store.pending_tasks(session.id)
        latest = await store.latest_feedback(session.id)

    assert latest is not None
    assert latest.content == "Prefer simpler assays."
    assert ctx.current_feedback == "Prefer simpler assays."
    assert any(
        task.agent == "evolution" and task.action == "evolve_top_hypotheses"
        for task in pending
    )
    assert pending[0].priority == int(TaskPriority.EVOLUTION)


@pytest.mark.asyncio
async def test_supervisor_stagnation_threshold_is_configurable(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "stagnation_cfg.sqlite") as store:
        session = await store.create_session("goal")
        await store.save_research_plan(ResearchPlan(session_id=session.id, goal=session.goal))
        hypotheses = await _reviewed_hypotheses(store, session.id, count=5)
        baselines = (1200, 1206, 1212, 1218)
        for avg in baselines:
            for hypothesis in hypotheses:
                hypothesis_id = hypothesis.id
                assert hypothesis_id is not None
                await store.db.execute(
                    "UPDATE hypotheses SET elo = ? WHERE id = ?",
                    (avg, hypothesis_id),
                )
            await store.db.commit()
            await store.add_elo_checkpoint(session.id, top_k=5)

        loose = Supervisor(
            store=store,
            config=make_config(max_ideas=10, elo_stagnation_threshold=10.0),
            llm_router=StaticRouter(),
            agents={
                "ranking": StubAgent(name="ranking"),
                "evolution": StubAgent(name="evolution"),
                "metareview": StubAgent(name="metareview"),
            },
        )
        strict = Supervisor(
            store=store,
            config=make_config(max_ideas=10, elo_stagnation_threshold=1.0),
            llm_router=StaticRouter(),
            agents={
                "ranking": StubAgent(name="ranking"),
                "evolution": StubAgent(name="evolution"),
                "metareview": StubAgent(name="metareview"),
            },
        )

        loose_stale = await loose._elo_stagnated(session.id)
        strict_stale = await strict._elo_stagnated(session.id)

    assert loose_stale is True
    assert strict_stale is False


@pytest.mark.asyncio
async def test_supervisor_does_not_enqueue_evolution_at_max_ideas(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "max.sqlite") as store:
        session = await store.create_session("goal")
        await _reviewed_hypotheses(store, session.id, count=5)
        supervisor = Supervisor(
            store=store,
            config=make_config(max_ideas=5),
            llm_router=StaticRouter(),
            agents={
                "evolution": StubAgent(name="evolution"),
                "metareview": StubAgent(name="metareview"),
            },
        )
        queue = TaskQueue(store, session.id)
        task = await queue.enqueue(
            Task(session_id=session.id, agent="metareview", action="generate_system_feedback")
        )
        result = AgentResult(
            kind=AgentResultKind.FEEDBACK_GENERATED,
            payload={"round": 1, "content": "Feedback."},
        )

        await supervisor._handle_result(queue, task, result)
        pending = await store.pending_tasks(session.id)

    assert not any(
        task.agent == "evolution" and task.action == "evolve_top_hypotheses"
        for task in pending
    )


@pytest.mark.asyncio
async def test_supervisor_export_includes_phase6_feedback_overview_and_evolution_lineage(
    tmp_path: Path,
) -> None:
    async with SQLiteStore(tmp_path / "export_phase6.sqlite") as store:
        session = await store.create_session("goal")
        await store.save_research_plan(ResearchPlan(session_id=session.id, goal=session.goal))
        parent = await store.add_hypothesis(
            Hypothesis(session_id=session.id, content="Parent content", summary="Parent")
        )
        assert parent.id is not None
        child = await store.add_hypothesis(
            Hypothesis(
                session_id=session.id,
                content="Child content",
                summary="Child",
                parent_ids=[parent.id],
                source_strategy="evolution:simplification",
                meta_review_round=2,
            )
        )
        assert child.id is not None
        await store.add_feedback(
            SystemFeedback(session_id=session.id, round=2, content="Prefer leaner assays.")
        )
        await store.add_overview(
            ResearchOverview(
                session_id=session.id,
                round=1,
                content="Final overview content.",
                top_hypothesis_ids=[child.id, parent.id],
            )
        )
        supervisor = Supervisor(
            store=store,
            config=make_config(max_ideas=10),
            llm_router=StaticRouter(),
            agents={},
        )

        markdown = await supervisor.export_markdown(session.id)

    assert "# Co-Scientist Report" in markdown
    assert "## Final Research Overview" in markdown
    assert "Final overview content." in markdown
    assert f"Top hypothesis ids: {child.id}, {parent.id}" in markdown
    assert "## Latest System Feedback" in markdown
    assert "Round: `2`" in markdown
    assert "Prefer leaner assays." in markdown
    assert f"Parent hypothesis ids: `{parent.id}`" in markdown
    assert "Meta-review round: `2`" in markdown


@pytest.mark.asyncio
async def test_supervisor_ranking_target_idle_fallback_enqueues_metareview(
    tmp_path: Path,
) -> None:
    async with SQLiteStore(tmp_path / "idle_metareview.sqlite") as store:
        session = await store.create_session("goal")
        await store.save_research_plan(ResearchPlan(session_id=session.id, goal=session.goal))
        hypotheses = await _reviewed_hypotheses(store, session.id, count=2)
        assert hypotheses[0].id is not None and hypotheses[1].id is not None
        await store.add_match_and_update_elo(
            _match(session.id, hypotheses[0].id, hypotheses[1].id, hypotheses[0].id)
        )
        await store.add_match_and_update_elo(
            _match(session.id, hypotheses[0].id, hypotheses[1].id, hypotheses[0].id)
        )
        supervisor = Supervisor(
            store=store,
            config=make_config(max_ideas=5, max_matches_per_idea=1),
            llm_router=StaticRouter(),
            agents={
                "ranking": StubAgent(name="ranking"),
                "evolution": StubAgent(name="evolution"),
                "metareview": StubAgent(name="metareview"),
            },
        )
        queue = TaskQueue(store, session.id)

        await supervisor._maybe_enqueue_ranking(queue, session.id)
        pending = await store.pending_tasks(session.id)

    assert len(pending) == 1
    assert pending[0].agent == "metareview"
    assert pending[0].action == "generate_system_feedback"
    assert pending[0].priority == int(TaskPriority.META_REVIEW)


@pytest.mark.asyncio
async def test_supervisor_ranking_target_idle_fallback_enqueues_evolution_without_metareview(
    tmp_path: Path,
) -> None:
    async with SQLiteStore(tmp_path / "idle_evolution.sqlite") as store:
        session = await store.create_session("goal")
        await store.save_research_plan(ResearchPlan(session_id=session.id, goal=session.goal))
        hypotheses = await _reviewed_hypotheses(store, session.id, count=2)
        assert hypotheses[0].id is not None and hypotheses[1].id is not None
        await store.add_match_and_update_elo(
            _match(session.id, hypotheses[0].id, hypotheses[1].id, hypotheses[0].id)
        )
        await store.add_match_and_update_elo(
            _match(session.id, hypotheses[0].id, hypotheses[1].id, hypotheses[0].id)
        )
        supervisor = Supervisor(
            store=store,
            config=make_config(max_ideas=5, max_matches_per_idea=1),
            llm_router=StaticRouter(),
            agents={
                "ranking": StubAgent(name="ranking"),
                "evolution": StubAgent(name="evolution"),
            },
        )
        queue = TaskQueue(store, session.id)

        await supervisor._maybe_enqueue_ranking(queue, session.id)
        pending = await store.pending_tasks(session.id)

    assert len(pending) == 1
    assert pending[0].agent == "evolution"
    assert pending[0].action == "evolve_top_hypotheses"
    assert pending[0].priority == int(TaskPriority.EVOLUTION)


@pytest.mark.asyncio
async def test_supervisor_final_overview_waits_for_max_ideas_and_active_work(
    tmp_path: Path,
) -> None:
    async with SQLiteStore(tmp_path / "overview_wait.sqlite") as store:
        session = await store.create_session("goal")
        await store.save_research_plan(ResearchPlan(session_id=session.id, goal=session.goal))
        hypotheses = await _reviewed_hypotheses(store, session.id, count=2)
        assert hypotheses[0].id is not None and hypotheses[1].id is not None
        await store.add_match_and_update_elo(
            _match(session.id, hypotheses[0].id, hypotheses[1].id, hypotheses[0].id)
        )
        await store.add_match_and_update_elo(
            _match(session.id, hypotheses[0].id, hypotheses[1].id, hypotheses[0].id)
        )
        supervisor = Supervisor(
            store=store,
            config=make_config(max_ideas=2, max_matches_per_idea=1),
            llm_router=StaticRouter(),
            agents={
                "ranking": StubAgent(name="ranking"),
                "metareview": StubAgent(name="metareview"),
            },
        )
        queue = TaskQueue(store, session.id)
        active_review = await queue.enqueue(
            Task(
                session_id=session.id,
                agent="reflection",
                action="full_review",
                priority=int(TaskPriority.REFLECTION),
            )
        )

        await supervisor._maybe_enqueue_final_overview(queue, session.id)
        pending = await store.pending_tasks(session.id)
        assert [task.id for task in pending] == [active_review.id]

        assert active_review.id is not None
        await queue.mark_done(active_review.id)
        await supervisor._maybe_enqueue_final_overview(queue, session.id)
        pending = await store.pending_tasks(session.id)
        assert len(pending) == 1
        assert pending[0].agent == "metareview"
        assert pending[0].action == "generate_final_overview"

        await supervisor._maybe_enqueue_final_overview(queue, session.id)
        pending = await store.pending_tasks(session.id)

    assert len(pending) == 1


async def _reviewed_hypotheses(
    store: SQLiteStore,
    session_id: str,
    *,
    count: int,
) -> list[Hypothesis]:
    hypotheses = []
    for index in range(count):
        hypothesis = await store.add_hypothesis(
            Hypothesis(
                session_id=session_id,
                content=f"Hypothesis {index}",
                summary=f"H{index}",
                elo=1200,
            )
        )
        assert hypothesis.id is not None
        await store.add_review(
            Review(
                session_id=session_id,
                hypothesis_id=hypothesis.id,
                type="full",
                score=7,
                content="Review.",
            )
        )
        hypotheses.append(hypothesis)
    return hypotheses


def _match(session_id: str, hypo_a_id: int, hypo_b_id: int, winner_id: int) -> Match:
    return Match(
        session_id=session_id,
        hypo_a_id=hypo_a_id,
        hypo_b_id=hypo_b_id,
        winner_id=winner_id,
        transcript="Winner has stronger support.",
    )
