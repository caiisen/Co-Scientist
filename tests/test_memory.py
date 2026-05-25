from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from co_scientist.memory import (
    Hypothesis,
    Match,
    ResearchPlan,
    Review,
    SQLiteStore,
    Task,
    TaskPriority,
    TaskStatus,
)
from co_scientist.memory.store import _json_loads, _task_json_loads
from co_scientist.tools.models import Citation


@pytest.mark.asyncio
async def test_memory_crud_elo_transaction_and_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "session.sqlite"
    async with SQLiteStore(db_path) as store:
        session = await store.create_session("discover a better assay")
        first = await store.add_hypothesis(
            Hypothesis(
                session_id=session.id,
                content="Full markdown for hypothesis A",
                summary="Hypothesis A",
                detailed_description="Detailed A",
                mechanism="Mechanism A",
                impacted_pathways=["pathway-a"],
                experimental_plan="Run experiment A",
                safety_notes="No issue",
                testable_predictions=["Prediction A"],
                source_strategy="literature_review",
            )
        )
        second = await store.add_hypothesis(
            Hypothesis(
                session_id=session.id,
                content="Full markdown for hypothesis B",
                summary="Hypothesis B",
                mechanism="Mechanism B",
            )
        )

        assert first.id is not None
        loaded = await store.get_hypothesis(first.id)
        assert loaded is not None
        assert loaded.impacted_pathways == ["pathway-a"]
        assert loaded.testable_predictions == ["Prediction A"]

        review = await store.add_review(
            Review(
                session_id=session.id,
                hypothesis_id=first.id,
                type="full",
                score=8.5,
                content="Solid but needs validation.",
            )
        )
        assert review.id is not None
        reviews = await store.reviews_for_hypothesis(first.id)
        assert [item.id for item in reviews] == [review.id]

        match = await store.add_match_and_update_elo(
            Match(
                session_id=session.id,
                hypo_a_id=first.id,
                hypo_b_id=second.id,
                winner_id=first.id,
                transcript="A is more testable.",
            )
        )
        assert match.id is not None
        top = await store.top_k_by_elo(session.id, k=2)
        assert [item.id for item in top] == [first.id, second.id]
        assert top[0].elo == 1216
        assert top[1].elo == 1184

    async with SQLiteStore(db_path) as reopened:
        restored = await reopened.get_session(session.id)
        assert restored is not None
        assert restored.goal == "discover a better assay"
        assert await reopened.count_hypotheses(session.id) == 2
        assert await reopened.count_matches(session.id) == 1
        assert await reopened.count_reviews(session.id) == 1


@pytest.mark.asyncio
async def test_research_plan_and_list_session_hypotheses(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "plan.sqlite") as store:
        session = await store.create_session("goal")
        plan = await store.save_research_plan(
            ResearchPlan(
                session_id=session.id,
                goal=session.goal,
                preferences=["novel"],
                attributes=["mechanistic"],
                constraints=["safe"],
                idea_attributes=["testable"],
            )
        )
        await store.add_hypothesis(Hypothesis(session_id=session.id, content="A", summary="A"))
        await store.add_hypothesis(Hypothesis(session_id=session.id, content="B", summary="B"))

        loaded = await store.get_research_plan(session.id)
        hypotheses = await store.list_session_hypotheses(session.id)

    assert loaded == plan
    assert [hypothesis.summary for hypothesis in hypotheses] == ["A", "B"]


@pytest.mark.asyncio
async def test_pending_tasks_order_and_status_counts(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "tasks.sqlite") as store:
        session = await store.create_session("task ordering")
        low = await store.add_task(
            Task(
                session_id=session.id,
                agent="metareview",
                action="summarize",
                priority=int(TaskPriority.LOW),
            )
        )
        high = await store.add_task(
            Task(
                session_id=session.id,
                agent="reflection",
                action="review",
                priority=int(TaskPriority.REFLECTION),
            )
        )

        pending = await store.pending_tasks(session.id)
        assert [task.id for task in pending] == [high.id, low.id]

        assert high.id is not None
        await store.mark_task_status(high.id, TaskStatus.RUNNING)
        counts = await store.tasks_by_status(session.id)
        assert counts[TaskStatus.PENDING] == 1
        assert counts[TaskStatus.RUNNING] == 1


@pytest.mark.asyncio
async def test_add_match_rejects_winner_outside_pair(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "bad_winner.sqlite") as store:
        session = await store.create_session("bad winner")
        first = await store.add_hypothesis(
            Hypothesis(session_id=session.id, content="A", summary="A")
        )
        second = await store.add_hypothesis(
            Hypothesis(session_id=session.id, content="B", summary="B")
        )
        assert first.id is not None
        assert second.id is not None

        with pytest.raises(ValueError, match="winner_id"):
            await store.add_match_and_update_elo(
                Match(
                    session_id=session.id,
                    hypo_a_id=first.id,
                    hypo_b_id=second.id,
                    winner_id=999,
                    transcript="Invalid winner.",
                )
            )


@pytest.mark.asyncio
async def test_add_match_rejects_unknown_hypothesis(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "missing_hypothesis.sqlite") as store:
        session = await store.create_session("missing hypothesis")
        first = await store.add_hypothesis(
            Hypothesis(session_id=session.id, content="A", summary="A")
        )
        assert first.id is not None

        with pytest.raises(KeyError, match="unknown hypothesis"):
            await store.add_match_and_update_elo(
                Match(
                    session_id=session.id,
                    hypo_a_id=first.id,
                    hypo_b_id=999,
                    winner_id=first.id,
                    transcript="Missing opponent.",
                )
            )


@pytest.mark.asyncio
async def test_add_match_rejects_cross_session_hypotheses(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "cross_session_match.sqlite") as store:
        first_session = await store.create_session("first")
        second_session = await store.create_session("second")
        first = await store.add_hypothesis(
            Hypothesis(session_id=first_session.id, content="A", summary="A")
        )
        second = await store.add_hypothesis(
            Hypothesis(session_id=second_session.id, content="B", summary="B")
        )
        assert first.id is not None
        assert second.id is not None

        with pytest.raises(ValueError, match="match session"):
            await store.add_match_and_update_elo(
                Match(
                    session_id=first_session.id,
                    hypo_a_id=first.id,
                    hypo_b_id=second.id,
                    winner_id=first.id,
                    transcript="Invalid cross-session match.",
                )
            )


@pytest.mark.asyncio
async def test_add_citations_batch_dedupes_and_links_artifacts(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "citation_batch.sqlite") as store:
        session = await store.create_session("citations")
        citation_ids = await store.add_citations_batch(
            [
                Citation(source="pubmed", title="Paper A", pmid="1"),
                Citation(source="semantic_scholar", title="Paper A duplicate", pmid="1"),
                Citation(source="arxiv", title="Paper B", arxiv_id="2401.00001"),
            ],
            session_id=session.id,
        )
        await store.add_citation_links(
            citation_ids,
            session_id=session.id,
            artifact_type="review",
            artifact_id=7,
        )

        assert len(citation_ids) == 2
        async with store.db.execute(
            "SELECT source, title, dedupe_key FROM citations WHERE session_id = ? ORDER BY id",
            (session.id,),
        ) as cursor:
            rows = await cursor.fetchall()
        assert [(row["source"], row["dedupe_key"]) for row in rows] == [
            ("pubmed", "pmid:1"),
            ("arxiv", "arxiv:2401.00001"),
        ]
        links = await store.citation_links_for_artifact(
            session_id=session.id,
            artifact_type="review",
            artifact_id=7,
        )
        assert [(link["evidence_index"], link["title"]) for link in links] == [
            (1, "Paper A"),
            (2, "Paper B"),
        ]


@pytest.mark.asyncio
async def test_batch_upserts_embeddings_and_proximity_edges(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "proximity_batch.sqlite") as store:
        session = await store.create_session("batch")
        first = await store.add_hypothesis(
            Hypothesis(session_id=session.id, content="A", summary="A")
        )
        second = await store.add_hypothesis(
            Hypothesis(session_id=session.id, content="B", summary="B")
        )
        assert first.id is not None
        assert second.id is not None

        await store.upsert_hypothesis_embeddings_batch(
            session_id=session.id,
            embeddings={first.id: [1.0, 0.0], second.id: [0.5, 0.5]},
        )
        await store.upsert_proximity_edges_batch(
            session_id=session.id,
            edges=[(second.id, first.id, 0.75)],
        )

        assert await store.embeddings_for_session(session.id) == {
            first.id: [1.0, 0.0],
            second.id: [0.5, 0.5],
        }
        assert await store.proximity_edges_for_session(session.id) == {
            (first.id, second.id): 0.75
        }


def test_hypothesis_rejects_negative_elo() -> None:
    with pytest.raises(ValidationError):
        Hypothesis(session_id="s", content="A", summary="A", elo=-1)


def test_json_loads_is_strict_outside_task_legacy_fields() -> None:
    assert _json_loads(None, {"fallback": True}) == {"fallback": True}
    assert _json_loads("null", {"fallback": True}) is None
    with pytest.raises(json.JSONDecodeError):
        _json_loads("NULL", {"fallback": True})


def test_task_json_loads_treats_legacy_null_strings_as_default() -> None:
    assert _task_json_loads(None, {"fallback": True}) == {"fallback": True}
    assert _task_json_loads("", {"fallback": True}) == {"fallback": True}
    assert _task_json_loads("NULL", {"fallback": True}) == {"fallback": True}
    assert _task_json_loads("null", {"fallback": True}) is None
