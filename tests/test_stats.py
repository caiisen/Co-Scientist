from __future__ import annotations

from pathlib import Path

import pytest

from co_scientist.memory import Hypothesis, Match, SQLiteStore, Task, TaskPriority, TaskStatus
from co_scientist.supervisor import collect_session_stats


@pytest.mark.asyncio
async def test_collect_session_stats(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "stats.sqlite") as store:
        session = await store.create_session("stats")
        first = await store.add_hypothesis(
            Hypothesis(session_id=session.id, content="A", summary="A")
        )
        second = await store.add_hypothesis(
            Hypothesis(session_id=session.id, content="B", summary="B")
        )
        assert first.id is not None
        assert second.id is not None
        await store.add_match_and_update_elo(
            Match(
                session_id=session.id,
                hypo_a_id=first.id,
                hypo_b_id=second.id,
                winner_id=second.id,
                transcript="B wins.",
            )
        )
        task = await store.add_task(
            Task(
                session_id=session.id,
                agent="reflection",
                action="review",
                priority=int(TaskPriority.REFLECTION),
            )
        )
        assert task.id is not None
        await store.mark_task_status(task.id, TaskStatus.RUNNING)

        stats = await collect_session_stats(store, session.id, top_k=1)

        assert stats.hypothesis_count == 2
        assert stats.match_count == 1
        assert stats.matches_per_idea == 0.5
        assert [hypothesis.id for hypothesis in stats.top_hypotheses] == [second.id]
        assert stats.tasks_by_status[TaskStatus.RUNNING] == 1
