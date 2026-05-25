from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from co_scientist.memory import SQLiteStore, Task, TaskPriority, TaskStatus
from co_scientist.supervisor import TaskQueue


@pytest.mark.asyncio
async def test_task_queue_dequeues_by_priority_without_duplicates(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "queue.sqlite") as store:
        session = await store.create_session("queue")
        queue = TaskQueue(store, session.id)
        low = await queue.enqueue(
            Task(
                session_id=session.id,
                agent="meta",
                action="feedback",
                priority=int(TaskPriority.LOW),
            )
        )
        high = await queue.enqueue(
            Task(
                session_id=session.id,
                agent="reflection",
                action="review",
                priority=int(TaskPriority.REFLECTION),
            )
        )

        first = await queue.dequeue()
        second = await queue.dequeue()

        assert first.id == high.id
        assert second.id == low.id
        assert first.status == TaskStatus.RUNNING
        assert second.status == TaskStatus.RUNNING


@pytest.mark.asyncio
async def test_task_queue_restores_pending_and_marks_terminal_states(tmp_path: Path) -> None:
    db_path = tmp_path / "queue_restore.sqlite"
    async with SQLiteStore(db_path) as store:
        session = await store.create_session("queue restore")
        pending = await store.add_task(
            Task(
                session_id=session.id,
                agent="ranking",
                action="match",
                priority=int(TaskPriority.RANKING),
            )
        )
        done = await store.add_task(
            Task(
                session_id=session.id,
                agent="meta",
                action="feedback",
                priority=int(TaskPriority.LOW),
            )
        )
        assert done.id is not None
        await store.mark_task_status(done.id, TaskStatus.DONE)

    async with SQLiteStore(db_path) as reopened:
        queue = TaskQueue(reopened, session.id)
        await queue.load_pending()
        assert queue.qsize() == 1

        restored = await queue.dequeue()
        assert restored.id == pending.id
        assert restored.status == TaskStatus.RUNNING
        assert restored.id is not None

        completed = await queue.mark_done(restored.id, {"ok": True})
        assert completed.status == TaskStatus.DONE
        assert completed.result_json == {"ok": True}

        failed = await queue.enqueue(
            Task(
                session_id=session.id,
                agent="reflection",
                action="review",
                priority=int(TaskPriority.REFLECTION),
            )
        )
        assert failed.id is not None
        await queue.mark_failed(failed.id, "model timeout", result_json={"raw_text": "timeout"})
        loaded_failed = await reopened.get_task(failed.id)
        assert loaded_failed is not None
        assert loaded_failed.status == TaskStatus.FAILED
        assert loaded_failed.error == "model timeout"
        assert loaded_failed.result_json == {"raw_text": "timeout"}


@pytest.mark.asyncio
async def test_load_pending_recovers_orphan_running_tasks(tmp_path: Path) -> None:
    db_path = tmp_path / "queue_orphan.sqlite"
    async with SQLiteStore(db_path) as store:
        session = await store.create_session("orphan restore")
        orphan = await store.add_task(
            Task(
                session_id=session.id,
                agent="generation",
                action="create",
                priority=int(TaskPriority.USER),
            )
        )
        assert orphan.id is not None
        await store.mark_task_status(orphan.id, TaskStatus.RUNNING)

    async with SQLiteStore(db_path) as reopened:
        queue = TaskQueue(reopened, session.id)
        await queue.load_pending()

        restored = await queue.dequeue()
        assert restored.id == orphan.id
        assert restored.status == TaskStatus.RUNNING


@pytest.mark.asyncio
async def test_task_done_waits_for_terminal_state(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "queue_join.sqlite") as store:
        session = await store.create_session("join")
        queue = TaskQueue(store, session.id)
        task = await queue.enqueue(
            Task(
                session_id=session.id,
                agent="reflection",
                action="review",
                priority=int(TaskPriority.REFLECTION),
            )
        )

        running = await queue.dequeue()
        assert running.id == task.id

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(queue._queue.join(), timeout=0.01)

        assert running.id is not None
        await queue.mark_done(running.id)
        await asyncio.wait_for(queue._queue.join(), timeout=0.1)


@pytest.mark.asyncio
async def test_mark_terminal_state_before_dequeue_keeps_join_accounting(
    tmp_path: Path,
) -> None:
    async with SQLiteStore(tmp_path / "queue_predequeue_terminal.sqlite") as store:
        session = await store.create_session("predequeue terminal")
        queue = TaskQueue(store, session.id)
        done = await queue.enqueue(
            Task(
                session_id=session.id,
                agent="reflection",
                action="review",
                priority=int(TaskPriority.REFLECTION),
            )
        )
        failed = await queue.enqueue(
            Task(
                session_id=session.id,
                agent="ranking",
                action="match",
                priority=int(TaskPriority.RANKING),
            )
        )
        assert done.id is not None
        assert failed.id is not None

        await queue.mark_done(done.id)
        await queue.mark_failed(failed.id, "cancelled externally")

        await asyncio.wait_for(queue._queue.join(), timeout=0.1)

        loaded_done = await store.get_task(done.id)
        loaded_failed = await store.get_task(failed.id)
        assert loaded_done is not None
        assert loaded_failed is not None
        assert loaded_done.status == TaskStatus.DONE
        assert loaded_failed.status == TaskStatus.FAILED
