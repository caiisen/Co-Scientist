from __future__ import annotations

import asyncio

from co_scientist.memory.models import Task, TaskStatus
from co_scientist.memory.store import SQLiteStore


class TaskQueue:
    def __init__(self, store: SQLiteStore, session_id: str) -> None:
        self.store = store
        self.session_id = session_id
        self._queue: asyncio.PriorityQueue[tuple[int, float, int]] = asyncio.PriorityQueue()
        self._queued_ids: set[int] = set()
        self._in_progress_ids: set[int] = set()
        self._completed_queued_ids: set[int] = set()
        self._lock = asyncio.Lock()

    async def load_pending(self) -> None:
        async with self._lock:
            await self.store.reset_running_tasks(self.session_id)
            for task in await self.store.pending_tasks(self.session_id):
                self._put(task)

    async def enqueue(self, task: Task) -> Task:
        if task.session_id != self.session_id:
            raise ValueError("task session_id does not match queue session_id")
        if task.status != TaskStatus.PENDING:
            raise ValueError("only pending tasks can be enqueued")
        stored = await self.store.add_task(task)
        async with self._lock:
            self._put(stored)
        return stored

    async def enqueue_unique_action(
        self,
        task: Task,
        *,
        exclude_task_id: int | None = None,
    ) -> Task | None:
        if task.session_id != self.session_id:
            raise ValueError("task session_id does not match queue session_id")
        if task.status != TaskStatus.PENDING:
            raise ValueError("only pending tasks can be enqueued")
        async with self._lock:
            if await self.store.has_active_task(
                self.session_id,
                agent=task.agent,
                action=task.action,
                exclude_task_id=exclude_task_id,
            ):
                return None
            stored = await self.store.add_task(task)
            self._put(stored)
            return stored

    async def dequeue(self) -> Task:
        while True:
            _, _, task_id = await self._queue.get()
            async with self._lock:
                self._queued_ids.discard(task_id)
                completed_before_dequeue = task_id in self._completed_queued_ids
            task = await self.store.get_task(task_id)
            if task is None or task.status != TaskStatus.PENDING:
                if completed_before_dequeue:
                    async with self._lock:
                        self._completed_queued_ids.discard(task_id)
                else:
                    self._queue.task_done()
                continue
            running = await self.store.mark_task_status(task_id, TaskStatus.RUNNING)
            async with self._lock:
                self._in_progress_ids.add(task_id)
            return running

    async def mark_done(self, task_id: int, result_json: dict | None = None) -> Task:
        task = await self.store.mark_task_status(
            task_id,
            TaskStatus.DONE,
            result_json=result_json,
        )
        await self._mark_queue_item_finished(task_id)
        return task

    async def mark_failed(
        self,
        task_id: int,
        error: str,
        result_json: dict | None = None,
    ) -> Task:
        task = await self.store.mark_task_status(
            task_id,
            TaskStatus.FAILED,
            result_json=result_json,
            error=error,
        )
        await self._mark_queue_item_finished(task_id)
        return task

    def qsize(self) -> int:
        return self._queue.qsize()

    async def join(self) -> None:
        await self._queue.join()

    def _put(self, task: Task) -> None:
        if task.id is None or task.id in self._queued_ids:
            return
        self._queued_ids.add(task.id)
        self._queue.put_nowait((-task.priority, task.created_at.timestamp(), task.id))

    async def _mark_queue_item_finished(self, task_id: int) -> None:
        async with self._lock:
            if task_id in self._in_progress_ids:
                self._in_progress_ids.discard(task_id)
                self._queue.task_done()
                return
            if task_id in self._completed_queued_ids:
                return
            if task_id in self._queued_ids:
                self._completed_queued_ids.add(task_id)
                self._queue.task_done()
