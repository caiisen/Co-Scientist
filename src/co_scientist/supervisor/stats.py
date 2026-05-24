from __future__ import annotations

from pydantic import BaseModel

from co_scientist.memory.models import Hypothesis, TaskStatus
from co_scientist.memory.store import SQLiteStore


class SessionStats(BaseModel):
    session_id: str
    hypothesis_count: int
    match_count: int
    matches_per_idea: float
    top_hypotheses: list[Hypothesis]
    tasks_by_status: dict[TaskStatus, int]


async def collect_session_stats(
    store: SQLiteStore,
    session_id: str,
    *,
    top_k: int = 5,
) -> SessionStats:
    hypothesis_count = await store.count_hypotheses(session_id)
    match_count = await store.count_matches(session_id)
    matches_per_idea = match_count / hypothesis_count if hypothesis_count else 0.0
    return SessionStats(
        session_id=session_id,
        hypothesis_count=hypothesis_count,
        match_count=match_count,
        matches_per_idea=matches_per_idea,
        top_hypotheses=await store.top_k_by_elo(session_id, k=top_k),
        tasks_by_status=await store.tasks_by_status(session_id),
    )
