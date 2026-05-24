from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite

from co_scientist.memory.elo import update_elo
from co_scientist.memory.models import (
    Hypothesis,
    Match,
    ResearchOverview,
    Review,
    Session,
    SystemFeedback,
    Task,
    TaskStatus,
    utc_now,
)

if TYPE_CHECKING:
    from co_scientist.tools.models import Citation

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    return json.loads(value)


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _dt_text(value: datetime) -> str:
    return value.isoformat()


class SQLiteStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None

    async def __aenter__(self) -> SQLiteStore:
        await self.connect()
        await self.init_schema()
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("SQLiteStore is not connected")
        return self._db

    async def connect(self) -> None:
        if self._db is not None:
            return
        if self.db_path != Path(":memory:"):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self.db.execute("PRAGMA foreign_keys = ON")

    async def close(self) -> None:
        if self._db is None:
            return
        await self._db.close()
        self._db = None

    async def init_schema(self) -> None:
        schema = SCHEMA_PATH.read_text(encoding="utf-8")
        await self.db.executescript(schema)
        await self.db.commit()

    async def create_session(
        self,
        goal: str,
        *,
        session_id: str | None = None,
        config_json: dict[str, Any] | None = None,
    ) -> Session:
        session = Session(
            id=session_id or str(uuid.uuid4()),
            goal=goal,
            config_json=config_json or {},
        )
        await self.db.execute(
            """
            INSERT INTO sessions (id, goal, config_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                session.id,
                session.goal,
                _json_dumps(session.config_json),
                _dt_text(session.created_at),
            ),
        )
        await self.db.commit()
        return session

    async def get_session(self, session_id: str) -> Session | None:
        async with self.db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return Session(
            id=row["id"],
            goal=row["goal"],
            config_json=_json_loads(row["config_json"], {}),
            created_at=_dt(row["created_at"]),
        )

    async def add_hypothesis(self, hypothesis: Hypothesis) -> Hypothesis:
        cursor = await self.db.execute(
            """
            INSERT INTO hypotheses (
              session_id, content, summary, detailed_description, mechanism,
              impacted_pathways, experimental_plan, safety_notes,
              testable_predictions, elo, parent_ids, source_strategy, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                hypothesis.session_id,
                hypothesis.content,
                hypothesis.summary,
                hypothesis.detailed_description,
                hypothesis.mechanism,
                _json_dumps(hypothesis.impacted_pathways),
                hypothesis.experimental_plan,
                hypothesis.safety_notes,
                _json_dumps(hypothesis.testable_predictions),
                hypothesis.elo,
                _json_dumps(hypothesis.parent_ids),
                hypothesis.source_strategy,
                _dt_text(hypothesis.created_at),
            ),
        )
        await self.db.commit()
        return hypothesis.model_copy(update={"id": cursor.lastrowid})

    async def get_hypothesis(self, hypothesis_id: int) -> Hypothesis | None:
        async with self.db.execute(
            "SELECT * FROM hypotheses WHERE id = ?",
            (hypothesis_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return self._row_to_hypothesis(row) if row else None

    async def top_k_by_elo(self, session_id: str, *, k: int = 10) -> list[Hypothesis]:
        async with self.db.execute(
            """
            SELECT * FROM hypotheses
            WHERE session_id = ?
            ORDER BY elo DESC, created_at ASC
            LIMIT ?
            """,
            (session_id, k),
        ) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_hypothesis(row) for row in rows]

    async def add_review(self, review: Review) -> Review:
        cursor = await self.db.execute(
            """
            INSERT INTO reviews (session_id, hypothesis_id, type, score, content, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                review.session_id,
                review.hypothesis_id,
                review.type,
                review.score,
                review.content,
                _dt_text(review.created_at),
            ),
        )
        await self.db.commit()
        return review.model_copy(update={"id": cursor.lastrowid})

    async def add_task(self, task: Task) -> Task:
        now = task.updated_at or utc_now()
        cursor = await self.db.execute(
            """
            INSERT INTO tasks (
              session_id, agent, action, target_id, priority, status,
              payload_json, result_json, error, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task.session_id,
                task.agent,
                task.action,
                task.target_id,
                task.priority,
                task.status.value,
                _json_dumps(task.payload_json),
                _json_dumps(task.result_json) if task.result_json is not None else None,
                task.error,
                _dt_text(task.created_at),
                _dt_text(now),
            ),
        )
        await self.db.commit()
        return task.model_copy(update={"id": cursor.lastrowid, "updated_at": now})

    async def get_task(self, task_id: int) -> Task | None:
        async with self.db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)) as cursor:
            row = await cursor.fetchone()
        return self._row_to_task(row) if row else None

    async def pending_tasks(self, session_id: str, *, limit: int | None = None) -> list[Task]:
        sql = """
            SELECT * FROM tasks
            WHERE session_id = ? AND status = ?
            ORDER BY priority DESC, created_at ASC
        """
        params: tuple[Any, ...]
        if limit is not None:
            sql += " LIMIT ?"
            params = (session_id, TaskStatus.PENDING.value, limit)
        else:
            params = (session_id, TaskStatus.PENDING.value)
        async with self.db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_task(row) for row in rows]

    async def reset_running_tasks(self, session_id: str) -> int:
        now = utc_now()
        cursor = await self.db.execute(
            """
            UPDATE tasks
            SET status = ?, updated_at = ?
            WHERE session_id = ? AND status = ?
            """,
            (
                TaskStatus.PENDING.value,
                _dt_text(now),
                session_id,
                TaskStatus.RUNNING.value,
            ),
        )
        await self.db.commit()
        return cursor.rowcount

    async def tasks_by_status(self, session_id: str) -> dict[TaskStatus, int]:
        async with self.db.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM tasks
            WHERE session_id = ?
            GROUP BY status
            """,
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return {TaskStatus(row["status"]): row["count"] for row in rows}

    async def mark_task_status(
        self,
        task_id: int,
        status: TaskStatus,
        *,
        result_json: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> Task:
        now = utc_now()
        await self.db.execute(
            """
            UPDATE tasks
            SET status = ?, result_json = ?, error = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                status.value,
                _json_dumps(result_json) if result_json is not None else None,
                error,
                _dt_text(now),
                task_id,
            ),
        )
        await self.db.commit()
        task = await self.get_task(task_id)
        if task is None:
            raise KeyError(f"unknown task id: {task_id}")
        return task

    async def add_match_and_update_elo(self, match: Match, *, k: int = 32) -> Match:
        if match.winner_id not in {match.hypo_a_id, match.hypo_b_id}:
            raise ValueError("winner_id must be hypo_a_id or hypo_b_id")

        async with self.db.execute(
            "SELECT id, session_id, elo FROM hypotheses WHERE id IN (?, ?)",
            (match.hypo_a_id, match.hypo_b_id),
        ) as cursor:
            rows = await cursor.fetchall()
        ratings = {row["id"]: row["elo"] for row in rows}
        if set(ratings) != {match.hypo_a_id, match.hypo_b_id}:
            raise KeyError("match references unknown hypothesis")
        hypothesis_sessions = {row["session_id"] for row in rows}
        if hypothesis_sessions != {match.session_id}:
            raise ValueError("match hypotheses must belong to match session")

        loser_id = match.hypo_b_id if match.winner_id == match.hypo_a_id else match.hypo_a_id
        new_winner, new_loser = update_elo(ratings[match.winner_id], ratings[loser_id], k=k)

        try:
            cursor = await self.db.execute(
                """
                INSERT INTO matches (
                  session_id, hypo_a_id, hypo_b_id, winner_id, transcript, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    match.session_id,
                    match.hypo_a_id,
                    match.hypo_b_id,
                    match.winner_id,
                    match.transcript,
                    _dt_text(match.created_at),
                ),
            )
            await self.db.execute(
                "UPDATE hypotheses SET elo = ? WHERE id = ?",
                (new_winner, match.winner_id),
            )
            await self.db.execute(
                "UPDATE hypotheses SET elo = ? WHERE id = ?",
                (new_loser, loser_id),
            )
        except Exception:
            await self.db.rollback()
            raise
        await self.db.commit()
        return match.model_copy(update={"id": cursor.lastrowid})

    async def add_feedback(self, feedback: SystemFeedback) -> SystemFeedback:
        cursor = await self.db.execute(
            """
            INSERT INTO feedback (session_id, round, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (feedback.session_id, feedback.round, feedback.content, _dt_text(feedback.created_at)),
        )
        await self.db.commit()
        return feedback.model_copy(update={"id": cursor.lastrowid})

    async def add_overview(self, overview: ResearchOverview) -> ResearchOverview:
        cursor = await self.db.execute(
            """
            INSERT INTO overview (session_id, round, content, top_hypothesis_ids, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                overview.session_id,
                overview.round,
                overview.content,
                _json_dumps(overview.top_hypothesis_ids),
                _dt_text(overview.created_at),
            ),
        )
        await self.db.commit()
        return overview.model_copy(update={"id": cursor.lastrowid})

    async def add_citation(
        self,
        *,
        source: str,
        title: str,
        session_id: str | None = None,
        url: str | None = None,
        doi: str | None = None,
        pmid: str | None = None,
        arxiv_id: str | None = None,
        semantic_scholar_id: str | None = None,
        year: int | None = None,
        raw_json: dict[str, Any] | None = None,
    ) -> int:
        cursor = await self.db.execute(
            """
            INSERT INTO citations (
              session_id, source, title, url, doi, pmid, arxiv_id,
              semantic_scholar_id, year, raw_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                source,
                title,
                url,
                doi,
                pmid,
                arxiv_id,
                semantic_scholar_id,
                year,
                _json_dumps(raw_json or {}),
                _dt_text(utc_now()),
            ),
        )
        await self.db.commit()
        return int(cursor.lastrowid)

    async def add_citations_batch(
        self,
        citations: list[Citation],
        *,
        session_id: str | None = None,
    ) -> int:
        if not citations:
            return 0
        now = _dt_text(utc_now())
        await self.db.executemany(
            """
            INSERT INTO citations (
              session_id, source, title, url, doi, pmid, arxiv_id,
              semantic_scholar_id, year, raw_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    session_id,
                    citation.source,
                    citation.title,
                    citation.url,
                    citation.doi,
                    citation.pmid,
                    citation.arxiv_id,
                    citation.semantic_scholar_id,
                    citation.year,
                    _json_dumps(citation.raw_json),
                    now,
                )
                for citation in citations
            ],
        )
        await self.db.commit()
        return len(citations)

    async def set_tool_cache(
        self,
        *,
        cache_key: str,
        source: str,
        query: str,
        max_results: int,
        options_hash: str,
        status: str,
        result_json: dict[str, Any],
        expires_at: datetime,
    ) -> None:
        now = utc_now()
        await self.db.execute(
            """
            INSERT INTO tool_cache (
              cache_key, source, query, max_results, options_hash,
              status, result_json, created_at, expires_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
              status = excluded.status,
              result_json = excluded.result_json,
              created_at = excluded.created_at,
              expires_at = excluded.expires_at
            """,
            (
                cache_key,
                source,
                query,
                max_results,
                options_hash,
                status,
                _json_dumps(result_json),
                _dt_text(now),
                _dt_text(expires_at),
            ),
        )
        await self.db.commit()

    async def get_tool_cache(
        self,
        cache_key: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        current = now or utc_now()
        async with self.db.execute(
            """
            SELECT result_json FROM tool_cache
            WHERE cache_key = ? AND expires_at > ?
            """,
            (cache_key, _dt_text(current)),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return _json_loads(row["result_json"], {})

    async def purge_expired_tool_cache(self, *, now: datetime | None = None) -> int:
        current = now or utc_now()
        cursor = await self.db.execute(
            "DELETE FROM tool_cache WHERE expires_at <= ?",
            (_dt_text(current),),
        )
        await self.db.commit()
        return cursor.rowcount

    async def count_hypotheses(self, session_id: str) -> int:
        return await self._count("hypotheses", session_id)

    async def count_matches(self, session_id: str) -> int:
        return await self._count("matches", session_id)

    async def _count(self, table: str, session_id: str) -> int:
        async with self.db.execute(
            f"SELECT COUNT(*) AS count FROM {table} WHERE session_id = ?",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return int(row["count"])

    def _row_to_hypothesis(self, row: aiosqlite.Row) -> Hypothesis:
        return Hypothesis(
            id=row["id"],
            session_id=row["session_id"],
            content=row["content"],
            summary=row["summary"],
            detailed_description=row["detailed_description"],
            mechanism=row["mechanism"],
            impacted_pathways=_json_loads(row["impacted_pathways"], []),
            experimental_plan=row["experimental_plan"],
            safety_notes=row["safety_notes"],
            testable_predictions=_json_loads(row["testable_predictions"], []),
            elo=row["elo"],
            parent_ids=_json_loads(row["parent_ids"], []),
            source_strategy=row["source_strategy"],
            created_at=_dt(row["created_at"]),
        )

    def _row_to_task(self, row: aiosqlite.Row) -> Task:
        return Task(
            id=row["id"],
            session_id=row["session_id"],
            agent=row["agent"],
            action=row["action"],
            target_id=row["target_id"],
            priority=row["priority"],
            status=TaskStatus(row["status"]),
            payload_json=_json_loads(row["payload_json"], {}),
            result_json=_json_loads(row["result_json"], None),
            error=row["error"],
            created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]),
        )
