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
    ResearchPlan,
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


def _task_json_loads(value: str | None, default: Any) -> Any:
    if value is None or value == "" or value == "NULL":
        return default
    return _json_loads(value, default)


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
        await self._migrate_schema()
        await self.db.commit()

    async def _migrate_schema(self) -> None:
        await self._ensure_column("citations", "dedupe_key", "TEXT")
        await self._ensure_column("hypotheses", "meta_review_round", "INTEGER")
        await self.db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_matches_session_pair
              ON matches(session_id, hypo_a_id, hypo_b_id)
            """
        )
        await self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS elo_checkpoints (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
              match_id INTEGER REFERENCES matches(id) ON DELETE SET NULL,
              top_k INTEGER NOT NULL,
              avg_elo REAL NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        await self.db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_elo_checkpoints_session_created
              ON elo_checkpoints(session_id, created_at DESC, id DESC)
            """
        )
        await self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS hypothesis_embeddings (
              hypothesis_id INTEGER PRIMARY KEY REFERENCES hypotheses(id) ON DELETE CASCADE,
              session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
              embedding_json TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )
        await self.db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_hypothesis_embeddings_session
              ON hypothesis_embeddings(session_id)
            """
        )
        await self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS proximity_edges (
              session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
              hypo_a_id INTEGER NOT NULL REFERENCES hypotheses(id) ON DELETE CASCADE,
              hypo_b_id INTEGER NOT NULL REFERENCES hypotheses(id) ON DELETE CASCADE,
              similarity REAL NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY(session_id, hypo_a_id, hypo_b_id),
              CHECK(hypo_a_id < hypo_b_id)
            )
            """
        )
        await self.db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_proximity_edges_session_similarity
              ON proximity_edges(session_id, similarity DESC)
            """
        )
        await self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS private_corpus_chunks (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
              path TEXT NOT NULL,
              title TEXT NOT NULL,
              chunk_index INTEGER NOT NULL,
              content TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              mtime REAL NOT NULL,
              embedding_json TEXT,
              updated_at TEXT NOT NULL,
              UNIQUE(session_id, path, chunk_index)
            )
            """
        )
        await self.db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_private_corpus_chunks_session
              ON private_corpus_chunks(session_id)
            """
        )
        await self._ensure_column(
            "private_corpus_chunks",
            "file_size",
            "INTEGER NOT NULL DEFAULT 0",
        )
        await self.migrate_citation_dedupe_keys()
        session_ids = await self._session_ids_with_citations()
        for session_id in session_ids:
            await self.deduplicate_existing_citations(session_id)
        await self.db.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_citations_session_dedupe
              ON citations(session_id, dedupe_key)
            """
        )

    async def _ensure_column(self, table: str, column: str, definition: str) -> None:
        async with self.db.execute(f"PRAGMA table_info({table})") as cursor:
            rows = await cursor.fetchall()
        if any(row["name"] == column for row in rows):
            return
        await self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    async def _session_ids_with_citations(self) -> list[str]:
        async with self.db.execute(
            """
            SELECT DISTINCT session_id FROM citations
            WHERE session_id IS NOT NULL
            """
        ) as cursor:
            rows = await cursor.fetchall()
        return [str(row["session_id"]) for row in rows]

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

    async def update_session_goal(self, session_id: str, goal: str) -> None:
        cursor = await self.db.execute(
            "UPDATE sessions SET goal = ? WHERE id = ?",
            (goal, session_id),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"unknown session: {session_id}")
        await self.db.commit()

    async def save_research_plan(self, plan: ResearchPlan) -> ResearchPlan:
        await self.db.execute(
            """
            INSERT INTO research_plans (
              session_id, goal, preferences, attributes, constraints, idea_attributes, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
              goal = excluded.goal,
              preferences = excluded.preferences,
              attributes = excluded.attributes,
              constraints = excluded.constraints,
              idea_attributes = excluded.idea_attributes,
              created_at = excluded.created_at
            """,
            (
                plan.session_id,
                plan.goal,
                _json_dumps(plan.preferences),
                _json_dumps(plan.attributes),
                _json_dumps(plan.constraints),
                _json_dumps(plan.idea_attributes),
                _dt_text(plan.created_at),
            ),
        )
        await self.db.commit()
        return plan

    async def get_research_plan(self, session_id: str) -> ResearchPlan | None:
        async with self.db.execute(
            "SELECT * FROM research_plans WHERE session_id = ?",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return ResearchPlan(
            session_id=row["session_id"],
            goal=row["goal"],
            preferences=_json_loads(row["preferences"], []),
            attributes=_json_loads(row["attributes"], []),
            constraints=_json_loads(row["constraints"], []),
            idea_attributes=_json_loads(row["idea_attributes"], []),
            created_at=_dt(row["created_at"]),
        )

    async def add_hypothesis(self, hypothesis: Hypothesis) -> Hypothesis:
        cursor = await self.db.execute(
            """
            INSERT INTO hypotheses (
              session_id, content, summary, detailed_description, mechanism,
              impacted_pathways, experimental_plan, safety_notes,
              testable_predictions, elo, parent_ids, source_strategy,
              meta_review_round, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                hypothesis.meta_review_round,
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

    async def list_session_hypotheses(self, session_id: str) -> list[Hypothesis]:
        async with self.db.execute(
            """
            SELECT * FROM hypotheses
            WHERE session_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_hypothesis(row) for row in rows]

    async def list_reviewed_hypotheses(self, session_id: str) -> list[Hypothesis]:
        async with self.db.execute(
            """
            SELECT DISTINCT hypotheses.*
            FROM hypotheses
            JOIN reviews ON reviews.hypothesis_id = hypotheses.id
            WHERE hypotheses.session_id = ?
            ORDER BY hypotheses.created_at ASC, hypotheses.id ASC
            """,
            (session_id,),
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

    async def reviews_for_hypothesis(self, hypothesis_id: int) -> list[Review]:
        async with self.db.execute(
            """
            SELECT * FROM reviews
            WHERE hypothesis_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (hypothesis_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_review(row) for row in rows]

    async def latest_review_for_hypothesis(self, hypothesis_id: int) -> Review | None:
        async with self.db.execute(
            """
            SELECT * FROM reviews
            WHERE hypothesis_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (hypothesis_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return self._row_to_review(row) if row else None

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

    async def has_active_task(
        self,
        session_id: str,
        *,
        agent: str,
        action: str,
        exclude_task_id: int | None = None,
    ) -> bool:
        id_filter = "" if exclude_task_id is None else "AND id != ?"
        params: tuple[Any, ...] = (
            session_id,
            agent,
            action,
            TaskStatus.PENDING.value,
            TaskStatus.RUNNING.value,
        )
        if exclude_task_id is not None:
            params = params + (exclude_task_id,)
        async with self.db.execute(
            f"""
            SELECT 1 FROM tasks
            WHERE session_id = ?
              AND agent = ?
              AND action = ?
              AND status IN (?, ?)
              {id_filter}
            LIMIT 1
            """,
            params,
        ) as cursor:
            row = await cursor.fetchone()
        return row is not None

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

    async def reset_tournament_for_goal_revision(self, session_id: str) -> int:
        """Reset goal-dependent derived state before re-reviewing hypotheses."""
        if await self.get_session(session_id) is None:
            raise ValueError(f"unknown session: {session_id}")
        if await self.has_running_tasks(session_id):
            raise ValueError(
                "cannot revise goal while tasks are running; stop the supervisor first"
            )
        now = _dt_text(utc_now())
        try:
            await self.db.execute(
                """
                UPDATE hypotheses
                SET elo = 1200, meta_review_round = NULL
                WHERE session_id = ?
                """,
                (session_id,),
            )
            await self.db.execute("DELETE FROM matches WHERE session_id = ?", (session_id,))
            await self.db.execute(
                "DELETE FROM elo_checkpoints WHERE session_id = ?",
                (session_id,),
            )
            await self.db.execute(
                "DELETE FROM proximity_edges WHERE session_id = ?",
                (session_id,),
            )
            await self.db.execute("DELETE FROM feedback WHERE session_id = ?", (session_id,))
            await self.db.execute("DELETE FROM overview WHERE session_id = ?", (session_id,))
            cursor = await self.db.execute(
                """
                UPDATE tasks
                SET status = ?, updated_at = ?
                WHERE session_id = ? AND status = ?
                """,
                (
                    TaskStatus.CANCELLED.value,
                    now,
                    session_id,
                    TaskStatus.PENDING.value,
                ),
            )
        except Exception:
            await self.db.rollback()
            raise
        await self.db.commit()
        return cursor.rowcount

    async def has_running_tasks(self, session_id: str) -> bool:
        async with self.db.execute(
            """
            SELECT 1 FROM tasks
            WHERE session_id = ? AND status = ?
            LIMIT 1
            """,
            (session_id, TaskStatus.RUNNING.value),
        ) as cursor:
            row = await cursor.fetchone()
        return row is not None

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

    async def match_counts_by_hypothesis(self, session_id: str) -> dict[int, int]:
        async with self.db.execute(
            """
            SELECT hypothesis_id, COUNT(*) AS count
            FROM (
              SELECT hypo_a_id AS hypothesis_id
              FROM matches
              WHERE session_id = ?
              UNION ALL
              SELECT hypo_b_id AS hypothesis_id
              FROM matches
              WHERE session_id = ?
            )
            GROUP BY hypothesis_id
            """,
            (session_id, session_id),
        ) as cursor:
            rows = await cursor.fetchall()
        return {int(row["hypothesis_id"]): int(row["count"]) for row in rows}

    async def match_counts_by_pair(self, session_id: str) -> dict[tuple[int, int], int]:
        async with self.db.execute(
            """
            SELECT
              CASE WHEN hypo_a_id < hypo_b_id THEN hypo_a_id ELSE hypo_b_id END AS a_id,
              CASE WHEN hypo_a_id < hypo_b_id THEN hypo_b_id ELSE hypo_a_id END AS b_id,
              COUNT(*) AS count
            FROM matches
            WHERE session_id = ?
            GROUP BY a_id, b_id
            """,
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return {(int(row["a_id"]), int(row["b_id"])): int(row["count"]) for row in rows}

    async def upsert_hypothesis_embedding(
        self,
        *,
        session_id: str,
        hypothesis_id: int,
        embedding: list[float],
    ) -> None:
        await self.upsert_hypothesis_embeddings_batch(
            session_id=session_id,
            embeddings={hypothesis_id: embedding},
        )

    async def upsert_hypothesis_embeddings_batch(
        self,
        *,
        session_id: str,
        embeddings: dict[int, list[float]],
    ) -> None:
        if not embeddings:
            return
        now = _dt_text(utc_now())
        await self.db.executemany(
            """
            INSERT INTO hypothesis_embeddings (
              hypothesis_id, session_id, embedding_json, updated_at
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(hypothesis_id) DO UPDATE SET
              embedding_json = excluded.embedding_json,
              updated_at = excluded.updated_at
            """,
            [
                (hypothesis_id, session_id, _json_dumps(embedding), now)
                for hypothesis_id, embedding in embeddings.items()
            ],
        )
        await self.db.commit()

    async def embeddings_for_session(self, session_id: str) -> dict[int, list[float]]:
        async with self.db.execute(
            """
            SELECT hypothesis_id, embedding_json
            FROM hypothesis_embeddings
            WHERE session_id = ?
            """,
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return {
            int(row["hypothesis_id"]): [
                float(item) for item in _json_loads(row["embedding_json"], [])
            ]
            for row in rows
        }

    async def upsert_proximity_edge(
        self,
        *,
        session_id: str,
        hypo_a_id: int,
        hypo_b_id: int,
        similarity: float,
    ) -> None:
        await self.upsert_proximity_edges_batch(
            session_id=session_id,
            edges=[(hypo_a_id, hypo_b_id, similarity)],
        )

    async def upsert_proximity_edges_batch(
        self,
        *,
        session_id: str,
        edges: list[tuple[int, int, float]],
    ) -> None:
        if not edges:
            return
        now = _dt_text(utc_now())
        rows = []
        for hypo_a_id, hypo_b_id, similarity in edges:
            a_id, b_id = sorted((hypo_a_id, hypo_b_id))
            if a_id == b_id:
                raise ValueError("proximity edge requires two distinct hypotheses")
            rows.append((session_id, a_id, b_id, similarity, now))
        await self.db.executemany(
            """
            INSERT INTO proximity_edges (
              session_id, hypo_a_id, hypo_b_id, similarity, updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id, hypo_a_id, hypo_b_id) DO UPDATE SET
              similarity = excluded.similarity,
              updated_at = excluded.updated_at
            """,
            rows,
        )
        await self.db.commit()

    async def proximity_edges_for_session(self, session_id: str) -> dict[tuple[int, int], float]:
        async with self.db.execute(
            """
            SELECT hypo_a_id, hypo_b_id, similarity
            FROM proximity_edges
            WHERE session_id = ?
            """,
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return {
            (int(row["hypo_a_id"]), int(row["hypo_b_id"])): float(row["similarity"])
            for row in rows
        }

    async def replace_private_corpus_file_chunks(
        self,
        *,
        session_id: str,
        path: str,
        chunks: list[dict[str, Any]],
    ) -> list[int]:
        now = _dt_text(utc_now())
        ids: list[int] = []
        try:
            await self.db.execute(
                """
                DELETE FROM private_corpus_chunks
                WHERE session_id = ? AND path = ?
                """,
                (session_id, path),
            )
            for chunk in chunks:
                cursor = await self.db.execute(
                    """
                    INSERT INTO private_corpus_chunks (
                      session_id, path, title, chunk_index, content,
                      content_hash, mtime, file_size, embedding_json, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        path,
                        chunk["title"],
                        int(chunk["chunk_index"]),
                        chunk["content"],
                        chunk["content_hash"],
                        float(chunk["mtime"]),
                        int(chunk.get("file_size", 0)),
                        (
                            _json_dumps(chunk["embedding"])
                            if chunk.get("embedding") is not None
                            else None
                        ),
                        now,
                    ),
                )
                ids.append(int(cursor.lastrowid))
        except Exception:
            await self.db.rollback()
            raise
        await self.db.commit()
        return ids

    async def private_corpus_file_state(
        self,
        *,
        session_id: str,
        path: str,
    ) -> dict[str, Any] | None:
        async with self.db.execute(
            """
            SELECT mtime, file_size, content_hash
            FROM private_corpus_chunks
            WHERE session_id = ? AND path = ?
            ORDER BY chunk_index ASC
            """,
            (session_id, path),
        ) as cursor:
            rows = await cursor.fetchall()
        if not rows:
            return None
        return {
            "mtime": float(rows[0]["mtime"]),
            "file_size": int(rows[0]["file_size"]),
            "content_hashes": [str(row["content_hash"]) for row in rows],
        }

    async def list_private_corpus_chunks(self, session_id: str) -> list[dict[str, Any]]:
        async with self.db.execute(
            """
            SELECT *
            FROM private_corpus_chunks
            WHERE session_id = ?
            ORDER BY path ASC, chunk_index ASC
            """,
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            {
                **dict(row),
                "embedding": (
                    [float(item) for item in _json_loads(row["embedding_json"], [])]
                    if row["embedding_json"]
                    else None
                ),
            }
            for row in rows
        ]

    async def update_private_corpus_embeddings(
        self,
        *,
        session_id: str,
        embeddings: dict[int, list[float]],
    ) -> None:
        if not embeddings:
            return
        now = _dt_text(utc_now())
        await self.db.executemany(
            """
            UPDATE private_corpus_chunks
            SET embedding_json = ?, updated_at = ?
            WHERE session_id = ? AND id = ?
            """,
            [
                (_json_dumps(embedding), now, session_id, chunk_id)
                for chunk_id, embedding in embeddings.items()
            ],
        )
        await self.db.commit()

    async def count_private_corpus_chunks(self, session_id: str) -> int:
        return await self._count("private_corpus_chunks", session_id)

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

    async def latest_feedback(self, session_id: str) -> SystemFeedback | None:
        async with self.db.execute(
            """
            SELECT * FROM feedback
            WHERE session_id = ?
            ORDER BY round DESC, id DESC
            LIMIT 1
            """,
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return self._row_to_feedback(row) if row else None

    async def count_feedback(self, session_id: str) -> int:
        return await self._count("feedback", session_id)

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

    async def latest_overview(self, session_id: str) -> ResearchOverview | None:
        async with self.db.execute(
            """
            SELECT * FROM overview
            WHERE session_id = ?
            ORDER BY round DESC, id DESC
            LIMIT 1
            """,
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return self._row_to_overview(row) if row else None

    async def recent_reviews(self, session_id: str, *, limit: int = 20) -> list[Review]:
        async with self.db.execute(
            """
            SELECT * FROM reviews
            WHERE session_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_review(row) for row in rows]

    async def recent_matches(self, session_id: str, *, limit: int = 20) -> list[Match]:
        async with self.db.execute(
            """
            SELECT * FROM matches
            WHERE session_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_match(row) for row in rows]

    async def add_elo_checkpoint(
        self,
        session_id: str,
        *,
        match_id: int | None = None,
        top_k: int = 5,
    ) -> dict[str, Any] | None:
        top = await self.top_k_by_elo(session_id, k=top_k)
        if not top:
            return None
        avg_elo = sum(hypothesis.elo for hypothesis in top) / len(top)
        now = _dt_text(utc_now())
        cursor = await self.db.execute(
            """
            INSERT INTO elo_checkpoints (session_id, match_id, top_k, avg_elo, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, match_id, top_k, avg_elo, now),
        )
        await self.db.commit()
        return {
            "id": int(cursor.lastrowid),
            "session_id": session_id,
            "match_id": match_id,
            "top_k": top_k,
            "avg_elo": avg_elo,
            "created_at": now,
        }

    async def latest_elo_checkpoints(
        self,
        session_id: str,
        *,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        async with self.db.execute(
            """
            SELECT * FROM elo_checkpoints
            WHERE session_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

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
        from co_scientist.tools.models import Citation

        citation = Citation(
            source=source,
            title=title,
            url=url,
            doi=doi,
            pmid=pmid,
            arxiv_id=arxiv_id,
            semantic_scholar_id=semantic_scholar_id,
            year=year,
            raw_json=raw_json or {},
        )
        ids = await self.add_citations_batch([citation], session_id=session_id)
        return ids[0]

    async def add_citations_batch(
        self,
        citations: list[Citation],
        *,
        session_id: str | None = None,
    ) -> list[int]:
        if not citations:
            return []
        now = _dt_text(utc_now())
        ids: list[int] = []
        seen: set[str] = set()
        for citation in citations:
            dedupe_key = citation.dedupe_key()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            await self.db.execute(
                """
                INSERT INTO citations (
                  session_id, dedupe_key, source, title, url, doi, pmid, arxiv_id,
                  semantic_scholar_id, year, raw_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, dedupe_key) DO UPDATE SET
                  title = COALESCE(NULLIF(excluded.title, ''), citations.title),
                  url = COALESCE(excluded.url, citations.url),
                  doi = COALESCE(excluded.doi, citations.doi),
                  pmid = COALESCE(excluded.pmid, citations.pmid),
                  arxiv_id = COALESCE(excluded.arxiv_id, citations.arxiv_id),
                  semantic_scholar_id = COALESCE(
                    excluded.semantic_scholar_id,
                    citations.semantic_scholar_id
                  ),
                  year = COALESCE(excluded.year, citations.year),
                  raw_json = excluded.raw_json
                """,
                (
                    session_id,
                    dedupe_key,
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
                ),
            )
            async with self.db.execute(
                """
                SELECT id FROM citations
                WHERE session_id IS ? AND dedupe_key = ?
                """,
                (session_id, dedupe_key),
            ) as select_cursor:
                row = await select_cursor.fetchone()
            if row is None:
                raise RuntimeError("failed to resolve citation id after upsert")
            ids.append(int(row["id"]))
        await self.db.commit()
        return ids

    async def add_citation_links(
        self,
        citation_ids: list[int],
        *,
        session_id: str,
        artifact_type: str,
        artifact_id: int,
        source_task_id: int | None = None,
    ) -> int:
        if not citation_ids:
            return 0
        now = _dt_text(utc_now())
        await self.db.executemany(
            """
            INSERT OR IGNORE INTO citation_links (
              session_id, citation_id, artifact_type, artifact_id,
              source_task_id, evidence_index, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    session_id,
                    citation_id,
                    artifact_type,
                    artifact_id,
                    source_task_id,
                    index,
                    now,
                )
                for index, citation_id in enumerate(citation_ids, start=1)
            ],
        )
        await self.db.commit()
        return len(citation_ids)

    async def add_citations_for_artifact(
        self,
        citations: list[Citation],
        *,
        session_id: str,
        artifact_type: str,
        artifact_id: int,
        source_task_id: int | None = None,
    ) -> list[int]:
        citation_ids = await self.add_citations_batch(citations, session_id=session_id)
        await self.add_citation_links(
            citation_ids,
            session_id=session_id,
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            source_task_id=source_task_id,
        )
        return citation_ids

    async def list_citations(self, session_id: str) -> list[dict[str, Any]]:
        async with self.db.execute(
            """
            SELECT * FROM citations
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def citation_links_for_artifact(
        self,
        *,
        session_id: str,
        artifact_type: str,
        artifact_id: int,
    ) -> list[dict[str, Any]]:
        async with self.db.execute(
            """
            SELECT citation_links.evidence_index, citations.*
            FROM citation_links
            JOIN citations ON citations.id = citation_links.citation_id
            WHERE citation_links.session_id = ?
              AND citation_links.artifact_type = ?
              AND citation_links.artifact_id = ?
            ORDER BY citation_links.evidence_index ASC
            """,
            (session_id, artifact_type, artifact_id),
        ) as cursor:
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def migrate_citation_dedupe_keys(self) -> None:
        async with self.db.execute(
            """
            SELECT id, source, title, url, doi, pmid, arxiv_id,
                   semantic_scholar_id, year, raw_json
            FROM citations
            WHERE dedupe_key IS NULL OR dedupe_key = ''
            """
        ) as cursor:
            rows = await cursor.fetchall()
        if not rows:
            return
        from co_scientist.tools.models import Citation

        for row in rows:
            citation = Citation(
                source=row["source"],
                title=row["title"],
                url=row["url"],
                doi=row["doi"],
                pmid=row["pmid"],
                arxiv_id=row["arxiv_id"],
                semantic_scholar_id=row["semantic_scholar_id"],
                year=row["year"],
                raw_json=_json_loads(row["raw_json"], {}),
            )
            await self.db.execute(
                "UPDATE citations SET dedupe_key = ? WHERE id = ?",
                (citation.dedupe_key(), row["id"]),
            )
        await self.db.commit()

    async def deduplicate_existing_citations(self, session_id: str) -> int:
        await self.migrate_citation_dedupe_keys()
        async with self.db.execute(
            """
            SELECT dedupe_key, MIN(id) AS keep_id, COUNT(*) AS count
            FROM citations
            WHERE session_id = ?
            GROUP BY dedupe_key
            HAVING count > 1
            """,
            (session_id,),
        ) as cursor:
            duplicate_groups = await cursor.fetchall()
        removed = 0
        for group in duplicate_groups:
            keep_id = int(group["keep_id"])
            async with self.db.execute(
                """
                SELECT id FROM citations
                WHERE session_id = ? AND dedupe_key = ? AND id != ?
                """,
                (session_id, group["dedupe_key"], keep_id),
            ) as cursor:
                duplicate_rows = await cursor.fetchall()
            duplicate_ids = [int(row["id"]) for row in duplicate_rows]
            for duplicate_id in duplicate_ids:
                await self.db.execute(
                    "UPDATE citation_links SET citation_id = ? WHERE citation_id = ?",
                    (keep_id, duplicate_id),
                )
            if duplicate_ids:
                placeholders = ",".join("?" for _ in duplicate_ids)
                await self.db.execute(
                    f"DELETE FROM citations WHERE id IN ({placeholders})",
                    duplicate_ids,
                )
                removed += len(duplicate_ids)
        await self.db.commit()
        return removed

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

    async def count_reviews(self, session_id: str) -> int:
        return await self._count("reviews", session_id)

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
            meta_review_round=row["meta_review_round"],
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
            payload_json=_task_json_loads(row["payload_json"], {}),
            result_json=_task_json_loads(row["result_json"], None),
            error=row["error"],
            created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]),
        )

    def _row_to_review(self, row: aiosqlite.Row) -> Review:
        return Review(
            id=row["id"],
            session_id=row["session_id"],
            hypothesis_id=row["hypothesis_id"],
            type=row["type"],
            score=row["score"],
            content=row["content"],
            created_at=_dt(row["created_at"]),
        )

    def _row_to_match(self, row: aiosqlite.Row) -> Match:
        return Match(
            id=row["id"],
            session_id=row["session_id"],
            hypo_a_id=row["hypo_a_id"],
            hypo_b_id=row["hypo_b_id"],
            winner_id=row["winner_id"],
            transcript=row["transcript"],
            created_at=_dt(row["created_at"]),
        )

    def _row_to_feedback(self, row: aiosqlite.Row) -> SystemFeedback:
        return SystemFeedback(
            id=row["id"],
            session_id=row["session_id"],
            round=row["round"],
            content=row["content"],
            created_at=_dt(row["created_at"]),
        )

    def _row_to_overview(self, row: aiosqlite.Row) -> ResearchOverview:
        return ResearchOverview(
            id=row["id"],
            session_id=row["session_id"],
            round=row["round"],
            content=row["content"],
            top_hypothesis_ids=_json_loads(row["top_hypothesis_ids"], []),
            created_at=_dt(row["created_at"]),
        )
