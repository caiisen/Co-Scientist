from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import aiohttp

from co_scientist.agents.base import Agent, AgentContext
from co_scientist.agents.generation import GenerationAgent, hypothesis_from_payload
from co_scientist.agents.proximity import ProximityAgent
from co_scientist.agents.ranking import RankingAgent
from co_scientist.agents.reflection import ReflectionAgent, review_from_payload
from co_scientist.agents.results import AgentResult, AgentResultKind
from co_scientist.config import AppConfig
from co_scientist.llm.client import LLMRouter
from co_scientist.memory.models import Match, Task, TaskPriority
from co_scientist.memory.store import SQLiteStore
from co_scientist.supervisor.planner import create_research_plan
from co_scientist.supervisor.task_queue import TaskQueue
from co_scientist.tools.models import Citation


class Supervisor:
    def __init__(
        self,
        *,
        store: SQLiteStore,
        config: AppConfig,
        llm_router: LLMRouter | None = None,
        agents: dict[str, Agent] | None = None,
        verbose: bool = False,
        event_sink: Callable[[str], None] | None = None,
    ) -> None:
        self.store = store
        self.config = config
        self.llm_router = llm_router or LLMRouter(config.llm)
        self.verbose = verbose
        self.event_sink = event_sink or print
        self.agents = agents or {
            "generation": GenerationAgent(),
            "reflection": ReflectionAgent(),
            "proximity": ProximityAgent(),
            "ranking": RankingAgent(),
        }

    async def start(self, goal: str) -> str:
        session = await self.store.create_session(
            goal,
            config_json=_redacted_config_json(self.config),
        )
        self._log(
            "[phase:start] session="
            f"{session.id} max_ideas={self.config.runtime.max_ideas} "
            f"max_matches_per_idea={self.config.runtime.max_matches_per_idea} "
            f"workers={self.config.runtime.worker_concurrency}"
        )
        self._log(f"[phase:start:input] goal={_truncate(goal)}")
        await self.store.purge_expired_tool_cache()
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.runtime.request_timeout_seconds)
        ) as http_session:
            ctx = self._context(session.id, http_session=http_session)
            self._log("[phase:planner] creating research plan")
            plan = await create_research_plan(goal, ctx)
            self._log(
                "[phase:planner:output] "
                f"preferences={_compact_json(plan.preferences)} "
                f"attributes={_compact_json(plan.attributes)} "
                f"constraints={_compact_json(plan.constraints)} "
                f"idea_attributes={_compact_json(plan.idea_attributes)}"
            )
            queue = TaskQueue(self.store, session.id)
            task = await queue.enqueue(
                Task(
                    session_id=session.id,
                    agent="generation",
                    action="create_initial_hypotheses",
                    priority=int(TaskPriority.USER),
                )
            )
            self._log(f"[queue] enqueued {_task_name(task)} id={task.id}")
            await self._run_queue(queue, ctx)
        self._log(f"[phase:done] session={session.id}")
        return session.id

    async def resume(self, session_id: str) -> None:
        session = await self.store.get_session(session_id)
        if session is None:
            raise ValueError(f"unknown session: {session_id}")
        self._log(f"[phase:resume] session={session_id}")
        await self.store.purge_expired_tool_cache()
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.runtime.request_timeout_seconds)
        ) as http_session:
            ctx = self._context(session_id, http_session=http_session)
            if await self.store.get_research_plan(session_id) is None:
                self._log("[phase:planner] missing plan; creating research plan")
                plan = await create_research_plan(session.goal, ctx)
                self._log(
                    "[phase:planner:output] "
                    f"preferences={_compact_json(plan.preferences)} "
                    f"attributes={_compact_json(plan.attributes)} "
                    f"constraints={_compact_json(plan.constraints)} "
                    f"idea_attributes={_compact_json(plan.idea_attributes)}"
                )
            queue = TaskQueue(self.store, session_id)
            await queue.load_pending()
            self._log(f"[queue] loaded pending tasks count={queue.qsize()}")
            await self._run_queue(queue, ctx)
        self._log(f"[phase:done] session={session_id}")

    async def export_markdown(self, session_id: str) -> str:
        session = await self.store.get_session(session_id)
        if session is None:
            raise ValueError(f"unknown session: {session_id}")
        plan = await self.store.get_research_plan(session_id)
        hypotheses = await self.store.list_session_hypotheses(session_id)
        citations = await self.store.list_citations(session_id)
        reference_numbers = {
            int(citation["id"]): index
            for index, citation in enumerate(citations, start=1)
        }

        lines = [
            f"# Co-Scientist Phase 5 Report: {session_id}",
            "",
            "## Goal",
            session.goal,
            "",
        ]
        if plan is not None:
            lines.extend(
                [
                    "## Research Plan",
                    _list_section("Preferences", plan.preferences),
                    _list_section("Attributes", plan.attributes),
                    _list_section("Constraints", plan.constraints),
                    _list_section("Idea Attributes", plan.idea_attributes),
                    "",
                ]
            )
        lines.append("## Hypotheses and Full Reviews")
        for index, hypothesis in enumerate(hypotheses, start=1):
            lines.extend(
                [
                    "",
                    f"### {index}. {hypothesis.summary}",
                    "",
                    hypothesis.content,
                    "",
                    f"Source strategy: `{hypothesis.source_strategy or 'unknown'}`",
                ]
            )
            if hypothesis.id is None:
                continue
            lines.append(f"Elo: `{hypothesis.elo}`")
            reviews = await self.store.reviews_for_hypothesis(hypothesis.id)
            for review in reviews:
                links = await self.store.citation_links_for_artifact(
                    session_id=session_id,
                    artifact_type="review",
                    artifact_id=review.id or 0,
                )
                lines.extend(
                    [
                        "",
                        f"#### {review.type.title()} Review",
                        "",
                        (
                            f"Score: {review.score}"
                            if review.score is not None
                            else "Score: not parsed"
                        ),
                        "",
                        review.content,
                    ]
                )
                if links:
                    lines.extend(
                        [
                            "",
                            "Evidence references: "
                            + ", ".join(
                                f"[{link['evidence_index']}] -> [R{reference_numbers[link['id']]}]"
                                for link in links
                                if int(link["id"]) in reference_numbers
                            ),
                        ]
                    )
        stats = await self._tournament_stats(session_id)
        lines.extend(
            [
                "",
                "## Tournament Summary",
                "",
                f"Matches: {stats['match_count']}",
                f"Matches per idea: {stats['matches_per_idea']:.2f}",
                "",
                "Top hypotheses by Elo:",
            ]
        )
        for hypothesis in await self.store.top_k_by_elo(session_id, k=5):
            lines.append(f"- `{hypothesis.elo}` {hypothesis.summary}")
        if citations:
            lines.extend(["", "## References"])
            for index, citation in enumerate(citations, start=1):
                lines.extend(["", _format_reference(index, citation)])
        return "\n".join(lines).rstrip() + "\n"

    def _context(
        self,
        session_id: str,
        *,
        http_session: aiohttp.ClientSession | None,
    ) -> AgentContext:
        return AgentContext(
            store=self.store,
            llm_router=self.llm_router,
            config=self.config,
            session_id=session_id,
            http_session=http_session,
        )

    async def _run_queue(self, queue: TaskQueue, ctx: AgentContext) -> None:
        workers = [
            asyncio.create_task(self._run_worker(queue, ctx))
            for _ in range(self.config.runtime.worker_concurrency)
        ]
        try:
            await queue.join()
        finally:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

    async def _run_worker(self, queue: TaskQueue, ctx: AgentContext) -> None:
        while True:
            try:
                await self._run_one(queue, ctx)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log(f"[worker:error] {type(exc).__name__}: {_truncate(str(exc))}")
                await asyncio.sleep(0)

    async def _run_one(self, queue: TaskQueue, ctx: AgentContext) -> None:
        task = await queue.dequeue()
        self._log(
            f"[task:start] id={task.id} {_task_name(task)} target={task.target_id} "
            f"input={_compact_json(task.payload_json)}"
        )
        try:
            agent = self.agents[task.agent]
            result = await agent.execute(task, ctx)
            self._log(
                f"[task:output] id={task.id} kind={result.kind.value} "
                f"payload={_compact_json(_summarize_payload(result.payload))} "
                f"raw={_truncate(result.raw_text or '')}"
            )
            await self._handle_result(queue, task, result)
            self._log(f"[task:done] id={task.id} {_task_name(task)}")
        except Exception as exc:
            if task.id is None:
                raise
            await queue.mark_failed(task.id, str(exc))
            self._log(
                f"[task:failed] id={task.id} {_task_name(task)} "
                f"error={type(exc).__name__}: {_truncate(str(exc))}"
            )

    async def _handle_result(
        self,
        queue: TaskQueue,
        task: Task,
        result: AgentResult,
    ) -> None:
        if task.id is None:
            raise ValueError("stored task must have id")
        if not result.ok:
            await queue.mark_failed(
                task.id,
                result.parse_error or "agent parse error",
                result_json=_result_debug_json(result),
            )
            return

        stored_ids: list[int] = []
        if result.kind == AgentResultKind.HYPOTHESIS_CREATED:
            stored_ids = await self._store_hypotheses_and_enqueue_reviews(queue, task, result)
        elif result.kind == AgentResultKind.REVIEW_COMPLETED:
            stored_review = await self.store.add_review(
                review_from_payload(task.session_id, result.payload)
            )
            stored_ids = [stored_review.id] if stored_review.id is not None else []
            if stored_review.id is not None:
                await self.store.add_citations_for_artifact(
                    result.citations,
                    session_id=task.session_id,
                    artifact_type="review",
                    artifact_id=stored_review.id,
                    source_task_id=task.id,
                )
                if "proximity" in self.agents:
                    enqueued = await queue.enqueue_unique_action(
                        Task(
                            session_id=task.session_id,
                            agent="proximity",
                            action="update_proximity_graph",
                            priority=int(TaskPriority.PROXIMITY),
                        )
                    )
                    if enqueued is not None:
                        self._log(f"[queue] enqueued {_task_name(enqueued)} id={enqueued.id}")
        elif result.kind == AgentResultKind.PROXIMITY_UPDATED:
            await self._maybe_enqueue_ranking(queue, task.session_id, exclude_task_id=task.id)
        ranking_audit: dict[str, Any] = {}
        if result.kind == AgentResultKind.RANKING_DECISION:
            stored_match = await self.store.add_match_and_update_elo(
                Match(
                    session_id=task.session_id,
                    hypo_a_id=int(result.payload["hypo_a_id"]),
                    hypo_b_id=int(result.payload["hypo_b_id"]),
                    winner_id=int(result.payload["winner_id"]),
                    transcript=str(result.payload["transcript"]),
                )
            )
            stored_ids = [stored_match.id] if stored_match.id is not None else []
            ranking_audit = {
                "match_id": stored_match.id,
                "hypo_a_id": result.payload.get("hypo_a_id"),
                "hypo_b_id": result.payload.get("hypo_b_id"),
                "winner_id": result.payload.get("winner_id"),
            }
            await self._maybe_enqueue_ranking(queue, task.session_id, exclude_task_id=task.id)

        await queue.mark_done(
            task.id,
            _result_debug_json(result) | {"stored_ids": stored_ids} | ranking_audit,
        )

    async def _store_hypotheses_and_enqueue_reviews(
        self,
        queue: TaskQueue,
        task: Task,
        result: AgentResult,
    ) -> list[int]:
        payloads = result.payload.get("hypotheses", [result.payload])
        hypotheses = [
            hypothesis_from_payload(task.session_id, payload)
            for payload in payloads
        ]
        stored_ids: list[int] = []
        for candidate, payload in zip(hypotheses, payloads, strict=True):
            hypothesis = await self.store.add_hypothesis(candidate)
            if hypothesis.id is None:
                continue
            stored_ids.append(hypothesis.id)
            await self.store.add_citations_for_artifact(
                _citations_from_payload(payload),
                session_id=task.session_id,
                artifact_type="hypothesis",
                artifact_id=hypothesis.id,
                source_task_id=task.id,
            )
            review_task = await queue.enqueue(
                Task(
                    session_id=task.session_id,
                    agent="reflection",
                    action="full_review",
                    target_id=hypothesis.id,
                    priority=int(TaskPriority.REFLECTION),
                )
            )
            self._log(f"[queue] enqueued {_task_name(review_task)} id={review_task.id}")
        return stored_ids

    async def _maybe_enqueue_ranking(
        self,
        queue: TaskQueue,
        session_id: str,
        *,
        exclude_task_id: int | None = None,
    ) -> None:
        if "ranking" not in self.agents:
            return
        if await self._ranking_target_reached(session_id):
            return
        enqueued = await queue.enqueue_unique_action(
            Task(
                session_id=session_id,
                agent="ranking",
                action="run_tournament_match",
                priority=int(TaskPriority.RANKING),
            ),
            exclude_task_id=exclude_task_id,
        )
        if enqueued is not None:
            self._log(f"[queue] enqueued {_task_name(enqueued)} id={enqueued.id}")

    async def _ranking_target_reached(self, session_id: str) -> bool:
        hypotheses = await self.store.list_reviewed_hypotheses(session_id)
        if len(hypotheses) < 2:
            return True
        match_count = await self.store.count_matches(session_id)
        return match_count >= len(hypotheses) * self.config.runtime.max_matches_per_idea

    async def _tournament_stats(self, session_id: str) -> dict[str, float | int]:
        hypothesis_count = await self.store.count_hypotheses(session_id)
        match_count = await self.store.count_matches(session_id)
        return {
            "match_count": match_count,
            "matches_per_idea": match_count / hypothesis_count if hypothesis_count else 0.0,
        }

    def _log(self, message: str) -> None:
        if self.verbose:
            self.event_sink(message)


def _list_section(title: str, items: list[str]) -> str:
    if not items:
        return f"**{title}:** not specified"
    return f"**{title}:**\n" + "\n".join(f"- {item}" for item in items)


def _result_debug_json(result: AgentResult) -> dict[str, Any]:
    data: dict[str, Any] = {"kind": result.kind.value}
    if result.raw_text is not None:
        data["raw_text"] = result.raw_text
    if result.parse_error is not None:
        data["parse_error"] = result.parse_error
    return data


def _task_name(task: Task) -> str:
    return f"{task.agent}.{task.action}"


def _compact_json(value: Any, *, max_chars: int = 360) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        text = str(value)
    return _truncate(text, max_chars=max_chars)


def _truncate(text: str, *, max_chars: int = 500) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "hypotheses" and isinstance(value, list):
            summary[key] = [
                {
                    "summary": item.get("summary"),
                    "source_strategy": item.get("source_strategy"),
                    "query_variant": item.get("query_variant"),
                }
                if isinstance(item, dict)
                else str(item)
                for item in value
            ]
        elif key == "content" and isinstance(value, str):
            summary[key] = _truncate(value, max_chars=180)
        elif key == "transcript" and isinstance(value, str):
            summary[key] = _truncate(value, max_chars=180)
        elif key == "edges" and isinstance(value, list):
            summary[key] = f"{len(value)} edge(s)"
        else:
            summary[key] = value
    return summary


def _citations_from_payload(payload: dict[str, Any]) -> list[Citation]:
    citations = payload.get("citations") or []
    return [Citation.model_validate(citation) for citation in citations]


def _format_reference(index: int, citation: dict[str, Any]) -> str:
    parts = [f"[R{index}] {citation['title']}"]
    if citation.get("year"):
        parts.append(str(citation["year"]))
    identifiers = []
    if citation.get("doi"):
        identifiers.append(f"DOI: {citation['doi']}")
    if citation.get("pmid"):
        identifiers.append(f"PMID: {citation['pmid']}")
    if citation.get("arxiv_id"):
        identifiers.append(f"arXiv: {citation['arxiv_id']}")
    if citation.get("semantic_scholar_id"):
        identifiers.append(f"Semantic Scholar: {citation['semantic_scholar_id']}")
    if citation.get("url"):
        identifiers.append(citation["url"])
    if identifiers:
        parts.append("; ".join(identifiers))
    return ". ".join(parts) + "."


def _redacted_config_json(config: AppConfig) -> dict[str, Any]:
    return _redact_secrets(config.model_dump())


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if key == "api_key" and item:
                redacted[key] = "<redacted>"
            else:
                redacted[key] = _redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value


async def run_new_session(
    *,
    db_path: str | Path,
    config: AppConfig,
    goal: str,
    supervisor: Supervisor | None = None,
    verbose: bool = False,
) -> str:
    async with SQLiteStore(db_path) as store:
        runner = supervisor or Supervisor(store=store, config=config, verbose=verbose)
        return await runner.start(goal)


async def run_resume_session(
    *,
    db_path: str | Path,
    config: AppConfig,
    session_id: str,
    supervisor: Supervisor | None = None,
    verbose: bool = False,
) -> None:
    async with SQLiteStore(db_path) as store:
        runner = supervisor or Supervisor(store=store, config=config, verbose=verbose)
        await runner.resume(session_id)


async def export_session_markdown(
    *,
    db_path: str | Path,
    config: AppConfig,
    session_id: str,
    supervisor: Supervisor | None = None,
) -> str:
    async with SQLiteStore(db_path) as store:
        runner = supervisor or Supervisor(store=store, config=config)
        return await runner.export_markdown(session_id)


def result_json_kind(value: dict[str, Any] | None) -> str | None:
    if value is None:
        return None
    kind = value.get("kind")
    return str(kind) if kind is not None else None
