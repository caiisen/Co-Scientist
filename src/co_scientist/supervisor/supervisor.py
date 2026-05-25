from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import aiohttp

from co_scientist.agents.base import Agent, AgentContext
from co_scientist.agents.generation import GenerationAgent, hypothesis_from_payload
from co_scientist.agents.reflection import ReflectionAgent, review_from_payload
from co_scientist.agents.results import AgentResult, AgentResultKind
from co_scientist.config import AppConfig
from co_scientist.llm.client import LLMRouter
from co_scientist.memory.models import Task, TaskPriority
from co_scientist.memory.store import SQLiteStore
from co_scientist.supervisor.planner import create_research_plan
from co_scientist.supervisor.task_queue import TaskQueue


class Supervisor:
    def __init__(
        self,
        *,
        store: SQLiteStore,
        config: AppConfig,
        llm_router: LLMRouter | None = None,
        agents: dict[str, Agent] | None = None,
    ) -> None:
        self.store = store
        self.config = config
        self.llm_router = llm_router or LLMRouter(config.llm)
        self.agents = agents or {
            "generation": GenerationAgent(),
            "reflection": ReflectionAgent(),
        }

    async def start(self, goal: str) -> str:
        session = await self.store.create_session(goal, config_json=self.config.model_dump())
        await self.store.purge_expired_tool_cache()
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.runtime.request_timeout_seconds)
        ) as http_session:
            ctx = self._context(session.id, http_session=http_session)
            await create_research_plan(goal, ctx)
            queue = TaskQueue(self.store, session.id)
            await queue.enqueue(
                Task(
                    session_id=session.id,
                    agent="generation",
                    action="create_initial_hypotheses",
                    priority=int(TaskPriority.USER),
                )
            )
            await self._run_queue(queue, ctx)
        return session.id

    async def resume(self, session_id: str) -> None:
        session = await self.store.get_session(session_id)
        if session is None:
            raise ValueError(f"unknown session: {session_id}")
        await self.store.purge_expired_tool_cache()
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.runtime.request_timeout_seconds)
        ) as http_session:
            ctx = self._context(session_id, http_session=http_session)
            if await self.store.get_research_plan(session_id) is None:
                await create_research_plan(session.goal, ctx)
            queue = TaskQueue(self.store, session_id)
            await queue.load_pending()
            await self._run_queue(queue, ctx)

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
            f"# Co-Scientist Phase 4 Report: {session_id}",
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
        while queue.qsize() > 0:
            tasks = [
                asyncio.create_task(self._run_one(queue, ctx))
                for _ in range(min(self.config.runtime.worker_concurrency, queue.qsize()))
            ]
            if tasks:
                await asyncio.gather(*tasks)

    async def _run_one(self, queue: TaskQueue, ctx: AgentContext) -> None:
        task = await queue.dequeue()
        try:
            agent = self.agents[task.agent]
            result = await agent.execute(task, ctx)
            await self._handle_result(queue, task, result)
        except Exception as exc:
            if task.id is None:
                raise
            await queue.mark_failed(task.id, str(exc))

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

        await queue.mark_done(
            task.id,
            _result_debug_json(result) | {"stored_ids": stored_ids},
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
        for candidate in hypotheses:
            hypothesis = await self.store.add_hypothesis(candidate)
            if hypothesis.id is None:
                continue
            stored_ids.append(hypothesis.id)
            await queue.enqueue(
                Task(
                    session_id=task.session_id,
                    agent="reflection",
                    action="full_review",
                    target_id=hypothesis.id,
                    priority=int(TaskPriority.REFLECTION),
                )
            )
        for hypothesis_id in stored_ids:
            await self.store.add_citations_for_artifact(
                result.citations,
                session_id=task.session_id,
                artifact_type="hypothesis",
                artifact_id=hypothesis_id,
                source_task_id=task.id,
            )
        return stored_ids


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


async def run_new_session(
    *,
    db_path: str | Path,
    config: AppConfig,
    goal: str,
    supervisor: Supervisor | None = None,
) -> str:
    async with SQLiteStore(db_path) as store:
        runner = supervisor or Supervisor(store=store, config=config)
        return await runner.start(goal)


async def run_resume_session(
    *,
    db_path: str | Path,
    config: AppConfig,
    session_id: str,
    supervisor: Supervisor | None = None,
) -> None:
    async with SQLiteStore(db_path) as store:
        runner = supervisor or Supervisor(store=store, config=config)
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
