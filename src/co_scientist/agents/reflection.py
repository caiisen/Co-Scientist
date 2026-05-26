from __future__ import annotations

from co_scientist.memory.models import Review, Task
from co_scientist.tools.literature import search_literature
from co_scientist.tools.query import build_literature_query

from .base import Agent, AgentContext
from .evidence import LiteratureSearch, search_evidence
from .generation import infer_search_domain
from .parsers import parse_review_score
from .results import AgentResult, AgentResultKind


class ReflectionAgent(Agent):
    name = "reflection"
    system_prompt = "You critically review scientific hypotheses."

    def __init__(self, *, literature_search: LiteratureSearch = search_literature) -> None:
        super().__init__()
        self.literature_search = literature_search

    async def execute(self, task: Task, ctx: AgentContext) -> AgentResult:
        if task.action != "full_review":
            return AgentResult(
                kind=AgentResultKind.NOOP,
                payload={"agent": self.name, "ignored_action": task.action},
            )
        if task.target_id is None:
            raise ValueError("full_review task requires target_id")

        hypothesis = await ctx.store.get_hypothesis(task.target_id)
        if hypothesis is None:
            raise ValueError(f"unknown hypothesis: {task.target_id}")
        plan = await ctx.store.get_research_plan(ctx.session_id)
        if plan is None:
            raise ValueError(f"missing research plan for session {ctx.session_id}")

        evidence = await search_evidence(
            self.literature_search,
            [
                build_literature_query(plan.goal, hypothesis.summary),
                build_literature_query(plan.goal),
                plan.goal,
            ],
            source_text=f"{plan.goal}\n\nHypothesis summary:\n{hypothesis.summary}",
            query_client=ctx.llm_for(self.name),
            domain=infer_search_domain(plan),
            config=ctx.config.search,
            store=ctx.store,
            session_id=ctx.session_id,
            persist_citations=False,
            http_session=ctx.http_session,
            embedding_client=ctx.llm_for(self.name),
        )
        prompt = self.render_prompt(
            ctx,
            "reflection_full_review",
            goal=plan.goal,
            preferences=_format_list(plan.preferences),
            hypothesis=hypothesis.content,
            evidence=evidence.format_evidence_pack(max_items=ctx.config.search.max_results),
        )
        text = await self.chat(ctx, self.build_messages(ctx, user_prompt=prompt))
        parsed_score = parse_review_score(text)
        return AgentResult(
            kind=AgentResultKind.REVIEW_COMPLETED,
            payload={
                "hypothesis_id": hypothesis.id,
                "type": "full",
                "score": parsed_score.value if parsed_score.ok else None,
                "content": text,
            },
            citations=evidence.citations,
            raw_text=text,
        )


def _format_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- Not specified"


def review_from_payload(session_id: str, payload: dict) -> Review:
    score = payload.get("score")
    return Review(
        session_id=session_id,
        hypothesis_id=int(payload["hypothesis_id"]),
        type=str(payload["type"]),
        score=float(score) if score is not None else None,
        content=str(payload["content"]),
    )
