from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from co_scientist.memory.models import Hypothesis, ResearchPlan, Task
from co_scientist.tools.literature import search_literature, search_literature_with_fallbacks
from co_scientist.tools.models import ToolResult
from co_scientist.tools.query import build_literature_query

from .base import Agent, AgentContext
from .parsers import parse_hypothesis_block, summarize_hypothesis
from .results import AgentResult, AgentResultKind

LiteratureSearch = Callable[..., Awaitable[ToolResult]]


@dataclass(frozen=True)
class GenerationSpec:
    strategy: str
    template_name: str


class GenerationAgent(Agent):
    name = "generation"
    system_prompt = "You generate specific, testable scientific hypotheses."

    def __init__(
        self,
        *,
        literature_search: LiteratureSearch = search_literature,
        debate_min_turns: int = 3,
        debate_max_turns: int = 5,
    ) -> None:
        super().__init__()
        self.literature_search = literature_search
        self.debate_min_turns = debate_min_turns
        self.debate_max_turns = debate_max_turns

    async def execute(self, task: Task, ctx: AgentContext) -> AgentResult:
        if task.action != "create_initial_hypotheses":
            return AgentResult(
                kind=AgentResultKind.NOOP,
                payload={"agent": self.name, "ignored_action": task.action},
            )

        plan = await ctx.store.get_research_plan(ctx.session_id)
        if plan is None:
            raise ValueError(f"missing research plan for session {ctx.session_id}")

        results = []
        citations = []
        for spec in _initial_specs():
            result = await self._run_strategy(ctx, plan, spec)
            if result.parse_error:
                return result
            results.append(result.payload)
            citations.extend(result.citations)

        return AgentResult(
            kind=AgentResultKind.HYPOTHESIS_CREATED,
            payload={"hypotheses": results},
            citations=citations,
        )

    async def _run_strategy(
        self,
        ctx: AgentContext,
        plan: ResearchPlan,
        spec: GenerationSpec,
    ) -> AgentResult:
        if spec.strategy == "scientific_debate":
            return await self._scientific_debate(ctx, plan)

        evidence = await _search_evidence(
            self.literature_search,
            [build_literature_query(plan.goal), plan.goal],
            domain="biomed",
            config=ctx.config.search,
            store=ctx.store,
            session_id=ctx.session_id,
            persist_citations=False,
            http_session=ctx.http_session,
        )
        variables = _base_variables(plan) | {
            "source_hypothesis": "None",
            "instructions": f"Use the {spec.strategy} strategy.",
            "articles_with_reasoning": evidence.format_evidence_pack(
                max_items=ctx.config.search.max_results,
            ),
            "context": "No prior overview is available in Phase 4.",
        }
        prompt = self.render_prompt(ctx, spec.template_name, **variables)
        text = await self.chat(ctx, self.build_messages(ctx, user_prompt=prompt))
        return _hypothesis_result(
            text,
            strategy=spec.strategy,
            citations=evidence.citations,
        )

    async def _scientific_debate(
        self,
        ctx: AgentContext,
        plan: ResearchPlan,
    ) -> AgentResult:
        transcript = ""
        last_text = ""
        for turn in range(1, self.debate_max_turns + 1):
            instructions = (
                "Continue the discussion."
                if turn > 1
                else "Initiate the discussion and propose three distinct hypotheses."
            )
            if turn >= self.debate_min_turns:
                instructions += " You may now end with a final HYPOTHESIS."
            if turn == self.debate_max_turns:
                instructions += " You must end with a final HYPOTHESIS."

            prompt = self.render_prompt(
                ctx,
                "generation_scientific_debate",
                **_base_variables(plan),
                instructions=instructions,
                reviews_overview="No reviews are available in Phase 4 initial generation.",
                transcript=transcript or "No prior transcript.",
            )
            last_text = await self.chat(ctx, self.build_messages(ctx, user_prompt=prompt))
            transcript = f"{transcript}\n\nTurn {turn}:\n{last_text}".strip()
            parsed = parse_hypothesis_block(last_text)
            if turn >= self.debate_min_turns and parsed.ok:
                return _hypothesis_result(
                    last_text,
                    strategy="scientific_debate",
                    citations=[],
                    raw_text=transcript,
                )

        parsed = parse_hypothesis_block(last_text)
        return AgentResult(
            kind=AgentResultKind.HYPOTHESIS_CREATED,
            raw_text=transcript,
            parse_error=parsed.error or "scientific debate did not produce a hypothesis",
        )


def _initial_specs() -> list[GenerationSpec]:
    return [
        GenerationSpec("literature_review", "generation_literature_review"),
        GenerationSpec("scientific_debate", "generation_scientific_debate"),
        GenerationSpec("iterative_assumptions", "generation_iterative_assumptions"),
        GenerationSpec("research_expansion", "generation_research_expansion"),
        GenerationSpec("literature_review", "generation_literature_review"),
    ]


def _base_variables(plan: ResearchPlan) -> dict[str, str]:
    return {
        "goal": plan.goal,
        "preferences": _format_list(plan.preferences),
        "attributes": _format_list(plan.attributes),
        "constraints": _format_list(plan.constraints),
        "idea_attributes": _format_list(plan.idea_attributes),
    }


def _format_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- Not specified"


def _hypothesis_result(
    text: str,
    *,
    strategy: str,
    citations: list[Any],
    raw_text: str | None = None,
) -> AgentResult:
    parsed = parse_hypothesis_block(text)
    if not parsed.ok or not isinstance(parsed.value, str):
        return AgentResult(
            kind=AgentResultKind.HYPOTHESIS_CREATED,
            raw_text=raw_text or text,
            parse_error=parsed.error,
        )
    content = parsed.value
    return AgentResult(
        kind=AgentResultKind.HYPOTHESIS_CREATED,
        payload={
            "content": content,
            "summary": summarize_hypothesis(content),
            "source_strategy": strategy,
        },
        citations=citations,
        raw_text=raw_text or text,
    )


def hypothesis_from_payload(session_id: str, payload: dict[str, Any]) -> Hypothesis:
    return Hypothesis(
        session_id=session_id,
        content=str(payload["content"]),
        summary=str(payload["summary"]),
        source_strategy=str(payload["source_strategy"]),
    )


async def _search_evidence(searcher: LiteratureSearch, queries: list[str], **kwargs) -> ToolResult:
    if searcher is search_literature:
        return await search_literature_with_fallbacks(queries, **kwargs)
    return await searcher(queries[0], **kwargs)
