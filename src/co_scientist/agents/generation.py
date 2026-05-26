from __future__ import annotations

import asyncio
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
    query_variant: str = "summary"
    source_strategy: str | None = None
    instructions: str | None = None


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
        errors = []
        specs = _initial_specs()
        strategy_results = await asyncio.gather(
            *(self._run_strategy(ctx, plan, spec) for spec in specs)
        )
        for spec, result in zip(specs, strategy_results, strict=True):
            if result.parse_error:
                errors.append(
                    {
                        "strategy": spec.strategy,
                        "query_variant": spec.query_variant,
                        "error": result.parse_error,
                        "raw_text": result.raw_text,
                    }
                )
                continue
            results.append(result.payload)
            citations.extend(result.citations)

        if not results:
            return AgentResult(
                kind=AgentResultKind.HYPOTHESIS_CREATED,
                payload={"hypotheses": [], "errors": errors},
                raw_text="All generation strategies failed to produce a hypothesis.",
                parse_error="no hypotheses generated",
            )

        return AgentResult(
            kind=AgentResultKind.HYPOTHESIS_CREATED,
            payload={"hypotheses": results, "errors": errors},
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
            _evidence_queries(plan, spec.query_variant),
            domain=infer_search_domain(plan),
            config=ctx.config.search,
            store=ctx.store,
            session_id=ctx.session_id,
            persist_citations=False,
            http_session=ctx.http_session,
            embedding_client=ctx.llm_for(self.name),
        )
        variables = _base_variables(plan) | {
            "source_hypothesis": "None",
            "instructions": spec.instructions or f"Use the {spec.strategy} strategy.",
            "articles_with_reasoning": evidence.format_evidence_pack(
                max_items=ctx.config.search.max_results,
            ),
            "context": "No prior overview is available in Phase 4.",
        }
        prompt = self.render_prompt(ctx, spec.template_name, **variables)
        text = await self.chat(ctx, self.build_messages(ctx, user_prompt=prompt))
        return _hypothesis_result(
            text,
            strategy=spec.source_strategy or spec.strategy,
            citations=evidence.citations,
            query_variant=spec.query_variant,
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
        GenerationSpec(
            strategy="literature_review",
            template_name="generation_literature_review",
            query_variant="summary",
            source_strategy="literature_review",
            instructions=(
                "Use the literature_review strategy with a compressed keyword search. "
                "Prioritize direct evidence from the retrieved articles."
            ),
        ),
        GenerationSpec("scientific_debate", "generation_scientific_debate"),
        GenerationSpec("iterative_assumptions", "generation_iterative_assumptions"),
        GenerationSpec("research_expansion", "generation_research_expansion"),
        GenerationSpec(
            strategy="literature_review",
            template_name="generation_literature_review",
            query_variant="goal",
            source_strategy="literature_review",
            instructions=(
                "Use the literature_review strategy with the full goal as the primary query. "
                "Look for complementary evidence that may be missed by keyword compression."
            ),
        ),
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


def _evidence_queries(plan: ResearchPlan, variant: str) -> list[str]:
    summary_query = build_literature_query(plan.goal)
    if variant == "goal":
        return [plan.goal, summary_query]
    return [summary_query, plan.goal]


def infer_search_domain(plan: ResearchPlan) -> str:
    text = " ".join(
        [plan.goal, *plan.attributes, *plan.idea_attributes, *plan.preferences]
    ).lower()
    if any(term in text for term in ("arxiv", "computer science", "algorithm", "software")):
        return "cs"
    if any(term in text for term in ("physics", "quantum", "particle", "cosmology")):
        return "physics"
    if any(term in text for term in ("math", "mathematics", "theorem", "proof")):
        return "math"
    return "biomed"


def _hypothesis_result(
    text: str,
    *,
    strategy: str,
    citations: list[Any],
    query_variant: str | None = None,
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
            "query_variant": query_variant,
            "citations": [citation.model_dump() for citation in citations],
        },
        citations=citations,
        raw_text=raw_text or text,
    )


def hypothesis_from_payload(session_id: str, payload: dict[str, Any]) -> Hypothesis:
    raw_round = payload.get("meta_review_round")
    return Hypothesis(
        session_id=session_id,
        content=str(payload["content"]),
        summary=str(payload["summary"]),
        source_strategy=str(payload["source_strategy"]),
        parent_ids=[int(parent_id) for parent_id in payload.get("parent_ids", [])],
        meta_review_round=int(raw_round) if raw_round is not None else None,
    )


async def _search_evidence(searcher: LiteratureSearch, queries: list[str], **kwargs) -> ToolResult:
    if searcher is search_literature:
        return await search_literature_with_fallbacks(queries, **kwargs)
    return await searcher(queries[0], **kwargs)
