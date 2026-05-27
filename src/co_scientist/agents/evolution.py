from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from co_scientist.memory.models import Hypothesis, ResearchPlan, Task
from co_scientist.tools.literature import search_literature
from co_scientist.tools.models import ToolResult
from co_scientist.tools.query import build_literature_query

from .base import Agent, AgentContext
from .evidence import LiteratureSearch, search_evidence
from .generation import infer_search_domain
from .parsers import parse_hypothesis_block, summarize_hypothesis
from .results import AgentResult, AgentResultKind


@dataclass(frozen=True)
class EvolutionSpec:
    strategy: str
    template_name: str


EVOLUTION_SPECS = [
    EvolutionSpec("grounding_enhancement", "evolution_grounding_enhancement"),
    EvolutionSpec("feasibility_refinement", "evolution_feasibility"),
    EvolutionSpec("inspiration_from_existing", "evolution_inspiration_from_existing"),
    EvolutionSpec("combination", "evolution_combination"),
    EvolutionSpec("simplification", "evolution_simplification"),
    EvolutionSpec("out_of_box", "evolution_out_of_box"),
]


class EvolutionAgent(Agent):
    name = "evolution"
    system_prompt = "You evolve top scientific hypotheses without modifying originals."

    def __init__(self, *, literature_search: LiteratureSearch = search_literature) -> None:
        super().__init__()
        self.literature_search = literature_search

    async def execute(self, task: Task, ctx: AgentContext) -> AgentResult:
        if task.action != "evolve_top_hypotheses":
            return AgentResult(
                kind=AgentResultKind.NOOP,
                payload={"agent": self.name, "ignored_action": task.action},
            )

        plan = await ctx.store.get_research_plan(ctx.session_id)
        if plan is None:
            raise ValueError(f"missing research plan for session {ctx.session_id}")

        current_count = await ctx.store.count_hypotheses(ctx.session_id)
        remaining_slots = max(ctx.config.runtime.max_ideas - current_count, 0)
        if remaining_slots <= 0:
            return AgentResult(
                kind=AgentResultKind.HYPOTHESIS_CREATED,
                payload={"hypotheses": [], "errors": [], "reason": "max ideas reached"},
            )

        top_hypotheses = await ctx.store.top_k_by_elo(ctx.session_id, k=5)
        top_hypotheses = [
            hypothesis for hypothesis in top_hypotheses if hypothesis.id is not None
        ]
        if not top_hypotheses:
            return AgentResult(
                kind=AgentResultKind.HYPOTHESIS_CREATED,
                payload={"hypotheses": [], "errors": []},
                parse_error="no hypotheses available for evolution",
            )

        meta_round = await ctx.store.count_feedback(ctx.session_id)
        specs = [
            spec
            for spec in EVOLUTION_SPECS[:remaining_slots]
            if not _spec_requires_skip(spec, top_hypotheses)
        ]
        strategy_results = await asyncio.gather(
            *(
                self._run_strategy(ctx, plan, top_hypotheses, spec, meta_round)
                for spec in specs
            )
        )

        results: list[dict[str, Any]] = []
        citations = []
        errors = []
        for spec, strategy_result in zip(specs, strategy_results, strict=True):
            if strategy_result.parse_error:
                errors.append(
                    {
                        "strategy": spec.strategy,
                        "error": strategy_result.parse_error,
                        "raw_text": strategy_result.raw_text,
                    }
                )
                continue
            results.append(strategy_result.payload)
            citations.extend(strategy_result.citations)

        if not results:
            return AgentResult(
                kind=AgentResultKind.HYPOTHESIS_CREATED,
                payload={"hypotheses": [], "errors": errors},
                parse_error="no evolved hypotheses generated",
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
        top_hypotheses: list[Hypothesis],
        spec: EvolutionSpec,
        meta_round: int,
    ) -> AgentResult:
        parents = _parents_for_strategy(spec.strategy, top_hypotheses)
        evidence = ToolResult(source="literature")
        if spec.strategy == "grounding_enhancement":
            evidence = await search_evidence(
                self.literature_search,
                [
                    build_literature_query(plan.goal, parents[0].summary),
                    build_literature_query(plan.goal),
                    plan.goal,
                ],
                source_text=f"{plan.goal}\n\nHypothesis summary:\n{parents[0].summary}",
                query_client=ctx.llm_for(self.name),
                domain=infer_search_domain(plan),
                config=ctx.config.search,
                store=ctx.store,
                session_id=ctx.session_id,
                persist_citations=False,
                http_session=ctx.http_session,
                embedding_client=ctx.embedding_llm_for(self.name),
            )

        variables = _base_variables(plan) | {
            "hypothesis": parents[0].content,
            "hypotheses": _format_hypotheses(parents),
            "evidence": evidence.format_evidence_pack(max_items=ctx.config.search.max_results),
        }
        prompt = self.render_prompt(ctx, spec.template_name, **variables)
        text = await self.chat(ctx, self.build_messages(ctx, user_prompt=prompt))
        return _hypothesis_result(
            text,
            strategy=spec.strategy,
            parent_ids=[int(parent.id or 0) for parent in parents if parent.id is not None],
            citations=evidence.citations,
            meta_review_round=meta_round,
        )


def _spec_requires_skip(spec: EvolutionSpec, top_hypotheses: list[Hypothesis]) -> bool:
    if spec.strategy == "combination" and len(top_hypotheses) < 2:
        return True
    return False


def _parents_for_strategy(strategy: str, top_hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    if strategy == "feasibility_refinement":
        return [top_hypotheses[min(1, len(top_hypotheses) - 1)]]
    if strategy == "inspiration_from_existing":
        return [top_hypotheses[min(2, len(top_hypotheses) - 1)]]
    if strategy == "combination":
        return top_hypotheses[: min(3, len(top_hypotheses))]
    if strategy == "simplification":
        return [top_hypotheses[min(3, len(top_hypotheses) - 1)]]
    if strategy == "out_of_box":
        return top_hypotheses
    return [top_hypotheses[0]]


def _base_variables(plan: ResearchPlan) -> dict[str, str]:
    return {
        "goal": plan.goal,
        "preferences": _format_list(plan.preferences or plan.idea_attributes),
        "attributes": _format_list(plan.attributes),
        "constraints": _format_list(plan.constraints),
        "idea_attributes": _format_list(plan.idea_attributes),
    }


def _format_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- Not specified"


def _format_hypotheses(hypotheses: list[Hypothesis]) -> str:
    lines = []
    for index, hypothesis in enumerate(hypotheses, start=1):
        lines.append(
            f"Hypothesis {index} (Elo {hypothesis.elo}):\n"
            f"Summary: {hypothesis.summary}\n"
            f"Body:\n{hypothesis.content}"
        )
    return "\n\n".join(lines)


def _hypothesis_result(
    text: str,
    *,
    strategy: str,
    parent_ids: list[int],
    citations: list[Any],
    meta_review_round: int,
) -> AgentResult:
    parsed = parse_hypothesis_block(text)
    if not parsed.ok or not isinstance(parsed.value, str):
        return AgentResult(
            kind=AgentResultKind.HYPOTHESIS_CREATED,
            raw_text=text,
            parse_error=parsed.error,
        )
    content = parsed.value
    return AgentResult(
        kind=AgentResultKind.HYPOTHESIS_CREATED,
        payload={
            "content": content,
            "summary": summarize_hypothesis(content),
            "source_strategy": f"evolution:{strategy}",
            "parent_ids": parent_ids,
            "meta_review_round": meta_review_round,
            "citations": [citation.model_dump() for citation in citations],
        },
        citations=citations,
        raw_text=text,
    )
