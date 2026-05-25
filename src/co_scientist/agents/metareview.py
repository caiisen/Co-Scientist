from __future__ import annotations

from co_scientist.memory.models import ResearchOverview, SystemFeedback, Task

from .base import Agent, AgentContext
from .results import AgentResult, AgentResultKind


class MetaReviewAgent(Agent):
    name = "metareview"
    system_prompt = "You synthesize system-level scientific review feedback."

    async def execute(self, task: Task, ctx: AgentContext) -> AgentResult:
        if task.action == "generate_system_feedback":
            return await self._generate_system_feedback(ctx)
        if task.action == "generate_final_overview":
            return await self._generate_final_overview(ctx)
        return AgentResult(
            kind=AgentResultKind.NOOP,
            payload={"agent": self.name, "ignored_action": task.action},
        )

    async def _generate_system_feedback(self, ctx: AgentContext) -> AgentResult:
        plan = await ctx.store.get_research_plan(ctx.session_id)
        if plan is None:
            raise ValueError(f"missing research plan for session {ctx.session_id}")

        previous = await ctx.store.latest_feedback(ctx.session_id)
        reviews = await ctx.store.recent_reviews(ctx.session_id, limit=20)
        matches = await ctx.store.recent_matches(ctx.session_id, limit=20)
        round_number = await ctx.store.count_feedback(ctx.session_id) + 1
        prompt = self.render_prompt(
            ctx,
            "metareview",
            goal=plan.goal,
            preferences=_format_list(plan.preferences or plan.idea_attributes),
            previous_feedback=previous.content if previous else "None.",
            reviews=_format_reviews(reviews),
            matches=_format_matches(matches),
        )
        text = await self.chat(ctx, self.build_messages(ctx, user_prompt=prompt))
        return AgentResult(
            kind=AgentResultKind.FEEDBACK_GENERATED,
            payload={"round": round_number, "content": text},
            raw_text=text,
        )

    async def _generate_final_overview(self, ctx: AgentContext) -> AgentResult:
        plan = await ctx.store.get_research_plan(ctx.session_id)
        if plan is None:
            raise ValueError(f"missing research plan for session {ctx.session_id}")

        top = await ctx.store.top_k_by_elo(ctx.session_id, k=10)
        latest_overview = await ctx.store.latest_overview(ctx.session_id)
        round_number = latest_overview.round + 1 if latest_overview else 1
        prompt = self.render_prompt(
            ctx,
            "metareview_overview",
            goal=plan.goal,
            preferences=_format_list(plan.preferences or plan.idea_attributes),
            hypotheses=_format_hypotheses(top),
        )
        text = await self.chat(ctx, self.build_messages(ctx, user_prompt=prompt))
        return AgentResult(
            kind=AgentResultKind.OVERVIEW_GENERATED,
            payload={
                "round": round_number,
                "content": text,
                "top_hypothesis_ids": [hypothesis.id for hypothesis in top if hypothesis.id],
            },
            raw_text=text,
        )


def feedback_from_payload(session_id: str, payload: dict) -> SystemFeedback:
    return SystemFeedback(
        session_id=session_id,
        round=int(payload["round"]),
        content=str(payload["content"]),
    )


def overview_from_payload(session_id: str, payload: dict) -> ResearchOverview:
    return ResearchOverview(
        session_id=session_id,
        round=int(payload["round"]),
        content=str(payload["content"]),
        top_hypothesis_ids=[int(item) for item in payload.get("top_hypothesis_ids", [])],
    )


def _format_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- Not specified"


def _format_reviews(reviews) -> str:
    if not reviews:
        return "No reviews available."
    return "\n\n".join(
        f"Review {review.id} for hypothesis {review.hypothesis_id} "
        f"({review.type}, score={review.score}):\n{review.content}"
        for review in reviews
    )


def _format_matches(matches) -> str:
    if not matches:
        return "No matches available."
    return "\n\n".join(
        f"Match {match.id}: hypothesis {match.winner_id} won over "
        f"{match.hypo_a_id}/{match.hypo_b_id}.\n{match.transcript}"
        for match in matches
    )


def _format_hypotheses(hypotheses) -> str:
    if not hypotheses:
        return "No hypotheses available."
    return "\n\n".join(
        f"{index}. id={hypothesis.id} elo={hypothesis.elo}\n"
        f"Summary: {hypothesis.summary}\n{hypothesis.content}"
        for index, hypothesis in enumerate(hypotheses, start=1)
    )
