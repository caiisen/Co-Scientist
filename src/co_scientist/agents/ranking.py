from __future__ import annotations

import math
from dataclasses import dataclass

from co_scientist.memory.models import Hypothesis, Task

from .base import Agent, AgentContext
from .parsers import parse_better_idea
from .results import AgentResult, AgentResultKind


@dataclass(frozen=True)
class CandidatePair:
    first: Hypothesis
    second: Hypothesis
    score: float
    pair_match_count: int
    elo_gap: int


class RankingAgent(Agent):
    name = "ranking"
    system_prompt = "You compare scientific hypotheses and choose the better idea."

    async def execute(self, task: Task, ctx: AgentContext) -> AgentResult:
        if task.action not in {"run_tournament_match", "run_tournament_batch"}:
            return AgentResult(
                kind=AgentResultKind.NOOP,
                payload={"agent": self.name, "ignored_action": task.action},
            )

        hypotheses = await ctx.store.list_reviewed_hypotheses(ctx.session_id)
        hypotheses = [hypothesis for hypothesis in hypotheses if hypothesis.id is not None]
        if len(hypotheses) < 2:
            return AgentResult(
                kind=AgentResultKind.NOOP,
                payload={"reason": "not enough reviewed hypotheses"},
            )

        match_count = await ctx.store.count_matches(ctx.session_id)
        target_matches = len(hypotheses) * ctx.config.runtime.max_matches_per_idea
        if match_count >= target_matches:
            return AgentResult(
                kind=AgentResultKind.NOOP,
                payload={"reason": "match target reached", "target_matches": target_matches},
            )

        pair = await choose_pair(ctx, hypotheses)
        if pair is None:
            return AgentResult(kind=AgentResultKind.NOOP, payload={"reason": "no candidate pair"})

        mode = _ranking_mode(hypotheses, pair.first, pair.second)
        plan = await ctx.store.get_research_plan(ctx.session_id)
        if plan is None:
            raise ValueError(f"missing research plan for session {ctx.session_id}")

        review_1 = await ctx.store.latest_review_for_hypothesis(pair.first.id or 0)
        review_2 = await ctx.store.latest_review_for_hypothesis(pair.second.id or 0)
        prompt = self.render_prompt(
            ctx,
            "ranking_debate" if mode == "debate" else "ranking_pairwise",
            goal=plan.goal,
            preferences=_format_list(plan.preferences or plan.idea_attributes),
            hypothesis_1=pair.first.content,
            hypothesis_2=pair.second.content,
            review_1=review_1.content if review_1 else "No review available.",
            review_2=review_2.content if review_2 else "No review available.",
            notes=_ranking_notes(pair, match_count, target_matches),
        )
        text = await self.chat(ctx, self.build_messages(ctx, user_prompt=prompt))
        parsed = parse_better_idea(text)
        if not parsed.ok or parsed.value not in {1, 2}:
            return AgentResult(
                kind=AgentResultKind.RANKING_DECISION,
                raw_text=text,
                parse_error=parsed.error or "invalid ranking decision",
            )

        winner_id = pair.first.id if parsed.value == 1 else pair.second.id
        return AgentResult(
            kind=AgentResultKind.RANKING_DECISION,
            payload={
                "hypo_a_id": pair.first.id,
                "hypo_b_id": pair.second.id,
                "winner_id": winner_id,
                "mode": mode,
                "pair_score": pair.score,
                "pair_match_count": pair.pair_match_count,
                "elo_gap": pair.elo_gap,
                "transcript": text,
            },
            raw_text=text,
        )


async def choose_pair(ctx: AgentContext, hypotheses: list[Hypothesis]) -> CandidatePair | None:
    match_counts = await ctx.store.match_counts_by_hypothesis(ctx.session_id)
    pair_counts = await ctx.store.match_counts_by_pair(ctx.session_id)
    similarities = await ctx.store.proximity_edges_for_session(ctx.session_id)
    quantiles = _elo_quantiles(hypotheses)
    max_seen_matches = max([*match_counts.values(), ctx.config.runtime.max_matches_per_idea, 1])

    candidates: list[CandidatePair] = []
    for index, first in enumerate(hypotheses):
        if first.id is None:
            continue
        for second in hypotheses[index + 1 :]:
            if second.id is None:
                continue
            key = tuple(sorted((first.id, second.id)))
            similarity = similarities.get(key, 0.0)
            first_matches = match_counts.get(first.id, 0)
            second_matches = match_counts.get(second.id, 0)
            freshness = 1.0 - min(first_matches, second_matches) / max_seen_matches
            elo_gap = abs(first.elo - second.elo)
            elo_closeness = 1.0 - min(elo_gap / 400.0, 1.0)
            top_boost = (quantiles[first.id] + quantiles[second.id]) / 2
            pair_match_count = pair_counts.get(key, 0)
            pair_novelty = 1.0 if pair_match_count == 0 else 0.0
            repeat_penalty = min(pair_match_count, 3) / 3
            score = (
                0.35 * similarity
                + 0.20 * freshness
                + 0.20 * elo_closeness
                + 0.15 * top_boost
                + 0.10 * pair_novelty
                - 0.30 * repeat_penalty
            )
            candidates.append(
                CandidatePair(
                    first=first,
                    second=second,
                    score=score,
                    pair_match_count=pair_match_count,
                    elo_gap=elo_gap,
                )
            )

    if not candidates:
        return None
    return max(
        candidates,
        key=lambda candidate: (
            candidate.score,
            -candidate.pair_match_count,
            -candidate.elo_gap,
            -(candidate.first.id or 0),
            -(candidate.second.id or 0),
        ),
    )


def _ranking_mode(hypotheses: list[Hypothesis], first: Hypothesis, second: Hypothesis) -> str:
    ranked_ids = [hypothesis.id for hypothesis in sorted(hypotheses, key=lambda item: -item.elo)]
    top_count = max(2, math.ceil(len(ranked_ids) * 0.30))
    top_ids = set(ranked_ids[:top_count])
    if first.id in top_ids and second.id in top_ids:
        return "debate"
    return "pairwise"


def _elo_quantiles(hypotheses: list[Hypothesis]) -> dict[int, float]:
    ranked = sorted(hypotheses, key=lambda item: (-item.elo, item.created_at, item.id or 0))
    if len(ranked) == 1:
        return {int(ranked[0].id or 0): 1.0}
    return {
        int(hypothesis.id or 0): 1.0 - index / (len(ranked) - 1)
        for index, hypothesis in enumerate(ranked)
    }


def _format_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- Not specified"


def _ranking_notes(pair: CandidatePair, match_count: int, target_matches: int) -> str:
    return (
        f"Current tournament match {match_count + 1} of target {target_matches}. "
        f"This pair has been compared {pair.pair_match_count} times. "
        f"Their Elo gap is {pair.elo_gap}."
    )
