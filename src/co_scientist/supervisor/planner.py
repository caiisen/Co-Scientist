from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from co_scientist.agents.base import AgentContext
from co_scientist.agents.parsers import parse_json_object
from co_scientist.memory.models import ResearchPlan


async def create_research_plan(goal: str, ctx: AgentContext) -> ResearchPlan:
    prompt = ctx.prompt_store.render("planner_research_plan", goal=goal)
    response = await ctx.llm_for("planner").chat(
        [
            {
                "role": "system",
                "content": "You turn research goals into structured Co-Scientist plans.",
            },
            {"role": "user", "content": prompt},
        ]
    )
    parsed = parse_json_object(response)
    if not parsed.ok or not isinstance(parsed.value, dict):
        raise ValueError(parsed.error or "planner did not return a JSON object")

    data = _normalize_plan(parsed.value)
    try:
        plan = ResearchPlan(session_id=ctx.session_id, goal=goal, **data)
    except ValidationError as exc:
        raise ValueError(f"invalid research plan: {exc}") from exc
    return await ctx.store.save_research_plan(plan)


def _normalize_plan(value: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "preferences": _string_list(value.get("preferences")),
        "attributes": _string_list(value.get("attributes")),
        "constraints": _string_list(value.get("constraints")),
        "idea_attributes": _string_list(value.get("idea_attributes")),
    }


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError("planner fields must be strings or lists")
