from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from co_scientist.llm.client import LLMClient

_log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a research assistant. Analyze the user's research goal and output a JSON object.
Output ONLY raw JSON — no markdown code blocks, no explanation.

Required schema:
{
  "core_concepts": ["<English concept 1>", ...],
  "specific_aspect": "<what specifically the user focuses on>",
  "synonyms": {
    "<concept>": ["<synonym1>", "<synonym2>"]
  },
  "temporal": "<recent | foundational | both>"
}

Rules:
- core_concepts must be in English (2-4 most discriminative terms)
- synonyms should cover alternate phrasings and abbreviation expansions used in academic literature
"""


@dataclass
class ParsedGoal:
    core_concepts: list[str]
    specific_aspect: str
    synonyms: dict[str, list[str]] = field(default_factory=dict)
    temporal: str = "recent"


async def parse_goal(goal: str, client: LLMClient) -> ParsedGoal:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]
    text = await client.chat(messages, temperature=0)
    _log.debug("parse_goal raw response: %s", text)
    parsed = _parse_response(text)
    _log.info("parse_goal result: concepts=%s", parsed.core_concepts)
    return parsed


def _parse_response(text: str) -> ParsedGoal:
    raw = _extract_json(text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return ParsedGoal(core_concepts=[], specific_aspect="")

    return ParsedGoal(
        core_concepts=data.get("core_concepts") or [],
        specific_aspect=data.get("specific_aspect") or "",
        synonyms=data.get("synonyms") or {},
        temporal=data.get("temporal") or "recent",
    )


def _extract_json(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        return match.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return text[start : end + 1]
    return text
