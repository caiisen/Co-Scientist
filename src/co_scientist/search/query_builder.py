from __future__ import annotations

import json
import re

from co_scientist.llm.client import LLMClient

from .goal_parser import ParsedGoal

_PLATFORM_RULES = {
    "pubmed": """\
PubMed rules:
- Use field tags: [tiab] (title+abstract), [mh] (MeSH heading)
- Boolean operators: AND, OR. Group synonyms in parentheses with OR.
- Date filter example: "2020/01/01"[dp]:3000[dp]
- Example: (CRISPR-Cas9[mh] OR "gene editing"[tiab]) AND ("CAR-T"[tiab] OR
  "chimeric antigen receptor"[tiab]) AND "off-target effects"[tiab]
""",
    "arxiv": """\
arXiv rules:
- Field prefixes: ti: (title), abs: (abstract), cat: (category)
- Always include at least one cat: filter.
  Examples: cs.AI, cs.LG, cs.CL, q-bio.GN, physics.bio-ph, math.ST
- Boolean: AND, OR, ANDNOT
- Example: ti:(attention mechanism) AND abs:(protein structure) AND cat:cs.LG
""",
    "semantic_scholar": """\
Semantic Scholar rules:
- Natural language query, no boolean syntax needed
- Concise phrase of 3-8 key terms works best
- Example: CRISPR off-target safety CAR-T hematologic
""",
    "openalex": """\
OpenAlex rules:
- Natural language query (similar to Semantic Scholar but covers all disciplines)
- Works well for interdisciplinary topics
- Example: gene editing safety blood cancer immunotherapy
""",
    "tavily": """\
Tavily rules:
- Full natural language sentence, optionally include year(s) for recency
- Best for recent developments, news, and practical/applied content
- Example: CRISPR CAR-T off-target effects safety improvements 2024 2025
""",
}

_SYSTEM_PROMPT_TEMPLATE = """\
You are a literature search expert.
Generate platform-specific database queries for the research goal.
Output ONLY raw JSON — no markdown code blocks, no explanation.

Platform syntax rules:
{rules}

Generate exactly {n} queries per platform, covering three angles:
1. Core concepts query (the main topic)
2. Method/technique query (specific techniques or approaches)
3. Problem/challenge query (the underlying problem or broader context)

Output schema:
{{
  "<platform_name>": ["<query1>", "<query2>", "<query3>"],
  ...
}}
"""


async def build_queries(
    parsed: ParsedGoal,
    platforms: list[str],
    client: LLMClient,
    n: int = 3,
) -> dict[str, list[str]]:
    rules = "\n".join(
        f"[{p.upper()}]\n{_PLATFORM_RULES[p]}"
        for p in platforms
        if p in _PLATFORM_RULES
    )
    system = _SYSTEM_PROMPT_TEMPLATE.format(rules=rules, n=n)

    synonyms_str = "; ".join(
        f"{k}: {', '.join(v)}" for k, v in (parsed.synonyms or {}).items()
    )
    user = (
        f"Research goal: {parsed.specific_aspect or ', '.join(parsed.core_concepts)}\n"
        f"Core concepts: {', '.join(parsed.core_concepts)}\n"
        f"Synonyms: {synonyms_str}\n"
        f"Platforms needed: {', '.join(p for p in platforms if p in _PLATFORM_RULES)}"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    text = await client.chat(messages, temperature=0)
    return _parse_response(text, platforms)


def _parse_response(text: str, platforms: list[str]) -> dict[str, list[str]]:
    raw = _extract_json(text)
    try:
        data: dict[str, list[str]] = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    # Keep only known platforms and ensure list[str]
    result: dict[str, list[str]] = {}
    for p in platforms:
        queries = data.get(p) or []
        result[p] = [str(q) for q in queries if q]
    return result


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
