from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from co_scientist.config import SearchConfig
from co_scientist.memory.store import SQLiteStore
from co_scientist.tools.literature import (
    search_literature,
    search_literature_by_source_queries,
    search_literature_with_fallbacks,
)
from co_scientist.tools.models import ToolResult
from co_scientist.tools.query import build_literature_query

LiteratureSearch = Callable[..., Awaitable[ToolResult]]


async def search_evidence(
    searcher: LiteratureSearch,
    queries: list[str],
    *,
    source_text: str,
    query_client: Any | None = None,
    domain: str = "biomed",
    max_results: int | None = None,
    config: SearchConfig | None = None,
    store: SQLiteStore | None = None,
    session_id: str | None = None,
    persist_citations: bool = True,
    http_session: Any | None = None,
    embedding_client: object | None = None,
) -> ToolResult:
    if searcher is not search_literature:
        return await searcher(
            queries[0],
            domain=domain,
            max_results=max_results,
            config=config,
            store=store,
            session_id=session_id,
            persist_citations=persist_citations,
            http_session=http_session,
            embedding_client=embedding_client,
        )

    cfg = config or SearchConfig(max_results=5)
    if query_client is None or not _source_specific_search_enabled(domain, cfg):
        return await search_literature_with_fallbacks(
            queries,
            domain=domain,
            max_results=max_results,
            config=cfg,
            store=store,
            session_id=session_id,
            persist_citations=persist_citations,
            http_session=http_session,
            embedding_client=embedding_client,
        )

    source_queries = await build_source_queries(
        source_text,
        queries,
        query_client=query_client,
    )
    return await search_literature_by_source_queries(
        source_queries,
        domain=domain,
        max_results=max_results,
        config=cfg,
        store=store,
        session_id=session_id,
        persist_citations=persist_citations,
        http_session=http_session,
        embedding_client=embedding_client,
    )


async def build_source_queries(
    source_text: str,
    fallback_queries: list[str],
    *,
    query_client: Any,
) -> dict[str, list[str]]:
    keyword_query = await _build_llm_query(
        source_text,
        query_client=query_client,
        style="keywords",
    )
    pubmed_query = await _build_llm_query(
        source_text,
        query_client=query_client,
        style="pubmed",
    )
    lexical_query = build_literature_query(source_text)

    keyword_fallbacks = [keyword_query, lexical_query, *fallback_queries]
    pubmed_fallbacks = [pubmed_query, keyword_query, lexical_query, *fallback_queries]
    return {
        "pubmed": _dedupe_queries(pubmed_fallbacks),
        "semantic_scholar": _dedupe_queries(keyword_fallbacks),
        "tavily": _dedupe_queries(keyword_fallbacks),
        "arxiv": _dedupe_queries(keyword_fallbacks),
        "private_corpus": _dedupe_queries(keyword_fallbacks),
    }


async def _build_llm_query(source_text: str, *, query_client: Any, style: str) -> str:
    try:
        response = await query_client.chat(
            _query_messages(source_text, style),
            temperature=0.0,
        )
    except Exception:
        return ""
    return _parse_llm_query(response)


def _query_messages(source_text: str, style: str) -> list[dict[str, str]]:
    if style == "pubmed":
        instruction = (
            "Create one PubMed-compatible advanced search query. Use Boolean syntax with "
            "AND/OR groups, include scientific names and important synonyms, and avoid "
            "overly narrow location/use constraints when they would suppress recall."
        )
    else:
        instruction = (
            "Create one broad literature/web search keyword query for Semantic Scholar and "
            "web search. Use 8 to 14 high-signal English keywords or short phrases. Do not "
            "use Boolean operators, parentheses, field tags, JSON, quotes, markdown, or a "
            "full sentence."
        )
    return [
        {
            "role": "system",
            "content": (
                "You create concise literature search queries. Return only the search query "
                "text. Do not return JSON, quotes, markdown, or explanation."
            ),
        },
        {
            "role": "user",
            "content": f"{instruction}\n\nResearch context:\n{source_text}",
        },
    ]


def _parse_llm_query(response: str) -> str:
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    json_fragment = re.search(r"""["']query["']\s*:\s*["']([^"'}]+)""", text)
    if json_fragment:
        return _clean_query_text(json_fragment.group(1))
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _clean_query_text(text)
    if isinstance(payload, dict):
        query = payload.get("query")
        if isinstance(query, str):
            return _clean_query_text(query)
        keywords = payload.get("keywords")
        if isinstance(keywords, list):
            return _clean_query_text(" ".join(str(item) for item in keywords))
    return ""


def _clean_query_text(text: str) -> str:
    text = re.sub(r"^[\s\"']*query[\s\"']*:\s*", "", text.strip(), flags=re.IGNORECASE)
    text = text.strip().strip("\"'")
    return re.sub(r"\s+", " ", text).strip()


def _dedupe_queries(queries: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = " ".join(query.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _source_specific_search_enabled(domain: str, config: SearchConfig) -> bool:
    if domain == "biomed":
        return config.pubmed_enabled or config.semantic_scholar_enabled or config.tavily_enabled
    if domain in {"preprint", "cs", "math", "physics"}:
        return config.arxiv_enabled or config.semantic_scholar_enabled or config.tavily_enabled
    return config.semantic_scholar_enabled or config.tavily_enabled
