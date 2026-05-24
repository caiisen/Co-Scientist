from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp

from co_scientist.tools.models import Citation, SearchDocument, ToolResult
from co_scientist.tools.registry import register_tool

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
DEFAULT_FIELDS = "paperId,title,abstract,authors,venue,year,url,externalIds,citationCount"

FetchJson = Callable[[str, dict[str, Any], dict[str, str], int], Awaitable[dict[str, Any]]]


async def _fetch_json(
    url: str,
    params: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: int,
) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, params=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()


@register_tool("semantic_scholar")
async def search(
    query: str,
    *,
    max_results: int = 5,
    timeout_seconds: int = 60,
    api_key: str | None = None,
    api_key_env: str = "SEMANTIC_SCHOLAR_API_KEY",
    fetch_json: FetchJson | None = None,
) -> ToolResult:
    fetch = fetch_json or _fetch_json
    headers = {}
    resolved_key = api_key or os.getenv(api_key_env)
    if resolved_key:
        headers["x-api-key"] = resolved_key
    params = {"query": query, "limit": max_results, "fields": DEFAULT_FIELDS}
    try:
        payload = await fetch(SEMANTIC_SCHOLAR_SEARCH_URL, params, headers, timeout_seconds)
        return ToolResult.from_documents(
            source="semantic_scholar",
            documents=parse_search_payload(payload),
        )
    except (aiohttp.ClientError, TimeoutError) as exc:
        return ToolResult.from_documents(
            source="semantic_scholar",
            documents=[],
            errors=[str(exc)],
        )


def parse_search_payload(payload: dict[str, Any]) -> list[SearchDocument]:
    documents: list[SearchDocument] = []
    for item in payload.get("data", []):
        title = item.get("title") or "Untitled Semantic Scholar record"
        external_ids = item.get("externalIds") or {}
        authors = [
            author.get("name")
            for author in item.get("authors", [])
            if isinstance(author, dict) and author.get("name")
        ]
        citation = Citation(
            source="semantic_scholar",
            title=title,
            url=item.get("url"),
            doi=external_ids.get("DOI"),
            pmid=external_ids.get("PubMed"),
            arxiv_id=external_ids.get("ArXiv"),
            semantic_scholar_id=item.get("paperId"),
            year=item.get("year"),
            raw_json=item,
        )
        documents.append(
            SearchDocument(
                source="semantic_scholar",
                title=title,
                abstract_or_snippet=item.get("abstract"),
                authors=authors,
                venue=item.get("venue"),
                year=item.get("year"),
                score=item.get("citationCount"),
                citation=citation,
            )
        )
    return documents
