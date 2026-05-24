from __future__ import annotations

import asyncio
import os
from typing import Any

from co_scientist.tools.models import (
    Citation,
    SearchDocument,
    ToolResult,
    extract_arxiv_id,
    extract_doi,
    extract_pmid,
)
from co_scientist.tools.registry import register_tool

TAVILY_RUNTIME_ERRORS = (ConnectionError, TimeoutError)


@register_tool("tavily")
async def search(
    query: str,
    *,
    max_results: int = 5,
    api_key: str | None = None,
    api_key_env: str = "TAVILY_API_KEY",
    client: Any | None = None,
) -> ToolResult:
    resolved_key = api_key or os.getenv(api_key_env)
    if client is None and not resolved_key:
        return ToolResult.from_documents(
            source="tavily",
            documents=[],
            errors=[f"{api_key_env} is not set"],
        )
    try:
        tavily_client = client or _build_tavily_client(resolved_key)
        payload = await asyncio.to_thread(
            tavily_client.search,
            query=query,
            max_results=max_results,
            include_raw_content=False,
        )
        return ToolResult.from_documents(source="tavily", documents=parse_tavily_payload(payload))
    except TAVILY_RUNTIME_ERRORS as exc:
        return ToolResult.from_documents(source="tavily", documents=[], errors=[str(exc)])


def _build_tavily_client(api_key: str | None) -> Any:
    from tavily import TavilyClient

    return TavilyClient(api_key=api_key)


def parse_tavily_payload(payload: dict[str, Any]) -> list[SearchDocument]:
    documents: list[SearchDocument] = []
    for item in payload.get("results", []):
        title = item.get("title") or item.get("url") or "Untitled web result"
        content = item.get("content") or item.get("raw_content")
        url = item.get("url")
        doi = extract_doi(title, content, url)
        pmid = extract_pmid(title, content, url)
        arxiv_id = extract_arxiv_id(title, content, url)
        citation = Citation(
            source="tavily",
            title=title,
            url=url,
            doi=doi,
            pmid=pmid,
            arxiv_id=arxiv_id,
            raw_json=item,
        )
        documents.append(
            SearchDocument(
                source="tavily",
                title=title,
                abstract_or_snippet=content,
                score=item.get("score"),
                citation=citation,
            )
        )
    return documents
