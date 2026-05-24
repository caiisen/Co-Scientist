from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp

from co_scientist.tools.models import Citation, SearchDocument, ToolResult, extract_arxiv_id
from co_scientist.tools.registry import register_tool

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

FetchText = Callable[[str, dict[str, Any], int], Awaitable[str]]


async def _fetch_text(url: str, params: dict[str, Any], timeout_seconds: int) -> str:
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.text()


@register_tool("arxiv")
async def search(
    query: str,
    *,
    max_results: int = 5,
    timeout_seconds: int = 60,
    fetch_text: FetchText | None = None,
) -> ToolResult:
    fetch = fetch_text or _fetch_text
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    try:
        xml_text = await fetch(ARXIV_API_URL, params, timeout_seconds)
        return ToolResult.from_documents(source="arxiv", documents=parse_arxiv_feed(xml_text))
    except (aiohttp.ClientError, TimeoutError, ET.ParseError) as exc:
        return ToolResult.from_documents(source="arxiv", documents=[], errors=[str(exc)])


def parse_arxiv_feed(xml_text: str) -> list[SearchDocument]:
    root = ET.fromstring(xml_text)
    documents: list[SearchDocument] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        title = _entry_text(entry, "atom:title") or "Untitled arXiv record"
        summary = _entry_text(entry, "atom:summary")
        published = _entry_text(entry, "atom:published")
        year = int(published[:4]) if published and published[:4].isdigit() else None
        entry_id = _entry_text(entry, "atom:id")
        arxiv_id = extract_arxiv_id(entry_id)
        authors = [
            _entry_text(author, "atom:name") or ""
            for author in entry.findall("atom:author", ATOM_NS)
        ]
        authors = [author for author in authors if author]
        citation = Citation(
            source="arxiv",
            title=" ".join(title.split()),
            url=entry_id,
            arxiv_id=arxiv_id,
            year=year,
            raw_json={"entry_id": entry_id, "published": published},
        )
        documents.append(
            SearchDocument(
                source="arxiv",
                title=citation.title,
                abstract_or_snippet=summary,
                authors=authors,
                venue="arXiv",
                year=year,
                citation=citation,
            )
        )
    return documents


def _entry_text(entry: ET.Element, path: str) -> str | None:
    child = entry.find(path, ATOM_NS)
    if child is None or child.text is None:
        return None
    return " ".join(child.text.split())
