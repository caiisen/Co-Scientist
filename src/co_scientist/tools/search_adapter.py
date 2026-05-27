from __future__ import annotations

from co_scientist.search.platforms.base import SearchResult
from co_scientist.tools.models import Citation, SearchDocument, ToolResult


def to_tool_result(results: list[SearchResult]) -> ToolResult:
    """Convert search module SearchResult list to the project's ToolResult type."""
    documents: list[SearchDocument] = []
    for r in results:
        citation = Citation(
            source=r.source,
            title=r.title,
            url=r.url,
            doi=r.doi,
            year=r.year,
        )
        documents.append(
            SearchDocument(
                source=r.source,
                title=r.title,
                abstract_or_snippet=r.abstract or "",
                year=r.year,
                score=r.score if r.score else None,
                citation=citation,
            )
        )
    return ToolResult.from_documents(source="enhanced_search", documents=documents)
