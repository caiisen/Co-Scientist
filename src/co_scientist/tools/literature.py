from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import aiohttp

from co_scientist.config import SearchConfig
from co_scientist.memory.store import SQLiteStore
from co_scientist.tools import arxiv, pubmed, semantic_scholar, web_search
from co_scientist.tools.cache import ToolCache
from co_scientist.tools.models import SearchDocument, ToolResult, ToolStatus

SearchCallable = Callable[..., Awaitable[ToolResult]]


async def search_literature(
    query: str,
    *,
    domain: str = "biomed",
    max_results: int | None = None,
    config: SearchConfig | None = None,
    store: SQLiteStore | None = None,
    session_id: str | None = None,
    persist_citations: bool = True,
    http_session: aiohttp.ClientSession | None = None,
    source_searchers: dict[str, SearchCallable] | None = None,
) -> ToolResult:
    cfg = config or SearchConfig(max_results=5)
    limit = max_results or cfg.max_results
    cache = (
        ToolCache(
            store,
            ttl_seconds=cfg.cache_ttl_seconds,
            failed_ttl_seconds=cfg.failed_cache_ttl_seconds,
        )
        if store is not None
        else None
    )
    documents: list[SearchDocument] = []
    errors: list[str] = []

    searchers = source_searchers or _default_searchers()
    misses: list[tuple[str, int, SearchCallable, dict[str, str]]] = []
    for source in _sources_for_domain(domain, cfg):
        searcher = searchers.get(source)
        if searcher is None:
            continue
        source_limit = _source_limit(source, cfg, limit)
        options = {"domain": domain}
        cached = (
            await cache.get(
                source=source,
                query=query,
                max_results=source_limit,
                options=options,
            )
            if cache
            else None
        )
        if cached is not None:
            _merge_source_result(source, cached, documents, errors)
            continue
        misses.append((source, source_limit, searcher, options))

    if misses:
        results = await asyncio.gather(
            *[
                _call_searcher(
                    source,
                    searcher,
                    query,
                    max_results=source_limit,
                    http_session=http_session,
                )
                for source, source_limit, searcher, _ in misses
            ]
        )
        for (source, source_limit, _, options), result in zip(misses, results, strict=True):
            if cache:
                await cache.set(
                    source=source,
                    query=query,
                    max_results=source_limit,
                    result=result,
                    options=options,
                )
            _merge_source_result(source, result, documents, errors)

    deduped = dedupe_documents(documents)[:limit]
    if store is not None and persist_citations:
        await _persist_citations(store, deduped, session_id=session_id)

    status = ToolStatus.OK
    if errors and deduped:
        status = ToolStatus.DEGRADED
    elif errors and not deduped:
        status = ToolStatus.FAILED
    return ToolResult(
        source="literature",
        status=status,
        documents=deduped,
        citations=[document.citation for document in deduped],
        errors=errors,
    )


async def search_literature_with_fallbacks(
    queries: list[str],
    **kwargs,
) -> ToolResult:
    first_result: ToolResult | None = None
    errors: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = " ".join(query.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result = await search_literature(normalized, **kwargs)
        if first_result is None:
            first_result = result
        if result.documents:
            return result
        errors.extend(f"{normalized}: {error}" for error in result.errors)
    if first_result is not None:
        if errors:
            first_result.errors.extend(errors)
        return first_result
    return ToolResult(
        source="literature",
        status=ToolStatus.FAILED,
        errors=["no literature queries provided"],
    )


async def _call_searcher(
    source: str,
    searcher: SearchCallable,
    query: str,
    *,
    max_results: int,
    http_session: aiohttp.ClientSession | None,
) -> ToolResult:
    if http_session is not None and source in {"semantic_scholar", "arxiv"}:
        return await searcher(query, max_results=max_results, session=http_session)
    return await searcher(query, max_results=max_results)


def _merge_source_result(
    source: str,
    result: ToolResult,
    documents: list[SearchDocument],
    errors: list[str],
) -> None:
    documents.extend(result.documents)
    errors.extend(f"{source}: {error}" for error in result.errors)


def dedupe_documents(documents: list[SearchDocument]) -> list[SearchDocument]:
    seen: set[str] = set()
    deduped: list[SearchDocument] = []
    for document in sorted(documents, key=_document_sort_key):
        key = document.citation.dedupe_key()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(document)
    return deduped


def _document_sort_key(document: SearchDocument) -> tuple[int, float, int]:
    source_priority = {
        "pubmed": 0,
        "semantic_scholar": 1,
        "arxiv": 2,
        "tavily": 3,
    }.get(document.source, 9)
    score = document.score if document.score is not None else 0
    year = document.year if document.year is not None else 0
    return (source_priority, -float(score), -year)


def _sources_for_domain(domain: str, config: SearchConfig) -> list[str]:
    normalized = domain.lower()
    sources: list[str]
    if normalized == "biomed":
        sources = ["pubmed", "semantic_scholar", "tavily"]
    elif normalized in {"preprint", "cs", "math", "physics"}:
        sources = ["arxiv", "semantic_scholar", "tavily"]
    else:
        sources = ["semantic_scholar", "tavily"]
    enabled = {
        "pubmed": config.pubmed_enabled,
        "semantic_scholar": config.semantic_scholar_enabled,
        "arxiv": config.arxiv_enabled,
        "tavily": config.tavily_enabled,
    }
    return [source for source in sources if enabled[source]]


def _source_limit(source: str, config: SearchConfig, fallback: int) -> int:
    return {
        "pubmed": config.pubmed_max_results,
        "semantic_scholar": config.semantic_scholar_max_results,
        "arxiv": config.arxiv_max_results,
        "tavily": config.tavily_max_results,
    }.get(source) or fallback


def _default_searchers() -> dict[str, SearchCallable]:
    return {
        "pubmed": pubmed.search,
        "semantic_scholar": semantic_scholar.search,
        "arxiv": arxiv.search,
        "tavily": web_search.search,
    }


async def _persist_citations(
    store: SQLiteStore,
    documents: list[SearchDocument],
    *,
    session_id: str | None,
) -> None:
    await store.add_citations_batch(
        [document.citation for document in documents],
        session_id=session_id,
    )
