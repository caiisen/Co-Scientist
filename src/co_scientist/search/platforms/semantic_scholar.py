from __future__ import annotations

import aiohttp

from .base import SearchPlatform, SearchResult

_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "title,abstract,year,citationCount,externalIds,url"


class SemanticScholarPlatform(SearchPlatform):
    def __init__(self, api_key: str | None = None, timeout: float = 12.0) -> None:
        self._api_key = api_key
        self._timeout = aiohttp.ClientTimeout(total=timeout)

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["x-api-key"] = self._api_key

        params = {"query": query, "limit": max_results, "fields": _FIELDS}
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.get(_BASE_URL, params=params, headers=headers) as resp:
                resp.raise_for_status()
                data = await resp.json()

        papers = data.get("data") or []
        results: list[SearchResult] = []
        for p in papers:
            title = p.get("title") or ""
            abstract = p.get("abstract") or ""
            year = p.get("year")
            url = p.get("url")
            doi = (p.get("externalIds") or {}).get("DOI")
            citation_count = p.get("citationCount") or 0
            if title:
                results.append(SearchResult(
                    title=title,
                    abstract=abstract[:1200],
                    year=year,
                    source="semantic_scholar",
                    url=url,
                    doi=doi,
                    score=float(citation_count),
                ))
        return results
