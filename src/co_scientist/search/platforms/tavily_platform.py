from __future__ import annotations

import aiohttp

from .base import SearchPlatform, SearchResult

_API_URL = "https://api.tavily.com/search"


class TavilyPlatform(SearchPlatform):
    def __init__(self, api_key: str, timeout: float = 12.0) -> None:
        self._api_key = api_key
        self._timeout = aiohttp.ClientTimeout(total=timeout)

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        payload = {
            "api_key": self._api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
            "include_answer": False,
        }
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.post(_API_URL, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()

        results: list[SearchResult] = []
        for item in data.get("results") or []:
            title = item.get("title") or ""
            content = item.get("content") or ""
            url = item.get("url")
            score = float(item.get("score") or 0.0)
            if title:
                results.append(SearchResult(
                    title=title,
                    abstract=content[:1200],
                    year=None,
                    source="tavily",
                    url=url,
                    doi=None,
                    score=score,
                ))
        return results
