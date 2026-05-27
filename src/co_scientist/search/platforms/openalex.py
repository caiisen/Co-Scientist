from __future__ import annotations

import aiohttp

from .base import SearchPlatform, SearchResult

_BASE_URL = "https://api.openalex.org/works"
_SELECT = "title,abstract_inverted_index,publication_year,cited_by_count,doi,id"


class OpenAlexPlatform(SearchPlatform):
    def __init__(self, email: str | None = None, timeout: float = 12.0) -> None:
        self._email = email
        self._timeout = aiohttp.ClientTimeout(total=timeout)

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        params: dict[str, object] = {
            "search": query,
            "select": _SELECT,
            "sort": "relevance_score:desc",
            "per_page": max_results,
        }
        if self._email:
            params["mailto"] = self._email

        headers = {"Accept-Encoding": "gzip, deflate"}
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.get(_BASE_URL, params=params, headers=headers) as resp:
                resp.raise_for_status()
                data = await resp.json()

        results: list[SearchResult] = []
        for work in data.get("results") or []:
            title = work.get("title") or ""
            abstract = self._decode_abstract(work.get("abstract_inverted_index"))
            year = work.get("publication_year")
            citation_count = work.get("cited_by_count") or 0
            raw_doi = work.get("doi") or ""
            doi = (
                raw_doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
                or None
            )
            openalex_id = work.get("id") or ""
            url = openalex_id if openalex_id.startswith("http") else None
            if title:
                results.append(SearchResult(
                    title=title,
                    abstract=abstract[:1200],
                    year=year,
                    source="openalex",
                    url=url,
                    doi=doi,
                    score=float(citation_count),
                ))
        return results

    @staticmethod
    def _decode_abstract(inv_index: dict[str, list[int]] | None) -> str:
        if not inv_index:
            return ""
        pos_word: list[tuple[int, str]] = []
        for word, positions in inv_index.items():
            for pos in positions:
                pos_word.append((pos, word))
        pos_word.sort(key=lambda x: x[0])
        return " ".join(w for _, w in pos_word)
