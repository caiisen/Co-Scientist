from __future__ import annotations

import xml.etree.ElementTree as ET

import aiohttp

from .base import SearchPlatform, SearchResult

_ATOM = "http://www.w3.org/2005/Atom"
_ARXIV = "http://arxiv.org/schemas/atom"
_BASE_URL = "http://export.arxiv.org/api/query"


class ArXivPlatform(SearchPlatform):
    def __init__(self, timeout: float = 12.0) -> None:
        self._timeout = aiohttp.ClientTimeout(total=timeout)

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        params = {
            "search_query": query,
            "max_results": max_results,
            "sortBy": "relevance",
        }
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.get(_BASE_URL, params=params) as resp:
                resp.raise_for_status()
                xml_text = await resp.text()
        return self._parse_atom(xml_text)

    def _parse_atom(self, xml_text: str) -> list[SearchResult]:
        root = ET.fromstring(xml_text)
        results: list[SearchResult] = []
        for entry in root.findall(f"{{{_ATOM}}}entry"):
            title = (entry.findtext(f"{{{_ATOM}}}title") or "").strip()
            abstract = (entry.findtext(f"{{{_ATOM}}}summary") or "").strip()
            published = entry.findtext(f"{{{_ATOM}}}published") or ""
            year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else None
            arxiv_id = (entry.findtext(f"{{{_ATOM}}}id") or "").strip()
            url = arxiv_id if arxiv_id.startswith("http") else None

            # arXiv API doesn't reliably expose DOIs; use arxiv URL as canonical id
            if title:
                results.append(SearchResult(
                    title=title,
                    abstract=abstract[:1200],
                    year=year,
                    source="arxiv",
                    url=url,
                    doi=None,
                ))
        return results
