from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET

from .base import SearchPlatform, SearchResult


class PubMedPlatform(SearchPlatform):
    def __init__(self, api_key: str | None = None, email: str = "search@co-scientist.local") -> None:  # noqa: E501
        self._api_key = api_key
        self._email = email

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_search, query, max_results)

    def _sync_search(self, query: str, max_results: int) -> list[SearchResult]:
        from Bio import Entrez  # local import to avoid Entrez globals at module level

        Entrez.email = self._email
        if self._api_key:
            Entrez.api_key = self._api_key

        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        ids: list[str] = record.get("IdList", [])
        if not ids:
            return []

        handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="xml")
        xml_bytes = handle.read()
        handle.close()

        return self._parse_xml(xml_bytes if isinstance(xml_bytes, bytes) else xml_bytes.encode())

    def _parse_xml(self, xml_bytes: bytes) -> list[SearchResult]:
        root = ET.fromstring(xml_bytes)
        results: list[SearchResult] = []
        for article in root.findall(".//PubmedArticle"):
            title = article.findtext(".//ArticleTitle") or ""
            abstract_nodes = article.findall(".//AbstractText")
            abstract = " ".join((n.text or "") for n in abstract_nodes).strip()
            year_text = article.findtext(".//PubDate/Year")
            year = int(year_text) if year_text and year_text.isdigit() else None

            doi_elem = article.find(".//ArticleId[@IdType='doi']")
            doi = doi_elem.text.strip() if doi_elem is not None and doi_elem.text else None

            pmid_elem = article.find(".//ArticleId[@IdType='pubmed']")
            pmid = pmid_elem.text.strip() if pmid_elem is not None and pmid_elem.text else None
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None

            if title:
                results.append(SearchResult(
                    title=title,
                    abstract=abstract[:1200],
                    year=year,
                    source="pubmed",
                    url=url,
                    doi=doi,
                ))
        return results
