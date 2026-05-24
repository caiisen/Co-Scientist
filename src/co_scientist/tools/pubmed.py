from __future__ import annotations

import asyncio
import os
import threading
import urllib.error
import xml.etree.ElementTree as ET
from typing import Any

from Bio import Entrez

from co_scientist.tools.models import Citation, SearchDocument, ToolResult
from co_scientist.tools.registry import register_tool

_ENTREZ_LOCK = threading.Lock()


@register_tool("pubmed")
async def search(
    query: str,
    *,
    max_results: int = 5,
    email: str | None = None,
    api_key: str | None = None,
    email_env: str = "NCBI_EMAIL",
    api_key_env: str = "NCBI_API_KEY",
    entrez: Any = Entrez,
) -> ToolResult:
    try:
        documents = await asyncio.to_thread(
            _search_sync,
            query,
            max_results,
            email or os.getenv(email_env),
            api_key or os.getenv(api_key_env),
            entrez,
        )
        return ToolResult.from_documents(source="pubmed", documents=documents)
    except (urllib.error.URLError, ET.ParseError) as exc:
        return ToolResult.from_documents(source="pubmed", documents=[], errors=[str(exc)])


def _search_sync(
    query: str,
    max_results: int,
    email: str | None,
    api_key: str | None,
    entrez: Any,
) -> list[SearchDocument]:
    with _ENTREZ_LOCK:
        if email:
            entrez.email = email
        if api_key:
            entrez.api_key = api_key

        handle = entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        try:
            search_result = entrez.read(handle)
        finally:
            handle.close()
        ids = list(search_result.get("IdList", []))
        if not ids:
            return []

        handle = entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="xml")
        try:
            fetch_result = entrez.read(handle)
        finally:
            handle.close()
    return parse_pubmed_payload(fetch_result)


def parse_pubmed_payload(payload: dict[str, Any]) -> list[SearchDocument]:
    articles = payload.get("PubmedArticle", [])
    documents: list[SearchDocument] = []
    for article in articles:
        medline = article.get("MedlineCitation", {})
        article_data = medline.get("Article", {})
        pmid = _stringify(medline.get("PMID"))
        title = _stringify(article_data.get("ArticleTitle")) or "Untitled PubMed record"
        abstract = _abstract_text(article_data.get("Abstract", {}))
        journal = article_data.get("Journal", {})
        venue = _stringify(journal.get("Title"))
        year = _publication_year(article_data, journal)
        doi = _article_doi(article_data)
        authors = _authors(article_data.get("AuthorList", []))
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
        raw_json = {"PMID": pmid, "ArticleTitle": title}
        citation = Citation(
            source="pubmed",
            title=title,
            url=url,
            doi=doi,
            pmid=pmid,
            year=year,
            raw_json=raw_json,
        )
        documents.append(
            SearchDocument(
                source="pubmed",
                title=title,
                abstract_or_snippet=abstract,
                authors=authors,
                venue=venue,
                year=year,
                citation=citation,
            )
        )
    return documents


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _abstract_text(abstract: dict[str, Any]) -> str | None:
    parts = abstract.get("AbstractText", []) if isinstance(abstract, dict) else []
    text = " ".join(str(part) for part in parts if part)
    return text or None


def _publication_year(article_data: dict[str, Any], journal: dict[str, Any]) -> int | None:
    pub_date = (
        journal.get("JournalIssue", {})
        .get("PubDate", {})
    )
    for candidate in (
        pub_date.get("Year"),
        article_data.get("ArticleDate", [{}])[0].get("Year")
        if article_data.get("ArticleDate")
        else None,
    ):
        if candidate and str(candidate).isdigit():
            return int(candidate)
    return None


def _article_doi(article_data: dict[str, Any]) -> str | None:
    for item in article_data.get("ELocationID", []):
        attributes = getattr(item, "attributes", {})
        if attributes.get("EIdType") == "doi":
            return str(item)
    for item in article_data.get("ArticleIdList", []):
        attributes = getattr(item, "attributes", {})
        if attributes.get("IdType") == "doi":
            return str(item)
    return None


def _authors(author_list: list[Any]) -> list[str]:
    authors: list[str] = []
    for author in author_list:
        if not isinstance(author, dict):
            continue
        collective = author.get("CollectiveName")
        if collective:
            authors.append(str(collective))
            continue
        last = author.get("LastName")
        initials = author.get("Initials")
        if last and initials:
            authors.append(f"{last} {initials}")
        elif last:
            authors.append(str(last))
    return authors
