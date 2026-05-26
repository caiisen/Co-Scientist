from __future__ import annotations

import re
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse, urlunparse

from pydantic import BaseModel, Field


class ToolStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    FAILED = "failed"


class Citation(BaseModel):
    source: str
    title: str
    url: str | None = None
    doi: str | None = None
    pmid: str | None = None
    arxiv_id: str | None = None
    semantic_scholar_id: str | None = None
    year: int | None = None
    raw_json: dict[str, Any] = Field(default_factory=dict)

    def dedupe_key(self) -> str:
        if self.doi:
            return f"doi:{normalize_identifier(self.doi)}"
        if self.pmid:
            return f"pmid:{normalize_identifier(self.pmid)}"
        if self.arxiv_id:
            return f"arxiv:{normalize_identifier(self.arxiv_id)}"
        if self.semantic_scholar_id:
            return f"semantic_scholar:{normalize_identifier(self.semantic_scholar_id)}"
        if self.url:
            return f"url:{normalize_url(self.url)}"
        return f"title:{normalize_title(self.title)}:{self.year or ''}"


class SearchDocument(BaseModel):
    source: str
    title: str
    abstract_or_snippet: str | None = None
    authors: list[str] = Field(default_factory=list)
    venue: str | None = None
    year: int | None = None
    score: float | None = None
    citation: Citation

    def evidence_text(self, *, max_chars: int | None = None) -> str:
        identifiers = []
        if self.citation.doi:
            identifiers.append(f"DOI: {self.citation.doi}")
        if self.citation.pmid:
            identifiers.append(f"PMID: {self.citation.pmid}")
        if self.citation.arxiv_id:
            identifiers.append(f"arXiv: {self.citation.arxiv_id}")
        if self.citation.semantic_scholar_id:
            identifiers.append(f"Semantic Scholar: {self.citation.semantic_scholar_id}")
        if self.citation.url:
            identifiers.append(self.citation.url)

        header_parts = [self.title]
        if self.year:
            header_parts.append(str(self.year))
        if self.venue:
            header_parts.append(self.venue)
        header_parts.append(self.source)

        body = normalize_text(self.abstract_or_snippet or "No abstract or snippet available.")
        if max_chars is not None:
            body = truncate_text(body, max_chars)
        suffix = f"\nIdentifiers: {'; '.join(identifiers)}" if identifiers else ""
        return f"{' | '.join(header_parts)}\n{body}{suffix}"


class ToolResult(BaseModel):
    source: str
    status: ToolStatus = ToolStatus.OK
    documents: list[SearchDocument] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @classmethod
    def from_documents(
        cls,
        *,
        source: str,
        documents: list[SearchDocument],
        errors: list[str] | None = None,
    ) -> ToolResult:
        status = ToolStatus.OK
        if errors and documents:
            status = ToolStatus.DEGRADED
        elif errors and not documents:
            status = ToolStatus.FAILED
        return cls(
            source=source,
            status=status,
            documents=documents,
            citations=[document.citation for document in documents],
            errors=errors or [],
        )

    def format_evidence_pack(
        self,
        *,
        max_items: int = 5,
        max_chars_per_item: int | None = None,
    ) -> str:
        if not self.documents:
            error_text = "; ".join(self.errors) if self.errors else "no results"
            return f"No evidence returned from {self.source}: {error_text}"

        lines = []
        for index, document in enumerate(self.documents[:max_items], start=1):
            lines.append(f"[{index}] {document.evidence_text(max_chars=max_chars_per_item)}")
        if self.errors:
            lines.append(f"Search warnings: {'; '.join(self.errors)}")
        return "\n\n".join(lines)


DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
ARXIV_RE = re.compile(r"\b(?:arXiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)\b", re.IGNORECASE)
PMID_RE = re.compile(r"\bPMID[:\s]+(\d+)\b", re.IGNORECASE)


def truncate_text(text: str, max_chars: int) -> str:
    normalized = normalize_text(text)
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def normalize_identifier(value: str) -> str:
    return value.strip().lower()


def normalize_title(value: str) -> str:
    return re.sub(r"\W+", " ", value).strip().lower()


def normalize_url(value: str) -> str:
    parsed = urlparse(value.strip())
    path = parsed.path.rstrip("/")
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", "", ""))


def extract_doi(*values: str | None) -> str | None:
    for value in values:
        if not value:
            continue
        match = DOI_RE.search(value)
        if match:
            return match.group(0).rstrip(".")
    return None


def extract_arxiv_id(*values: str | None) -> str | None:
    for value in values:
        if not value:
            continue
        match = ARXIV_RE.search(value)
        if match:
            return match.group(1)
    return None


def extract_pmid(*values: str | None) -> str | None:
    for value in values:
        if not value:
            continue
        match = PMID_RE.search(value)
        if match:
            return match.group(1)
    return None
