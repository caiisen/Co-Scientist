from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from co_scientist.config import SearchConfig
from co_scientist.memory import SQLiteStore
from co_scientist.tools.literature import (
    dedupe_documents,
    search_literature,
    search_literature_with_fallbacks,
)
from co_scientist.tools.models import Citation, SearchDocument, ToolResult, ToolStatus


def document(source: str, title: str, *, doi: str | None = None, pmid: str | None = None):
    return SearchDocument(
        source=source,
        title=title,
        citation=Citation(source=source, title=title, doi=doi, pmid=pmid),
    )


def test_dedupe_documents_prefers_pubmed_for_same_doi() -> None:
    documents = dedupe_documents(
        [
            document("semantic_scholar", "Paper via Semantic Scholar", doi="10.1000/x"),
            document("pubmed", "Paper via PubMed", doi="10.1000/x"),
        ]
    )

    assert len(documents) == 1
    assert documents[0].source == "pubmed"


@pytest.mark.asyncio
async def test_search_literature_aggregates_partial_failures_and_persists_citations(
    tmp_path: Path,
) -> None:
    calls = {"pubmed": 0}

    async def pubmed_search(query: str, *, max_results: int):
        calls["pubmed"] += 1
        return ToolResult.from_documents(
            source="pubmed",
            documents=[document("pubmed", "PubMed paper", pmid="123")],
        )

    async def semantic_search(query: str, *, max_results: int):
        return ToolResult.from_documents(
            source="semantic_scholar",
            documents=[],
            errors=["rate limited"],
        )

    async def tavily_search(query: str, *, max_results: int):
        return ToolResult.from_documents(
            source="tavily",
            documents=[document("tavily", "Web paper", doi="10.1000/web")],
        )

    async with SQLiteStore(tmp_path / "literature.sqlite") as store:
        session = await store.create_session("goal")
        result = await search_literature(
            "AML",
            config=SearchConfig(max_results=5),
            store=store,
            session_id=session.id,
            source_searchers={
                "pubmed": pubmed_search,
                "semantic_scholar": semantic_search,
                "tavily": tavily_search,
            },
        )
        cached = await search_literature(
            "AML",
            config=SearchConfig(max_results=5),
            store=store,
            session_id=session.id,
            source_searchers={
                "pubmed": pubmed_search,
                "semantic_scholar": semantic_search,
                "tavily": tavily_search,
            },
        )

    assert result.status == ToolStatus.DEGRADED
    assert len(result.documents) == 2
    assert calls["pubmed"] == 1
    assert cached.documents[0].citation.pmid == "123"


@pytest.mark.asyncio
async def test_search_literature_can_defer_citation_persistence(tmp_path: Path) -> None:
    async def pubmed_search(query: str, *, max_results: int):
        return ToolResult.from_documents(
            source="pubmed",
            documents=[document("pubmed", "PubMed paper", pmid="123")],
        )

    async with SQLiteStore(tmp_path / "literature_defer.sqlite") as store:
        session = await store.create_session("goal")
        result = await search_literature(
            "AML",
            config=SearchConfig(
                max_results=5,
                semantic_scholar_enabled=False,
                tavily_enabled=False,
            ),
            store=store,
            session_id=session.id,
            persist_citations=False,
            source_searchers={"pubmed": pubmed_search},
        )
        async with store.db.execute(
            "SELECT COUNT(*) AS count FROM citations WHERE session_id = ?",
            (session.id,),
        ) as cursor:
            row = await cursor.fetchone()

    assert result.status == ToolStatus.OK
    assert row["count"] == 0


@pytest.mark.asyncio
async def test_search_literature_runs_cache_misses_concurrently() -> None:
    active = 0
    all_running = asyncio.Event()

    async def searcher(query: str, *, max_results: int):
        nonlocal active
        active += 1
        if active == 3:
            all_running.set()
        await asyncio.wait_for(all_running.wait(), timeout=0.2)
        return ToolResult.from_documents(
            source="test",
            documents=[document("test", f"{query}-{active}-{max_results}")],
        )

    result = await search_literature(
        "AML",
        config=SearchConfig(max_results=3),
        source_searchers={
            "pubmed": searcher,
            "semantic_scholar": searcher,
            "tavily": searcher,
        },
    )

    assert result.status == ToolStatus.OK
    assert active == 3


@pytest.mark.asyncio
async def test_search_literature_does_not_swallow_source_code_bugs() -> None:
    async def buggy_searcher(query: str, *, max_results: int):
        raise TypeError("programming bug")

    with pytest.raises(TypeError, match="programming bug"):
        await search_literature(
            "AML",
            config=SearchConfig(
                max_results=1,
                semantic_scholar_enabled=False,
                tavily_enabled=False,
            ),
            source_searchers={"pubmed": buggy_searcher},
        )


@pytest.mark.asyncio
async def test_search_literature_with_fallbacks_uses_later_query(tmp_path: Path) -> None:
    calls = []

    async def pubmed_search(query: str, *, max_results: int):
        calls.append(query)
        documents = [] if query == "too narrow" else [document("pubmed", "Fallback", pmid="1")]
        return ToolResult.from_documents(source="pubmed", documents=documents)

    result = await search_literature_with_fallbacks(
        ["too narrow", "broader query"],
        config=SearchConfig(
            max_results=1,
            semantic_scholar_enabled=False,
            tavily_enabled=False,
        ),
        source_searchers={"pubmed": pubmed_search},
    )

    assert calls == ["too narrow", "broader query"]
    assert result.documents[0].title == "Fallback"


@pytest.mark.asyncio
async def test_search_literature_includes_private_corpus_markdown(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "paper.md").write_text(
        "# Cannabis latitude selection\n\n"
        "High-latitude cannabis selection favored fiber traits and flowering timing. "
        "Low-latitude populations retained medicinal chemotype variation.",
        encoding="utf-8",
    )

    async with SQLiteStore(tmp_path / "private.sqlite") as store:
        session = await store.create_session("goal")
        result = await search_literature(
            "high latitude cannabis fiber selection",
            config=SearchConfig(
                max_results=5,
                tavily_enabled=False,
                pubmed_enabled=False,
                semantic_scholar_enabled=False,
                arxiv_enabled=False,
                private_corpus_enabled=True,
                private_corpus_paths=[str(corpus_dir)],
            ),
            store=store,
            session_id=session.id,
        )
        chunk_count = await store.count_private_corpus_chunks(session.id)

    assert result.status == ToolStatus.OK
    assert result.documents
    assert result.documents[0].source == "private_corpus"
    assert "Cannabis latitude selection" in result.documents[0].title
    assert "chunk_id" not in result.documents[0].citation.raw_json
    assert chunk_count >= 1


class BrokenEmbeddingClient:
    async def embed(self, texts):
        raise RuntimeError("embedding unavailable")


@pytest.mark.asyncio
async def test_private_corpus_reports_embedding_fallback_as_degraded(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "paper.txt").write_text(
        "Cannabis fiber selection and latitude adaptation.",
        encoding="utf-8",
    )

    async with SQLiteStore(tmp_path / "private_degraded.sqlite") as store:
        session = await store.create_session("goal")
        result = await search_literature(
            "cannabis fiber latitude",
            config=SearchConfig(
                max_results=5,
                tavily_enabled=False,
                pubmed_enabled=False,
                semantic_scholar_enabled=False,
                arxiv_enabled=False,
                private_corpus_enabled=True,
                private_corpus_paths=[str(corpus_dir)],
            ),
            store=store,
            session_id=session.id,
            embedding_client=BrokenEmbeddingClient(),
        )

    assert result.status == ToolStatus.DEGRADED
    assert result.documents
    assert "embedding search failed" in result.errors[0]
