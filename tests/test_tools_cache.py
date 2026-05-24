from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from co_scientist.memory import SQLiteStore
from co_scientist.memory.models import utc_now
from co_scientist.tools.cache import ToolCache
from co_scientist.tools.models import Citation, SearchDocument, ToolResult, ToolStatus


@pytest.mark.asyncio
async def test_tool_cache_round_trips_result(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "cache.sqlite") as store:
        cache = ToolCache(store)
        result = ToolResult.from_documents(
            source="pubmed",
            documents=[
                SearchDocument(
                    source="pubmed",
                    title="A paper",
                    citation=Citation(source="pubmed", title="A paper", pmid="123"),
                )
            ],
        )

        await cache.set(source="pubmed", query="AML", max_results=5, result=result)
        restored = await cache.get(source="pubmed", query="AML", max_results=5)

    assert restored is not None
    assert restored.documents[0].citation.pmid == "123"


@pytest.mark.asyncio
async def test_failed_tool_cache_uses_short_ttl_and_can_be_purged(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "failed-cache.sqlite") as store:
        cache = ToolCache(store, ttl_seconds=100, failed_ttl_seconds=1)
        result = ToolResult(source="pubmed", status=ToolStatus.FAILED, errors=["rate limited"])

        await cache.set(source="pubmed", query="AML", max_results=5, result=result)
        async with store.db.execute("SELECT created_at, expires_at FROM tool_cache") as cursor:
            row = await cursor.fetchone()
        created_at = datetime.fromisoformat(row["created_at"])
        expires_at = datetime.fromisoformat(row["expires_at"])
        assert timedelta(0) < expires_at - created_at <= timedelta(seconds=2)

        deleted = await store.purge_expired_tool_cache(now=utc_now() + timedelta(seconds=2))

    assert deleted == 1
