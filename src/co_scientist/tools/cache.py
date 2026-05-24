from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from typing import Any

from co_scientist.memory.models import utc_now
from co_scientist.memory.store import SQLiteStore
from co_scientist.tools.models import ToolResult, ToolStatus


def options_hash(options: dict[str, Any] | None = None) -> str:
    payload = json.dumps(options or {}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_key(
    *,
    source: str,
    query: str,
    max_results: int,
    options: dict[str, Any] | None = None,
) -> str:
    payload = json.dumps(
        {
            "source": source,
            "query": query,
            "max_results": max_results,
            "options_hash": options_hash(options),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ToolCache:
    def __init__(
        self,
        store: SQLiteStore,
        *,
        ttl_seconds: int = 604800,
        failed_ttl_seconds: int = 300,
    ) -> None:
        self.store = store
        self.ttl_seconds = ttl_seconds
        self.failed_ttl_seconds = failed_ttl_seconds

    async def get(
        self,
        *,
        source: str,
        query: str,
        max_results: int,
        options: dict[str, Any] | None = None,
    ) -> ToolResult | None:
        key = cache_key(source=source, query=query, max_results=max_results, options=options)
        cached = await self.store.get_tool_cache(key)
        if cached is None:
            return None
        return ToolResult.model_validate(cached)

    async def set(
        self,
        *,
        source: str,
        query: str,
        max_results: int,
        result: ToolResult,
        options: dict[str, Any] | None = None,
    ) -> None:
        ttl = self.failed_ttl_seconds if result.status == ToolStatus.FAILED else self.ttl_seconds
        if ttl <= 0:
            return
        key = cache_key(source=source, query=query, max_results=max_results, options=options)
        await self.store.set_tool_cache(
            cache_key=key,
            source=source,
            query=query,
            max_results=max_results,
            options_hash=options_hash(options),
            status=result.status.value,
            result_json=result.model_dump(mode="json"),
            expires_at=utc_now() + timedelta(seconds=ttl),
        )
