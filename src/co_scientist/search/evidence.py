from __future__ import annotations

import asyncio
import logging

from co_scientist.llm.client import LLMClient

from .config import SearchConfig
from .fusion import fuse_and_deduplicate
from .goal_parser import parse_goal
from .platforms.arxiv import ArXivPlatform
from .platforms.base import SearchPlatform, SearchResult
from .platforms.openalex import OpenAlexPlatform
from .platforms.pubmed import PubMedPlatform
from .platforms.semantic_scholar import SemanticScholarPlatform
from .platforms.tavily_platform import TavilyPlatform
from .query_builder import build_queries

logger = logging.getLogger(__name__)


async def search_for_goal(
    goal: str,
    llm_client: LLMClient,
    config: SearchConfig | None = None,
) -> list[SearchResult]:
    """Search across academic databases and the web for the given research goal.

    Returns up to config.top_k_final de-duplicated, RRF-ranked results.
    """
    if config is None:
        config = SearchConfig()

    parsed = await parse_goal(goal, llm_client)
    platforms = config.enabled_platforms
    logger.info("Platforms: %s", platforms)
    queries = await build_queries(parsed, platforms, llm_client, n=config.queries_per_platform)

    platform_instances = _build_platforms(config, platforms)

    tasks: list[tuple[str, asyncio.Task[list[SearchResult]]]] = []
    for platform_name, platform in platform_instances.items():
        for query in queries.get(platform_name, []):
            coro = platform.search(query, config.max_results_per_query)
            task = asyncio.ensure_future(
                asyncio.wait_for(coro, timeout=config.timeout_seconds)
            )
            tasks.append((platform_name, task))

    all_results = await asyncio.gather(*[t for _, t in tasks], return_exceptions=True)

    ranked_lists: list[list[SearchResult]] = []
    for (platform_name, _), result in zip(tasks, all_results, strict=False):
        if isinstance(result, BaseException):
            logger.debug("Platform %s search failed: %s", platform_name, result)
        else:
            ranked_lists.append(result)

    fused = fuse_and_deduplicate(ranked_lists)
    return fused[: config.top_k_final]


def _build_platforms(config: SearchConfig, platforms: list[str]) -> dict[str, SearchPlatform]:
    instances: dict[str, SearchPlatform] = {}
    t = config.timeout_seconds
    for name in platforms:
        if name == "pubmed":
            instances[name] = PubMedPlatform(api_key=config.ncbi_api_key)
        elif name == "arxiv":
            instances[name] = ArXivPlatform(timeout=t)
        elif name == "semantic_scholar":
            instances[name] = SemanticScholarPlatform(api_key=config.s2_api_key, timeout=t)
        elif name == "openalex":
            instances[name] = OpenAlexPlatform(email=config.openalex_email, timeout=t)
        elif name == "tavily" and config.tavily_api_key:
            instances[name] = TavilyPlatform(api_key=config.tavily_api_key, timeout=t)
    return instances
