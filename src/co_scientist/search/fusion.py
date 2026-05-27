from __future__ import annotations

import re
from collections import defaultdict

from .platforms.base import SearchResult


def fuse_and_deduplicate(
    ranked_lists: list[list[SearchResult]],
    k: int = 60,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion across all ranked lists, with DOI-aware deduplication.

    score(d) = Σ_i  1 / (k + rank_i + 1)
    """
    scores: dict[str, float] = defaultdict(float)
    key_to_result: dict[str, SearchResult] = {}

    for ranked in ranked_lists:
        for rank, result in enumerate(ranked):
            key = _result_key(result)
            scores[key] += 1.0 / (k + rank + 1)
            if key not in key_to_result:
                key_to_result[key] = result

    sorted_keys = sorted(scores, key=lambda x: -scores[x])
    return [key_to_result[ck] for ck in sorted_keys]


def _result_key(result: SearchResult) -> str:
    if result.doi:
        doi = result.doi.lower()
        doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
        return f"doi:{doi}"
    # For arXiv, use the URL (contains arXiv ID) as canonical key
    if result.source == "arxiv" and result.url:
        # Normalize arxiv URL: strip version suffix e.g. v1, v2
        url = re.sub(r"v\d+$", "", result.url.rstrip("/"))
        return f"url:{url}"
    # Fallback: normalized title prefix
    title = re.sub(r"[^\w\s]", "", result.title.lower()).strip()
    return f"title:{title[:80]}"
