from __future__ import annotations

import os

from pydantic import BaseModel, Field

ALL_PLATFORMS: list[str] = ["pubmed", "arxiv", "openalex", "semantic_scholar", "tavily"]


class SearchConfig(BaseModel):
    ncbi_api_key: str | None = Field(default_factory=lambda: os.getenv("NCBI_API_KEY"))
    s2_api_key: str | None = Field(default_factory=lambda: os.getenv("S2_API_KEY"))
    tavily_api_key: str = Field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    openalex_email: str | None = Field(default_factory=lambda: os.getenv("OPENALEX_EMAIL"))

    enabled_platforms: list[str] = Field(default_factory=lambda: list(ALL_PLATFORMS))

    max_results_per_query: int = 5
    queries_per_platform: int = 3
    top_k_final: int = 20
    timeout_seconds: float = 12.0
