from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    title: str
    abstract: str
    year: int | None
    source: str
    url: str | None
    doi: str | None
    score: float = field(default=0.0)


class SearchPlatform(ABC):
    @abstractmethod
    async def search(self, query: str, max_results: int) -> list[SearchResult]: ...
