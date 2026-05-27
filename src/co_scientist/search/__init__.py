from .config import SearchConfig
from .evidence import search_for_goal
from .platforms.base import SearchResult

__all__ = ["search_for_goal", "SearchResult", "SearchConfig"]
