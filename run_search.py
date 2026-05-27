import asyncio
import logging
import os
import sys
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Load key=value pairs from a .env file into os.environ (skips already-set vars)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler("search.log", mode="w"), logging.StreamHandler(sys.stdout)],
)

from co_scientist.config import load_config
from co_scientist.llm.client import LLMClient, LLMRouter
from co_scientist.search import SearchConfig, search_for_goal

GOAL = (
    "What impact did the selection pressure that caused the differentiation between cannabis in high latitudes of China "
    "and cannabis in low latitudes play in the differentiation of cannabis in high latitudes to European fiber and the "
    "differentiation of cannabis in low latitudes to medicinal use in South Asia?"
)


async def main() -> None:
    config = load_config()
    client = LLMClient(config.llm.providers["deepseek"])

    search_config = SearchConfig()
    log = logging.getLogger(__name__)
    log.info("Starting search for goal...")
    log.info("Goal: %s", GOAL)
    log.info("LLM provider: deepseek (%s)", config.llm.providers["deepseek"].chat_model)
    log.info("API keys loaded — TAVILY:%s  NCBI:%s  S2:%s",
             "yes" if search_config.tavily_api_key else "MISSING",
             "yes" if search_config.ncbi_api_key else "no (rate-limited)",
             "yes" if search_config.s2_api_key else "no (rate-limited)")

    results = await search_for_goal(GOAL, client, search_config)

    print(f"\n{'='*70}")
    print(f"Found {len(results)} results\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r.title}")
        print(f"    Source: {r.source} | Year: {r.year} | DOI: {r.doi}")
        print(f"    URL: {r.url}")
        print(f"    Abstract: {r.abstract[:300]}...")
        print()


asyncio.run(main())
