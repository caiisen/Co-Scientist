from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import aiohttp  # noqa: E402

from co_scientist.config import load_config  # noqa: E402
from co_scientist.llm.client import LLMRouter  # noqa: E402
from co_scientist.memory.store import SQLiteStore  # noqa: E402
from co_scientist.tools.literature import search_literature_with_fallbacks  # noqa: E402
from co_scientist.tools.models import ToolResult, ToolStatus  # noqa: E402
from co_scientist.tools.query import build_literature_query  # noqa: E402

DOMAINS = ("biomed", "cs", "physics", "math", "preprint")
DEFAULT_DB_PATH = PROJECT_ROOT / "runs" / "search_probe.sqlite"
DEFAULT_GOAL_PATH = PROJECT_ROOT / "goal.txt"
SESSION_ID = "search_goal_probe"
SOURCE_ENABLED_FIELDS = {
    "pubmed": "pubmed_enabled",
    "semantic_scholar": "semantic_scholar_enabled",
    "arxiv": "arxiv_enabled",
    "tavily": "tavily_enabled",
    "private_corpus": "private_corpus_enabled",
}


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    asyncio.run(run_probe(args))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe configured literature and web search sources for a goal file.",
    )
    parser.add_argument(
        "--goal-file",
        type=Path,
        default=DEFAULT_GOAL_PATH,
        help="Goal text file to search from. Defaults to goal.txt in the repo root.",
    )
    parser.add_argument(
        "--domain",
        choices=DOMAINS,
        default="biomed",
        help="Search domain used by the literature aggregator.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Maximum aggregate documents to return. Defaults to configured search.max_results.",
    )
    parser.add_argument(
        "--query-mode",
        choices=("llm", "lexical"),
        default="llm",
        help="Use an LLM or the local lexical extractor for the compressed fallback query.",
    )
    parser.add_argument(
        "--llm-query-max-tokens",
        type=int,
        default=None,
        help="Maximum tokens for LLM query generation. Defaults to the configured provider.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="SQLite DB for probe cache and citations.",
    )
    return parser.parse_args()


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = clean_env_value(value.strip())


def clean_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value.split(" #", 1)[0].strip()


async def build_probe_query(
    goal: str,
    config: Any,
    mode: str,
    max_tokens: int | None,
    warnings: list[str],
    *,
    style: str,
) -> str:
    lexical_query = build_literature_query(goal)
    if mode == "lexical":
        return lexical_query

    provider = config.llm.provider_for_agent("generation")
    if not provider.resolved_api_key:
        warnings.append(
            f"{provider.api_key_env or 'LLM API key'} is not set; using lexical query"
        )
        return lexical_query

    try:
        client = LLMRouter(config.llm).client_for("generation")
        response = await client.chat(
            query_messages(goal, style),
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        warnings.append(f"LLM query generation failed: {exc}; using lexical query")
        return lexical_query

    llm_query = parse_llm_query(response)
    if not llm_query:
        preview = truncate(response, 160) or "<empty>"
        warnings.append(
            f"LLM returned an empty query; raw response: {preview}; using lexical query"
        )
        return lexical_query
    return llm_query


def query_messages(goal: str, style: str) -> list[dict[str, str]]:
    if style == "pubmed":
        instruction = (
            "Create one PubMed-compatible advanced search query. Use Boolean syntax with "
            "AND/OR groups, include scientific names and important synonyms, and avoid "
            "overly narrow location/use constraints when they would suppress recall."
        )
    else:
        instruction = (
            "Create one broad literature/web search keyword query for Semantic Scholar and "
            "Tavily. Use 8 to 14 high-signal English keywords or short phrases. Do not use "
            "Boolean operators, parentheses, field tags, JSON, quotes, markdown, or a full "
            "sentence."
        )
    return [
        {
            "role": "system",
            "content": (
                "You create concise literature search queries. Return only the search query "
                "text. Do not return JSON, quotes, markdown, or explanation."
            ),
        },
        {
            "role": "user",
            "content": f"{instruction}\n\nGoal:\n{goal}",
        },
    ]


def parse_llm_query(response: str) -> str:
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    json_fragment = re.search(r"""["']query["']\s*:\s*["']([^"'}]+)""", text)
    if json_fragment:
        return clean_query_text(json_fragment.group(1))
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return clean_query_text(text)
    if isinstance(payload, dict):
        query = payload.get("query")
        if isinstance(query, str):
            return clean_query_text(query)
        keywords = payload.get("keywords")
        if isinstance(keywords, list):
            return clean_query_text(" ".join(str(item) for item in keywords))
    return ""


def clean_query_text(text: str) -> str:
    text = re.sub(r"^[\s\"']*query[\s\"']*:\s*", "", text.strip(), flags=re.IGNORECASE)
    text = text.strip().strip("\"'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


async def run_probe(args: argparse.Namespace) -> None:
    config = load_config()
    goal_file = resolve_path(args.goal_file)
    db_path = resolve_path(args.db_path)
    goal = goal_file.read_text(encoding="utf-8").strip()
    if not goal:
        raise SystemExit(f"goal file is empty: {goal_file}")

    query_warnings: list[str] = []
    source_queries = await build_source_queries(
        goal,
        config,
        args.query_mode,
        args.llm_query_max_tokens,
        query_warnings,
    )

    print_header("Search Goal Probe")
    print(f"Goal file: {goal_file}")
    print(f"DB path: {db_path}")
    print(f"Domain: {args.domain}")
    print(f"Max results: {args.max_results or config.search.max_results}")
    print(f"Query mode: {args.query_mode}")
    if args.query_mode == "llm" and args.llm_query_max_tokens is not None:
        print(f"LLM query max tokens: {args.llm_query_max_tokens}")
    print(f"Request timeout: {config.runtime.request_timeout_seconds}s")
    print()
    print_search_config(config.search, args.domain)
    print_env_presence(config.search)
    if query_warnings:
        print("Query warnings:")
        for warning in query_warnings:
            print(f"- {warning}")
        print()
    enabled_sources = sources_for_domain(args.domain, config.search)
    print_queries(source_queries, enabled_sources)

    timeout = aiohttp.ClientTimeout(total=config.runtime.request_timeout_seconds)
    async with SQLiteStore(db_path) as store:
        session = await store.get_session(SESSION_ID)
        if session is None:
            await store.create_session(goal, session_id=SESSION_ID)
        elif session.goal != goal:
            await store.update_session_goal(SESSION_ID, goal)

        async with aiohttp.ClientSession(timeout=timeout) as http_session:
            result = await search_sources(
                source_queries,
                domain=args.domain,
                max_results=args.max_results,
                config=config.search,
                store=store,
                session_id=SESSION_ID,
                http_session=http_session,
            )

    print_header("Result")
    print(f"Status: {result.status}")
    print(f"Documents: {len(result.documents)}")
    if result.errors:
        print("Warnings/errors:")
        for error in result.errors:
            print(f"- {error}")
    else:
        print("Warnings/errors: none")
    print_source_counts(result.documents, enabled_sources)
    print_documents(result.documents, enabled_sources)


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def print_header(title: str) -> None:
    print(f"== {title} ==")


def print_search_config(search: Any, domain: str) -> None:
    print("Enabled sources:")
    for source in sources_for_domain(domain, search):
        print(f"- {source}")
    disabled = disabled_sources_for_domain(domain, search)
    if disabled:
        print("Configured off for this domain:")
        for source in disabled:
            print(f"- {source}")
    print()


def sources_for_domain(domain: str, search: Any) -> list[str]:
    sources = domain_sources(domain)
    if search.private_corpus_enabled:
        sources.append("private_corpus")
    return [source for source in sources if source_enabled(source, search)]


def disabled_sources_for_domain(domain: str, search: Any) -> list[str]:
    sources = domain_sources(domain)
    if search.private_corpus_enabled:
        sources.append("private_corpus")
    return [source for source in sources if not source_enabled(source, search)]


def domain_sources(domain: str) -> list[str]:
    if domain == "biomed":
        return ["pubmed", "semantic_scholar", "tavily"]
    if domain in {"preprint", "cs", "math", "physics"}:
        return ["arxiv", "semantic_scholar", "tavily"]
    return ["semantic_scholar", "tavily"]


def source_enabled(source: str, search: Any) -> bool:
    return bool(getattr(search, SOURCE_ENABLED_FIELDS[source]))


async def build_source_queries(
    goal: str,
    config: Any,
    mode: str,
    max_tokens: int | None,
    warnings: list[str],
) -> dict[str, list[str]]:
    keyword_query = await build_probe_query(
        goal,
        config,
        mode,
        max_tokens,
        warnings,
        style="keywords",
    )
    pubmed_query = await build_probe_query(
        goal,
        config,
        mode,
        max_tokens,
        warnings,
        style="pubmed",
    )
    lexical_query = build_literature_query(goal)
    return {
        "pubmed": dedupe_queries([pubmed_query, keyword_query, lexical_query, goal]),
        "semantic_scholar": dedupe_queries([keyword_query, lexical_query, goal]),
        "tavily": dedupe_queries([keyword_query, lexical_query, goal]),
        "arxiv": dedupe_queries([keyword_query, lexical_query, goal]),
        "private_corpus": dedupe_queries([keyword_query, lexical_query, goal]),
    }


def dedupe_queries(queries: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = " ".join(query.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


async def search_sources(
    source_queries: dict[str, list[str]],
    *,
    domain: str,
    max_results: int | None,
    config: Any,
    store: SQLiteStore,
    session_id: str,
    http_session: aiohttp.ClientSession,
) -> ToolResult:
    results: list[ToolResult] = []
    for source in sources_for_domain(domain, config):
        source_config = config_for_single_source(config, source, max_results=max_results)
        source_result = await search_literature_with_fallbacks(
            source_queries[source],
            domain=domain,
            max_results=max_results,
            config=source_config,
            store=store,
            session_id=session_id,
            persist_citations=True,
            http_session=http_session,
        )
        results.append(source_result)
    return merge_results(results)


def config_for_single_source(config: Any, source: str, *, max_results: int | None) -> Any:
    updates = {
        field: candidate == source
        for candidate, field in SOURCE_ENABLED_FIELDS.items()
    }
    if max_results is not None:
        updates[f"{source}_max_results"] = max_results
    return config.model_copy(update=updates)


def merge_results(results: list[ToolResult]) -> ToolResult:
    documents = []
    seen = set()
    for result in results:
        for document in result.documents:
            key = document.citation.dedupe_key()
            if key in seen:
                continue
            seen.add(key)
            documents.append(document)

    citations = [document.citation for document in documents]
    errors: list[str] = []
    for result in results:
        errors.extend(result.errors)

    status = ToolStatus.OK
    if errors and documents:
        status = ToolStatus.DEGRADED
    elif errors and not documents:
        status = ToolStatus.FAILED
    return ToolResult(
        source="literature",
        status=status,
        documents=documents,
        citations=citations,
        errors=errors,
    )


def print_env_presence(search: Any) -> None:
    print("Environment keys:")
    for label, env_name in [
        ("Tavily", search.tavily_api_key_env),
        ("NCBI email", search.ncbi_email_env),
        ("NCBI API key", search.ncbi_api_key_env),
        ("Semantic Scholar", search.semantic_scholar_api_key_env),
    ]:
        state = "present" if os.getenv(env_name) else "missing"
        print(f"- {label}: {env_name} {state}")
    print()


def print_queries(source_queries: dict[str, list[str]], enabled_sources: list[str]) -> None:
    print("Fallback queries by source:")
    for source in enabled_sources:
        queries = source_queries.get(source)
        if not queries:
            continue
        print(f"[{source}]")
        for index, query in enumerate(queries, start=1):
            print(f"{index}. {query}")
    print()


def print_source_counts(documents: list[Any], expected_sources: list[str]) -> None:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for document in documents:
        grouped[document.source].append(document)

    print("Source counts:")
    for source in expected_sources:
        print(f"- {source}: {len(grouped[source])}")


def print_documents(documents: list[Any], expected_sources: list[str]) -> None:
    if not documents:
        print()
        print("No documents returned.")
        return

    grouped: dict[str, list[Any]] = defaultdict(list)
    for document in documents:
        grouped[document.source].append(document)

    print()
    print_header("Documents")
    for source in expected_sources:
        print(f"\n[{source}]")
        source_documents = grouped[source]
        if not source_documents:
            print("No documents returned.")
            continue
        for index, document in enumerate(source_documents, start=1):
            print_document(index, document)


def print_document(index: int, document: Any) -> None:
    citation = document.citation
    identifiers = [
        f"DOI {citation.doi}" if citation.doi else None,
        f"PMID {citation.pmid}" if citation.pmid else None,
        f"arXiv {citation.arxiv_id}" if citation.arxiv_id else None,
        (
            f"Semantic Scholar {citation.semantic_scholar_id}"
            if citation.semantic_scholar_id
            else None
        ),
        citation.url,
    ]
    identifier_text = "; ".join(value for value in identifiers if value) or "none"
    year = document.year if document.year is not None else "n/a"
    score = f"{document.score:.3g}" if document.score is not None else "n/a"
    snippet = truncate(document.abstract_or_snippet or "", 320) or "No snippet."
    print(f"{index}. {document.title}")
    print(f"   source: {document.source} | year: {year} | score: {score}")
    print(f"   ids: {identifier_text}")
    print(f"   snippet: {snippet}")


def truncate(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


if __name__ == "__main__":
    main()
