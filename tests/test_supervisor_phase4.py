from __future__ import annotations

from pathlib import Path

import pytest

from co_scientist.agents.generation import GenerationAgent
from co_scientist.agents.reflection import ReflectionAgent
from co_scientist.config import AppConfig, LLMConfig, ProviderConfig, RuntimeConfig, SearchConfig
from co_scientist.llm.client import LLMRouter
from co_scientist.memory import SQLiteStore
from co_scientist.supervisor import Supervisor
from co_scientist.tools.models import Citation, SearchDocument, ToolResult


class SequenceClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    async def chat(self, messages, **kwargs):
        if not self.responses:
            raise AssertionError("unexpected chat call")
        return self.responses.pop(0)


class StaticRouter(LLMRouter):
    def __init__(self, client: SequenceClient) -> None:
        self.client = client

    def client_for(self, agent=None):
        return self.client


def make_config() -> AppConfig:
    return AppConfig(
        runtime=RuntimeConfig(
            max_ideas=5,
            max_matches_per_idea=1,
            worker_concurrency=2,
            request_timeout_seconds=30,
        ),
        search=SearchConfig(
            max_results=1,
            semantic_scholar_enabled=False,
            tavily_enabled=False,
            arxiv_enabled=False,
        ),
        llm=LLMConfig(
            default_provider="test",
            providers={"test": ProviderConfig(chat_model="test-chat")},
        ),
    )


async def fake_literature_search(*args, **kwargs) -> ToolResult:
    return ToolResult.from_documents(
        source="literature",
        documents=[
            SearchDocument(
                source="pubmed",
                title="Paper",
                year=2026,
                citation=Citation(
                    source="pubmed",
                    title="Paper",
                    pmid="1",
                    year=2026,
                    url="https://pubmed.ncbi.nlm.nih.gov/1/",
                ),
            )
        ],
    )


@pytest.mark.asyncio
async def test_supervisor_runs_phase4_and_exports(tmp_path: Path) -> None:
    client = SequenceClient(
        [
            '{"preferences":["novel"],"attributes":[],"constraints":[],"idea_attributes":[]}',
            "HYPOTHESIS\nH1",
            "turn one",
            "turn two",
            "HYPOTHESIS\nH2",
            "HYPOTHESIS\nH3",
            "HYPOTHESIS\nH4",
            "HYPOTHESIS\nH5",
            "Review 1\nOverall score: 6/10",
            "Review 2\nOverall score: 7/10",
            "Review 3\nOverall score: 8/10",
            "Review 4\nOverall score: 9/10",
            "Review 5\nOverall score: 5/10",
        ]
    )
    config = make_config()
    async with SQLiteStore(tmp_path / "supervisor.sqlite") as store:
        supervisor = Supervisor(
            store=store,
            config=config,
            llm_router=StaticRouter(client),
            agents={
                "generation": GenerationAgent(literature_search=fake_literature_search),
                "reflection": ReflectionAgent(literature_search=fake_literature_search),
            },
        )

        session_id = await supervisor.start("goal")
        markdown = await supervisor.export_markdown(session_id)

        assert await store.count_hypotheses(session_id) == 5
        assert await store.count_reviews(session_id) == 5
        assert len(await store.list_citations(session_id)) == 1
        assert "# Co-Scientist Phase 4 Report" in markdown
        assert "H1" in markdown
        assert "Evidence references: [1] -> [R1]" in markdown
        assert "## References" in markdown
        assert "[R1] Paper. 2026. PMID: 1; https://pubmed.ncbi.nlm.nih.gov/1/." in markdown
