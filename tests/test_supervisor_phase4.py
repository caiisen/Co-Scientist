from __future__ import annotations

from pathlib import Path

import pytest

from co_scientist.agents.base import Agent
from co_scientist.agents.generation import GenerationAgent
from co_scientist.agents.reflection import ReflectionAgent
from co_scientist.agents.results import AgentResult, AgentResultKind
from co_scientist.config import AppConfig, LLMConfig, ProviderConfig, RuntimeConfig, SearchConfig
from co_scientist.llm.client import LLMRouter
from co_scientist.memory import ResearchPlan, SQLiteStore, Task
from co_scientist.supervisor import Supervisor
from co_scientist.supervisor.task_queue import TaskQueue
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

    def embedding_client_for(self, agent=None):
        return self.client


class BucketedGenerationAgent(Agent):
    name = "generation"

    async def execute(self, task: Task, ctx) -> AgentResult:
        return AgentResult(
            kind=AgentResultKind.HYPOTHESIS_CREATED,
            payload={
                "hypotheses": [
                    {
                        "content": "H1",
                        "summary": "H1",
                        "source_strategy": "test",
                        "citations": [
                            Citation(source="pubmed", title="Paper A", pmid="1").model_dump()
                        ],
                    },
                    {
                        "content": "H2",
                        "summary": "H2",
                        "source_strategy": "test",
                        "citations": [
                            Citation(source="pubmed", title="Paper B", pmid="2").model_dump()
                        ],
                    },
                ]
            },
        )


def make_config(*, max_ideas: int = 5) -> AppConfig:
    return AppConfig(
        runtime=RuntimeConfig(
            max_ideas=max_ideas,
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
        assert "# Co-Scientist Report" in markdown
        assert "H1" in markdown
        assert "Evidence references: [1] -> [R1]" in markdown
        assert "## References" in markdown
        assert "[R1] Paper. 2026. PMID: 1; https://pubmed.ncbi.nlm.nih.gov/1/." in markdown


@pytest.mark.asyncio
async def test_supervisor_links_hypothesis_citations_by_payload_bucket(tmp_path: Path) -> None:
    config = make_config()
    async with SQLiteStore(tmp_path / "supervisor_bucketed.sqlite") as store:
        session = await store.create_session("goal")
        await store.save_research_plan(ResearchPlan(session_id=session.id, goal=session.goal))
        supervisor = Supervisor(
            store=store,
            config=config,
            llm_router=StaticRouter(SequenceClient([])),
            agents={"generation": BucketedGenerationAgent()},
        )
        queue = TaskQueue(store, session.id)
        task = await queue.enqueue(
            Task(session_id=session.id, agent="generation", action="create_initial_hypotheses")
        )
        ctx = supervisor._context(session.id, http_session=None)

        result = await BucketedGenerationAgent().execute(task, ctx)
        await supervisor._handle_result(queue, task, result)
        hypotheses = await store.list_session_hypotheses(session.id)

        first_links = await store.citation_links_for_artifact(
            session_id=session.id,
            artifact_type="hypothesis",
            artifact_id=hypotheses[0].id or 0,
        )
        second_links = await store.citation_links_for_artifact(
            session_id=session.id,
            artifact_type="hypothesis",
            artifact_id=hypotheses[1].id or 0,
        )

    assert [link["pmid"] for link in first_links] == ["1"]
    assert [link["pmid"] for link in second_links] == ["2"]


@pytest.mark.asyncio
async def test_supervisor_does_not_store_more_than_max_ideas(tmp_path: Path) -> None:
    config = make_config(max_ideas=1)
    async with SQLiteStore(tmp_path / "supervisor_max_ideas.sqlite") as store:
        session = await store.create_session("goal")
        await store.save_research_plan(ResearchPlan(session_id=session.id, goal=session.goal))
        supervisor = Supervisor(
            store=store,
            config=config,
            llm_router=StaticRouter(SequenceClient([])),
            agents={"generation": BucketedGenerationAgent()},
        )
        queue = TaskQueue(store, session.id)
        task = await queue.enqueue(
            Task(session_id=session.id, agent="generation", action="create_initial_hypotheses")
        )
        ctx = supervisor._context(session.id, http_session=None)

        result = await BucketedGenerationAgent().execute(task, ctx)
        await supervisor._handle_result(queue, task, result, ctx)
        hypotheses = await store.list_session_hypotheses(session.id)

    assert [hypothesis.summary for hypothesis in hypotheses] == ["H1"]
