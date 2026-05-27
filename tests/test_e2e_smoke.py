from __future__ import annotations

import json
from pathlib import Path

import pytest

from co_scientist.config import (
    AppConfig,
    LLMConfig,
    ObservabilityConfig,
    ProviderConfig,
    RuntimeConfig,
    SearchConfig,
)
from co_scientist.llm.client import LLMCallMetadata, LLMRouter
from co_scientist.memory import SQLiteStore, TaskStatus
from co_scientist.supervisor import Supervisor


class TemplateStubClient:
    def __init__(self) -> None:
        self.counter = 0
        self.last_call: LLMCallMetadata | None = None

    async def chat(self, messages, **kwargs):
        self.counter += 1
        prompt = messages[-1]["content"]
        self.last_call = LLMCallMetadata(model="stub-chat", latency_seconds=0.001)
        if "Return only JSON with these keys" in prompt:
            return json.dumps(
                {
                    "preferences": ["plausibility", "novelty", "testability", "safety"],
                    "attributes": ["biomedical mechanism"],
                    "constraints": ["use literature-grounded reasoning"],
                    "idea_attributes": ["specific hypothesis", "validation plan"],
                }
            )
        if "better idea:" in prompt:
            return "Hypothesis 1 is more testable and better grounded.\nbetter idea: 1"
        if "final research overview" in prompt or "Top ranked hypotheses" in prompt:
            return "Overview: prioritize the top-ranked mechanisms and validate them in assays."
        if "meta-analysis" in prompt or "system-wide feedback" in prompt:
            return "System feedback: prefer concrete mechanisms and simple validation."
        if "End with exactly one final line in this format:" in prompt:
            return "Review: plausible, novel enough, testable, and safe.\nOverall score: 8/10"
        return (
            "Discussion converges on a concrete proposal.\n\n"
            f"HYPOTHESIS:\nCandidate mechanism {self.counter}: test a literature-grounded "
            "selection-pressure explanation with measurable intermediate phenotypes."
        )

    async def embed(self, texts, **kwargs):
        self.last_call = LLMCallMetadata(model="stub-embed", latency_seconds=0.001)
        return [[float((len(text) % 7) + 1), 1.0, 0.5] for text in texts]


class StaticRouter(LLMRouter):
    def __init__(self) -> None:
        self.client = TemplateStubClient()

    def client_for(self, agent=None):
        return self.client

    def embedding_client_for(self, agent=None):
        return self.client


def make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        runtime=RuntimeConfig(
            max_ideas=5,
            max_matches_per_idea=1,
            worker_concurrency=1,
            request_timeout_seconds=30,
            elo_stagnation_threshold=5.0,
            elo_stagnation_window=4,
        ),
        search=SearchConfig(
            max_results=1,
            tavily_enabled=False,
            pubmed_enabled=False,
            semantic_scholar_enabled=False,
            arxiv_enabled=False,
        ),
        llm=LLMConfig(
            default_provider="test",
            providers={
                "test": ProviderConfig(
                    chat_model="test-chat",
                    embedding_model="test-embed",
                )
            },
        ),
        observability=ObservabilityConfig(
            metrics_enabled=True,
            runs_dir=str(tmp_path / "runs"),
        ),
    )


@pytest.mark.asyncio
async def test_e2e_smoke_runs_supervisor_to_overview_and_export(tmp_path: Path) -> None:
    db_path = tmp_path / "smoke.sqlite"
    config = make_config(tmp_path)
    async with SQLiteStore(db_path) as store:
        supervisor = Supervisor(
            store=store,
            config=config,
            llm_router=StaticRouter(),
        )

        session_id = await supervisor.start("Explain cannabis latitude selection pressure.")
        report = await supervisor.export_markdown(session_id)
        task_counts = await store.tasks_by_status(session_id)
        hypothesis_count = await store.count_hypotheses(session_id)
        review_count = await store.count_reviews(session_id)
        match_count = await store.count_matches(session_id)
        overview = await store.latest_overview(session_id)

    metrics_path = tmp_path / "runs" / session_id / "metrics.jsonl"
    metrics = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert hypothesis_count >= 5
    assert review_count >= 5
    assert match_count >= 5
    assert overview is not None
    assert "# Co-Scientist Report" in report
    assert "Final Research Overview" in report
    assert task_counts.get(TaskStatus.PENDING, 0) == 0
    assert task_counts.get(TaskStatus.RUNNING, 0) == 0
    assert sum(event["event"] == "session.start" for event in metrics) == 1
    assert sum(event["event"] == "session.done" for event in metrics) == 1
    assert any(event["event"] == "task.start" for event in metrics)
    assert any(event["event"] == "task.done" for event in metrics)
    assert any(event["event"] == "llm.chat" for event in metrics)
    assert any(
        event["event"] == "task.done" and event.get("result_kind") == "overview_generated"
        for event in metrics
    )
