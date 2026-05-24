from __future__ import annotations

import pytest

from co_scientist.tools import web_search
from co_scientist.tools.models import ToolStatus


class FakeTavilyClient:
    def search(self, **kwargs):
        return {
            "results": [
                {
                    "title": "Relevant web result",
                    "url": "https://example.org/10.1000/example",
                    "content": "Mentions PMID: 123 and arXiv:2412.12345",
                    "score": 0.9,
                }
            ]
        }


class BuggyTavilyClient:
    def search(self, **kwargs):
        raise TypeError("programming bug")


def test_parse_tavily_payload_extracts_identifiers() -> None:
    documents = web_search.parse_tavily_payload(FakeTavilyClient().search())

    assert documents[0].citation.doi == "10.1000/example"
    assert documents[0].citation.pmid == "123"
    assert documents[0].citation.arxiv_id == "2412.12345"


@pytest.mark.asyncio
async def test_tavily_missing_key_fails_without_importing_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    result = await web_search.search("AML")

    assert result.status == ToolStatus.FAILED
    assert "TAVILY_API_KEY" in result.errors[0]


@pytest.mark.asyncio
async def test_tavily_search_uses_injected_client() -> None:
    result = await web_search.search("AML", client=FakeTavilyClient())

    assert result.status == ToolStatus.OK
    assert result.documents[0].title == "Relevant web result"


@pytest.mark.asyncio
async def test_tavily_search_does_not_swallow_code_bugs() -> None:
    with pytest.raises(TypeError, match="programming bug"):
        await web_search.search("AML", client=BuggyTavilyClient())
