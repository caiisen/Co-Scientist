from __future__ import annotations

import pytest

from co_scientist.tools import semantic_scholar
from co_scientist.tools.models import ToolStatus

PAYLOAD = {
    "data": [
        {
            "paperId": "paper-1",
            "title": "Drug repurposing in AML",
            "abstract": "A structured abstract.",
            "authors": [{"name": "A Author"}],
            "venue": "Journal",
            "year": 2025,
            "url": "https://semanticscholar.org/paper-1",
            "externalIds": {"DOI": "10.1000/example", "PubMed": "123"},
            "citationCount": 7,
        }
    ]
}


def test_parse_semantic_scholar_payload() -> None:
    documents = semantic_scholar.parse_search_payload(PAYLOAD)

    assert len(documents) == 1
    assert documents[0].citation.semantic_scholar_id == "paper-1"
    assert documents[0].citation.doi == "10.1000/example"
    assert documents[0].authors == ["A Author"]


@pytest.mark.asyncio
async def test_semantic_scholar_search_uses_injected_fetcher() -> None:
    async def fetch_json(url: str, params: dict, headers: dict, timeout_seconds: int) -> dict:
        assert params["query"] == "AML"
        assert params["limit"] == 1
        return PAYLOAD

    result = await semantic_scholar.search("AML", max_results=1, fetch_json=fetch_json)

    assert result.status == ToolStatus.OK
    assert result.documents[0].citation.pmid == "123"


@pytest.mark.asyncio
async def test_semantic_scholar_search_does_not_swallow_code_bugs() -> None:
    async def fetch_json(url: str, params: dict, headers: dict, timeout_seconds: int) -> dict:
        raise TypeError("programming bug")

    with pytest.raises(TypeError, match="programming bug"):
        await semantic_scholar.search("AML", fetch_json=fetch_json)
