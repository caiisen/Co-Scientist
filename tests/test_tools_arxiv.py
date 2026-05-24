from __future__ import annotations

import pytest

from co_scientist.tools import arxiv
from co_scientist.tools.models import ToolStatus

ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2412.12345v1</id>
    <published>2024-12-18T00:00:00Z</published>
    <title> A useful preprint </title>
    <summary> This is the abstract. </summary>
    <author><name>Ada Lovelace</name></author>
  </entry>
</feed>
"""


def test_parse_arxiv_feed() -> None:
    documents = arxiv.parse_arxiv_feed(ARXIV_XML)

    assert len(documents) == 1
    assert documents[0].title == "A useful preprint"
    assert documents[0].citation.arxiv_id == "2412.12345v1"
    assert documents[0].year == 2024


@pytest.mark.asyncio
async def test_arxiv_search_uses_injected_fetcher() -> None:
    async def fetch_text(url: str, params: dict, timeout_seconds: int) -> str:
        assert params["max_results"] == 1
        return ARXIV_XML

    result = await arxiv.search("test", max_results=1, fetch_text=fetch_text)

    assert result.status == ToolStatus.OK
    assert result.documents[0].citation.arxiv_id == "2412.12345v1"


@pytest.mark.asyncio
async def test_arxiv_search_does_not_swallow_code_bugs() -> None:
    async def fetch_text(url: str, params: dict, timeout_seconds: int) -> str:
        raise TypeError("programming bug")

    with pytest.raises(TypeError, match="programming bug"):
        await arxiv.search("test", fetch_text=fetch_text)
