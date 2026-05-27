from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from co_scientist.search.platforms.arxiv import ArXivPlatform
from co_scientist.search.platforms.openalex import OpenAlexPlatform
from co_scientist.search.platforms.semantic_scholar import SemanticScholarPlatform
from co_scientist.search.platforms.tavily_platform import TavilyPlatform

# ── arXiv ─────────────────────────────────────────────────────────────────────

_ARXIV_ATOM = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <title>Test Paper Title</title>
    <summary>This is the abstract text.</summary>
    <published>2023-01-02T00:00:00Z</published>
  </entry>
</feed>
"""


@pytest.mark.asyncio
async def test_arxiv_parse():
    platform = ArXivPlatform()

    mock_resp = AsyncMock()
    mock_resp.text = AsyncMock(return_value=_ARXIV_ATOM)
    mock_resp.raise_for_status = MagicMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("co_scientist.search.platforms.arxiv.aiohttp.ClientSession", return_value=mock_session):
        results = await platform.search("ti:transformer AND cat:cs.AI", 5)

    assert len(results) == 1
    r = results[0]
    assert r.title == "Test Paper Title"
    assert r.abstract == "This is the abstract text."
    assert r.year == 2023
    assert r.source == "arxiv"
    assert r.url == "http://arxiv.org/abs/2301.00001v1"


# ── Semantic Scholar ───────────────────────────────────────────────────────────

_S2_RESPONSE = {
    "data": [
        {
            "title": "CRISPR Study",
            "abstract": "We study CRISPR.",
            "year": 2022,
            "citationCount": 150,
            "externalIds": {"DOI": "10.1000/test"},
            "url": "https://semanticscholar.org/paper/abc",
        }
    ]
}


@pytest.mark.asyncio
async def test_semantic_scholar_parse():
    platform = SemanticScholarPlatform()

    mock_resp = AsyncMock()
    mock_resp.json = AsyncMock(return_value=_S2_RESPONSE)
    mock_resp.raise_for_status = MagicMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "co_scientist.search.platforms.semantic_scholar.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        results = await platform.search("CRISPR therapy", 5)

    assert len(results) == 1
    r = results[0]
    assert r.title == "CRISPR Study"
    assert r.doi == "10.1000/test"
    assert r.year == 2022
    assert r.score == 150.0


# ── OpenAlex ──────────────────────────────────────────────────────────────────

_OPENALEX_RESPONSE = {
    "results": [
        {
            "title": "OpenAlex Paper",
            "abstract_inverted_index": {"Hello": [0], "world": [1], "CRISPR": [2]},
            "publication_year": 2023,
            "cited_by_count": 42,
            "doi": "https://doi.org/10.1000/oa",
            "id": "https://openalex.org/W123",
        }
    ]
}


def test_openalex_decode_abstract():
    result = OpenAlexPlatform._decode_abstract({"world": [1], "Hello": [0], "CRISPR": [2]})
    assert result == "Hello world CRISPR"


def test_openalex_decode_abstract_empty():
    assert OpenAlexPlatform._decode_abstract(None) == ""
    assert OpenAlexPlatform._decode_abstract({}) == ""


@pytest.mark.asyncio
async def test_openalex_parse():
    platform = OpenAlexPlatform()

    mock_resp = AsyncMock()
    mock_resp.json = AsyncMock(return_value=_OPENALEX_RESPONSE)
    mock_resp.raise_for_status = MagicMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("co_scientist.search.platforms.openalex.aiohttp.ClientSession", return_value=mock_session):
        results = await platform.search("CRISPR therapy", 5)

    assert len(results) == 1
    r = results[0]
    assert r.doi == "10.1000/oa"
    assert r.year == 2023
    assert r.abstract == "Hello world CRISPR"


# ── Tavily ────────────────────────────────────────────────────────────────────

_TAVILY_RESPONSE = {
    "results": [
        {
            "title": "News Article",
            "content": "Some content here.",
            "url": "https://example.com",
            "score": 0.95,
        }
    ]
}


@pytest.mark.asyncio
async def test_tavily_parse():
    platform = TavilyPlatform(api_key="fake-key")

    mock_resp = AsyncMock()
    mock_resp.json = AsyncMock(return_value=_TAVILY_RESPONSE)
    mock_resp.raise_for_status = MagicMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("co_scientist.search.platforms.tavily_platform.aiohttp.ClientSession", return_value=mock_session):
        results = await platform.search("CRISPR latest 2024", 5)

    assert len(results) == 1
    r = results[0]
    assert r.title == "News Article"
    assert r.score == 0.95
    assert r.doi is None


# ── PubMed ────────────────────────────────────────────────────────────────────

_PUBMED_XML = b"""\
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <Article>
        <ArticleTitle>PubMed Test Paper</ArticleTitle>
        <Abstract>
          <AbstractText>Abstract text here.</AbstractText>
        </Abstract>
        <Journal>
          <JournalIssue>
            <PubDate><Year>2021</Year></PubDate>
          </JournalIssue>
        </Journal>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">12345678</ArticleId>
        <ArticleId IdType="doi">10.1000/pm</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


def test_pubmed_parse_xml():
    from co_scientist.search.platforms.pubmed import PubMedPlatform

    platform = PubMedPlatform()
    results = platform._parse_xml(_PUBMED_XML)

    assert len(results) == 1
    r = results[0]
    assert r.title == "PubMed Test Paper"
    assert r.abstract == "Abstract text here."
    assert r.year == 2021
    assert r.doi == "10.1000/pm"
    assert r.url == "https://pubmed.ncbi.nlm.nih.gov/12345678/"
