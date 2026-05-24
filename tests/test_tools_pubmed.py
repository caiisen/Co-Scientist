from __future__ import annotations

import pytest

from co_scientist.tools import pubmed
from co_scientist.tools.models import ToolStatus


class FakeHandle:
    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = False

    def close(self) -> None:
        self.closed = True


class EntrezString(str):
    def __new__(cls, value: str, attributes: dict | None = None):
        item = str.__new__(cls, value)
        item.attributes = attributes or {}
        return item


class FakeEntrez:
    email = None
    api_key = None

    @staticmethod
    def esearch(**kwargs):
        return FakeHandle("search")

    @staticmethod
    def efetch(**kwargs):
        return FakeHandle("fetch")

    @staticmethod
    def read(handle):
        if handle.name == "search":
            return {"IdList": ["123"]}
        return {
            "PubmedArticle": [
                {
                    "MedlineCitation": {
                        "PMID": "123",
                        "Article": {
                            "ArticleTitle": "AML drug repurposing",
                            "Abstract": {"AbstractText": ["Abstract part one.", "Part two."]},
                            "Journal": {
                                "Title": "Journal",
                                "JournalIssue": {"PubDate": {"Year": "2025"}},
                            },
                            "ELocationID": [
                                EntrezString("10.1000/example", {"EIdType": "doi"})
                            ],
                            "AuthorList": [{"LastName": "Doe", "Initials": "J"}],
                        },
                    }
                }
            ]
        }


class BuggyEntrez(FakeEntrez):
    @staticmethod
    def read(handle):
        raise TypeError("programming bug")


def test_parse_pubmed_payload() -> None:
    documents = pubmed.parse_pubmed_payload(FakeEntrez.read(FakeHandle("fetch")))

    assert documents[0].citation.pmid == "123"
    assert documents[0].citation.doi == "10.1000/example"
    assert documents[0].year == 2025
    assert documents[0].authors == ["Doe J"]


@pytest.mark.asyncio
async def test_pubmed_search_uses_injected_entrez() -> None:
    result = await pubmed.search("AML", max_results=1, email="test@example.org", entrez=FakeEntrez)

    assert result.status == ToolStatus.OK
    assert result.documents[0].title == "AML drug repurposing"
    assert FakeEntrez.email == "test@example.org"


@pytest.mark.asyncio
async def test_pubmed_search_does_not_swallow_code_bugs() -> None:
    with pytest.raises(TypeError, match="programming bug"):
        await pubmed.search("AML", max_results=1, entrez=BuggyEntrez)
