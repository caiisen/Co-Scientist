from __future__ import annotations

from co_scientist.tools.models import Citation, SearchDocument, ToolResult, ToolStatus


def test_citation_dedupe_key_prefers_stable_identifiers() -> None:
    citation = Citation(
        source="pubmed",
        title="A paper",
        url="https://example.org/paper",
        doi="10.1000/ABC",
        pmid="123",
    )

    assert citation.dedupe_key() == "doi:10.1000/abc"


def test_tool_result_formats_evidence_pack_and_serializes() -> None:
    citation = Citation(source="pubmed", title="AML paper", pmid="123", year=2025)
    document = SearchDocument(
        source="pubmed",
        title="AML paper",
        abstract_or_snippet=" ".join(["evidence"] * 100),
        venue="Nature",
        year=2025,
        citation=citation,
    )
    result = ToolResult.from_documents(source="pubmed", documents=[document])

    dumped = result.model_dump(mode="json")
    restored = ToolResult.model_validate(dumped)

    assert restored.status == ToolStatus.OK
    assert "PMID: 123" in restored.format_evidence_pack(max_chars_per_item=40)
    assert "..." in restored.format_evidence_pack(max_chars_per_item=40)
    assert "..." not in restored.format_evidence_pack()


def test_tool_result_status_from_errors() -> None:
    failed = ToolResult.from_documents(source="pubmed", documents=[], errors=["rate limited"])
    degraded = ToolResult.from_documents(
        source="pubmed",
        documents=[
            SearchDocument(
                source="pubmed",
                title="One paper",
                citation=Citation(source="pubmed", title="One paper"),
            )
        ],
        errors=["partial error"],
    )

    assert failed.status == ToolStatus.FAILED
    assert degraded.status == ToolStatus.DEGRADED
