from __future__ import annotations

from co_scientist.tools.query import build_literature_query


def test_build_literature_query_extracts_keywords() -> None:
    query = build_literature_query(
        "Propose a hypothesis of a transcriptional regulatory network that "
        "simultaneously regulates cannabis glandular trichome development and "
        "cannabinoid synthesis pathways"
    )

    assert query == "transcriptional regulatory Cannabis glandular trichome cannabinoid synthesis"
