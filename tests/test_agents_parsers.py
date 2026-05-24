from __future__ import annotations

from co_scientist.agents.parsers import (
    parse_better_idea,
    parse_hypothesis_block,
    parse_observation_verdict,
)


def test_parse_better_idea_accepts_idea_or_hypothesis() -> None:
    assert parse_better_idea("Rationale...\nbetter idea: 1").value == 1
    assert parse_better_idea("Conclusion: better hypothesis: 2").value == 2


def test_parse_better_idea_reports_failure() -> None:
    result = parse_better_idea("both are good")

    assert not result.ok
    assert result.error is not None


def test_parse_hypothesis_block_extracts_final_text() -> None:
    result = parse_hypothesis_block(
        "Discussion first.\nHYPOTHESIS\nNPC phosphorylation disrupts transport."
    )

    assert result.ok
    assert result.value == "NPC phosphorylation disrupts transport."


def test_parse_hypothesis_block_reports_failure() -> None:
    result = parse_hypothesis_block("No final block")

    assert not result.ok
    assert result.error is not None


def test_parse_observation_verdict_uses_last_verdict() -> None:
    text = """
    Intermediate note: hypothesis: neutral
    Final conclusion: hypothesis: missing piece
    """

    result = parse_observation_verdict(text)

    assert result.ok
    assert result.value == "missing piece"


def test_parse_observation_verdict_reports_failure() -> None:
    result = parse_observation_verdict("No verdict here")

    assert not result.ok
    assert result.error is not None
