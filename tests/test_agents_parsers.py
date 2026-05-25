from __future__ import annotations

from co_scientist.agents.parsers import (
    parse_better_idea,
    parse_hypothesis_block,
    parse_json_object,
    parse_observation_verdict,
    parse_review_score,
    summarize_hypothesis,
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


def test_parse_hypothesis_block_uses_last_marker_and_strips_bold() -> None:
    result = parse_hypothesis_block(
        "The instructions say **HYPOTHESIS** should appear later.\n"
        "Intermediate text.\n"
        "**HYPOTHESIS**\n\n"
        "Final concise hypothesis."
    )

    assert result.ok
    assert result.value == "Final concise hypothesis."


def test_parse_hypothesis_block_reports_failure() -> None:
    result = parse_hypothesis_block("No final block")

    assert not result.ok
    assert result.error is not None


def test_parse_json_object_extracts_fenced_json() -> None:
    result = parse_json_object('```json\n{"preferences": ["novel"]}\n```')

    assert result.ok
    assert result.value == {"preferences": ["novel"]}


def test_parse_review_score_uses_last_score() -> None:
    result = parse_review_score("Initial score: 4/10\nOverall score: 8.5 out of 10")

    assert result.ok
    assert result.value == 8.5


def test_parse_review_score_accepts_fraction_and_chinese_labels() -> None:
    assert parse_review_score("Final assessment: 8/10").value == 8
    assert parse_review_score("综合评分：7.5 分").value == 7.5


def test_summarize_hypothesis_uses_first_content_line() -> None:
    assert summarize_hypothesis("\n# Hypothesis: Target NPC transport\nMore detail") == (
        "Target NPC transport"
    )


def test_summarize_hypothesis_strips_wrapping_markdown() -> None:
    assert summarize_hypothesis("**Hypothesis:** **Target NPC transport**") == (
        "Target NPC transport"
    )
    assert summarize_hypothesis("**HYPOTHESIS**\n\n**Target NPC transport**") == (
        "Target NPC transport"
    )


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
