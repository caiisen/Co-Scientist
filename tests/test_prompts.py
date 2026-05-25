from __future__ import annotations

from pathlib import Path

import pytest

from co_scientist.utils.prompts import (
    PromptTemplateError,
    PromptTemplateStore,
    build_prompt_messages,
)


def test_prompt_store_loads_and_renders_template(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_text(
        "Goal: {goal}\nCriteria: {preferences}",
        encoding="utf-8",
    )
    store = PromptTemplateStore(tmp_path)

    rendered = store.render("sample", goal="find targets", preferences="novel and testable")

    assert rendered == "Goal: find targets\nCriteria: novel and testable"
    assert store.required_variables("sample") == {"goal", "preferences"}


def test_prompt_store_rejects_missing_variables(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_text("Goal: {goal}", encoding="utf-8")
    store = PromptTemplateStore(tmp_path)

    with pytest.raises(PromptTemplateError, match="goal"):
        store.render("sample")


def test_prompt_store_rejects_unknown_or_unsafe_template(tmp_path: Path) -> None:
    store = PromptTemplateStore(tmp_path)

    with pytest.raises(PromptTemplateError, match="unknown prompt template"):
        store.load("missing")

    with pytest.raises(PromptTemplateError, match="invalid prompt template name"):
        store.load("../missing")


def test_build_prompt_messages_merges_feedback_into_system_message() -> None:
    messages = build_prompt_messages(
        system_prompt="agent instructions",
        user_prompt="do the task",
        feedback="check BBB permeability",
    )

    assert messages == [
        {
            "role": "system",
            "content": (
                "agent instructions\n\n"
                "## Meta-review feedback for this run:\ncheck BBB permeability"
            ),
        },
        {"role": "user", "content": "do the task"},
    ]


def test_build_prompt_messages_omits_blank_feedback() -> None:
    messages = build_prompt_messages(
        system_prompt="agent instructions",
        user_prompt="do the task",
        feedback=" ",
    )

    assert messages == [
        {"role": "system", "content": "agent instructions"},
        {"role": "user", "content": "do the task"},
    ]


def test_note9_prompt_templates_keep_key_constraints() -> None:
    store = PromptTemplateStore()

    generation_debate = store.load("generation_scientific_debate")
    assert "typically 3-5 conversational turns" in generation_debate
    assert "maximum of 10 turns" in generation_debate
    assert "HYPOTHESIS" in generation_debate

    observation = store.load("reflection_observation")
    assert "would we see this observation if the hypothesis was true:" in observation
    assert "hypothesis: <already explained" in observation

    ranking_debate = store.load("ranking_debate")
    assert "maximum of 10" in ranking_debate
    assert "better idea: " in ranking_debate

    evolution = store.load("evolution_feasibility")
    assert "CORE CONTRIBUTION" in evolution

    metareview = store.load("metareview")
    assert "Refrain from evaluating individual proposals or reviews" in metareview


def test_phase4_full_review_prompt_requires_parseable_final_score() -> None:
    template = PromptTemplateStore().load("reflection_full_review")

    assert "Overall score: <number from 0 to 10>/10" in template
    assert "[1] or [2]" in template
