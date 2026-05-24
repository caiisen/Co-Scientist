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


def test_build_prompt_messages_injects_feedback_as_system_message() -> None:
    messages = build_prompt_messages(
        system_prompt="agent instructions",
        user_prompt="do the task",
        feedback="check BBB permeability",
    )

    assert messages == [
        {"role": "system", "content": "agent instructions"},
        {
            "role": "system",
            "content": "Meta-review feedback for this run:\ncheck BBB permeability",
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

