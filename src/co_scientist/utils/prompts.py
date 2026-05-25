from __future__ import annotations

import string
from pathlib import Path
from typing import Any

from co_scientist.config import PROJECT_ROOT
from co_scientist.llm.client import ChatMessage

PROMPTS_PATH = PROJECT_ROOT / "config" / "prompts"


class PromptTemplateError(ValueError):
    """Raised when a prompt template cannot be loaded or rendered."""


class StrictFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        raise PromptTemplateError(f"missing prompt variable: {key}")


class PromptTemplateStore:
    def __init__(self, prompt_dir: str | Path = PROMPTS_PATH) -> None:
        self.prompt_dir = Path(prompt_dir)
        self._cache: dict[str, str] = {}

    def load(self, template_name: str) -> str:
        normalized = normalize_template_name(template_name)
        if normalized not in self._cache:
            path = self.prompt_dir / f"{normalized}.txt"
            if not path.is_file():
                raise PromptTemplateError(f"unknown prompt template: {normalized}")
            self._cache[normalized] = path.read_text(encoding="utf-8").strip()
        return self._cache[normalized]

    def required_variables(self, template_name: str) -> set[str]:
        template = self.load(template_name)
        variables: set[str] = set()
        formatter = string.Formatter()
        for _, field_name, _, _ in formatter.parse(template):
            if field_name:
                variables.add(field_name)
        return variables

    def render(self, template_name: str, **variables: Any) -> str:
        template = self.load(template_name)
        required = self.required_variables(template_name)
        missing = sorted(required - variables.keys())
        if missing:
            raise PromptTemplateError(
                f"missing prompt variables for {normalize_template_name(template_name)}: "
                + ", ".join(missing)
            )
        try:
            return template.format_map(StrictFormatDict(variables))
        except KeyError as exc:
            raise PromptTemplateError(f"missing prompt variable: {exc.args[0]}") from exc


def normalize_template_name(template_name: str) -> str:
    normalized = template_name.strip()
    if normalized.endswith(".txt"):
        normalized = normalized[:-4]
    if not normalized or "/" in normalized or "\\" in normalized or ".." in normalized:
        raise PromptTemplateError(f"invalid prompt template name: {template_name!r}")
    return normalized


def render_prompt(
    template_name: str,
    *,
    template_store: PromptTemplateStore | None = None,
    **variables: Any,
) -> str:
    store = template_store or PromptTemplateStore()
    return store.render(template_name, **variables)


def build_prompt_messages(
    *,
    system_prompt: str,
    user_prompt: str,
    feedback: str | None = None,
) -> list[ChatMessage]:
    system_content = system_prompt
    if feedback and feedback.strip():
        system_content = (
            system_prompt
            + "\n\n## Meta-review feedback for this run:\n"
            + feedback.strip()
        )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_prompt},
    ]
