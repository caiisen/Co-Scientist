from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from co_scientist.tools.models import Citation


class AgentResultKind(StrEnum):
    HYPOTHESIS_CREATED = "hypothesis_created"
    REVIEW_COMPLETED = "review_completed"
    PROXIMITY_UPDATED = "proximity_updated"
    RANKING_DECISION = "ranking_decision"
    FEEDBACK_GENERATED = "feedback_generated"
    OVERVIEW_GENERATED = "overview_generated"
    NOOP = "noop"


class AgentResult(BaseModel):
    kind: AgentResultKind
    payload: dict[str, Any] = Field(default_factory=dict)
    citations: list[Citation] = Field(default_factory=list)
    raw_text: str | None = None
    parse_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.parse_error is None
