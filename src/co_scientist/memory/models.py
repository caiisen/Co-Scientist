from __future__ import annotations

from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(IntEnum):
    LOW = 10
    PROXIMITY = 30
    RANKING = 50
    EVOLUTION = 55
    META_REVIEW = 60
    REFLECTION = 70
    USER = 100


class ReviewType(StrEnum):
    INITIAL = "initial"
    FULL = "full"
    DEEP_VERIFICATION = "deep_verification"
    OBSERVATION = "observation"
    SIMULATION = "simulation"
    MANUAL = "manual"


class Session(BaseModel):
    id: str
    goal: str
    config_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class Hypothesis(BaseModel):
    id: int | None = None
    session_id: str
    content: str
    summary: str
    detailed_description: str | None = None
    mechanism: str | None = None
    impacted_pathways: list[str] = Field(default_factory=list)
    experimental_plan: str | None = None
    safety_notes: str | None = None
    testable_predictions: list[str] = Field(default_factory=list)
    elo: int = Field(default=1200, ge=0)
    parent_ids: list[int] = Field(default_factory=list)
    source_strategy: str | None = None
    meta_review_round: int | None = Field(default=None, ge=0)
    created_at: datetime = Field(default_factory=utc_now)


class Review(BaseModel):
    id: int | None = None
    session_id: str
    hypothesis_id: int
    type: ReviewType
    score: float | None = Field(default=None, ge=0, le=10)
    content: str
    created_at: datetime = Field(default_factory=utc_now)


class Match(BaseModel):
    id: int | None = None
    session_id: str
    hypo_a_id: int
    hypo_b_id: int
    winner_id: int
    transcript: str
    created_at: datetime = Field(default_factory=utc_now)


class Task(BaseModel):
    id: int | None = None
    session_id: str
    agent: str
    action: str
    target_id: int | None = None
    priority: int = int(TaskPriority.LOW)
    status: TaskStatus = TaskStatus.PENDING
    payload_json: dict[str, Any] = Field(default_factory=dict)
    result_json: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ResearchPlan(BaseModel):
    session_id: str
    goal: str
    preferences: list[str] = Field(default_factory=list)
    attributes: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    idea_attributes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class SystemFeedback(BaseModel):
    id: int | None = None
    session_id: str
    round: int
    content: str
    created_at: datetime = Field(default_factory=utc_now)


class ResearchOverview(BaseModel):
    id: int | None = None
    session_id: str
    round: int
    content: str
    top_hypothesis_ids: list[int] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
