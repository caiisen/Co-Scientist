from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

OBSERVATION_VERDICTS = {
    "already explained",
    "other explanations more likely",
    "missing piece",
    "neutral",
    "disproved",
}

BETTER_IDEA_RE = re.compile(
    r"\bbetter\s+(?:idea|hypothesis)\s*:\s*([12])\b",
    re.IGNORECASE,
)
HYPOTHESIS_MARKER_RE = re.compile(
    r"^[ \t>#*-]*\*{0,2}HYPOTHESIS\*{0,2}\s*:?[ \t]*",
    re.IGNORECASE | re.MULTILINE,
)
INLINE_HYPOTHESIS_MARKER_RE = re.compile(
    r"\b\*{0,2}HYPOTHESIS\*{0,2}\s*:\s*",
    re.IGNORECASE,
)
SCORE_RE = re.compile(
    r"(?:overall\s+)?(?:score|rating|评分|总分|得分)\s*"
    r"(?:from\s*)?(?:0\s*(?:-|to|/)\s*10)?\s*[:：=]?\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*(?:/|out\s+of|分，共)?\s*10?",
    re.IGNORECASE,
)
FRACTION_SCORE_RE = re.compile(r"\b([0-9]+(?:\.[0-9]+)?)\s*/\s*10\b")
OBSERVATION_RE = re.compile(
    r"\bhypothesis\s*:\s*"
    r"(already explained|other explanations more likely|missing piece|neutral|disproved)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParseResult:
    value: str | int | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def parse_better_idea(text: str) -> ParseResult:
    match = BETTER_IDEA_RE.search(text)
    if match is None:
        return ParseResult(None, "could not find 'better idea: <1 or 2>' decision")
    return ParseResult(int(match.group(1)))


def parse_hypothesis_block(text: str) -> ParseResult:
    stripped = text.strip()
    matches = list(HYPOTHESIS_MARKER_RE.finditer(stripped))
    if not matches:
        matches = list(INLINE_HYPOTHESIS_MARKER_RE.finditer(stripped))
        if not matches:
            return ParseResult(None, "could not find final HYPOTHESIS block")
    hypothesis = _strip_hypothesis_markers(stripped[matches[-1].end():])
    if not hypothesis:
        return ParseResult(None, "HYPOTHESIS block is empty")
    return ParseResult(hypothesis)


def parse_json_object(text: str) -> ParseResult:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"\A```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```\s*\Z", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ParseResult(None, "could not find JSON object")
    try:
        value: Any = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        return ParseResult(None, f"invalid JSON object: {exc}")
    if not isinstance(value, dict):
        return ParseResult(None, "parsed JSON is not an object")
    return ParseResult(value)


def parse_review_score(text: str) -> ParseResult:
    matches = SCORE_RE.findall(text)
    if matches:
        return _score_result(matches[-1])

    fraction_matches = FRACTION_SCORE_RE.findall(text)
    if fraction_matches:
        return _score_result(fraction_matches[-1])

    conclusion_lines = [
        line
        for line in text.splitlines()
        if any(marker in line.lower() for marker in ("score", "rating", "评分", "总分", "得分"))
    ]
    for line in reversed(conclusion_lines):
        number = re.search(r"([0-9]+(?:\.[0-9]+)?)", line)
        if number:
            return _score_result(number.group(1))

    return ParseResult(None, "could not find review score")


def _score_result(value: str) -> ParseResult:
    score = float(value)
    if score < 0 or score > 10:
        return ParseResult(None, f"review score out of range: {score}")
    return ParseResult(score)


def summarize_hypothesis(text: str, *, max_chars: int = 180) -> str:
    for line in text.splitlines():
        cleaned = _clean_summary_line(line)
        if cleaned:
            return _truncate(cleaned, max_chars)
    return _truncate(" ".join(text.split()), max_chars)


def parse_observation_verdict(text: str) -> ParseResult:
    matches = OBSERVATION_RE.findall(text)
    if not matches:
        return ParseResult(None, "could not find observation verdict")
    verdict = matches[-1].strip().lower()
    if verdict not in OBSERVATION_VERDICTS:
        return ParseResult(None, f"unknown observation verdict: {verdict}")
    return ParseResult(verdict)


def _clean_summary_line(line: str) -> str:
    cleaned = line.strip()
    cleaned = re.sub(r"\A(?:#{1,6}|[-*]|\d+[.)])\s+", "", cleaned).strip()
    cleaned = re.sub(r"\A\*{1,2}HYPOTHESIS:?\*{1,2}\s*:?\s*", "", cleaned, flags=re.IGNORECASE)
    if not cleaned:
        return ""
    cleaned = _strip_wrapping_markdown(cleaned)
    if cleaned.lower() == "hypothesis":
        return ""
    for prefix in ("summary:", "hypothesis:", "title:"):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            cleaned = _strip_wrapping_markdown(cleaned)
    return cleaned


def _truncate(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _strip_hypothesis_markers(text: str) -> str:
    stripped = text.strip()
    stripped = re.sub(r"\A\*+", "", stripped)
    stripped = re.sub(r"\*+\Z", "", stripped)
    return stripped.strip()


def _strip_wrapping_markdown(text: str) -> str:
    stripped = text.strip()
    while len(stripped) >= 2 and (
        (stripped.startswith("**") and stripped.endswith("**"))
        or (stripped.startswith("__") and stripped.endswith("__"))
    ):
        stripped = stripped[2:-2].strip()
    while len(stripped) >= 2 and (
        (stripped.startswith("*") and stripped.endswith("*"))
        or (stripped.startswith("_") and stripped.endswith("_"))
        or (stripped.startswith("`") and stripped.endswith("`"))
    ):
        stripped = stripped[1:-1].strip()
    return stripped
