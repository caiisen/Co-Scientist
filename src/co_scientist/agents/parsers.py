from __future__ import annotations

import re
from dataclasses import dataclass

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
HYPOTHESIS_RE = re.compile(r"\bHYPOTHESIS\b\s*:?\s*(.+)\Z", re.IGNORECASE | re.DOTALL)
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
    match = HYPOTHESIS_RE.search(text.strip())
    if match is None:
        return ParseResult(None, "could not find final HYPOTHESIS block")
    hypothesis = match.group(1).strip()
    if not hypothesis:
        return ParseResult(None, "HYPOTHESIS block is empty")
    return ParseResult(hypothesis)


def parse_observation_verdict(text: str) -> ParseResult:
    matches = OBSERVATION_RE.findall(text)
    if not matches:
        return ParseResult(None, "could not find observation verdict")
    verdict = matches[-1].strip().lower()
    if verdict not in OBSERVATION_VERDICTS:
        return ParseResult(None, f"unknown observation verdict: {verdict}")
    return ParseResult(verdict)
