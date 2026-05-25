from __future__ import annotations

import re

STOPWORDS = {
    "about",
    "also",
    "and",
    "are",
    "based",
    "between",
    "can",
    "could",
    "development",
    "does",
    "for",
    "from",
    "have",
    "hypotheses",
    "hypothesis",
    "into",
    "its",
    "may",
    "mechanism",
    "mechanisms",
    "network",
    "novel",
    "pathway",
    "pathways",
    "propose",
    "proposed",
    "regulates",
    "research",
    "should",
    "simultaneously",
    "that",
    "the",
    "their",
    "this",
    "through",
    "to",
    "using",
    "what",
    "which",
    "with",
}


def build_literature_query(*texts: str, max_terms: int = 12) -> str:
    raw = " ".join(text for text in texts if text).replace("_", " ")
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", raw)
    selected: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        normalized = token.strip("-").lower()
        if not normalized or normalized in STOPWORDS:
            continue
        if normalized.startswith("cs") and len(normalized) > 5:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(_canonical_term(token))
        if len(selected) >= max_terms:
            break
    if selected:
        return " ".join(selected)
    return " ".join(raw.split()[:max_terms])


def _canonical_term(term: str) -> str:
    if term.lower() == "cannabis":
        return "Cannabis"
    return term.strip("-")
