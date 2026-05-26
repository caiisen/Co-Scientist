from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)

from co_scientist.memory.models import Hypothesis, Task

from .base import Agent, AgentContext
from .results import AgentResult, AgentResultKind

EMBED_FALLBACK_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)


class ProximityAgent(Agent):
    name = "proximity"
    system_prompt = "You maintain hypothesis similarity relationships."

    async def execute(self, task: Task, ctx: AgentContext) -> AgentResult:
        if task.action != "update_proximity_graph":
            return AgentResult(
                kind=AgentResultKind.NOOP,
                payload={"agent": self.name, "ignored_action": task.action},
            )

        hypotheses = await ctx.store.list_session_hypotheses(ctx.session_id)
        hypotheses = [hypothesis for hypothesis in hypotheses if hypothesis.id is not None]
        if not hypotheses:
            return AgentResult(kind=AgentResultKind.PROXIMITY_UPDATED, payload={"edges": []})

        embeddings = await ctx.store.embeddings_for_session(ctx.session_id)
        missing = [hypothesis for hypothesis in hypotheses if hypothesis.id not in embeddings]
        embedding_source = "llm"
        embedding_error: str | None = None
        if missing:
            texts = [_embedding_text(hypothesis) for hypothesis in missing]
            try:
                vectors = await ctx.llm_for(self.name).embed(texts)
            except ValueError as exc:
                if str(exc) != "no embedding model configured for this provider":
                    raise
                embedding_source = "lexical_fallback"
                embedding_error = str(exc)
                vectors = [lexical_embedding(text) for text in texts]
            except EMBED_FALLBACK_ERRORS as exc:
                embedding_source = "lexical_fallback"
                embedding_error = str(exc)
                vectors = [lexical_embedding(text) for text in texts]
            new_embeddings: dict[int, list[float]] = {}
            for hypothesis, vector in zip(missing, vectors, strict=True):
                assert hypothesis.id is not None
                embeddings[hypothesis.id] = [float(value) for value in vector]
                new_embeddings[hypothesis.id] = embeddings[hypothesis.id]
            await ctx.store.upsert_hypothesis_embeddings_batch(
                session_id=ctx.session_id,
                embeddings=new_embeddings,
            )

        target_ids = _target_ids(task, hypotheses)
        edges: list[dict[str, Any]] = []
        edge_rows: list[tuple[int, int, float]] = []
        for first in hypotheses:
            if first.id is None:
                continue
            for second in hypotheses:
                if second.id is None or first.id >= second.id:
                    continue
                if target_ids and first.id not in target_ids and second.id not in target_ids:
                    continue
                if first.id not in embeddings or second.id not in embeddings:
                    continue
                similarity = cosine_similarity(embeddings[first.id], embeddings[second.id])
                edge_rows.append((first.id, second.id, similarity))
                edges.append(
                    {
                        "hypo_a_id": first.id,
                        "hypo_b_id": second.id,
                        "similarity": similarity,
                    }
                )
        await ctx.store.upsert_proximity_edges_batch(
            session_id=ctx.session_id,
            edges=edge_rows,
        )

        return AgentResult(
            kind=AgentResultKind.PROXIMITY_UPDATED,
            payload={
                "target_ids": sorted(target_ids),
                "edges": edges,
                "embedding_source": embedding_source,
                "embedding_error": embedding_error,
            },
        )


def cosine_similarity(first: list[float], second: list[float]) -> float:
    if len(first) != len(second):
        raise ValueError("embedding vectors must have the same length")
    dot = sum(a * b for a, b in zip(first, second, strict=True))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        return 0.0
    return dot / (first_norm * second_norm)


def lexical_embedding(text: str, *, dimensions: int = 128) -> list[float]:
    vector = [0.0] * dimensions
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        tokens = [text.lower()]
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    return vector


def _embedding_text(hypothesis: Hypothesis) -> str:
    return hypothesis.summary or hypothesis.content


def _target_ids(task: Task, hypotheses: list[Hypothesis]) -> set[int]:
    if task.target_id is not None:
        return {task.target_id}
    raw_target_ids = task.payload_json.get("target_ids")
    if isinstance(raw_target_ids, list):
        return {int(value) for value in raw_target_ids}
    return {int(hypothesis.id) for hypothesis in hypotheses if hypothesis.id is not None}
