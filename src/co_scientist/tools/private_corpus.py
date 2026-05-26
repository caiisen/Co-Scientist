from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from co_scientist.config import SearchConfig
from co_scientist.memory.store import SQLiteStore
from co_scientist.tools.models import Citation, SearchDocument, ToolResult, ToolStatus

SUPPORTED_SUFFIXES = {".md", ".txt"}
_INDEX_LOCKS: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


async def search(
    query: str,
    *,
    config: SearchConfig,
    store: SQLiteStore,
    session_id: str,
    max_results: int | None = None,
    embedding_client: Any | None = None,
) -> ToolResult:
    if not config.private_corpus_enabled or not config.private_corpus_paths:
        return ToolResult(source="private_corpus")

    async with _INDEX_LOCKS[_lock_key(session_id, config.private_corpus_paths)]:
        errors = await index_private_corpus(config=config, store=store, session_id=session_id)
    chunks = await store.list_private_corpus_chunks(session_id)
    limit = max_results or config.private_corpus_max_results
    if not chunks:
        return ToolResult(
            source="private_corpus",
            status=ToolStatus.FAILED if errors else ToolStatus.OK,
            errors=errors,
        )

    scored, score_errors = await _score_chunks(
        query,
        chunks,
        store=store,
        session_id=session_id,
        embedding_client=embedding_client,
    )
    errors.extend(score_errors)
    documents = [_chunk_to_document(chunk, score) for score, chunk in scored[:limit] if score > 0]
    status = ToolStatus.DEGRADED if errors and documents else ToolStatus.OK
    if errors and not documents:
        status = ToolStatus.FAILED
    return ToolResult(
        source="private_corpus",
        status=status,
        documents=documents,
        citations=[document.citation for document in documents],
        errors=errors,
    )


async def index_private_corpus(
    *,
    config: SearchConfig,
    store: SQLiteStore,
    session_id: str,
) -> list[str]:
    errors: list[str] = []
    for file_path in _iter_private_files(config.private_corpus_paths):
        try:
            stat = file_path.stat()
        except OSError as exc:
            errors.append(f"{file_path}: {exc}")
            continue
        path_text = str(file_path.resolve())
        state = await store.private_corpus_file_state(
            session_id=session_id,
            path=path_text,
        )
        if (
            state is not None
            and state["mtime"] == stat.st_mtime
            and state["file_size"] == stat.st_size
        ):
            continue

        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                errors.append(f"{file_path}: {exc}")
                continue
        except OSError as exc:
            errors.append(f"{file_path}: {exc}")
            continue

        title = _title_for_file(file_path, text)
        chunks = [
            {
                "title": title,
                "chunk_index": index,
                "content": chunk,
                "content_hash": _hash_text(chunk),
                "mtime": stat.st_mtime,
                "file_size": stat.st_size,
            }
            for index, chunk in enumerate(
                _chunk_text(
                    text,
                    chunk_chars=config.private_corpus_chunk_chars,
                    overlap=config.private_corpus_chunk_overlap,
                )
            )
            if chunk.strip()
        ]
        current_hashes = [chunk["content_hash"] for chunk in chunks]
        if state is not None and state["content_hashes"] == current_hashes:
            continue
        try:
            await store.replace_private_corpus_file_chunks(
                session_id=session_id,
                path=path_text,
                chunks=chunks,
            )
        except Exception as exc:
            errors.append(f"{file_path}: failed to index private corpus file: {exc}")
    return errors


def _iter_private_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
        elif path.is_dir():
            files.extend(
                child
                for child in sorted(path.rglob("*"))
                if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES
            )
    return files


def _chunk_text(text: str, *, chunk_chars: int, overlap: int) -> list[str]:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    step = max(chunk_chars - overlap, 1)
    while start < len(normalized):
        end = min(start + chunk_chars, len(normalized))
        chunks.append(normalized[start:end].strip())
        if end == len(normalized):
            break
        start += step
    return chunks


async def _score_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    *,
    store: SQLiteStore,
    session_id: str,
    embedding_client: Any | None,
) -> tuple[list[tuple[float, dict[str, Any]]], list[str]]:
    errors: list[str] = []
    if embedding_client is not None:
        try:
            missing = [chunk for chunk in chunks if chunk.get("embedding") is None]
            if missing:
                vectors = await embedding_client.embed(
                    [_chunk_embedding_text(chunk) for chunk in missing]
                )
                updates = {
                    int(chunk["id"]): [float(value) for value in vector]
                    for chunk, vector in zip(missing, vectors, strict=True)
                }
                await store.update_private_corpus_embeddings(
                    session_id=session_id,
                    embeddings=updates,
                )
                for chunk in chunks:
                    if int(chunk["id"]) in updates:
                        chunk["embedding"] = updates[int(chunk["id"])]
            query_vector = (await embedding_client.embed([query]))[0]
            return sorted(
                (
                    (
                        _cosine_similarity(
                            [float(value) for value in query_vector],
                            chunk["embedding"],
                        ),
                        chunk,
                    )
                    for chunk in chunks
                    if chunk.get("embedding") is not None
                ),
                key=lambda item: item[0],
                reverse=True,
            ), errors
        except Exception as exc:
            errors.append(
                f"private_corpus embedding search failed; using lexical fallback: "
                f"{type(exc).__name__}: {exc}"
            )

    query_terms = _terms(query)
    return sorted(
        ((_lexical_score(query_terms, chunk["content"]), chunk) for chunk in chunks),
        key=lambda item: item[0],
        reverse=True,
    ), errors


def _chunk_to_document(chunk: dict[str, Any], score: float) -> SearchDocument:
    title = f"{chunk['title']} (private chunk {int(chunk['chunk_index']) + 1})"
    path = str(chunk["path"])
    content = f"Source: {path}\n\n{chunk['content']}"
    citation = Citation(
        source="private_corpus",
        title=title,
        raw_json={
            "path": path,
            "chunk_index": int(chunk["chunk_index"]),
            "content_hash": str(chunk["content_hash"]),
        },
    )
    return SearchDocument(
        source="private_corpus",
        title=title,
        abstract_or_snippet=content,
        score=score,
        citation=citation,
    )


def _title_for_file(path: Path, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title
    return path.stem


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk_embedding_text(chunk: dict[str, Any]) -> str:
    return f"{chunk['title']}\n\n{chunk['content']}"


def _terms(text: str) -> set[str]:
    return {term for term in re.findall(r"\w+", text.lower()) if len(term) > 2}


def _lexical_score(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0
    text_terms = _terms(text)
    if not text_terms:
        return 0.0
    overlap = len(query_terms & text_terms)
    return overlap / math.sqrt(len(query_terms) * len(text_terms))


def _cosine_similarity(first: list[float], second: list[float]) -> float:
    if len(first) != len(second):
        return 0.0
    dot = sum(a * b for a, b in zip(first, second, strict=True))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        return 0.0
    return dot / (first_norm * second_norm)


def _lock_key(session_id: str, paths: list[str]) -> str:
    normalized_paths = "|".join(sorted(str(Path(path).expanduser()) for path in paths))
    return f"{session_id}:{normalized_paths}"
