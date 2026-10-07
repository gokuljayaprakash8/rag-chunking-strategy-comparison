#!/usr/bin/env python3
"""Controlled, retrieval-only RAG chunking comparison."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.embedding import (
        MODEL_ID,
        MODEL_REVISION,
        EmbeddingError,
        SentenceTransformerOnnx,
    )
else:
    from .embedding import (
        MODEL_ID,
        MODEL_REVISION,
        EmbeddingError,
        SentenceTransformerOnnx,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "experiment.json"
SOURCE_PATH = "data/diabetes_reference_document.md"
QUESTIONS_PATH = "data/questions.json"
RESULTS_PATH = "results"

STRATEGY_A = "A"
STRATEGY_B = "B"
TOP_K = 3
SIMILARITY = "cosine"
FIXED_SIZE = 500
FIXED_OVERLAP = 50
STRUCTURE_HIERARCHY = ["section", "paragraph", "sentence"]
EVALUATION_PROCEDURE = "manual_source_grounded"

JSONDict = dict[str, Any]
class ExperimentError(Exception):
    """An input or runtime error that should be shown clearly to the user."""


@dataclass(frozen=True)
class Chunk:
    strategy: str
    label: str
    index: int
    start_char: int
    end_char: int
    content: str
    content_sha256: str
    chunk_id: str
    heading_path: tuple[str, ...]
    headings_in_chunk: tuple[str, ...]

    def to_dict(self) -> JSONDict:
        return {
            "strategy": self.strategy,
            "label": self.label,
            "index": self.index,
            "chunk_id": self.chunk_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "content_sha256": self.content_sha256,
            "heading_path": list(self.heading_path),
            "headings_in_chunk": list(self.headings_in_chunk),
            "content": self.content,
        }


@dataclass(frozen=True)
class Heading:
    start_char: int
    level: int
    title: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _require_exact_keys(value: Any, expected: set[str], name: str) -> JSONDict:
    if not isinstance(value, dict):
        raise ExperimentError(f"{name} must be a JSON object.")
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing or extra:
        details = []
        if missing:
            details.append("missing: " + ", ".join(sorted(missing)))
        if extra:
            details.append("unexpected: " + ", ".join(sorted(extra)))
        raise ExperimentError(f"{name} has invalid fields ({'; '.join(details)}).")
    return value


def _non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentError(f"{name} must be supplied as a non-empty string.")
    return value


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> tuple[JSONDict, bytes]:
    config_path = config_path.expanduser().resolve()
    try:
        raw = config_path.read_bytes()
    except OSError as exc:
        raise ExperimentError(f"Cannot read configuration file {config_path}: {exc}") from exc

    try:
        config = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"Configuration must be valid UTF-8 JSON: {exc}") from exc

    validate_config(config)
    return config, raw


def validate_config(config: Any) -> None:
    expected_top_level = {
        "source_document",
        "questions_file",
        "results_directory",
        "top_k",
        "similarity",
        "strategies",
        "embedding",
        "evaluation",
    }
    config = _require_exact_keys(config, expected_top_level, "Experiment configuration")

    if config["source_document"] != SOURCE_PATH:
        raise ExperimentError(
            f"source_document must remain {SOURCE_PATH!r}; the experiment source cannot change."
        )
    if config["questions_file"] != QUESTIONS_PATH:
        raise ExperimentError(f"questions_file must be {QUESTIONS_PATH!r}.")
    if config["results_directory"] != RESULTS_PATH:
        raise ExperimentError(f"results_directory must be {RESULTS_PATH!r}.")
    if (
        isinstance(config["top_k"], bool)
        or not isinstance(config["top_k"], int)
        or config["top_k"] < 1
    ):
        raise ExperimentError(f"top_k must be exactly {TOP_K}.")
    if config["similarity"] != SIMILARITY:
        raise ExperimentError(f"similarity must be exactly {SIMILARITY!r}.")

    _validate_strategies(config["strategies"])
    _validate_embedding_config(config["embedding"])
    evaluation = _require_exact_keys(
        config["evaluation"], {"procedure"}, "evaluation"
    )
    if evaluation["procedure"] != EVALUATION_PROCEDURE:
        raise ExperimentError(
            f"evaluation.procedure must be {EVALUATION_PROCEDURE!r}."
        )


def _validate_strategies(value: Any) -> None:
    strategies = _require_exact_keys(value, {STRATEGY_A, STRATEGY_B}, "strategies")

    a = _require_exact_keys(
        strategies[STRATEGY_A],
        {"label", "method", "chunk_size", "overlap"},
        "strategies.A",
    )
    if a["label"] != "Strategy A — Basic / Naive":
        raise ExperimentError("Strategy A label must be 'Strategy A — Basic / Naive'.")
    if a["method"] != "fixed_size":
        raise ExperimentError("Strategy A method must be 'fixed_size'.")
    if (
        isinstance(a["chunk_size"], bool)
        or not isinstance(a["chunk_size"], int)
        or a["chunk_size"] != FIXED_SIZE
    ):
        raise ExperimentError(f"Strategy A chunk_size must be exactly {FIXED_SIZE}.")
    if (
        isinstance(a["overlap"], bool)
        or not isinstance(a["overlap"], int)
        or a["overlap"] != FIXED_OVERLAP
    ):
        raise ExperimentError(f"Strategy A overlap must be exactly {FIXED_OVERLAP}.")

    b = _require_exact_keys(
        strategies[STRATEGY_B],
        {
            "label",
            "method",
            "hierarchy",
            "preserve_markdown_headings",
            "preserve_natural_sections",
            "max_chunk_characters",
        },
        "strategies.B",
    )
    if b["label"] != "Strategy B — Advanced / Structure-aware":
        raise ExperimentError(
            "Strategy B label must be 'Strategy B — Advanced / Structure-aware'."
        )
    if b["method"] != "structure_aware":
        raise ExperimentError("Strategy B method must be 'structure_aware'.")
    if b["hierarchy"] != STRUCTURE_HIERARCHY:
        raise ExperimentError(
            "Strategy B hierarchy must be exactly section → paragraph → sentence."
        )
    if b["preserve_markdown_headings"] is not True:
        raise ExperimentError("Strategy B must preserve Markdown headings.")
    if b["preserve_natural_sections"] is not True:
        raise ExperimentError("Strategy B must preserve natural document sections.")
    max_chars = b["max_chunk_characters"]
    if max_chars is not None and (
        isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1
    ):
        raise ExperimentError(
            "Strategy B max_chunk_characters must be null or a positive integer."
        )


def _validate_embedding_config(value: Any) -> None:
    embedding = _require_exact_keys(
        value,
        {"model", "revision", "backend", "batch_size", "max_seq_length", "device"},
        "embedding",
    )
    if embedding["model"] != MODEL_ID:
        raise ExperimentError(f"embedding.model must be exactly {MODEL_ID!r}.")
    if embedding["revision"] != MODEL_REVISION:
        raise ExperimentError(
            f"embedding.revision must be exactly {MODEL_REVISION!r}."
        )
    if embedding["backend"] != "onnxruntime-cpu":
        raise ExperimentError("embedding.backend must be 'onnxruntime-cpu'.")
    if embedding["batch_size"] != 32:
        raise ExperimentError("embedding.batch_size must be exactly 32.")
    if embedding["max_seq_length"] != 256:
        raise ExperimentError("embedding.max_seq_length must be exactly 256.")
    if embedding["device"] != "CPUExecutionProvider":
        raise ExperimentError(
            "embedding.device must be 'CPUExecutionProvider'."
        )


def _load_source(path: Path) -> tuple[str, bytes]:
    try:
        source_bytes = path.read_bytes()
    except OSError as exc:
        raise ExperimentError(f"Cannot read source document {path}: {exc}") from exc
    if not source_bytes:
        raise ExperimentError(f"Source document is empty: {path}")
    try:
        # newline="" avoids universal-newline conversion, preserving source offsets.
        with path.open("r", encoding="utf-8", newline="") as handle:
            source = handle.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise ExperimentError(f"Source document must be readable UTF-8: {exc}") from exc
    if not source:
        raise ExperimentError(f"Source document decodes to empty text: {path}")
    return source, source_bytes


def _load_questions(path: Path) -> tuple[list[dict[str, str]], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ExperimentError(
            f"Cannot read the required official question file {path}: {exc}"
        ) from exc
    try:
        values = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"Question file must be valid UTF-8 JSON: {exc}") from exc

    if not isinstance(values, list) or len(values) != 10:
        raise ExperimentError("The official question file must contain exactly 10 questions.")

    questions: list[dict[str, str]] = []
    for index, question in enumerate(values, start=1):
        if not isinstance(question, str) or not question.strip():
            raise ExperimentError(
                f"Question Q{index} must be a non-empty string; wording is not altered."
            )
        # Preserve the supplied question exactly: no strip, normalization, or rewrite.
        questions.append({"id": f"Q{index}", "text": question})
    return questions, raw


def _markdown_headings(source: str) -> list[Heading]:
    headings: list[Heading] = []
    offset = 0
    fence_char: str | None = None
    fence_length = 0

    for raw_line in source.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        fence_match = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence_char is not None:
            if fence_match:
                marker = fence_match.group(1)
                if marker[0] == fence_char and len(marker) >= fence_length:
                    fence_char = None
                    fence_length = 0
            offset += len(raw_line)
            continue

        if fence_match:
            marker = fence_match.group(1)
            fence_char = marker[0]
            fence_length = len(marker)
            offset += len(raw_line)
            continue

        heading_match = re.match(r"^ {0,3}(#{1,6})(?=[ \t]|$)", line)
        if heading_match:
            level = len(heading_match.group(1))
            title = line[heading_match.end() :].strip()
            title = re.sub(r"[ \t]+#+[ \t]*$", "", title).strip()
            headings.append(Heading(offset, level, title))
        offset += len(raw_line)
    return headings


def _heading_paths(headings: list[Heading]) -> list[tuple[str, ...]]:
    stack: list[tuple[int, str]] = []
    paths: list[tuple[str, ...]] = []
    for heading in headings:
        while stack and stack[-1][0] >= heading.level:
            stack.pop()
        stack.append((heading.level, heading.title))
        paths.append(tuple(title for _, title in stack))
    return paths


def _active_heading_path(
    offset: int, headings: list[Heading], paths: list[tuple[str, ...]]
) -> tuple[str, ...]:
    active: tuple[str, ...] = ()
    for heading, path in zip(headings, paths):
        if heading.start_char > offset:
            break
        active = path
    return active


def _chunk_from_span(
    *,
    strategy: str,
    label: str,
    index: int,
    start: int,
    end: int,
    source: str,
    heading_path: tuple[str, ...],
    headings_in_chunk: tuple[str, ...],
) -> Chunk:
    content = source[start:end]
    content_hash = _sha256(content.encode("utf-8"))
    chunk_id = f"{strategy}-{index:04d}-{content_hash[:12]}"
    return Chunk(
        strategy=strategy,
        label=label,
        index=index,
        start_char=start,
        end_char=end,
        content=content,
        content_sha256=content_hash,
        chunk_id=chunk_id,
        heading_path=heading_path,
        headings_in_chunk=headings_in_chunk,
    )


def fixed_size_chunks(
    source: str,
    size: int = FIXED_SIZE,
    overlap: int = FIXED_OVERLAP,
) -> list[Chunk]:
    if size != FIXED_SIZE or overlap != FIXED_OVERLAP:
        raise ExperimentError(
            f"Strategy A is fixed at {FIXED_SIZE} characters with {FIXED_OVERLAP} overlap."
        )
    if overlap >= size:
        raise ExperimentError("Chunk overlap must be smaller than chunk size.")

    headings = _markdown_headings(source)
    paths = _heading_paths(headings)
    step = size - overlap
    chunks: list[Chunk] = []
    for start in range(0, len(source), step):
        end = min(start + size, len(source))
        included_headings = tuple(
            heading.title
            for heading in headings
            if start <= heading.start_char < end
        )
        chunks.append(
            _chunk_from_span(
                strategy=STRATEGY_A,
                label="Strategy A — Basic / Naive",
                index=len(chunks) + 1,
                start=start,
                end=end,
                source=source,
                heading_path=_active_heading_path(start, headings, paths),
                headings_in_chunk=included_headings,
            )
        )
    return chunks


def _paragraph_spans(source: str, start: int, end: int) -> list[tuple[int, int]]:
    """Split a span at blank-line boundaries while retaining every source character."""
    segment = source[start:end]
    separators = list(re.finditer(r"\r?\n[ \t\r]*\r?\n+", segment))
    spans: list[tuple[int, int]] = []
    cursor = start
    for separator in separators:
        boundary = start + separator.end()
        if boundary > cursor:
            spans.append((cursor, boundary))
            cursor = boundary
    if cursor < end:
        spans.append((cursor, end))
    return spans or [(start, end)]


def _sentence_spans(source: str, start: int, end: int) -> list[tuple[int, int]]:
    """Find conservative sentence boundaries without changing the original text."""
    segment = source[start:end]
    boundaries = [
        start + match.end()
        for match in re.finditer(r"(?<=[.!?])[ \t\r\n]+", segment)
    ]
    spans: list[tuple[int, int]] = []
    cursor = start
    for boundary in boundaries:
        if boundary > cursor:
            spans.append((cursor, boundary))
            cursor = boundary
    if cursor < end:
        spans.append((cursor, end))
    return spans or [(start, end)]


def _fit_structured_span(
    source: str,
    start: int,
    end: int,
    max_chars: int,
) -> list[tuple[int, int]]:
    """Fit a structure-aware section into semantic chunks.

    The section heading is attached to the first substantive unit rather
    than being emitted as a standalone chunk. This intentionally allows
    the first chunk to exceed max_chars when necessary to preserve the
    heading + first substantive unit together.
    """
    if end <= start:
        return []

    # Extract the section heading, if the section begins with a Markdown
    # heading. The heading is metadata/context and must stay attached to
    # the first substantive unit.
    first_line_end = source.find("\n", start, end)
    if first_line_end == -1:
        first_line_end = end

    first_line = source[start:first_line_end].strip()
    heading_end = first_line_end + 1 if first_line_end < end else end

    is_heading = first_line.startswith("#")
    body_start = heading_end if is_heading else start

    if body_start >= end:
        return []

    units = _paragraph_spans(source, body_start, end)

    # If paragraph detection gives us nothing, fall back to the whole body.
    if not units:
        units = [(body_start, end)]

    output: list[tuple[int, int]] = []

    pending_start: int | None = None
    pending_end: int | None = None

    def flush_pending() -> None:
        nonlocal pending_start, pending_end
        if pending_start is not None and pending_end is not None:
            output.append((pending_start, pending_end))
        pending_start = None
        pending_end = None

    first_substantive = True

    for unit_start, unit_end in units:
        if unit_end <= unit_start:
            continue

        # The first substantive unit receives the heading as context.
        effective_start = start if first_substantive and is_heading else unit_start

        if unit_end - effective_start > max_chars:
            # A heading + first paragraph is kept together even when it
            # exceeds max_chars. For an oversized ordinary paragraph,
            # split at sentence boundaries.
            if first_substantive and is_heading:
                sentences = _sentence_spans(source, unit_start, unit_end)

                if not sentences:
                    flush_pending()
                    output.append((start, unit_end))
                else:
                    first_sentence_start, first_sentence_end = sentences[0]
                    first_chunk_end = first_sentence_end

                    # Keep heading attached to the first sentence.
                    output.append((start, first_chunk_end))

                    sentence_start: int | None = None
                    sentence_end: int | None = None

                    for part_start, part_end in sentences[1:]:
                        if part_end - part_start > max_chars:
                            if sentence_start is not None and sentence_end is not None:
                                output.append((sentence_start, sentence_end))
                            sentence_start = None
                            sentence_end = None
                            output.append((part_start, part_end))
                            continue

                        if sentence_start is None:
                            sentence_start = part_start
                            sentence_end = part_end
                        elif part_end - sentence_start <= max_chars:
                            sentence_end = part_end
                        else:
                            output.append((sentence_start, sentence_end))
                            sentence_start = part_start
                            sentence_end = part_end

                    if sentence_start is not None and sentence_end is not None:
                        output.append((sentence_start, sentence_end))

            else:
                flush_pending()
                sentences = _sentence_spans(source, unit_start, unit_end)

                if not sentences:
                    output.append((unit_start, unit_end))
                else:
                    sentence_start: int | None = None
                    sentence_end: int | None = None

                    for part_start, part_end in sentences:
                        if part_end - part_start > max_chars:
                            if sentence_start is not None and sentence_end is not None:
                                output.append((sentence_start, sentence_end))
                            sentence_start = None
                            sentence_end = None
                            output.append((part_start, part_end))
                            continue

                        if sentence_start is None:
                            sentence_start = part_start
                            sentence_end = part_end
                        elif part_end - sentence_start <= max_chars:
                            sentence_end = part_end
                        else:
                            output.append((sentence_start, sentence_end))
                            sentence_start = part_start
                            sentence_end = part_end

                    if sentence_start is not None and sentence_end is not None:
                        output.append((sentence_start, sentence_end))

        else:
            if first_substantive and is_heading:
                # Heading + first paragraph are atomic.
                effective_start = start

            if pending_start is None:
                pending_start = effective_start
                pending_end = unit_end
            elif unit_end - pending_start <= max_chars:
                pending_end = unit_end
            else:
                flush_pending()
                pending_start = effective_start
                pending_end = unit_end

        first_substantive = False

    flush_pending()

    return output

def structure_aware_chunks(
    source: str, max_chunk_characters: int | None = None
) -> list[Chunk]:
    if max_chunk_characters is not None and (
        isinstance(max_chunk_characters, bool)
        or not isinstance(max_chunk_characters, int)
        or max_chunk_characters < 1
    ):
        raise ExperimentError(
            "Structure-aware max chunk size must be null or a positive integer."
        )

    headings = _markdown_headings(source)
    paths = _heading_paths(headings)
    spans: list[tuple[int, int, tuple[str, ...]]] = []
    if not headings:
        spans.append((0, len(source), ()))
    else:
        if headings[0].start_char > 0:
            spans.append((0, headings[0].start_char, ()))
        for index, heading in enumerate(headings):
            end = (
                headings[index + 1].start_char
                if index + 1 < len(headings)
                else len(source)
            )

            # Treat the Markdown heading as metadata/context for a
            # substantive section. Do not emit a heading-only chunk.
            line_end = source.find("\n", heading.start_char)
            body_start = len(source) if line_end == -1 else line_end + 1

            if source[body_start:end].strip():
                spans.append((heading.start_char, end, paths[index]))

    chunks: list[Chunk] = []
    for section_start, section_end, path in spans:
        if section_start == section_end:
            continue
        fitted_spans = _fit_structured_span(
            source, section_start, section_end, max_chunk_characters
        )
        for start, end in fitted_spans:
            included_headings = tuple(
                heading.title
                for heading in headings
                if start <= heading.start_char < end
            )
            chunks.append(
                _chunk_from_span(
                    strategy=STRATEGY_B,
                    label="Strategy B — Advanced / Structure-aware",
                    index=len(chunks) + 1,
                    start=start,
                    end=end,
                    source=source,
                    heading_path=path,
                    headings_in_chunk=included_headings,
                )
            )
    return chunks


def _vector_sha256(vector: Sequence[float]) -> str:
    stable = [float(value).hex() for value in vector]
    return _sha256(_canonical_json(stable))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    if left_array.shape != right_array.shape:
        raise ExperimentError(
            f"Embedding dimension mismatch ({left_array.shape} != {right_array.shape})."
        )
    left_norm = np.linalg.norm(left_array)
    right_norm = np.linalg.norm(right_array)
    if left_norm == 0 or right_norm == 0:
        raise ExperimentError("Cosine similarity is undefined for a zero vector.")
    score = float(np.dot(left_array, right_array) / (left_norm * right_norm))
    if not np.isfinite(score):
        raise ExperimentError("Cosine similarity produced a non-finite score.")
    return float(score)


def _manual_evaluation_fields() -> JSONDict:
    return {
        "assessment": "not_assessed",
        "justification": None,
    }


def _rank_context(
    question_vector: Sequence[float],
    chunks: list[Chunk],
    chunk_vectors: list[list[float]],
    question: dict[str, str],
    query_vector_sha256: str,
    top_k: int,
) -> JSONDict:
    ranked: list[tuple[float, int]] = []
    for index, chunk_vector in enumerate(chunk_vectors):
        ranked.append((cosine_similarity(question_vector, chunk_vector), index))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    retrieved: list[JSONDict] = []
    for rank, (score, index) in enumerate(ranked[:top_k], start=1):
        chunk = chunks[index]
        retrieved.append(
            {
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "content_sha256": chunk.content_sha256,
                "similarity": score,
                "heading_path": list(chunk.heading_path),
                "content": chunk.content,
            }
        )

    context = "\n\n".join(item["content"] for item in retrieved)
    return {
        "question_id": question["id"],
        "question": question["text"],
        "query_embedding_sha256": query_vector_sha256,
        "retrieved_top_" + str(top_k): retrieved,
        "retrieved_context": context,
        "answer_generation": {
            "status": "not_performed",
            "reason": (
                "No answer-generation model or procedure was supplied. "
                "The exact retrieved context is retained for manual assessment."
            ),
        },
        "manual_evaluation": _manual_evaluation_fields(),
    }


def _source_line_count(source: str) -> int:
    if not source:
        return 0
    return source.count("\n") + (0 if source.endswith("\n") else 1)


def run_experiment(config_path: Path = DEFAULT_CONFIG_PATH) -> Path:
    config, config_bytes = load_config(config_path)

    source_path = PROJECT_ROOT / config["source_document"]
    questions_path = PROJECT_ROOT / config["questions_file"]
    results_dir = PROJECT_ROOT / config["results_directory"]
    source, source_bytes = _load_source(source_path)
    questions, questions_bytes = _load_questions(questions_path)

    top_k = config["top_k"]

    chunks_by_strategy: dict[str, list[Chunk]] = {
        STRATEGY_A: fixed_size_chunks(
            source,
            size=config["strategies"][STRATEGY_A]["chunk_size"],
            overlap=config["strategies"][STRATEGY_A]["overlap"],
        ),
        STRATEGY_B: structure_aware_chunks(
            source,
            max_chunk_characters=config["strategies"][STRATEGY_B][
                "max_chunk_characters"
            ],
        ),
    }
    if any(not chunks for chunks in chunks_by_strategy.values()):
        raise ExperimentError("A chunking strategy produced no chunks.")
    if any(len(chunks) < top_k for chunks in chunks_by_strategy.values()):
        raise ExperimentError(
            f"Each strategy must produce at least {top_k} chunks for top-k retrieval."
        )

    embedding_config = config["embedding"]
    try:
        model = SentenceTransformerOnnx(
            model_id=embedding_config["model"],
            revision=embedding_config["revision"],
            batch_size=embedding_config["batch_size"],
            max_seq_length=embedding_config["max_seq_length"],
            device=embedding_config["device"],
        )
    except EmbeddingError as exc:
        raise ExperimentError(str(exc)) from exc

    # Encode each distinct query/chunk string once, then reuse exact vectors.
    all_texts = [question["text"] for question in questions]
    for strategy_key in (STRATEGY_A, STRATEGY_B):
        all_texts.extend(chunk.content for chunk in chunks_by_strategy[strategy_key])
    unique_texts = list(dict.fromkeys(all_texts))
    vectors = model.encode(unique_texts)
    if vectors.shape != (len(unique_texts), 384):
        raise ExperimentError(
            f"The embedding model returned an unexpected shape: {vectors.shape}."
        )
    vectors_by_text = dict(zip(unique_texts, vectors))
    query_vectors = [vectors_by_text[question["text"]] for question in questions]

    strategy_results: JSONDict = {}
    for strategy_key in (STRATEGY_A, STRATEGY_B):
        chunks = chunks_by_strategy[strategy_key]
        chunk_vectors = [vectors_by_text[chunk.content] for chunk in chunks]

        questions_results = [
            _rank_context(
                query_vector,
                chunks,
                chunk_vectors,
                question,
                _vector_sha256(query_vector),
                top_k,
            )
            for question, query_vector in zip(questions, query_vectors)
        ]
        strategy_results[strategy_key] = {
            "label": config["strategies"][strategy_key]["label"],
            "chunking_configuration": copy.deepcopy(
                config["strategies"][strategy_key]
            ),
            "chunk_count": len(chunks),
            "chunks": [
                {
                    **chunk.to_dict(),
                    "embedding_sha256": _vector_sha256(vector),
                }
                for chunk, vector in zip(chunks, chunk_vectors)
            ],
            "questions": questions_results,
            "chunk_statistics": {
                "count": len(chunks),
                "average_characters": float(
                    np.mean([len(chunk.content) for chunk in chunks])
                ),
                "minimum_characters": min(len(chunk.content) for chunk in chunks),
                "maximum_characters": max(len(chunk.content) for chunk in chunks),
            },
        }

    timestamp = datetime.now(timezone.utc)
    created_at = timestamp.isoformat()
    config_sha256 = _sha256(config_bytes)
    run_signature = _sha256(
        _canonical_json(
            {
                "configuration_sha256": config_sha256,
                "source_sha256": _sha256(source_bytes),
                "questions_sha256": _sha256(questions_bytes),
            }
        )
    )
    run_record = {
        "schema_version": 2,
        "run_signature_sha256": run_signature,
        "created_at_utc": created_at,
        "status": "retrieval_complete_manual_evaluation_pending",
        "configuration": copy.deepcopy(config),
        "configuration_sha256": config_sha256,
        "source_document": {
            "path": config["source_document"],
            "sha256": _sha256(source_bytes),
            "bytes": len(source_bytes),
            "characters": len(source),
            "lines": _source_line_count(source),
        },
        "question_set": {
            "path": config["questions_file"],
            "sha256": _sha256(questions_bytes),
            "count": len(questions),
            "items": copy.deepcopy(questions),
        },
        "embedding_pipeline": {
            **model.metadata(),
            "query_embeddings_generated_once_and_reused_for_both_strategies": True,
            "unique_input_text_count": len(unique_texts),
        },
        "retrieval_configuration": {
            "similarity": SIMILARITY,
            "top_k": config["top_k"],
            "tie_break": "source chunk order (ascending chunk index)",
            "reranking": False,
            "similarity_threshold": None,
        },
        "answering_and_evaluation": {
            "answer_generation": "not performed",
            "manual_evaluation_procedure": EVALUATION_PROCEDURE,
            "source_restriction": (
                "Judge only against the supplied source document; do not use "
                "external medical facts."
            ),
            "assessment_fields": [
                "assessment",
                "justification",
            ],
            "allowed_assessments": ["YES", "PARTIAL", "NO"],
            "rubric": {
                "YES": f"The top-{top_k} context contains sufficient source-supported evidence.",
                "PARTIAL": f"The top-{top_k} context contains some relevant but incomplete evidence.",
                "NO": f"The top-{top_k} context cannot support an answer from the source.",
            },
            "accuracy_rule": (
                "Percentage of the 10 questions assessed YES for each strategy; "
                "PARTIAL and NO are not counted as YES."
            ),
        },
        "strategies": strategy_results,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"rag_run_{timestamp.strftime('%Y%m%dT%H%M%S.%fZ')}_"
        f"{run_signature[:12]}.json"
    )
    output_path = results_dir / filename
    if output_path.exists():
        raise ExperimentError(f"Refusing to overwrite existing results: {output_path}")
    temporary_path = output_path.with_suffix(".json.tmp")
    try:
        temporary_path.write_bytes(_canonical_json(run_record) + b"\n")
        os.replace(temporary_path, output_path)
    except OSError as exc:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise ExperimentError(f"Cannot write experiment results: {exc}") from exc
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the controlled A/B RAG chunking retrieval experiment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"JSON experiment configuration (default: {DEFAULT_CONFIG_PATH})",
    )
    args = parser.parse_args(argv)

    try:
        output_path = run_experiment(args.config)
    except ExperimentError as exc:
        print(f"Experiment not run: {exc}", file=sys.stderr)
        return 2
    print(f"Experiment results written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
