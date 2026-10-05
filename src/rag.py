#!/usr/bin/env python3
"""Controlled, retrieval-only RAG chunking comparison.

No embedding model, answer generator, or question set is selected by this
module. Fill in config/experiment.json and provide data/questions.json before
running it. The embedding adapter is a user-supplied Python callable with this
interface:

    embed_text(text, *, model, parameters, seed) -> sequence[float]

The adapter must use the requested model and parameters and honor the supplied
seed. In deterministic mode it must provide deterministic embeddings.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import inspect
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


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
EmbeddingFunction = Callable[..., Sequence[float]]


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
        or config["top_k"] != TOP_K
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
    if a["label"] != "Strategy A — Fixed-size":
        raise ExperimentError("Strategy A label must be 'Strategy A — Fixed-size'.")
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
    if b["label"] != "Strategy B — Structure-aware":
        raise ExperimentError("Strategy B label must be 'Strategy B — Structure-aware'.")
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
        value, {"model", "adapter", "parameters", "reproducibility"}, "embedding"
    )
    _non_empty_string(embedding["model"], "embedding.model")

    adapter = _non_empty_string(embedding["adapter"], "embedding.adapter")
    if adapter.count(":") != 1:
        raise ExperimentError(
            "embedding.adapter must use the explicit 'python.module:callable' form."
        )
    module_name, function_name = adapter.split(":", 1)
    if (
        not module_name
        or not function_name
        or not all(part.isidentifier() for part in module_name.split("."))
        or not function_name.isidentifier()
    ):
        raise ExperimentError(
            "embedding.adapter must use the explicit 'python.module:callable' form."
        )

    if not isinstance(embedding["parameters"], dict):
        raise ExperimentError(
            "embedding.parameters must be explicitly supplied as a JSON object "
            "(use {} only if the selected adapter needs no parameters)."
        )

    reproducibility = _require_exact_keys(
        embedding["reproducibility"], {"mode", "seed"}, "embedding.reproducibility"
    )
    mode = reproducibility["mode"]
    seed = reproducibility["seed"]
    if not isinstance(mode, str) or mode not in {"deterministic", "seeded"}:
        raise ExperimentError(
            "embedding.reproducibility.mode must be explicitly set to "
            "'deterministic' or 'seeded'."
        )
    if mode == "deterministic" and seed is not None:
        raise ExperimentError(
            "Use seed: null in deterministic mode; no random seed is applied."
        )
    if mode == "seeded" and (
        isinstance(seed, bool) or not isinstance(seed, int)
    ):
        raise ExperimentError("Seeded mode requires an explicit integer seed.")


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
                label="Strategy A — Fixed-size",
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
    source: str, start: int, end: int, max_chars: int | None
) -> list[tuple[int, int]]:
    """Keep sections intact unless an explicit size cap requires finer boundaries."""
    if max_chars is None or end - start <= max_chars:
        return [(start, end)]

    units = _paragraph_spans(source, start, end)
    output: list[tuple[int, int]] = []
    pending_start: int | None = None
    pending_end: int | None = None

    def flush_pending() -> None:
        nonlocal pending_start, pending_end
        if pending_start is not None and pending_end is not None:
            output.append((pending_start, pending_end))
        pending_start = None
        pending_end = None

    for unit_start, unit_end in units:
        if unit_end - unit_start > max_chars:
            flush_pending()
            sentences = _sentence_spans(source, unit_start, unit_end)
            sentence_start: int | None = None
            sentence_end: int | None = None
            for part_start, part_end in sentences:
                if part_end - part_start > max_chars:
                    if sentence_start is not None and sentence_end is not None:
                        output.append((sentence_start, sentence_end))
                    sentence_start = None
                    sentence_end = None
                    # Never cut an overlong sentence arbitrarily.
                    output.append((part_start, part_end))
                    continue
                if sentence_start is None:
                    sentence_start, sentence_end = part_start, part_end
                elif part_end - sentence_start <= max_chars:
                    sentence_end = part_end
                else:
                    output.append((sentence_start, sentence_end or part_end))
                    sentence_start, sentence_end = part_start, part_end
            if sentence_start is not None and sentence_end is not None:
                output.append((sentence_start, sentence_end))
            continue

        if pending_start is None:
            pending_start, pending_end = unit_start, unit_end
        elif unit_end - pending_start <= max_chars:
            pending_end = unit_end
        else:
            flush_pending()
            pending_start, pending_end = unit_start, unit_end

    flush_pending()
    return output or [(start, end)]


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
                    label="Strategy B — Structure-aware",
                    index=len(chunks) + 1,
                    start=start,
                    end=end,
                    source=source,
                    heading_path=path,
                    headings_in_chunk=included_headings,
                )
            )
    return chunks


def _seed_for_text(mode: str, base_seed: int | None, text: str) -> int | None:
    if mode == "deterministic":
        return None
    assert base_seed is not None
    material = f"{base_seed}\0{text}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _load_embedding_adapter(
    adapter_spec: str,
) -> tuple[EmbeddingFunction, JSONDict]:
    module_name, function_name = adapter_spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise ExperimentError(
            f"Cannot import embedding adapter module {module_name!r}: {exc}"
        ) from exc
    function = getattr(module, function_name, None)
    if not callable(function):
        raise ExperimentError(
            f"Embedding adapter {adapter_spec!r} is not an importable callable."
        )

    source_path: str | None = None
    source_hash: str | None = None
    try:
        adapter_file = inspect.getsourcefile(function)
        if adapter_file and Path(adapter_file).is_file():
            adapter_bytes = Path(adapter_file).read_bytes()
            source_path = str(Path(adapter_file).resolve())
            source_hash = _sha256(adapter_bytes)
    except (OSError, TypeError):
        pass
    return function, {
        "callable": adapter_spec,
        "source_file": source_path,
        "source_sha256": source_hash,
        "module_version": getattr(module, "__version__", None),
    }


def _embed(
    adapter: EmbeddingFunction,
    text: str,
    embedding_config: JSONDict,
) -> list[float]:
    reproducibility = embedding_config["reproducibility"]
    seed = _seed_for_text(
        reproducibility["mode"], reproducibility["seed"], text
    )
    try:
        raw_vector = adapter(
            text,
            model=embedding_config["model"],
            parameters=copy.deepcopy(embedding_config["parameters"]),
            seed=seed,
        )
        if isinstance(raw_vector, (str, bytes, bytearray)):
            raise TypeError("adapter must return a numeric sequence, not text or bytes")
        vector = [float(value) for value in raw_vector]
    except Exception as exc:
        raise ExperimentError(f"Embedding adapter failed: {exc}") from exc
    if not vector:
        raise ExperimentError("Embedding adapter returned an empty vector.")
    if any(not math.isfinite(value) for value in vector):
        raise ExperimentError("Embedding adapter returned a non-finite vector value.")
    return vector


def _vector_sha256(vector: Sequence[float]) -> str:
    stable = [float(value).hex() for value in vector]
    return _sha256(_canonical_json(stable))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ExperimentError(
            f"Embedding dimension mismatch ({len(left)} != {len(right)})."
        )
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ExperimentError("Cosine similarity is undefined for a zero vector.")
    score = math.fsum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    if not math.isfinite(score):
        raise ExperimentError("Cosine similarity produced a non-finite score.")
    return score


def _manual_evaluation_fields() -> JSONDict:
    return {
        "source_answerability": "not_assessed",
        "retrieved_content_support": "not_assessed",
        "retrieval_quality": "not_assessed",
        "assessor_notes": None,
        "allowed_source_answerability_values": [
            "answerable_from_source",
            "not_answerable_from_source",
            "uncertain",
            "not_assessed",
        ],
        "allowed_retrieved_content_support_values": [
            "supported_by_retrieved_content",
            "not_supported_by_retrieved_content",
            "not_assessed",
        ],
        "allowed_retrieval_quality_values": [
            "correct",
            "incorrect_or_incomplete_retrieval",
            "not_assessed",
        ],
    }


def _rank_context(
    question_vector: Sequence[float],
    chunks: list[Chunk],
    chunk_vectors: list[list[float]],
    question: dict[str, str],
    query_vector_sha256: str,
) -> JSONDict:
    ranked: list[tuple[float, int]] = []
    for index, chunk_vector in enumerate(chunk_vectors):
        ranked.append((cosine_similarity(question_vector, chunk_vector), index))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    retrieved: list[JSONDict] = []
    for rank, (score, index) in enumerate(ranked[:TOP_K], start=1):
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
        "retrieved_top_3": retrieved,
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

    adapter, adapter_metadata = _load_embedding_adapter(config["embedding"]["adapter"])

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
    if any(len(chunks) < TOP_K for chunks in chunks_by_strategy.values()):
        raise ExperimentError(
            f"Each strategy must produce at least {TOP_K} chunks for top-k retrieval."
        )

    # Query embeddings are generated once and reused unchanged for A and B.
    query_vectors = [
        _embed(adapter, question["text"], config["embedding"])
        for question in questions
    ]
    query_dimensions = {len(vector) for vector in query_vectors}
    if len(query_dimensions) != 1:
        raise ExperimentError("The embedding adapter returned inconsistent query dimensions.")

    strategy_results: JSONDict = {}
    expected_dimension = next(iter(query_dimensions))
    for strategy_key in (STRATEGY_A, STRATEGY_B):
        chunks = chunks_by_strategy[strategy_key]
        chunk_vectors = [
            _embed(adapter, chunk.content, config["embedding"]) for chunk in chunks
        ]
        if any(len(vector) != expected_dimension for vector in chunk_vectors):
            raise ExperimentError(
                "Query and document embeddings must have identical dimensions."
            )

        questions_results = [
            _rank_context(
                query_vector,
                chunks,
                chunk_vectors,
                question,
                _vector_sha256(query_vector),
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
        "schema_version": 1,
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
            "model": config["embedding"]["model"],
            "parameters": copy.deepcopy(config["embedding"]["parameters"]),
            "reproducibility": copy.deepcopy(
                config["embedding"]["reproducibility"]
            ),
            "call_contract": "embed_text(text, *, model, parameters, seed)",
            "query_embeddings_generated_once_and_reused_for_both_strategies": True,
            "seed_policy": (
                "None in deterministic mode; in seeded mode, the first 8 bytes "
                "of SHA-256(f'{base_seed}\\0{text}') as an unsigned integer."
            ),
            "adapter": adapter_metadata,
        },
        "retrieval_configuration": {
            "similarity": SIMILARITY,
            "top_k": TOP_K,
            "tie_break": "source chunk order (ascending chunk index)",
            "reranking": False,
            "similarity_threshold": None,
        },
        "answering_and_evaluation": {
            "answer_generation": "not performed",
            "manual_evaluation_procedure": EVALUATION_PROCEDURE,
            "assessment_fields": [
                "source_answerability",
                "retrieved_content_support",
                "retrieval_quality",
                "assessor_notes",
            ],
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
