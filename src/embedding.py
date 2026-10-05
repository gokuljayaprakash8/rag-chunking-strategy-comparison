"""Deterministic local embedding pipeline for the pinned MiniLM model revision."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_FILES = {
    "onnx_model": "onnx/model.onnx",
    "tokenizer": "tokenizer.json",
    "sentence_config": "sentence_bert_config.json",
    "modules": "modules.json",
    "pooling_config": "1_Pooling/config.json",
    "model_config": "config.json",
}


class EmbeddingError(RuntimeError):
    """Raised when the pinned model or its expected pipeline cannot be loaded."""


def _download_asset(filename: str) -> Path:
    cache_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{MODEL_ID.replace('/', '--')}"
        / "snapshots"
        / MODEL_REVISION
    )
    cache_path = cache_dir / filename
    if cache_path.is_file() and cache_path.stat().st_size:
        return cache_path.resolve()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://huggingface.co/{MODEL_ID}/resolve/"
        f"{MODEL_REVISION}/{filename}"
    )
    request = Request(
        url,
        headers={"User-Agent": "controlled-rag-chunking-experiment/1.0"},
    )
    temporary_path = cache_path.with_name(cache_path.name + ".download")
    try:
        with urlopen(request, timeout=300) as response, temporary_path.open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        if temporary_path.stat().st_size == 0:
            raise EmbeddingError(f"Downloaded an empty model asset: {filename}.")
        os.replace(temporary_path, cache_path)
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        if isinstance(exc, EmbeddingError):
            raise
        raise EmbeddingError(
            f"Could not download pinned model asset {filename!r}: {exc}"
        ) from exc
    return cache_path


class SentenceTransformerOnnx:
    """Run the official model ONNX graph and its configured pooling modules."""

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        revision: str = MODEL_REVISION,
        batch_size: int = 32,
        max_seq_length: int = 256,
        device: str = "CPUExecutionProvider",
    ) -> None:
        if model_id != MODEL_ID:
            raise EmbeddingError(f"Only the required model {MODEL_ID!r} is allowed.")
        if revision != MODEL_REVISION:
            raise EmbeddingError(
                f"Only the pinned model revision {MODEL_REVISION!r} is allowed."
            )
        if batch_size != 32 or max_seq_length != 256:
            raise EmbeddingError("The embedding batch and sequence lengths are fixed.")
        if device != "CPUExecutionProvider":
            raise EmbeddingError("The embedding provider is fixed to CPU.")

        self.model_id = model_id
        self.revision = revision
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.device = device

        self.asset_paths = {
            key: _download_asset(filename)
            for key, filename in MODEL_FILES.items()
        }

        self._validate_sentence_transformer_config()
        self.tokenizer = Tokenizer.from_file(str(self.asset_paths["tokenizer"]))
        pad_id = self.tokenizer.token_to_id("[PAD]")
        if pad_id is None:
            raise EmbeddingError("The pinned tokenizer has no [PAD] token.")
        self.tokenizer.enable_truncation(
            max_length=self.max_seq_length,
            stride=0,
            strategy="longest_first",
            direction="right",
        )
        self.tokenizer.enable_padding(
            direction="right",
            pad_id=pad_id,
            pad_type_id=0,
            pad_token="[PAD]",
        )

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            self.session = ort.InferenceSession(
                str(self.asset_paths["onnx_model"]),
                sess_options=options,
                providers=[self.device],
            )
        except Exception as exc:
            raise EmbeddingError(f"Could not load the official ONNX graph: {exc}") from exc

        self.input_names = {item.name for item in self.session.get_inputs()}
        self.output_names = {item.name for item in self.session.get_outputs()}
        if not {"input_ids", "attention_mask"}.issubset(self.input_names):
            raise EmbeddingError(
                f"Unexpected ONNX inputs: {sorted(self.input_names)}."
            )
        if "last_hidden_state" not in self.output_names:
            raise EmbeddingError(
                f"Expected last_hidden_state output, got {sorted(self.output_names)}."
            )

    def _validate_sentence_transformer_config(self) -> None:
        try:
            sentence_config = json.loads(
                self.asset_paths["sentence_config"].read_text(encoding="utf-8")
            )
            modules = json.loads(
                self.asset_paths["modules"].read_text(encoding="utf-8")
            )
            pooling = json.loads(
                self.asset_paths["pooling_config"].read_text(encoding="utf-8")
            )
            model_config = json.loads(
                self.asset_paths["model_config"].read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EmbeddingError(f"Invalid model configuration asset: {exc}") from exc

        if sentence_config.get("max_seq_length") != self.max_seq_length:
            raise EmbeddingError(
                "The pinned model's max_seq_length differs from the configured value."
            )
        module_types = [
            entry.get("type")
            for entry in sorted(modules, key=lambda item: item.get("idx", -1))
        ]
        expected_types = [
            "sentence_transformers.models.Transformer",
            "sentence_transformers.models.Pooling",
            "sentence_transformers.models.Normalize",
        ]
        if module_types != expected_types:
            raise EmbeddingError(
                f"Unexpected SentenceTransformer module sequence: {module_types!r}."
            )
        if pooling.get("pooling_mode_mean_tokens") is not True:
            raise EmbeddingError("The pinned model is not configured for mean pooling.")
        if any(
            pooling.get(key) is not False
            for key in (
                "pooling_mode_cls_token",
                "pooling_mode_max_tokens",
                "pooling_mode_mean_sqrt_len_tokens",
            )
        ):
            raise EmbeddingError("The pinned model has unexpected pooling modes.")
        if model_config.get("hidden_size") != 384:
            raise EmbeddingError("The pinned model hidden size is not 384.")

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized masked-mean embeddings in input order."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        batches: list[np.ndarray] = []
        for offset in range(0, len(texts), self.batch_size):
            batch_texts = texts[offset : offset + self.batch_size]
            encodings = self.tokenizer.encode_batch(batch_texts)
            input_ids = np.asarray([item.ids for item in encodings], dtype=np.int64)
            attention_mask = np.asarray(
                [item.attention_mask for item in encodings], dtype=np.int64
            )
            token_type_ids = np.asarray(
                [item.type_ids for item in encodings], dtype=np.int64
            )
            available_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
            feeds = {
                name: available_inputs[name]
                for name in self.input_names
                if name in available_inputs
            }
            if feeds.keys() != self.input_names:
                raise EmbeddingError(
                    f"Unsupported ONNX input names: {sorted(self.input_names)}."
                )
            try:
                token_embeddings = self.session.run(
                    ["last_hidden_state"], feeds
                )[0]
            except Exception as exc:
                raise EmbeddingError(f"ONNX inference failed: {exc}") from exc

            mask = attention_mask.astype(np.float32, copy=False)
            denominator = mask.sum(axis=1, keepdims=True)
            if np.any(denominator <= 0):
                raise EmbeddingError("Tokenizer produced an empty attention mask.")
            pooled = (token_embeddings * mask[:, :, None]).sum(axis=1) / denominator
            norms = np.linalg.norm(pooled, ord=2, axis=1, keepdims=True)
            if np.any(norms <= 0):
                raise EmbeddingError("The model produced a zero embedding.")
            normalized = pooled / np.maximum(norms, 1e-12)
            if not np.isfinite(normalized).all():
                raise EmbeddingError("The model produced non-finite embeddings.")
            batches.append(normalized.astype(np.float32, copy=False))

        return np.concatenate(batches, axis=0)

    def metadata(self) -> dict[str, Any]:
        assets: dict[str, dict[str, Any]] = {}
        for key, path in self.asset_paths.items():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assets[MODEL_FILES[key]] = {
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        return {
            "model": self.model_id,
            "revision": self.revision,
            "backend": "official repository ONNX transformer graph",
            "provider": self.device,
            "batch_size": self.batch_size,
            "max_seq_length": self.max_seq_length,
            "tokenization": (
                "Pinned tokenizer.json; right truncation at max_seq_length; "
                "right padding to each batch's longest sequence."
            ),
            "pooling_and_normalization": (
                "Attention-mask-weighted mean pooling over non-padding tokens, "
                "followed by L2 normalization, matching modules.json and "
                "1_Pooling/config.json."
            ),
            "determinism": (
                "CPUExecutionProvider, one intra-op and inter-op thread, "
                "sequential ONNX execution; no sampling or random seed."
            ),
            "libraries": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "onnxruntime": ort.__version__,
                "tokenizers": metadata_version("tokenizers"),
            },
            "assets": assets,
        }


def metadata_version(package: str) -> str:
    """Read installed package metadata without importing another dependency."""
    from importlib.metadata import version

    return version(package)
