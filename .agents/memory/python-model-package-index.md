---
name: Python model package index
description: Handles Replit's Python resolver routing for embedding-model packages that are absent from the configured CPU wheel index.
---

In this workspace, the Python package resolver routed `sentence-transformers` and `transformers` to the configured PyTorch CPU wheel index, which had no compatible Linux distributions for those packages. The supported package installer continued to regenerate these source mappings after edits.

**Why:** Repeated package resolution failed during a controlled run that required one exact embedding model. The model's own repository provides an official ONNX transformer graph, tokenizer, and SentenceTransformer pooling configuration, so that route preserved the requested model without substituting another model or bypassing package controls.

**How to apply:** If this resolver limitation recurs, verify that the exact model revision includes an ONNX graph and tokenizer plus its pooling/normalization settings. Use that same revision with ONNX Runtime, its tokenizer, masked mean pooling, and L2 normalization; record the revision and asset hashes.
