# Controlled RAG chunking experiment

`src/rag.py` implements retrieval-only comparison of the same source Markdown and
the same 10 supplied questions with two chunking strategies:

- **A — Fixed-size:** exactly 500 characters with 50 characters of overlap.
- **B — Structure-aware:** preserves Markdown heading-defined sections and
  records each chunk's heading path. No maximum size is imposed unless
  `max_chunk_characters` is explicitly set; if set, splitting uses paragraph
  boundaries first and sentence boundaries only for paragraphs that exceed the
  cap. Any sentence still longer than the cap remains intact.

Both strategies use the one embedding configuration in `config/experiment.json`,
the same adapter call, cosine similarity, and top-k of 3. Query embeddings are
computed once and reused for both strategies. Retrieval has no similarity
threshold or reranker. Results record the full configuration, source and
question hashes, chunk text and character offsets, embedding fingerprints,
retrieved chunks, scores, and the exact assembled context.

## Required user-supplied inputs

The experiment intentionally cannot run with the current configuration:

1. Set `embedding.model` to the exact chosen model identifier.
2. Set `embedding.adapter` to an importable `python.module:callable`.
3. Supply the adapter's explicit `parameters` object. Use `{}` only if the
   selected adapter requires no additional parameters.
4. Set reproducibility mode to `deterministic` with `seed: null`, or to `seeded`
   with an explicit integer seed. The adapter must honor the supplied model,
   parameters, and seed.
5. Create `data/questions.json` as a JSON array containing exactly 10 strings
   in official assignment order. Enter each question verbatim; do not change
   Question 3's wording.

No question examples or answer text are included. The framework does not generate
answers because no answer-generation model or procedure was supplied. It
preserves retrieved context for manual source-grounded assessment, with separate
fields for source answerability, support by retrieved content, and incorrect or
incomplete retrieval.

Run from the project root with:

```sh
python src/rag.py
```

The command validates all inputs before embedding anything and writes one
auditable JSON run record under `results/` only after both strategies complete.
The Markdown source is read as UTF-8 without newline conversion and is never
modified.
