# Controlled RAG chunking experiment

This is a retrieval-only comparison over `data/diabetes_reference_document.md`
and the ten verbatim questions in `data/questions.json`. No answer generation or
outside medical source is used.

## Fixed configuration

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`, pinned to the
  revision recorded in `config/experiment.json`. The official repository ONNX
  transformer graph and tokenizer are used with the model's configured
  attention-masked mean pooling and L2 normalization.
- **Similarity:** NumPy cosine similarity; **top-k:** 3; no reranker or threshold.
- **A — Basic / Naive:** ordinary 500-character slicing with 50-character overlap.
- **B — Advanced / Structure-aware:** Markdown heading-defined sections are
  preserved as chunks. The configured size cap is `null`, so each natural
  section remains intact; paragraphs and sentences are only split if a cap is
  deliberately configured.
- Query embeddings are generated once and reused for both strategies. The model
  runs locally through ONNX Runtime's CPU provider.

Each run writes a machine-readable JSON record to `results/`, including source
and question hashes, chunk contents and offsets, model asset hashes, top-three
IDs and scores, and the exact retrieved context. The run file is manually
annotated against the source after retrieval. Q9 is recorded as absent from the
reference; no outside glucose target is supplied.

Run from the project root:

```sh
python src/rag.py
```

The final question-by-question comparison, accuracy calculation, and analysis
are in `README.md`.
