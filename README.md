# Controlled RAG Chunking Strategy Comparison

## Hypothesis

Preserving document structure and semantic boundaries will improve retrieval of relevant evidence compared with fixed-size chunking when retrieval depth is held constant.

## Scope and method

This project compares a basic fixed-size RAG chunking strategy with a structure-aware chunking strategy using the same source document, questions, embedding model, similarity calculation and retrieval procedure.

This is a retrieval-only experiment. No answer-generation model, external medical source, reranker or similarity threshold is used.

Fixed configuration:

- Source: `data/diabetes_reference_document.md` (7,535 characters)
- Questions: the exact 10 questions in `data/questions.json`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- Inference: official ONNX graph and tokenizer from the same model revision (the `sentence-transformers` package was unavailable in the environment)
- Token limit: 256 tokens; attention-masked mean pooling; L2 normalization
- Similarity: NumPy cosine similarity
- Retrieval depths: k=3 and k=5
- Reranker: none. Similarity threshold: none. Answer generation: none.

## Chunking strategies

**Strategy A, Basic / Naive.** Fixed 500-character chunks with a 50-character overlap (step 450). This follows the assignment's example baseline and ignores document structure.

**Strategy B, Structure-Aware.** The document is split hierarchically: section (Markdown heading), then paragraph (blank line), then sentence, with a 500-character target.

- A heading stays attached to the first unit under it. It is never emitted as its own chunk.
- If a section fits within 500 characters it stays whole. Otherwise paragraphs are packed up to the target. A paragraph that is too long is split at sentence boundaries.
- A single sentence is never cut, so a chunk can exceed 500 characters (maximum 590).

I chose this approach because the reference document is organised into numbered sections, so headings are the author's own topic boundaries. It is deterministic and easy to inspect, and it avoids adding a second model or another similarity-based decision as an extra variable.

## Evaluation

Each retrieved top-k context was read and judged against the supplied reference:

- **YES**: the retrieved text contains sufficient evidence for the full requested answer.
- **PARTIAL**: it contains only part of the requested information.
- **NO**: it cannot support the answer, or the reference does not contain it.

Accuracy is YES count divided by 10. PARTIAL and NO do not count as correct. Judging was manual and not blind to the strategy.

Two source limitations apply:

- **Q3** is PARTIAL for both strategies: the reference names metformin as usually first-line but does not give three first-line oral drugs.
- **Q9** is NO for both: the reference does not give the requested target glucose range.

No outside medical information was used.

## Results

| Strategy | k | Chunks | Mean chars | Min–max chars | YES | PARTIAL | NO | Accuracy | Answerable-only (Q9 excluded) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A: Basic / Naive | 3 | 17 | 490.29 | 335–500 | 7 | 1 | 2 | 70% | 7/9 |
| B: Structure-Aware | 3 | 24 | 309.50 | 74–590 | 7 | 2 | 1 | 70% | 7/9 |
| A: Basic / Naive | 5 | 17 | 490.29 | 335–500 | 8 | 1 | 1 | 80% | 8/9 |
| B: Structure-Aware | 5 | 24 | 309.50 | 74–590 | 8 | 1 | 1 | 80% | 8/9 |

| Question | A, k=3 | B, k=3 | A, k=5 | B, k=5 |
|---|---|---|---|---|
| Q1 Diagnostic criteria | YES | YES | YES | YES |
| Q2 Prediabetes HbA1c range | YES | YES | YES | YES |
| Q3 Three first-line oral medications | PARTIAL | PARTIAL | PARTIAL | PARTIAL |
| Q4 Classic hyperglycemia symptoms | YES | YES | YES | YES |
| Q5 Lifestyle modifications | YES | PARTIAL | YES | YES |
| Q6 Microvascular complications | YES | YES | YES | YES |
| Q7 Macrovascular complications | NO | YES | YES | YES |
| Q8 HbA1c monitoring frequency | YES | YES | YES | YES |
| Q9 Blood-glucose target range | NO | NO | NO | NO |
| Q10 Emergency symptoms | YES | YES | YES | YES |

## What differs, and why

At k=3 the strategies differ on two questions, in opposite directions.

- **Q7 (B wins).** Strategy A's fixed cut mixed the end of the microvascular list with the start of the macrovascular section, so the chunk holding the macrovascular list (A-0011) scored 0.498 and ranked 5th, behind a purely microvascular chunk (0.711). Strategy B kept the macrovascular section as its own chunk, which ranked 2nd. At k=5, A also retrieves it.
- **Q5 (A wins).** Strategy B split the lifestyle section into a short heading-plus-introduction chunk and a separate bullet-list chunk. The short chunk ranked first without the list, and the list chunk ranked 4th, so B's top 3 contained only two of the five lifestyle items. Strategy A's rank-1 chunk contained the heading and the full list. At k=5 B retrieves the list chunk.

## Analysis

The two strategies tie: 70% at k=3 and 80% at k=5. Where the cut falls decided each difference. Strategy A's fixed boundary broke the complications section, and Strategy B's heading boundary separated the lifestyle introduction from its list. The hypothesis, that structure-aware chunking would retrieve better, is therefore not supported on this document and question set.

**Trade-offs.** Strategy A is simple and format-independent, with 17 relatively large chunks, but a fixed cut can place two topics in one chunk. Strategy B keeps topics together and is easier to inspect, but it needs a more complex parser, produces more vectors to embed and compare (24 versus 17), and depends on the document having clear headings. Increasing k from 3 to 5 removed the observed difference by letting the lower-ranked chunk in, at the cost of passing more text downstream.

**Limitations.** Ten questions is a small sample: one question is ten percentage points. Judging was manual and not blind. Only one document was used. B attaches a heading only to the first paragraph under it, which can leave a short heading-plus-introduction chunk that ranks highly without containing the answer. Headings with no text of their own (the document title and `## 5.` and `## 6.`) are not emitted as chunks and appear only in each chunk's `heading_path` metadata.

**Next steps.** Attach the heading to every sub-chunk, set a minimum chunk size, test k=1 and k=2, and add more questions and documents.

## Reproduction

```bash
python3 src/rag.py --config config/experiment.json
python3 src/rag.py --config config/experiment_k5.json
```

Both configurations use the same document, questions, embedding model and evaluation procedure; only the retrieval depth differs.

Final run files:

- `results/rag_run_20261007T062827.360367Z_cc70ba5ff65d.json`: k=3
- `results/rag_run_20261007T062905.399087Z_f49b824d1ec6.json`: k=5

Each run file records the retrieved chunk text, chunk IDs, source offsets, heading paths, similarity scores and model fingerprints. The run files mark manual evaluation as `not_assessed`; my judgments are in `results/manual_evaluation.md`.
