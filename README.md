# Controlled RAG Chunking Strategy Comparison

## Scope and method

This retrieval-only experiment compares a basic fixed-size RAG chunking strategy with a structure-aware strategy while keeping the source document, questions, embedding model, similarity calculation, and retrieval procedure fixed.

The source is `data/diabetes_reference_document.md` and the ten official assignment questions are in `data/questions.json`. No answer-generation model, external medical source, reranker, or similarity threshold is used.

Both strategies use `sentence-transformers/all-MiniLM-L6-v2`, pinned to revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Inference uses the official ONNX graph and tokenizer from the same model repository and revision because the `sentence-transformers` Python package was unavailable in the environment. The configured 256-token limit, attention-masked mean pooling, and L2 normalization are applied. Inference runs with ONNX Runtime on CPU using one thread. NumPy cosine similarity is used for retrieval.

Two retrieval depths are evaluated: k=3 and k=5. Query embeddings are generated once and reused.

## Strategy A — Basic / Naive

The document is divided into fixed 500-character chunks with a 50-character overlap. This follows the assignment's example baseline and deliberately does not consider semantic boundaries.

## Strategy B — Advanced / Structure-Aware

The document is parsed hierarchically using:

1. headings/sections as the highest-level boundaries;
2. paragraphs as the next boundary;
3. sentences only when a paragraph must be divided to stay near the 500-character target.

Headings are retained with their associated content, and heading-only fragments are not emitted. A semantic unit can exceed the target when necessary rather than being split arbitrarily.

This approach was chosen to test whether preserving document structure can improve retrieval without introducing a more complicated semantic-similarity chunking algorithm. It is deterministic, reproducible, and straightforward to inspect and explain.

## Evaluation

Evaluation is retrieval-only and source-grounded. For each question, the retrieved context is manually classified as:

- **YES** — sufficient evidence to support the requested answer from the supplied reference.
- **PARTIAL** — relevant but incomplete evidence.
- **NO** — the supplied reference cannot support the requested answer.

Accuracy is calculated as YES count divided by 10. PARTIAL and NO do not count as correct.

Q3 is PARTIAL for both strategies because the supplied reference identifies metformin as usually first-line but does not provide three named first-line oral medications. Q9 is NO for both because the supplied reference does not provide the requested adult blood-glucose target range. No outside medical information was substituted for either source limitation.

## Summary

| Strategy | k | Chunks | Mean characters | Min–max characters | YES | PARTIAL | NO | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A — Basic / Naive | 3 | 17 | 490.29 | 335–500 | 7 | 1 | 2 | 70% |
| B — Structure-Aware | 3 | 26 | 285.69 | 16–590 | 8 | 1 | 1 | 80% |
| A — Basic / Naive | 5 | 17 | 490.29 | 335–500 | 8 | 1 | 1 | 80% |
| B — Structure-Aware | 5 | 26 | 285.69 | 16–590 | 8 | 1 | 1 | 80% |

## Question-by-question results

| ID | Official question | A k=3 | B k=3 | A k=5 | B k=5 |
|---|---|---|---|---|---|
| Q1 | What are the two main diagnostic criteria for Type 2 diabetes based on HbA1c and fasting plasma glucose? | YES | YES | YES | YES |
| Q2 | What HbA1c range indicates prediabetes? | YES | YES | YES | YES |
| Q3 | Name three first-line oral medications used to treat Type 2 diabetes. | PARTIAL | PARTIAL | PARTIAL | PARTIAL |
| Q4 | What are the classic symptoms of hyperglycemia? | YES | YES | YES | YES |
| Q5 | What lifestyle modifications are recommended for managing Type 2 diabetes? | YES | YES | YES | YES |
| Q6 | Name two microvascular complications of diabetes. | YES | YES | YES | YES |
| Q7 | Name two macrovascular complications of diabetes. | NO | YES | YES | YES |
| Q8 | How often should HbAlc be monitored in someone with stable Type 2 diabetes? | YES | YES | YES | YES |
| Q9 | What blood glucose target range is recommended for most adults with Type 2 diabetes? | NO | NO | NO | NO |
| Q10 | What symptoms should prompt someone to seek emergency medical care? | YES | YES | YES | YES |

## Analysis

Strategy B was chosen to test whether respecting document structure improves retrieval compared with arbitrary character boundaries. The hierarchy is deliberately simple: headings define sections, paragraphs are the next boundary, and sentences are used only when a paragraph must be divided to stay near the 500-character target. This preserves semantic relationships while keeping the experiment reproducible and interpretable.

The main trade-off is complexity versus semantic coherence. Strategy A is simpler to implement and produces 17 chunks averaging 490.29 characters. Strategy B produces 26 chunks averaging 285.69 characters because it preserves meaningful boundaries rather than filling each chunk to a fixed character boundary. The additional chunks increase the number of embeddings and cosine comparisons required during retrieval. The structure-aware parser is also more complex because it must identify headings, paragraphs, and sentence boundaries.

At k=3, Strategy B reaches 80% retrieval accuracy compared with 70% for Strategy A. Q7 is the clearest difference: Strategy B retrieves the macrovascular-complications evidence at rank 2, while Strategy A does not retrieve it within the top three. At k=5, both strategies reach 80%. Strategy A eventually retrieves the same macrovascular evidence at rank 5. Therefore, the k=3 difference should not be interpreted as proof that Strategy B is universally superior.

The result demonstrates a trade-off rather than a universal winner: structure-aware chunking improved retrieval at k=3 on this ten-question evaluation, while increasing chunk count, processing work, and parser complexity. Increasing k to 5 removed the observed accuracy difference in this experiment.

## Results

### Chunk statistics

| Strategy | k | Chunks | Mean chars | Min | Max | YES | PARTIAL | NO | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A — Basic / Naive | 3 | 17 | 490.29 | 335 | 500 | 7 | 1 | 2 | 70% |
| B — Structure-Aware | 3 | 26 | 285.69 | 16 | 590 | 8 | 1 | 1 | 80% |
| A — Basic / Naive | 5 | 17 | 490.29 | 335 | 500 | 8 | 1 | 1 | 80% |
| B — Structure-Aware | 5 | 26 | 285.69 | 16 | 590 | 8 | 1 | 1 | 80% |

### Question-by-question retrieval results

| Question | A k=3 | B k=3 | A k=5 | B k=5 |
|---|---|---|---|---|
| Q1 Diagnostic criteria | YES | YES | YES | YES |
| Q2 Prediabetes HbA1c range | YES | YES | YES | YES |
| Q3 Three first-line oral medications | PARTIAL | PARTIAL | PARTIAL | PARTIAL |
| Q4 Classic hyperglycemia symptoms | YES | YES | YES | YES |
| Q5 Lifestyle modifications | YES | YES | YES | YES |
| Q6 Microvascular complications | YES | YES | YES | YES |
| Q7 Macrovascular complications | NO | YES | YES | YES |
| Q8 HbA1c monitoring frequency | YES | YES | YES | YES |
| Q9 Blood glucose target range | NO | NO | NO | NO |
| Q10 Emergency symptoms | YES | YES | YES | YES |

Full retrieved chunk text, chunk IDs, similarity scores, source offsets, heading paths, embedding fingerprints, exact assembled context, manual justification, and model asset hashes are preserved in the final run JSON files. The source-grounded manual judgments are recorded in `results/manual_evaluation.md`.

## Final runs

- `results/rag_run_20261006T134057.247099Z_cc70ba5ff65d.json` — Strategy A/B, k=3
- `results/rag_run_20261006T134136.403182Z_f49b824d1ec6.json` — Strategy A/B, k=5

## Reproduction

From the project root:

```bash
python3 src/rag.py --config config/experiment.json
python3 src/rag.py --config config/experiment_k5.json