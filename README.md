# Controlled RAG chunking experiment

## Scope and method

This retrieval-only experiment uses exactly `data/diabetes_reference_document.md`
and the ten official questions in `data/questions.json`. No answer-generation
model, external medical source, reranker, or similarity threshold is used.

Both strategies use `sentence-transformers/all-MiniLM-L6-v2`, pinned to revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, NumPy cosine similarity, and top-k
3. This environment could not resolve the `sentence-transformers` Python
package, so inference uses the **official ONNX graph and tokenizer from the same
model repository and revision**. The model's configured 256-token limit,
attention-masked mean pooling, and L2 normalization are applied. Inference runs
on ONNX Runtime's CPU provider with one thread; queries and chunks share the
same pipeline.

- **A — Basic / Naive:** ordinary 500-character slices with 50-character overlap.
- **B — Advanced / Structure-aware:** Markdown heading-defined sections are
  preserved as chunks. The configured size cap is `null`, so sections remain
  intact; paragraph/sentence splitting is not triggered.

Manual assessments judge only whether each retrieved top-three context supports
an answer from the reference: YES, PARTIAL, or NO. Accuracy is YES count divided
by 10; PARTIAL and NO do not count as YES. Q9 is marked NO because the reference
does not provide the requested adult blood-glucose target range. No outside
range is substituted.

## Summary

| Strategy | Chunks | Mean characters | Min–max characters | YES | PARTIAL | NO | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| A — Basic / Naive | 17 | 490.29 | 335–500 | 7 | 1 | 2 | 70% |
| B — Advanced / Structure-aware | 16 | 470.94 | 21–888 | 8 | 1 | 1 | 80% |

## Question-by-question results

| ID | Official question | A | B |
|---|---|---|---|
| Q1 | What are the two main diagnostic criteria for Type 2 diabetes based on HbA1c and fasting plasma glucose? | YES | YES |
| Q2 | What HbA1c range indicates prediabetes? | YES | YES |
| Q3 | Name three first-line oral medications used to treat Type 2 diabetes. | PARTIAL | PARTIAL |
| Q4 | What are the classic symptoms of hyperglycemia? | YES | YES |
| Q5 | What lifestyle modifications are recommended for managing Type 2 diabetes? | YES | YES |
| Q6 | Name two microvascular complications of diabetes. | YES | YES |
| Q7 | Name two macrovascular complications of diabetes. | NO | YES |
| Q8 | How often should HbAlc be monitored in someone with stable Type 2 diabetes? | YES | YES |
| Q9 | What blood glucose target range is recommended for most adults with Type 2 diabetes? | NO | NO |
| Q10 | What symptoms should prompt someone to seek emergency medical care? | YES | YES |

Top-three chunk IDs and cosine scores for all 20 evaluations are in
also contains every retrieved chunk's text, source offsets, heading path,
similarity score, embedding fingerprints, exact assembled context, manual
justification, and model asset hashes.

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

Full retrieved chunk text, chunk IDs, similarity scores, offsets, and heading paths are preserved in the final run JSON files. The source-grounded manual judgments are recorded in `results/manual_evaluation.md`.

### Final runs

- `results/rag_run_20261006T123036.769040Z_cc70ba5ff65d.json` — Strategy A/B, k=3
- `results/rag_run_20261006T123055.709077Z_f49b824d1ec6.json` — Strategy A/B, k=5

## Run

From the project root:

```bash
python3 src/rag.py --config config/experiment.json
python3 src/rag.py --config config/experiment_k5.json
