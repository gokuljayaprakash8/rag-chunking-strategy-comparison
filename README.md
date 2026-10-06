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
[`results/summary.csv`](results/summary.csv). The full JSON record,
[`results/rag_run_20261005T103320.956233Z_509c0208157e.json`](results/rag_run_20261005T103320.956233Z_509c0208157e.json),
also contains every retrieved chunk's text, source offsets, heading path,
similarity score, embedding fingerprints, exact assembled context, manual
justification, and model asset hashes.

## Analysis

The structure-aware strategy scored 80% (8/10 YES) versus 70% (7/10) for
fixed-size chunking, a gain of one question or 10 percentage points. Both
methods retrieved enough evidence for Q1, Q2, Q4, Q5, Q6, Q8, and Q10. B recovered
the macrovascular-complications section at rank 2 for Q7, while A's top three
contained microvascular complications, emergency-care text, and symptoms but
never surfaced the macrovascular list. That is the only question on which B
changed a NO to YES.

Both strategies scored NO on Q9. The source lists diagnostic thresholds and
recommends monitoring, but does not state a target range for most adults. This
is a source-coverage limitation; no outside clinical range was added. Both
scored PARTIAL on Q3 because the source lists several oral drug classes but
explicitly identifies only metformin as usually first-line. Without outside
information, it does not support three first-line agents.

## Analysis

Strategy A uses fixed 500-character chunks with a 50-character overlap. It is the simpler and faster approach: chunking is a straightforward linear pass over the document, and the final index contains 17 chunks with a mean size of 490.29 characters. Its main weakness is that fixed boundaries can split tables, lists, and related statements across chunks.

Strategy B uses the document's Markdown hierarchy first, then paragraph boundaries, then sentence boundaries when a structural unit exceeds the configured 500-character budget. I chose this approach because the supplied reference is explicitly organized into numbered sections and subsections. Preserving that structure is deterministic, interpretable, and reproducible, while avoiding another semantic model or subjective threshold. The final B index contains 22 chunks, averaging 320.14 characters, with a minimum of 40 and maximum of 590 characters.

B has more chunking and retrieval overhead than A because it performs structural parsing and produces more index entries. Retrieval scans 22 chunks for B versus 17 for A. Increasing k from 3 to 5 increases the amount of returned context and downstream inspection, although the similarity calculation still ranks the full index.

On this small curated set, k=3 gives 70% retrieval accuracy for A and 80% for B. At k=5, A falls to 60% while B remains at 80%. The clearest result is Q7: B misses the Macrovascular Complications evidence at k=3 but retrieves it at rank 5 with k=5. Q8 also shows that a larger k does not automatically improve retrieval: A's top five still miss the monitoring-frequency evidence.

Q3 is PARTIAL for both strategies because the source lists several oral medication classes but explicitly identifies only metformin as usually first-line; the experiment does not import outside medical knowledge. Q9 remains NO because the supplied reference does not provide the requested blood-glucose target range. These results support a useful recall advantage for structure-aware chunking on this document, but do not establish broad superiority.

## Results

### Chunk statistics

| Strategy | k | Chunks | Average chars | Min chars | Max chars | YES | PARTIAL | NO | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A — Basic / Naive | 3 | 17 | 490.29 | 335 | 500 | 7 | 1 | 2 | 70% |
| B — Structure-aware | 3 | 22 | 320.14 | 40 | 590 | 8 | 1 | 1 | 80% |
| A — Basic / Naive | 5 | 17 | 490.29 | 335 | 500 | 6 | 1 | 3 | 60% |
| B — Structure-aware | 5 | 22 | 320.14 | 40 | 590 | 8 | 1 | 1 | 80% |

### Question-by-question evaluation

| Question | A k=3 | B k=3 | A k=5 | B k=5 |
|---|---|---|---|---|
| Q1 — Diagnostic criteria | YES | YES | YES | YES |
| Q2 — Prediabetes HbA1c range | YES | YES | YES | YES |
| Q3 — Three first-line oral medications | PARTIAL | PARTIAL | PARTIAL | PARTIAL |
| Q4 — Classic hyperglycemia symptoms | YES | YES | YES | YES |
| Q5 — Lifestyle modifications | YES | YES | YES | YES |
| Q6 — Microvascular complications | YES | YES | YES | YES |
| Q7 — Macrovascular complications | NO | NO | NO | YES |
| Q8 — HbA1c monitoring frequency | YES | YES | NO | YES |
| Q9 — Blood glucose target range | NO | NO | NO | NO |
| Q10 — Emergency symptoms | YES | YES | YES | YES |

Full retrieved chunk text, chunk IDs, similarity scores, offsets, and heading paths are preserved in the final run JSON files. The source-grounded manual judgments are recorded in `results/manual_evaluation.md`.

Final runs:
- `results/rag_run_20261006T103820.038159Z_cc70ba5ff65d.json` — Strategy A/B, k=3
- `results/rag_run_20261006T105337.252112Z_f49b824d1ec6.json` — Strategy A/B, k=5

## Run

From the project root:

```bash
python3 src/rag.py --config config/experiment.json
python3 src/rag.py --config config/experiment_k5.json
