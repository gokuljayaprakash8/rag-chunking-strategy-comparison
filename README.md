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

B's sections average 471 characters versus 490 for A, but vary from a
21-character heading-only chunk (`## 6. Complications`) to an 888-character
section. Short heading chunks and the document-title chunk sometimes occupy a
top-three slot, while long sections can combine distinct details. A's slices
sometimes split tables and lists, although the 50-character overlap and
top-three retrieval still surface useful evidence. These results favor B on
this small, curated set, especially for retrieving a whole natural section;
they do not establish broad superiority. The heading-only chunks and Q9's
absence also expose limits in this document-specific setup.

## Run

From the project root, run `python3 src/rag.py`. The first run downloads the
pinned public model assets to the local Hugging Face cache. Each run writes a
retrieval record to `results/` with manual fields pending; assess a new run's
top-three contexts against the source before treating it as finalized. The
result linked above is the completed source-only assessment.
