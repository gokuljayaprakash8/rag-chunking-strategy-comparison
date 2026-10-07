# Manual Source-Grounded Evaluation

Evaluation rule: a question is marked YES when the retrieved top-k chunks contain the correct answer in the supplied reference document. PARTIAL means the retrieved source contains only part of the requested information or the reference itself provides only partial information. NO means the retrieved top-k chunks do not contain the requested answer, or the reference document does not provide it.

## k=3

| Question | Strategy A | Strategy B | Notes |
|---|---|---|---|
| Q1 | YES | YES | Diagnostic evidence is retrieved. |
| Q2 | YES | YES | Prediabetes HbA1c evidence is retrieved. |
| Q3 | PARTIAL | PARTIAL | The reference identifies metformin as usually first-line but does not explicitly provide three named first-line oral medications. |
| Q4 | YES | YES | Symptom evidence is retrieved. |
| Q5 | YES | PARTIAL | Strategy A rank 1 holds the heading and the full lifestyle list. Strategy B's top-3 holds the heading-plus-introduction chunk (rank 1) but not the bullet-list chunk (B-0011, rank 4), so only 2 of the 5 lifestyle items are retrieved. |
| Q6 | YES | YES | Microvascular complication evidence is retrieved. |
| Q7 | NO | YES | Strategy A does not retrieve the macrovascular evidence in the top-3; Strategy B retrieves it at rank 2. |
| Q8 | YES | YES | Monitoring frequency is retrieved in the Monitoring and Follow-up evidence. |
| Q9 | NO | NO | The reference document does not provide the requested blood-glucose target range. |
| Q10 | YES | YES | Emergency-care evidence is retrieved. |

**k=3 summary:** A = 7/10 (70%); B = 7/10 (70%). The strategies tie; each wins one question (B wins Q7, A wins Q5).

## k = 5

| Question | Strategy A | Strategy B | Notes |
|---|---|---|---|
| Q1 | YES | YES | Diagnostic evidence is retrieved. |
| Q2 | YES | YES | Prediabetes HbA1c evidence is retrieved. |
| Q3 | PARTIAL | PARTIAL | Same source limitation as k=3. |
| Q4 | YES | YES | Symptom evidence is retrieved. |
| Q5 | YES | YES | Strategy B's list chunk (B-0011) enters the top-5 at rank 4. |
| Q6 | YES | YES | Microvascular complication evidence is retrieved. |
| Q7 | YES | YES | Strategy A retrieves the macrovascular evidence at rank 5; Strategy B retrieves it at rank 2. |
| Q8 | YES | YES | Strategy A retrieves the monitoring-frequency evidence at rank 1; Strategy B also retrieves it at rank 1. |
| Q9 | NO | NO | Source gap remains. |
| Q10 | YES | YES | Emergency-care evidence is retrieved. |

**k=5 summary:** A = 8/10 (80%); B = 8/10 (80%).

## Interpretation

At k=3 the strategies tie at 70%, but they fail on different questions. Strategy A misses Q7 because its fixed cut mixed the end of the microvascular list with the start of the macrovascular section, so the chunk holding the macrovascular list (A-0011, similarity 0.498) ranks 5th. Strategy B misses Q5 (PARTIAL) because its heading boundary separated the lifestyle introduction from its bullet list; the short introduction chunk ranks 1st and the list chunk ranks 4th.

At k=5 both strategies reach 80%: A-0011 enters A's top-5 at rank 5, and B-0011 enters B's top-5 at rank 4. Increasing k removed the observed difference.

The hypothesis that structure-aware chunking improves retrieval is not supported on this document and question set. With ten questions, one question is ten percentage points, and judging was manual and not blind to the strategy, so this is a small-sample observation.

Strategy B preserves section boundaries and is easier to inspect, but it needs a more complex parser and produces more chunks to embed and compare (24 versus 17). Increasing k also increases the amount of retrieved context passed downstream.

Q3 remains PARTIAL because the evaluation is restricted to the supplied reference document. Q9 remains NO because the requested blood-glucose target range is absent from the source; no external medical information was substituted.
