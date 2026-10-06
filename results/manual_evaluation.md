# Manual Source-Grounded Evaluation

Evaluation rule: a question is marked YES when the retrieved top-k chunks contain the correct answer in the supplied reference document. PARTIAL means the retrieved source contains only part of the requested information or the reference itself provides only partial information. NO means the retrieved top-k chunks do not contain the requested answer, or the reference document does not provide it.

## k=3

| Question | Strategy A | Strategy B | Notes |
|---|---|---|---|
| Q1 | YES | YES | Diagnostic evidence is retrieved. |
| Q2 | YES | YES | Prediabetes HbA1c evidence is retrieved. |
| Q3 | PARTIAL | PARTIAL | The reference identifies metformin as usually first-line but does not explicitly provide three named first-line oral medications. |
| Q4 | YES | YES | Symptom evidence is retrieved. |
| Q5 | YES | YES | Lifestyle evidence is retrieved. |
| Q6 | YES | YES | Microvascular complication evidence is retrieved. |
| Q7 | NO | YES | Strategy A does not retrieve the macrovascular evidence in the top-3; Strategy B retrieves it at rank 2. |
| Q8 | YES | YES | Monitoring frequency is retrieved in the Monitoring and Follow-up evidence. |
| Q9 | NO | NO | The reference document does not provide the requested blood-glucose target range. |
| Q10 | YES | YES | Emergency-care evidence is retrieved. |

**k=3 summary:** A = 7/10 (70%); B = 8/10 (80%).

## k = 5

| Question | Strategy A | Strategy B | Notes |
|---|---|---|---|
| Q1 | YES | YES | Diagnostic evidence is retrieved. |
| Q2 | YES | YES | Prediabetes HbA1c evidence is retrieved. |
| Q3 | PARTIAL | PARTIAL | Same source limitation as k=3. |
| Q4 | YES | YES | Symptom evidence is retrieved. |
| Q5 | YES | YES | Lifestyle evidence is retrieved. |
| Q6 | YES | YES | Microvascular complication evidence is retrieved. |
| Q7 | YES | YES | Strategy A retrieves the macrovascular evidence at rank 5; Strategy B retrieves it at rank 2. |
| Q8 | YES | YES | Strategy A retrieves the monitoring-frequency evidence at rank 1; Strategy B also retrieves it at rank 1. |
| Q9 | NO | NO | Source gap remains. |
| Q10 | YES | YES | Emergency-care evidence is retrieved. |

**k=5 summary:** A = 8/10 (80%); B = 8/10 (80%).

## Interpretation


Increasing k from 3 to 5 improves Strategy A from 70% to 80%, while Strategy B remains at 80%. The main change is Q7: the macrovascular evidence is outside Strategy A's top-3 but appears at rank 5, while Strategy B retrieves the corresponding macrovascular chunk at rank 2 even at k=3.

The corrected Strategy B implementation preserves Markdown headings and uses a structure-aware hierarchy of section, paragraph, and sentence boundaries. This avoids heading-only chunks while keeping headings available as retrieval context.

The experiment does not show a measurable accuracy advantage for Strategy B at k=5. Its main structural benefit is preserving semantic boundaries and heading context, while its smaller chunks increase the number of vectors that must be embedded and searched. Increasing k also increases the amount of retrieved context and therefore retrieval work, although the effect on this small dataset is limited.

Q3 remains PARTIAL because the evaluation is restricted to the supplied reference document. Q9 remains NO because the requested blood-glucose target range is absent from the source; no external medical information was substituted.
