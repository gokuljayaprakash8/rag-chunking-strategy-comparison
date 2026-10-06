# Manual Retrieval Evaluation

Evaluation is source-grounded and retrieval-only. A result is scored from the
retrieved chunk text, not from outside medical knowledge.

## Scoring

YES = retrieved evidence contains the complete answer supported by the reference.
PARTIAL = retrieved evidence contains only part of the requested answer.
NO = retrieved evidence does not contain the answer, or the reference itself
does not provide the requested information.

## Question-level judgments

### k=3

| Question | Strategy A | Strategy B | Notes |
|---|---|---|---|
| Q1 | YES | YES | Diagnostic criteria are present in the Diagnostic Criteria section. |
| Q2 | YES | YES | Prediabetes HbA1c range is present in Diagnostic Criteria. |
| Q3 | PARTIAL | PARTIAL | Oral medication classes are listed, but the source explicitly identifies metformin as usually first-line; it does not identify three first-line drugs. |
| Q4 | YES | YES | Classic hyperglycemia symptoms are present in Symptoms. |
| Q5 | YES | YES | Lifestyle modifications are present in Lifestyle Modifications. |
| Q6 | YES | YES | Microvascular complications are present in Microvascular Complications. |
| Q7 | NO | NO | The Macrovascular Complications evidence is not in the top-3 retrieved set. |
| Q8 | YES | YES | Monitoring frequency is present in Monitoring and Follow-up. |
| Q9 | NO | NO | The reference document does not provide the requested blood-glucose target range. |
| Q10 | YES | YES | Emergency symptoms are present in When to Seek Emergency Care. |

### k=5

| Question | Strategy A | Strategy B | Notes |
|---|---|---|---|
| Q1 | YES | YES | Diagnostic evidence is retrieved. |
| Q2 | YES | YES | Prediabetes HbA1c evidence is retrieved. |
| Q3 | PARTIAL | PARTIAL | Same source limitation as k=3. |
| Q4 | YES | YES | Symptom evidence is retrieved. |
| Q5 | YES | YES | Lifestyle evidence is retrieved. |
| Q6 | YES | YES | Microvascular evidence is retrieved. |
| Q7 | NO | YES | Strategy B retrieves Macrovascular Complications at rank 5; Strategy A still misses it. |
| Q8 | NO | YES | Strategy A's top-5 retrieved chunks do not contain the monitoring-frequency answer; Strategy B retrieves the Monitoring and Follow-up section. |
| Q9 | NO | NO | Source gap remains. |
| Q10 | YES | YES | Emergency-care evidence is retrieved. |

## Interpretation

For the answerable questions, k=5 improves Strategy B by recovering the
macrovascular evidence that k=3 missed. The experiment therefore demonstrates
a concrete recall benefit from increasing k, although the additional retrieved
chunks can also introduce less relevant context.

Q3 remains PARTIAL because the evaluation is restricted to the supplied
reference document. Q9 remains a deliberate source-gap result rather than
being filled with external medical knowledge.
