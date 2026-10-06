## Hypothesis

Hypothesis: preserving document structure and semantic boundaries will improve retrieval of relevant evidence compared with fixed-size chunking when retrieval depth is held constant.

Controlled RAG Chunking Strategy Comparison

Scope and Method

This project compares a basic fixed-size RAG chunking strategy with a structure-aware chunking strategy using the same source document, questions, embedding model, similarity calculation, and retrieval procedure.

This is a retrieval-only experiment. No answer-generation model, external medical source, reranker, or similarity threshold is used.

Fixed experimental configuration

- Source: "data/diabetes_reference_document.md"
- Questions: the exact 10 questions in "data/questions.json"
- Embedding model: "sentence-transformers/all-MiniLM-L6-v2"
- Model revision: "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
- Inference: official ONNX graph and tokenizer from the same model revision
- Token limit: 256 tokens
- Pooling: attention-masked mean pooling
- Normalization: L2 normalization
- Similarity: NumPy cosine similarity
- Retrieval depths: "k=3" and "k=5"
- Reranker: none
- Similarity threshold: none
- Answer generation: none

The same embedding and retrieval pipeline is used for both strategies.

Chunking Strategies

Strategy A — Basic / Naive

The document is divided into fixed 500-character chunks with 50-character overlap.

This follows the assignment's example baseline and deliberately does not consider semantic or document-structure boundaries.

Strategy B — Structure-Aware

The document is parsed hierarchically:

section → paragraph → sentence

Headings are retained with their associated content rather than emitted as heading-only chunks. Paragraphs are preserved when they fit within the configured target; when a paragraph needs to be divided, sentence boundaries are used.

The target maximum is 500 characters, but a semantic unit may exceed that target when necessary rather than being split arbitrarily.

This approach was chosen because it preserves document structure and semantic boundaries while remaining deterministic, reproducible, and straightforward to inspect and explain. It also avoids introducing a more complex semantic-similarity chunking algorithm, which would add another variable to the experiment.

Retrieval and Evaluation

For each question, a query embedding is compared with all chunk embeddings using cosine similarity. The highest-scoring "k" chunks are retrieved.

The experiment was evaluated manually against the supplied reference document.

- YES: the retrieved context contains sufficient evidence for the requested answer.
- PARTIAL: the context or the reference document provides only part of the requested information.
- NO: the retrieved context cannot support the requested answer, or the requested information is absent from the supplied reference.

Two source limitations are important:

- Q3 is PARTIAL because the reference identifies metformin as usually first-line but does not explicitly provide three named first-line oral medications.
- Q9 is NO because the supplied reference does not provide the requested blood-glucose target range.

No outside medical information was substituted for either case.

Summary Results

Strategy| k| Chunks| Mean characters| Min–max characters| YES| PARTIAL| NO| Accuracy
A — Basic / Naive| 3| 17| 490.29| 335–500| 7| 1| 2| 70%
B — Structure-Aware| 3| 26| 285.69| 16–590| 8| 1| 1| 80%
A — Basic / Naive| 5| 17| 490.29| 335–500| 8| 1| 1| 80%
B — Structure-Aware| 5| 26| 285.69| 16–590| 8| 1| 1| 80%

Question-by-Question Results

Question| A k=3| B k=3| A k=5| B k=5
Q1 Diagnostic criteria| YES| YES| YES| YES
Q2 Prediabetes HbA1c range| YES| YES| YES| YES
Q3 Three first-line oral medications| PARTIAL| PARTIAL| PARTIAL| PARTIAL
Q4 Classic hyperglycemia symptoms| YES| YES| YES| YES
Q5 Lifestyle modifications| YES| YES| YES| YES
Q6 Microvascular complications| YES| YES| YES| YES
Q7 Macrovascular complications| NO| YES| YES| YES
Q8 HbA1c monitoring frequency| YES| YES| YES| YES
Q9 Blood glucose target range| NO| NO| NO| NO
Q10 Emergency symptoms| YES| YES| YES| YES

Key retrieval difference

At "k=3", the clearest difference is Q7.

Strategy A does not retrieve the macrovascular-complications evidence within its top three chunks, while Strategy B retrieves the relevant macrovascular evidence at rank 2.

When retrieval depth increases to "k=5", Strategy A also retrieves the macrovascular evidence, at rank 5. Therefore the Strategy B advantage at "k=3" does not persist at "k=5".

Analysis

The results show a small retrieval advantage for Strategy B at "k=3", but not a universal advantage. Strategy B achieves 80% versus 70% for Strategy A at "k=3", primarily because its structure-aware chunks retrieve the macrovascular-complications evidence within the top three results. At "k=5", both strategies achieve 80%, because Strategy A retrieves that same evidence at rank 5.

The main trade-off is semantic coherence versus simplicity and retrieval cost. Strategy A creates only 17 relatively large chunks, with a mean size of 490.29 characters. It is simple to implement and requires fewer embeddings and fewer similarity comparisons. However, fixed character boundaries can separate related information or place useful terminology in less relevant chunks.

Strategy B creates 26 smaller chunks, with a mean size of 285.69 characters. Preserving headings, paragraphs, and sentence boundaries makes the chunks easier to interpret and keeps related information together. The cost is a more complex parser and more vectors to embed and search. Smaller chunks also increase the number of candidates considered during similarity search.

The structure-aware approach was chosen because it tests a meaningful improvement in chunk construction without adding a second semantic model or another similarity-based decision process. It is deterministic, reproducible, and easy to explain.

The experiment therefore supports a limited conclusion: for this document and these ten questions, structure-aware chunking improved top-3 retrieval on one question, but the advantage disappeared at top-5 retrieval. It should not be interpreted as evidence that Strategy B is universally superior.

Results and Reproducibility

The final run artifacts preserve retrieved chunk text, chunk IDs, source offsets, heading paths, similarity scores, and embedding/model fingerprints.

The source-grounded manual evaluation is stored in:

"results/manual_evaluation.md"

Final run files:

- "results/" — "k=3"
- "results/" — "k=5"

Reproduction

From the project root:

python3 src/rag.py --config config/experiment.json
python3 src/rag.py --config config/experiment_k5.json

Both configurations use the same source document, questions, embedding model, and evaluation procedure; only the retrieval depth differs.