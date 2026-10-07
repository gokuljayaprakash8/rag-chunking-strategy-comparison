# Experiment Methodology

## Hypothesis

Hypothesis: preserving document structure and semantic boundaries will improve retrieval of relevant evidence compared with fixed-size chunking when retrieval depth is held constant.

## Objective
Compare a basic fixed-size RAG chunking strategy with a structure-aware strategy while keeping the source document, embedding model, questions, similarity calculation, and retrieval procedure fixed.

## Fixed configuration
- Source: diabetes_reference_document.md
- Questions: exact 10 assignment questions
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Model revision: 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
- Runtime: ONNX Runtime CPU
- Pooling: attention-masked mean pooling
- Normalization: L2
- Similarity: NumPy cosine similarity
- Retrieval depths: k=3 and k=5
- Reranker: none
- Similarity threshold: none
- Answer generation: none; retrieval-only evaluation

## Strategy A — Basic / Naive
The document is divided into fixed 500-character chunks with a 50-character overlap. This follows the assignment example baseline and deliberately does not consider semantic boundaries.

## Strategy B — Structure-Aware
The document is parsed hierarchically using headings/sections, then paragraphs, then sentences when a paragraph must be divided to stay near the 500-character target. Headings are retained with associated content, and heading-only fragments are not emitted. A semantic unit can exceed the target when necessary rather than being split arbitrarily.

This was chosen to test whether preserving document structure can improve retrieval without a more complicated semantic-similarity chunking algorithm. It is deterministic, reproducible, and easy to explain and inspect.

## Retrieval and evaluation
Query embeddings are generated once and reused. Each query is compared with chunk embeddings using cosine similarity, and the highest-scoring k chunks are returned.

Each run writes JSON in results/ including source/question hashes, chunk contents and offsets, model asset hashes, retrieval rankings and similarities, and retrieved context.

Evaluation is source-grounded and manual. YES means sufficient evidence, PARTIAL means relevant but incomplete evidence, and NO means the supplied source cannot support the requested answer.

Q3 is PARTIAL because the source identifies metformin as usually first-line but does not provide three named first-line oral medications. Q9 is NO because the source does not provide the requested adult blood-glucose target range. No outside medical information is substituted.

## Final findings
At k=3 the strategies tie at 70% (7 YES each). They differ on two questions in opposite directions: Strategy B retrieves the Q7 macrovascular evidence at rank 2 while Strategy A ranks it 5th, and Strategy A retrieves the full Q5 lifestyle list at rank 1 while Strategy B's top three contain only the heading-plus-introduction chunk (the list chunk ranks 4th, so Q5 is PARTIAL).

At k=5 both strategies reach 80%. Increasing k removed the difference. The hypothesis that structure-aware chunking improves retrieval is not supported on this document and ten-question set, which is small and was judged manually, not blind to the strategy.

Strategy B produces more chunks (24 versus 17) and smaller average chunks (309.50 versus 490.29 characters). This keeps topics together but increases embedding and similarity-comparison work and requires a more complex parser than fixed character slicing.

## Reproduction
From the project root:

python3 src/rag.py --config config/experiment.json
python3 src/rag.py --config config/experiment_k5.json
