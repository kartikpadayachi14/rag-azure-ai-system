*Evaluation Goal
The goal of this evaluation was to ensure the system retrieves the correct technical context and stays "grounded" to prevent AI hallucinations.

* Methodology
I created a test suite of technical queries with expected keywords to verify the FAISS vector search performance:
Test Case 1: "How can I deploy ML models?" → Expected: "Azure"
Test Case 2: "What is attention in transformers?" → Expected: "attention"

* Performance Results
Retrieval Accuracy: 100% (1.0)
Search Metric: L2 Distance (Euclidean) via FAISS
Latent Space: Using all-MiniLM-L6-v2 for 384-dimensional dense embeddings

* Hallucination Reduction
To ensure engineering depth and reliability, I implemented a Strict Grounding Prompt:
Context Constraint: The model is explicitly told: "Answer the question using only the context below."
Fallback Mechanism: If the information is missing, the model is instructed to say: "I don't know," instead of making up an answer.
