from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np

# 1. Load Models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text2text-generation", model="google/flan-t5-base")


# 2. Knowledge Base
documents = [
    "Azure Machine Learning allows you to train and deploy ML models in the cloud.",
    "Transformers use attention mechanisms to process sequences in parallel.",
    "Cosine similarity measures the angle between two vectors.",
    "Neural networks are used in deep learning models.",
    "Cloud computing enables scalable infrastructure for AI applications."
]

doc_embeddings = embedding_model.encode(documents)

# 3. Ask a Question
query = input("Enter your question: ")

query_embedding = embedding_model.encode([query])


# 4. Retrieve Top Relevant Document
similarities = cosine_similarity(query_embedding, doc_embeddings)

top_k = 2

top_indices = np.argsort(similarities[0])[-top_k:]

retrieved_docs = [documents[i] for i in top_indices]

print("\nRetrieved Documents:")

for doc in retrieved_docs:
    print("-", doc)


# 5. Generate Final Answer
context = " ".join(retrieved_docs)

prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

response = generator(prompt, max_length=150)

print("\nFinal Answer:")

print(response[0]["generated_text"])
