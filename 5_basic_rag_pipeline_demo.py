from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

generator = pipeline("text2text-generation", model="google/flan-t5-base")

documents = [
    "Azure Machine Learning allows you to train and deploy ML models in the cloud.",
    "Transformers use attention mechanisms to process sequences in parallel.",
    "Cosine similarity measures the angle between two vectors.",
    "Neural networks are used in deep learning models.",
    "Cloud computing enables scalable infrastructure for AI applications."
]

doc_embeddings = embedding_model.encode(documents)

query = "What is attention in transformers?"

query_embedding = embedding_model.encode([query])

similarities = cosine_similarity(query_embedding, doc_embeddings)

top_k = 2

top_indices = np.argsort(similarities[0])[-top_k:]

retrieved_docs = [documents[i] for i in top_indices]

print("Retrieved Documents:")

for doc in retrieved_docs:
    print("-", doc)

context = " ".join(retrieved_docs)

prompt = f"""
Use the following context to answer the question.

context:
{context}

Question:
{query}

Answer:
"""

response = generator(prompt, max_length=150)

print(response[0]['generated_text'])
