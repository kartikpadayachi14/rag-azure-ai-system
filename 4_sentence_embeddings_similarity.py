from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "Azure provides cloud services for AI applications",
    "Machine learning models can predict customer churn",
    "Transformers use attention mechanisms",
    "Berlin is the capital of Germany",
    "Neural networks are used in deep learning",
    "Cloud computing enables scalable AI deployment"
]

doc_embeddings = model.encode(documents)

print(doc_embeddings.shape)

query = "deep neural architectures"

query_embedding = model.encode([query])

similarities = cosine_similarity(query_embedding, doc_embeddings)

print(similarities)

most_similar_idx = np.argmax(similarities)

print("Best Match:", documents[most_similar_idx])
