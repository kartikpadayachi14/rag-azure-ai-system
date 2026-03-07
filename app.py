print("VERSION 4 - NEW BUILD CONFIRMED")

from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import nest_asyncio


#Allow nested loops 
nest_asyncio.apply()


app = FastAPI()


embedding_model = None
generator = None
index = None

documents = [
    "Azure Machine Learning allows deployment of models in the cloud.",
    "Transformers use attention mechanisms to process sequential data efficiently.",
    "Vector databases enable efficient similarity search using embeddings.",
    "Cloud computing provides scalable AI infrastructure.",
    "Machine learning models can be deployed using REST APIs for real-time inference.",
    "Docker containers help package AI models with their dependencies for consistent deployment.",
    "Kubernetes orchestrates containerized applications at scale.",
    "Model versioning ensures reproducibility and tracking of experiments.",
    "Continuous integration and continuous deployment pipelines automate model updates.",
    "Feature engineering improves the predictive performance of machine learning systems.",
    "Neural networks learn hierarchical representations of data.",
    "Transfer learning allows models to leverage pre-trained knowledge.",
    "Fine-tuning adapts pre-trained models to specific tasks.",
    "Batch inference processes large datasets efficiently.",
    "Real-time inference is used in applications like chatbots and recommendation systems.",
    "Monitoring deployed models helps detect performance degradation.",
    "Model drift occurs when data distribution changes over time.",
    "Data preprocessing includes cleaning, normalization, and encoding.",
    "Hyperparameter tuning optimizes model performance.",
    "Embeddings convert text into numerical vector representations.",
    "Cosine similarity measures semantic similarity between vectors.",
    "L2 distance calculates Euclidean distance between embeddings.",
    "Distributed computing accelerates large-scale AI workloads.",
    "GPU acceleration significantly speeds up deep learning training.",
    "Scalable storage solutions are essential for big data applications."
]




test_cases = [
    {"question": "How can I deploy ML models?", "expected_keyword": "Azure"},
    {"question": "What enables similarity search?", "expected_keyword": "Vector"},
    {"question": "What improves predictive performance?", "expected_keyword": "Feature"},
    {"question": "How do transformers process sequential data?", "expected_keyword": "attention"}
]

def evaluate_retrieval(question, expected_keyword):
   query_embedding = embedding_model.encode([question])
   query_embedding = np.array(query_embedding).astype("float32")

   faiss.normalize_L2(query_embedding)

   distances, indices = index.search(query_embedding, 5)

   retrieved_docs = [documents[i] for i in indices[0]]
   print("\nQUESTION:", question)
   print("RETRIEVED DOCS:", retrieved_docs)

   for doc in retrieved_docs:
      if expected_keyword.lower() in doc.lower():
         return 1
      
   return 0


@app.get("/ask")
def ask(question: str):
    global embedding_model, generator, index

    try:

        if embedding_model is None:
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        if generator is None:
            generator = pipeline(
                "text2text-generation",
                model="google/flan-t5-small",
                device=-1
            )

        if index is None:
            doc_embeddings = embedding_model.encode(documents)
            doc_embeddings = np.array(doc_embeddings).astype("float32")

            faiss.normalize_L2(doc_embeddings)

            dimension = doc_embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(doc_embeddings)

            print("\n--- Running Retrieval Evaluation ---")

            score = 0
            for case in test_cases:
                score += evaluate_retrieval(case["question"], case["expected_keyword"])

            print("Retrieval Accuracy:", score / len(test_cases))

        query_embedding = embedding_model.encode([question])
        query_embedding = np.array(query_embedding).astype("float32")

        faiss.normalize_L2(query_embedding)

        distance, indices = index.search(query_embedding, 6)
        if distance[0][0] < 0.4:
             return {"answer": "I don't know"}
        
        retrieved_docs = [documents[i] for i in indices[0]]

        # choose the most relevant docs
        top_docs = retrieved_docs[:2]

        print("\nQUESTION:", question)
        print("TOP DOCS:", top_docs)

        context = "\n".join(top_docs)
        

        prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the context.
If the answer is not clearly stated, say "I don't know."

context:
{context}

Question:
{question}

Answer:
"""

        
        response = generator(prompt, max_new_tokens=50, do_sample=False)
        answer = response[0]["generated_text"].strip()
        return {"answer": answer}

    except Exception as e:
        return {"error": str(e)})}

