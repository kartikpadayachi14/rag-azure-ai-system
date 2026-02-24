print("VERSION 4 - NEW BUILD CONFIRMED")

from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import nest_asyncio
from pyngrok import ngrok
import uvicorn

#Allow nested loops 
nest_asyncio.apply()

#Initialize FastAPI
app = FastAPI()

#Initialize models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text-generation",model = "google/flan-t5-base")

#documents
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

# After your imports
# def chunk_text(text, chunk_size=200, overlap=50):
  #  words = text.split()
 #   chunks = []
 #   for i in range(0, len(words), chunk_size - overlap):
 #       chunk = words[i:i + chunk_size]
#        chunks.append(" ".join(chunk))
 #   return chunks

#documents =[]
#for doc in raw_documents:
#   chunk_list = chunk_text(doc,chunk_size = 10, overlap = 2)
#   documents.extend(chunk_list)

#print(f"Total chunks created: {len(documents)}")
#print(f"First chunk: {documents[0]}")

doc_embeddings = embedding_model.encode(documents)
doc_embeddings = np.array(doc_embeddings).astype("float32")

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)


   #for chunk in retrieved_chunks:
    #  if expected_keyword.lower() in chunk.lower():
     #    return 1
    #return 0


test_cases = [
    {"question": "How can I deploy ML models?", "expected_keyword": "Azure"},
    {"question": "What is attention in transformers?", "expected_keyword": "attention"}
]

def evaluate_retrieval(question, expected_keyword):
   query_embedding = embedding_model.encode([question])
   query_embedding = np.array(query_embedding).astype("float32")
   distances, indices = index.search(query_embedding, k =3)

   retrieved_docs = [documents[i] for i in indices[0]]
   print("\nEVALUATING QUESTION:", question)
   print("RETRIEVED DOCS:", retrieved_docs)

   for doc in retrieved_docs:
      if expected_keyword.lower() in doc.lower():
         return 1
   return 0

print("\n--- Running Retrieval Evaluation ---")
score =0 
for case in test_cases:
   score += evaluate_retrieval(case["question"],case["expected_keyword"])

print("\nRetrieval Accuracy:" ,score / len(test_cases))

@app.get("/ask")
def ask(question: str):
  query_embedding = embedding_model.encode([question])
  query_embedding = np.array(query_embedding).astype("float32")

  distance, indices = index.search(query_embedding, 2)
  retrieved_docs = [documents[i] for i in indices[0]]
  context = " ".join(retrieved_docs)

  prompt = f"""
  Answer the question using only the context below.
  If the answer is not in the context, say "I don't know."

  context:
  {context}

  Question:{question}
  """

  response = generator(prompt, max_new_tokens=100)

  answer = response[0]["generated_text"].strip()

  return {"answer": answer}

  #return{"answer": response[0]['generated_text']}

if __name__ == "__main__":
   ngrok.set_auth_token("39nrON5sXnBO9LoEieiC5mg4xBg_5tDzV1v12v4tcKFwvBHtn")

   public_url = ngrok.connect(8000)
   print(f"\n✅ SUCCESS! Follow these steps:")
   print(f"1. Open this link: {public_url.public_url}/docs")
   print(f"2. Click 'GET /ask', then 'Try it out', then 'Execute'\n")
   
   uvicorn.run(app, host="0.0.0.0", port=8000)
#config =uvicorn.Config(app, host="0.0.0.0", port=8000)
#server = uvicorn.Server(config)

#asyncio.get_event_loop().create_task(server.serve())
