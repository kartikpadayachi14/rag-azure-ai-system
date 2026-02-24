Project Goal
A Retrieval-Augmented Generation (RAG) system that uses vector search to provide grounded AI answers and prevent hallucinations.

* Tech Stack
AI: Sentence-Transformers, FAISS, FLAN-T5

Backend: Python, FastAPI

DevOps: Docker, Azure Container Apps

* How it Works
User asks a question.

FAISS finds relevant text chunks using vector embeddings.

LLM generates an answer based only on that context.

* Run with Docker
Bash
docker build -t rag-api .
docker run -p 8000:8000 rag-api

* Results
Retrieval Accuracy: 100% on tested technical queries.

Safety: Implemented prompt engineering to ensure "I don't know" responses for out-of-context questions.
