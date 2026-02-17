from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Model
class SimilarityRequest(BaseModel):
    docs: List[str]
    query: str

# Fake Embedding Generator (Quota-Free)
def fake_embedding(text: str):
    np.random.seed(len(text))  # deterministic
    return np.random.rand(256)

# Cosine Similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.post("/similarity")
def compute_similarity(request: SimilarityRequest):

    # Generate embeddings for docs
    doc_embeddings = [fake_embedding(doc) for doc in request.docs]

    # Generate embedding for query
    query_embedding = fake_embedding(request.query)

    scores = []

    for i, doc_embedding in enumerate(doc_embeddings):
        similarity = cosine_similarity(query_embedding, doc_embedding)
        scores.append((i, similarity))

    # Sort descending
    scores.sort(key=lambda x: x[1], reverse=True)

    # Get top 3 docs
    top_docs = [request.docs[i] for i, _ in scores[:3]]

    return {
        "matches": top_docs
    }
