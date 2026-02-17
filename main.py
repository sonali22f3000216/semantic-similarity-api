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

class SimilarityRequest(BaseModel):
    docs: List[str]
    query: str

# Simple Bag-of-Words Embedding
def embed(text, vocabulary):
    words = text.lower().split()
    vector = np.zeros(len(vocabulary))
    for word in words:
        if word in vocabulary:
            vector[vocabulary.index(word)] += 1
    return vector

def cosine_similarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.post("/similarity")
def compute_similarity(request: SimilarityRequest):

    # Build vocabulary from docs + query
    all_text = " ".join(request.docs + [request.query])
    vocabulary = list(set(all_text.lower().split()))

    # Embed query
    query_vector = embed(request.query, vocabulary)

    scores = []

    for i, doc in enumerate(request.docs):
        doc_vector = embed(doc, vocabulary)
        similarity = cosine_similarity(query_vector, doc_vector)
        scores.append((i, similarity))

    # Sort by similarity descending
    scores.sort(key=lambda x: x[1], reverse=True)

    top_docs = [request.docs[i] for i, _ in scores[:3]]

    return {
        "matches": top_docs
    }
