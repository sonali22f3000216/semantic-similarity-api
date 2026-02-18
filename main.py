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

# Lightweight semantic-style embedding
def simple_embed(text):
    text = text.lower()
    words = text.split()
    length = len(text)
    unique_words = len(set(words))
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    return np.array([length, unique_words, avg_word_len])

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.post("/similarity")
def compute_similarity(request: SimilarityRequest):

    query_embedding = simple_embed(request.query)

    scores = []

    for i, doc in enumerate(request.docs):
        doc_embedding = simple_embed(doc)
        similarity = cosine_similarity(query_embedding, doc_embedding)
        scores.append((i, similarity))

    # Sort by similarity descending
    scores.sort(key=lambda x: x[1], reverse=True)

    # Return top 3 documents
    top_docs = [request.docs[i] for i, _ in scores[:3]]

    return {"matches": top_docs}
