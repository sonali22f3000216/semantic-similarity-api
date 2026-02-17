from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from openai import OpenAI
import numpy as np
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SimilarityRequest(BaseModel):
    docs: List[str]
    query: str

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

@app.post("/similarity")
def compute_similarity(request: SimilarityRequest):

    # Generate document embeddings
    doc_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=request.docs
    )

    doc_embeddings = [item.embedding for item in doc_response.data]

    # Generate query embedding
    query_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=request.query
    )

    query_embedding = query_response.data[0].embedding

    scores = []

    for i, doc_embedding in enumerate(doc_embeddings):
        similarity = cosine_similarity(query_embedding, doc_embedding)
        scores.append((i, similarity))

    # Sort by similarity descending
    scores.sort(key=lambda x: x[1], reverse=True)

    # Top 3 matches
    top_matches = [request.docs[i] for i, _ in scores[:3]]

    return {
        "matches": top_matches
    }
