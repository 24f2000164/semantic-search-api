from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import uvicorn
import numpy as np
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
# code here is used to create a FastAPI application that calculates the similarity between a query and a list of documents using OpenAI's embeddings.
class SimilarityRequest(BaseModel):
    docs: List[str]
    query: str

class SimilarityResponse(BaseModel):
    matches: List[str]

@app.post("/similarity", response_model=SimilarityResponse)
async def calculate_similarity(req: SimilarityRequest):
    client = OpenAI(
        base_url="https://aipipe.org/openai/v1",
        api_key="eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZjIwMDAxNjRAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.ctFb6WLOXZQksR-pdWDAaE8Bfah5LCIJ-c7pY-8t41c"
        
        # Replace with your actual key
    )

    # Get embeddings for documents
    doc_embeddings = []
    for doc in req.docs:
        response = client.embeddings.create(  # ✅ Fixed typo
            input=doc,
            model="text-embedding-3-small"
        )
        doc_embeddings.append(response.data[0].embedding)

    # Get embedding for query
    response = client.embeddings.create(
        input=req.query,
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding

    # Compute cosine similarity
    docs_embedding_np = np.array(doc_embeddings)
    query_embedding_np = np.array(query_embedding).reshape(1, -1)

    similarity_scores = cosine_similarity(query_embedding_np, docs_embedding_np).flatten()
    top_3_indices = np.argsort(similarity_scores)[::-1][:3].tolist()
    # Get the matching documents
    top_3_matches = [req.docs[i] for i in top_3_indices]

    return SimilarityResponse(matches=top_3_matches)  # ✅ Fixed key name