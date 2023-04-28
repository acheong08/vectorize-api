"""
Fast API server for sentence vectorization
"""
from os import getenv
from typing import List

import uvicorn
from fastapi import FastAPI
from flask import jsonify
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

# jsonify

model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI()


class Sentences(BaseModel):
    """
    Base model for sentences
    """

    sentences: List[str]


class SemanticSearchRequest(BaseModel):
    """
    Base model for semantic search request
    """

    corpus: List[str]
    query: str
    num_results: int = 1
    mode: str


class Utilities:
    """
    Methods for vectorization and semantic search
    """

    @staticmethod
    def encode(sentences: str) -> List[float]:
        """
        Convert sentences to vectors
        """
        embeddings = model.encode(sentences)
        return embeddings.tolist()

    @staticmethod
    def semantic_search(corpus: List[str], query: str, num_results: int) -> list[dict]:
        """
        Get the most similar sentences from the corpus to the query
        """
        query_embedding = model.encode([query])
        # Check validity of num_results
        if num_results > len(corpus):
            num_results = len(corpus)
        corpus_embeddings = model.encode(corpus)
        closest_n: List[dict] = util.semantic_search(
            query_embedding,
            corpus_embeddings,
            top_k=num_results,
        )[0]
        return closest_n


# // Developer's Note: You can only vectorize but not the other way around.
# // It appears to be a one-way function.
@app.post("/api/vectorize")
async def encode(sentences: Sentences):
    """
    Convert sentences to vectors
    Input: {"sentences": ["I am a sentence", "I am another sentence"]}
    Output: {"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}
    """
    return Utilities.encode(sentences.sentences)


@app.post("/api/semantic_search")
async def semantic_search(request: SemanticSearchRequest):
    """
    Semantic search
    Input: {"corpus":["Google Chrome", "Firefox", "Eggshells", "Garbage"], "query": "Browser", "num_results": 2, "mode": "sentence"}
    Output: [{"score":0.7520363330841064,"sentence":"Firefox"},{"score":0.724408745765686,"sentence":"Google Chrome"}]
    """
    closest_n = Utilities.semantic_search(
        request.corpus,
        request.query,
        request.num_results,
    )
    results: List[dict] = []
    if request.mode == "sentence":
        # Return the string of the most similar sentences
        for n in closest_n:
            results.append(
                {
                    "score": n.get("score"),
                    "sentence": request.corpus[n.get("corpus_id")],
                },
            )
    elif request.mode == "number":
        # Return the nresults: List[dict] = []umber of the most similar sentences
        for n in closest_n:
            results.append(
                {
                    "score": n.get("score"),
                    "number": n.get("corpus_id"),
                },
            )
    else:
        return jsonify({"error": "Invalid mode"})
    return results


if __name__ == "__main__":
    host = getenv("HOST", "127.0.0.1")
    port = getenv("PORT", "8000")
    uvicorn.run(app, host=host, port=int(port))
