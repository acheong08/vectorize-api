"""
Fast API server for sentence vectorization
"""
from typing import List

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

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


class Utilities:
    @staticmethod
    def encode(sentences: str) -> List[float]:
        embeddings = model.encode(sentences)
        return embeddings.tolist()

    @staticmethod
    def semantic_search(corpus: List[str], query: str, num_results: int) -> list[list]:
        query_embedding = model.encode([query])
        # Check validity of num_results
        if num_results > len(corpus):
            num_results = len(corpus)
        corpus_embeddings = model.encode(corpus)
        closest_n = util.semantic_search(
            query_embedding, corpus_embeddings, top_k=num_results
        )
        # Return the string of the most similar sentences
        results = []
        for idx, score in zip(closest_n[0], closest_n[1]):
            results.append({"sentence": corpus[idx], "score": score})
        return results


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
    """
    results = Utilities.semantic_search(
        request.corpus, request.query, request.num_results
    )
    return results


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
