"""
Fast API server for sentence vectorization
"""
from typing import List

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI()


class Sentences(BaseModel):
    """
    Base model for sentences
    """

    sentences: List[str]


# // Developer's Note: You can only vectorize but not the other way around.
# // It appears to be a one-way function.
@app.post("/api/vectorize")
async def encode(sentences: Sentences):
    """
    Convert sentences to vectors
    Input: {"sentences": ["I am a sentence", "I am another sentence"]}
    Output: {"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}
    """
    embeddings = model.encode(sentences.sentences)
    return {"embeddings": embeddings.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
