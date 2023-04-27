from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI()


class Sentences(BaseModel):
    sentences: List[str]


# // Developer's Note: You can only vectorize but not the other way around. It appears to be a one-way function.
@app.post("/api/vectorize")
async def encode(sentences: Sentences):
    embeddings = model.encode(sentences.sentences)
    return {"embeddings": embeddings.tolist()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
