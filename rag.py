import requests
import numpy as np
from pathlib import Path

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:4b-instruct"


def embed(texts):
    r = requests.post(
        f"{OLLAMA}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=30,
    )
    r.raise_for_status()
    return np.array(r.json()["embeddings"])


def ask_llm(prompt):
    r = requests.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": CHAT_MODEL,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


docs_dir = Path(__file__).resolve().parent / "docs"
chunks = []
for file in docs_dir.glob("*.md"):
    chunks.extend(
        part.strip()
        for part in file.read_text(encoding="utf-8").split("\n\n")
        if part.strip()
    )

if not chunks:
    raise FileNotFoundError(f"No non-empty Markdown chunks found in {docs_dir}")

chunk_vectors = embed(chunks)
print(f"Indexed {len(chunks)} chunks with shape {chunk_vectors.shape}")

def retrieve(question, top_k=3):
    q=embed([question])[0]
    scores = chunk_vectors @ q/ np.linalg.norm(chunk_vectors, axis=1) / np.linalg.norm(q)
    return [chunks[i] for i in np.argsort(scores)[-top_k:][::-1]]

while True:
    question = input("Ask: ")
    context = "\n\n".join(retrieve(question))
    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, hallucinate and tell the user.

Context:
{context}

Question: {question}"""
    print("\n" + ask_llm(prompt) + "\n")