# RAG From Scratch

A basic Retrieval-Augmented Generation (RAG) pipeline built using Python and Ollama to understand document retrieval, embeddings, similarity search, and context-based LLM generation.

## Tech Stack
- Python
- NumPy
- Ollama
- `nomic-embed-text` — Embeddings
- `qwen3:4b-instruct` — LLM

## How It Works
1. Load Markdown documents and split them into chunks.
2. Generate embeddings for each chunk.
3. Embed the user's query.
4. Retrieve the Top-K relevant chunks using cosine similarity.
5. Pass the retrieved context and query to the LLM.
6. Generate a context-based response.

## Project Structure
```text
RAG/
├── docs/
│   └── file.md
├── rag.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Setup & Run
```bash
git clone https://github.com/msk-panigrahi/rag-from-scratch.git
cd rag-from-scratch
uv sync
ollama pull nomic-embed-text
ollama pull qwen3:4b-instruct
uv run rag.py
```

## Learning Goals
Understanding the fundamentals of RAG by implementing chunking, embeddings, vector similarity, retrieval, and LLM integration from scratch.

## Future Improvements
- PDF support
- Improved chunking
- ChromaDB integration
- Source citations
- FastAPI integration
