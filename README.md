# Local document RAG

This project indexes PDF and TXT files from `data/pdf` and `data/text_file`, stores
their chunks in ChromaDB, retrieves relevant passages, and answers with a Groq chat
model through a `ChatPromptTemplate`.

## Setup

Python 3.12 is required. From the project root:

```powershell
uv sync
Copy-Item .env.example .env
```

Set `GROQ_API_KEY` in `.env`. Never paste the key into a notebook, source file, or
Git. If a key has been exposed, revoke it in Groq and use a newly generated key.

## Index and ask

```powershell
uv run rag --index
uv run rag --question "What is artificial intelligence?"
```

Use `--index` again after adding, changing, or deleting source documents. Indexing
uses deterministic chunk IDs and synchronizes the active collection, so rerunning
it does not accumulate duplicate chunks. `pdf_chunks` is the active collection;
the older `pdf_documents` collection is legacy-only and is not queried by the app.

The first index/query may download the `all-MiniLM-L6-v2` embedding model. Groq is
only needed when asking a question; indexing can run without an API key.

## Project layout

- `src/rag/documents.py`: PDF/TXT loading and deduplicated chunking
- `src/rag/embeddings.py`: local sentence-transformer embeddings
- `src/rag/vector_store.py`: stable IDs, Chroma upsert/sync, and retrieval
- `src/rag/chain.py`: context formatting, source citations, and ChatPromptTemplate
- `src/rag/cli.py`: command-line entry point
- `notebook/documents.ipynb`: exploratory notebook and package-backed example
- `tests/`: offline tests for idempotent indexing and RAG behavior

Run the offline tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```