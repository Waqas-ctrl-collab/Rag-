# RAG Chat — Local AI over your documents

A fully offline Retrieval-Augmented Generation (RAG) application.  
Ask questions about your PDFs and text files and get answers with source citations — no internet, no API keys, everything runs on your own machine.

---

## How it works

```
Your PDFs / TXT files
        │
        ▼
  Document Loader          (PyMuPDF for PDF, TextLoader for TXT)
        │
        ▼
   Text Splitter           (RecursiveCharacterTextSplitter, 800 chars, 120 overlap)
        │
        ▼
  OllamaEmbeddings         (nomic-embed-text:latest  →  768-dim vectors, batched)
        │
        ▼
   ChromaDB                (persistent vector store, L2 distance, stable chunk IDs)
        │
   On every question:
        │
        ▼
  Embed the query          (nomic-embed-text)
        │
        ▼
  Top-K retrieval          (default top_k = 3)
        │
        ▼
  ChatPromptTemplate       (context + question)
        │
        ▼
  llama3.2:latest          (via Ollama, local LLM)
        │
        ▼
  Answer + Sources         (returned to the chat UI)
```

---

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.12 |
| [Ollama](https://ollama.com) | latest |
| uv | latest |

No GPU required — `llama3.2` (3.2B Q4) and `nomic-embed-text` (137M) both run on CPU.  
Minimum ~6 GB RAM recommended.

---

## Quick start

### 1. Install Ollama
Download and install from [https://ollama.com](https://ollama.com).

### 2. Add your documents
Drop PDF or TXT files into:
```
data/pdf/          ← PDF documents
data/text_file/    ← plain text files (.txt)
```
The project already includes three sample PDFs:
- `Artificial_Intelligence_Research_Report.pdf`
- `LangChain_Research_Report.pdf`
- `Machine_Learning_Research_Report.pdf`

### 3. Launch
Double-click **`start.bat`**.  

That's it. It will:
1. Start the Ollama server (if not running)
2. Pull `llama3.2:latest` and `nomic-embed-text:latest` (skipped if already downloaded)
3. Index your documents into ChromaDB (batched, shows progress)
4. Start the FastAPI backend on `http://localhost:8000`
5. Open your browser automatically

---

## Chat UI

The browser UI at `http://localhost:8000` includes:

- **Chat window** — type a question, get an answer with source citations
- **Sources panel** — each answer shows which document and page it came from
- **Re-index button** — drop new files into `data/pdf` or `data/text_file` and click to re-index without restarting
- **Clear chat** button
- Thinking timer — shows elapsed seconds while the LLM generates
- Enter to send, Shift+Enter for new line

---

## Project layout

```
Rag/
├── start.bat                   ← one-click launcher
├── frontend/
│   └── index.html              ← chat UI (pure HTML/CSS/JS, no framework)
├── src/rag/
│   ├── api.py                  ← FastAPI backend  (/chat  /index  /health)
│   ├── chain.py                ← RAG chain (retrieval → prompt → LLM → answer)
│   ├── cli.py                  ← command-line entry point
│   ├── config.py               ← Settings dataclass (model names, paths)
│   ├── documents.py            ← PDF/TXT loading and chunking
│   ├── embeddings.py           ← OllamaEmbeddings (batched, nomic-embed-text)
│   ├── indexing.py             ← orchestrates load → split → embed → store
│   ├── vector_store.py         ← ChromaDB wrapper (upsert, search, stable IDs)
│   ├── __init__.py
│   └── __main__.py
├── data/
│   ├── pdf/                    ← put your PDFs here
│   ├── text_file/              ← put your TXT files here
│   └── Vector_store/           ← ChromaDB persisted data (auto-created)
├── tests/
│   └── test_rag.py
├── pyproject.toml
└── uv.lock
```

---

## CLI usage

You can also use the command line directly (with the venv active):

```bash
# Index documents
.venv\Scripts\python.exe -m rag --index

# Ask a question
.venv\Scripts\python.exe -m rag --question "What is machine learning?"

# Index then ask in one command
.venv\Scripts\python.exe -m rag --index --question "What is LangChain?"

# Control how many chunks are retrieved
.venv\Scripts\python.exe -m rag --question "Explain transformers" --top-k 5
```

---

## API endpoints

The backend exposes three endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the chat UI |
| `GET` | `/health` | Returns `{"status":"ok"}` |
| `POST` | `/chat` | Ask a question, get answer + sources |
| `POST` | `/index` | Re-index all documents |

### `/chat` request body
```json
{
  "question": "What is artificial intelligence?",
  "top_k": 3
}
```

### `/chat` response
```json
{
  "answer": "Artificial intelligence is… [Source 1: AI_Report.pdf, page 2]",
  "sources": [
    { "source": "AI_Report.pdf", "page": 2 }
  ]
}
```

---

## Configuration

All settings live in `src/rag/config.py`:

```python
embedding_model_name: str = "nomic-embed-text:latest"   # Ollama embedding model
ollama_model_name:    str = "llama3.2:latest"            # Ollama LLM
ollama_base_url:      str = "http://localhost:11434"     # Ollama server URL
collection_name:      str = "pdf_chunks"                 # ChromaDB collection
```

To swap models, change these values and re-run `start.bat`.  
Any model available via `ollama pull` will work.

---

## Performance notes

| Operation | Typical time |
|---|---|
| Indexing 3 PDFs (67 chunks) | ~45 seconds |
| First chat response (model loading) | 30–60 seconds |
| Subsequent chat responses | 10–20 seconds |

Times are for CPU-only on a typical laptop. A machine with a GPU will be significantly faster.

Embeddings are sent to Ollama in **batches of 8 chunks** to avoid timeouts.  
The vector store uses **stable SHA-256 chunk IDs** so re-indexing never creates duplicates — it upserts changed chunks and removes stale ones.

---

## Running tests

```bash
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Ollama not found` | Install from [https://ollama.com](https://ollama.com) and restart |
| `Read timed out` during indexing | Ollama is overloaded — close other apps and retry |
| `Collection expecting dimension 384, got 768` | Delete `data/Vector_store/` and re-index |
| Backend window shows no activity | Check the RAG Backend terminal for the full error |
| First question takes very long | Normal — `llama3.2` loads into RAM on first use |
| Re-index button shows error | Make sure at least one PDF or TXT file exists in the data folders |
