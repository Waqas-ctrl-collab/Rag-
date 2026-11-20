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

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | **3.12 exactly** | [python.org](https://www.python.org/downloads/) |
| [Ollama](https://ollama.com) | latest | [ollama.com](https://ollama.com) |
| [uv](https://docs.astral.sh/uv/) | latest | `pip install uv` or see below |
| Git | any | [git-scm.com](https://git-scm.com) |

No GPU required — `llama3.2` (3.2B Q4) and `nomic-embed-text` (137M) both run on CPU.  
Minimum **6 GB RAM** recommended.

---

## Setup from scratch (new clone)

Follow these steps exactly after cloning the repo.

### Step 1 — Clone the repo

```bash
git clone https://github.com/Waqas-ctrl-collab/Rag.git
cd Rag
```

### Step 2 — Install uv

uv is the package manager used by this project. If you don't have it:

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal after installing so `uv` is on your PATH.

### Step 3 — Create the virtual environment and install dependencies

```bash
uv sync
```

This reads `pyproject.toml` and `uv.lock`, creates `.venv/`, and installs all packages including FastAPI, LangChain, ChromaDB, and PyMuPDF. It takes 2–5 minutes on first run.

### Step 4 — Install Ollama and pull the models

Download and install Ollama from [https://ollama.com](https://ollama.com), then pull the two models:

```bash
ollama pull llama3.2:latest
ollama pull nomic-embed-text:latest
```

`llama3.2` is ~2 GB and `nomic-embed-text` is ~274 MB. This only needs to be done once.

### Step 5 — Add your documents *(optional)*

The repo already includes four sample PDFs in `data/pdf/`:

- `The-Alchemist-Paulo-Coelho.pdf`
- `Artificial_Intelligence_Research_Report.pdf`
- `LangChain_Research_Report.pdf`
- `Machine_Learning_Research_Report.pdf`

To use your own documents, drop PDF or TXT files into:
```
data/pdf/          ← PDF documents
data/text_file/    ← plain text files (.txt)
```

### Step 6 — Launch

**Windows — double-click `start.bat`**

Or from the terminal:
```bash
start.bat
```

`start.bat` will automatically:
1. Start the Ollama server (if not already running)
2. Pull models if missing
3. Index your documents into ChromaDB (shows batch progress)
4. Start the FastAPI backend on `http://localhost:8000`
5. Open your browser

> **First launch takes longer** — Ollama loads the LLM into RAM (~30–60s). Subsequent launches are faster.

---

## Chat UI

The browser UI at `http://localhost:8000` includes:

- **Chat window** — type a question, get an answer with source citations
- **Sources panel** — each answer shows which document and page it came from
- **Re-index button** — drop new files into `data/pdf` or `data/text_file` and click to re-index without restarting
- **Clear chat** button
- **Thinking timer** — shows elapsed seconds while the LLM generates
- Enter to send · Shift+Enter for new line

---

## Project layout

```
Rag/
├── start.bat                   ← one-click launcher (Windows)
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
├── pyproject.toml              ← project metadata and dependencies
└── uv.lock                     ← locked dependency versions
```

---

## CLI usage

You can also use the command line directly without the browser UI:

```bash
# Index documents
.venv\Scripts\python.exe -m rag --index

# Ask a question
.venv\Scripts\python.exe -m rag --question "What is machine learning?"

# Index then ask in one command
.venv\Scripts\python.exe -m rag --index --question "What is LangChain?"

# Control how many chunks are retrieved (default is 3)
.venv\Scripts\python.exe -m rag --question "Explain transformers" --top-k 5
```

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the chat UI |
| `GET` | `/health` | Returns `{"status":"ok"}` |
| `POST` | `/chat` | Ask a question, returns answer + sources |
| `POST` | `/index` | Re-index all documents |

### `/chat` request
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

To swap models, change these two lines and re-run `start.bat`.  
Any model available via `ollama pull` will work — for example:

```python
ollama_model_name: str = "mistral:latest"
```

---

## Performance notes

| Operation | Typical time (CPU only) |
|---|---|
| Indexing 4 PDFs | ~60–90 seconds (batched, 8 chunks per request) |
| First chat response | 30–60 seconds (model loading into RAM) |
| Subsequent responses | 10–20 seconds |

A machine with a GPU will be significantly faster.

Embeddings are sent to Ollama in **batches of 8 chunks** to avoid HTTP timeouts.  
The vector store uses **stable SHA-256 chunk IDs** — re-indexing upserts changed chunks and removes stale ones, never creates duplicates.

---

## Running tests

```bash
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `uv: command not found` | Install uv — see Step 2 above |
| `python 3.12 not found` | Install Python 3.12 from [python.org](https://www.python.org/downloads/) |
| `Ollama not found` | Install from [ollama.com](https://ollama.com) and restart your terminal |
| `Read timed out` during indexing | Ollama is overloaded — close other apps and retry |
| `Collection expecting dimension 384, got 768` | Delete `data/Vector_store/` folder and re-index |
| Backend window shows no activity | Check the RAG Backend terminal window for the full error message |
| First question takes very long | Normal — `llama3.2` loads into RAM on first use |
| Re-index button shows error | Make sure at least one PDF or TXT file exists in the data folders |
| Port 8000 already in use | Close any other app using port 8000, or change the port in `start.bat` |

---

## Stack

| Layer | Technology |
|---|---|
| LLM | [Ollama](https://ollama.com) + `llama3.2:latest` |
| Embeddings | [Ollama](https://ollama.com) + `nomic-embed-text:latest` |
| Vector store | [ChromaDB](https://www.trychroma.com/) |
| Backend | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| RAG framework | [LangChain](https://python.langchain.com/) |
| PDF parsing | [PyMuPDF](https://pymupdf.readthedocs.io/) |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Frontend | Pure HTML / CSS / JS |
