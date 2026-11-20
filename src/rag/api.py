"""FastAPI backend that exposes the RAG chain over HTTP."""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .chain import create_rag_chain
from .config import create_ollama_chat_model, load_settings
from .embeddings import OllamaEmbeddings
from .indexing import index_corpus
from .vector_store import ChromaVectorStore


# ---------------------------------------------------------------------------
# Logging — shows in the backend terminal
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

# ---------------------------------------------------------------------------
# Shared state  (lock only protects the pointer swap, not heavy work)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_chain: Any = None
_settings = None


def _build_chain() -> None:
    """Build the RAG chain. Must NOT be called while _lock is held."""
    global _chain, _settings
    log.info("Building RAG chain …")
    _settings = load_settings()

    log.info("Connecting to vector store …")
    vector_store = ChromaVectorStore(
        _settings.vector_store_dir,
        collection_name=_settings.collection_name,
    )
    log.info("Vector store has %d chunks.", vector_store.count)

    log.info("Loading embedding model: %s", _settings.embedding_model_name)
    embedding_model = OllamaEmbeddings(
        _settings.embedding_model_name,
        _settings.ollama_base_url,
    )

    log.info("Loading LLM: %s", _settings.ollama_model_name)
    llm = create_ollama_chat_model(_settings)

    new_chain = create_rag_chain(
        vector_store=vector_store,
        embedding_model=embedding_model,
        llm=llm,
    )

    # Swap the pointer atomically
    with _lock:
        _chain = new_chain

    log.info("RAG chain ready.")


def _get_chain() -> Any:
    with _lock:
        return _chain


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build chain in a thread so it doesn't block the event loop
    # and does NOT hold _lock during the heavy work
    log.info("Server starting — building chain in background thread …")
    await run_in_threadpool(_build_chain)
    log.info("Startup complete.")
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="RAG Chat API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    top_k: int = 3


class SourceItem(BaseModel):
    source: str
    page: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


class IndexResponse(BaseModel):
    indexed: int
    message: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty.")
    if req.top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be greater than zero.")

    chain = _get_chain()
    if chain is None:
        raise HTTPException(status_code=503, detail="Chain not ready yet — try again in a few seconds.")

    log.info("Chat request: %s", req.question[:80])

    try:
        result = await run_in_threadpool(
            chain.invoke, {"query": req.question, "top_k": req.top_k}
        )
    except Exception as exc:
        log.exception("Chain error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    log.info("Chat response ready. Sources: %s", [s["source"] for s in result["sources"]])
    sources = [SourceItem(source=s["source"], page=s.get("page")) for s in result["sources"]]
    return ChatResponse(answer=result["answer"], sources=sources)


@app.post("/index", response_model=IndexResponse)
async def index_documents():
    log.info("Re-index requested …")
    settings = load_settings()

    vector_store = ChromaVectorStore(
        settings.vector_store_dir,
        collection_name=settings.collection_name,
    )
    embedding_model = OllamaEmbeddings(
        settings.embedding_model_name,
        settings.ollama_base_url,
    )

    try:
        log.info("Indexing documents …")
        count = await run_in_threadpool(index_corpus, settings, vector_store, embedding_model)
        log.info("Indexed %d chunks.", count)
    except Exception as exc:
        log.exception("Indexing error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Rebuild chain in background thread without holding the lock
    log.info("Rebuilding chain after re-index …")
    await run_in_threadpool(_build_chain)

    return IndexResponse(indexed=count, message=f"Successfully indexed {count} chunks.")
