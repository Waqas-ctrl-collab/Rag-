from .config import Settings
from .documents import load_documents, split_documents
from .embeddings import OllamaEmbeddings
from .vector_store import ChromaVectorStore


def index_corpus(
    settings: Settings,
    vector_store: ChromaVectorStore,
    embedding_model: OllamaEmbeddings,
) -> int:
    documents = load_documents(settings.source_dirs, settings.project_root)
    chunks = split_documents(documents)
    vectors = embedding_model.embed_documents(
        [chunk.page_content for chunk in chunks]
    )
    return vector_store.sync_documents(chunks, vectors)
