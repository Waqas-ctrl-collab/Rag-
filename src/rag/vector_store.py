import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import chromadb
from langchain_core.documents import Document


def stable_chunk_id(document: Document) -> str:
    identity = {
        "source": document.metadata.get("source", ""),
        "page": document.metadata.get("page", ""),
        "start_index": document.metadata.get("start_index", ""),
        "content": document.page_content,
    }
    encoded = json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ChromaVectorStore:
    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "pdf_chunks",
        client: Any | None = None,
    ) -> None:
        persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = (
            client
            if client is not None
            else chromadb.PersistentClient(path=str(persist_directory))
        )
        existing_names = {
            collection.name for collection in self.client.list_collections()
        }

        if collection_name in existing_names:
            self.collection = self.client.get_collection(name=collection_name)
        else:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={
                    "description": "RAG document chunks; normalized vectors with L2 distance",
                    "hnsw:space": "l2",
                },
            )

        metadata = self.collection.metadata or {}
        self.distance_metric = metadata.get("hnsw:space", "l2")
        if self.distance_metric not in {"l2", "cosine"}:
            raise ValueError(
                f"Unsupported Chroma distance metric: {self.distance_metric}"
            )

    @property
    def count(self) -> int:
        return self.collection.count()

    def sync_documents(
        self,
        documents: Iterable[Document],
        embeddings: list[list[float]],
    ) -> int:
        document_list = list(documents)
        if len(document_list) != len(embeddings):
            raise ValueError("The number of documents and embeddings must match.")

        records: dict[str, tuple[Document, list[float]]] = {}
        seen_content: set[str] = set()
        for document, embedding in zip(document_list, embeddings):
            if document.page_content in seen_content:
                continue
            seen_content.add(document.page_content)
            records[stable_chunk_id(document)] = (document, embedding)

        if not records:
            raise ValueError("Cannot sync an empty set of document chunks.")

        ids = list(records)
        documents_text = [records[doc_id][0].page_content for doc_id in ids]
        metadatas = [
            self._chroma_metadata(records[doc_id][0].metadata) for doc_id in ids
        ]
        vectors = [records[doc_id][1] for doc_id in ids]

        existing_ids = set(self.collection.get(include=["metadatas"])["ids"])
        self.collection.upsert(
            ids=ids,
            documents=documents_text,
            metadatas=metadatas,
            embeddings=vectors,
        )

        stale_ids = sorted(existing_ids.difference(ids))
        if stale_ids:
            self.collection.delete(ids=stale_ids)

        return len(ids)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if self.count == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count),
            include=["documents", "metadatas", "distances"],
        )

        documents = results["documents"][0] or []
        metadatas = results["metadatas"][0] or []
        distances = results["distances"][0] or []
        ids = results["ids"][0] or []

        matches: list[dict[str, Any]] = []
        for doc_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            similarity = (
                1.0 - distance / 2.0
                if self.distance_metric == "l2"
                else 1.0 - distance
            )
            matches.append(
                {
                    "id": doc_id,
                    "document": Document(
                        page_content=text,
                        metadata=metadata or {},
                    ),
                    "distance": float(distance),
                    "similarity": float(similarity),
                }
            )

        return matches

    @staticmethod
    def _chroma_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
        return {
            key: value
            for key, value in metadata.items()
            if isinstance(value, (str, int, float, bool))
        }
