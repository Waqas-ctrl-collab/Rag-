import uuid
import unittest
from pathlib import Path

import chromadb
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from rag.chain import create_rag_chain
from rag.vector_store import ChromaVectorStore


class FakeEmbeddingModel:
    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0]


class FakeVectorStore:
    def __init__(self, matches: list[dict]) -> None:
        self.matches = matches

    def search(self, query_embedding: list[float], top_k: int) -> list[dict]:
        return self.matches[:top_k]


class VectorStoreTests(unittest.TestCase):
    def test_sync_is_idempotent_and_removes_duplicate_chunks(self) -> None:
        store = ChromaVectorStore(
            Path("."),
            collection_name=f"test_chunks_{uuid.uuid4().hex}",
            client=chromadb.EphemeralClient(),
        )
        chunk = Document(
            page_content="A useful unique passage.",
            metadata={"source": "notes.txt", "start_index": 0},
        )

        first_count = store.sync_documents(
            [chunk, chunk],
            [[1.0, 0.0], [1.0, 0.0]],
        )
        second_count = store.sync_documents([chunk], [[1.0, 0.0]])

        self.assertEqual(first_count, 1)
        self.assertEqual(second_count, 1)
        self.assertEqual(store.count, 1)

    def test_sync_removes_chunks_not_present_in_latest_corpus(self) -> None:
        store = ChromaVectorStore(
            Path("."),
            collection_name=f"test_chunks_{uuid.uuid4().hex}",
            client=chromadb.EphemeralClient(),
        )
        old_chunk = Document(
            page_content="Old passage.",
            metadata={"source": "notes.txt", "start_index": 0},
        )
        current_chunk = Document(
            page_content="Current passage.",
            metadata={"source": "notes.txt", "start_index": 20},
        )

        store.sync_documents(
            [old_chunk, current_chunk],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        store.sync_documents([current_chunk], [[0.0, 1.0]])

        self.assertEqual(store.count, 1)
        result = store.collection.get(include=["documents"])
        self.assertEqual(result["documents"], ["Current passage."])


class RAGChainTests(unittest.TestCase):
    def test_chain_returns_answer_and_page_citation(self) -> None:
        match = {
            "document": Document(
                page_content="AI is a field of computer science.",
                metadata={"source": "intro.pdf", "page": 1},
            )
        }
        fake_llm = RunnableLambda(
            lambda _: AIMessage(content="AI is a field of computer science. [Source 1]")
        )
        chain = create_rag_chain(
            vector_store=FakeVectorStore([match]),
            embedding_model=FakeEmbeddingModel(),
            llm=fake_llm,
        )

        result = chain.invoke({"query": "What is AI?"})

        self.assertIn("[Source 1]", result["answer"])
        self.assertEqual(result["sources"], [{"source": "intro.pdf", "page": 2}])

    def test_chain_abstains_without_calling_llm_when_no_context_exists(self) -> None:
        llm_calls: list[bool] = []
        fake_llm = RunnableLambda(
            lambda _: llm_calls.append(True) or AIMessage(content="should not run")
        )
        chain = create_rag_chain(
            vector_store=FakeVectorStore([]),
            embedding_model=FakeEmbeddingModel(),
            llm=fake_llm,
        )

        result = chain.invoke({"query": "What is missing?"})

        self.assertEqual(result["sources"], [])
        self.assertIn("could not find", result["answer"])
        self.assertEqual(llm_calls, [])


if __name__ == "__main__":
    unittest.main()
