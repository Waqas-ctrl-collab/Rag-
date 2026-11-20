import requests


class OllamaEmbeddings:
    """Embeddings via a locally running Ollama server."""

    def __init__(
        self,
        model_name: str = "nomic-embed-text:latest",
        base_url: str = "http://localhost:11434",
        batch_size: int = 8,
        timeout: int = 120,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.timeout = timeout
        self._url = f"{base_url.rstrip('/')}/api/embed"

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Send one small batch to Ollama and return embeddings."""
        response = requests.post(
            self._url,
            json={"model": self.model_name, "input": texts},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        all_embeddings: list[list[float]] = []
        total = len(texts)
        for i in range(0, total, self.batch_size):
            batch = texts[i : i + self.batch_size]
            print(f"  Embedding batch {i // self.batch_size + 1}"
                  f"/{(total + self.batch_size - 1) // self.batch_size}"
                  f" ({len(batch)} chunks)…", flush=True)
            all_embeddings.extend(self._embed_batch(batch))
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        return self._embed_batch([query])[0]
