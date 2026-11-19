from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return [[float(value) for value in vector] for vector in vectors]

    def embed_query(self, query: str) -> list[float]:
        vectors = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [float(value) for value in vectors[0]]
