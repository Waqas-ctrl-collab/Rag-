from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    collection_name: str = "pdf_chunks"
    embedding_model_name: str = "nomic-embed-text:latest"
    ollama_model_name: str = "llama3.2:latest"
    ollama_base_url: str = "http://localhost:11434"

    @property
    def pdf_dir(self) -> Path:
        return self.project_root / "data" / "pdf"

    @property
    def text_dir(self) -> Path:
        return self.project_root / "data" / "text_file"

    @property
    def vector_store_dir(self) -> Path:
        return self.project_root / "data" / "Vector_store"

    @property
    def source_dirs(self) -> tuple[Path, Path]:
        return self.pdf_dir, self.text_dir


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings()


def create_ollama_chat_model(settings: Settings):
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_model_name,
        base_url=settings.ollama_base_url,
        temperature=0.2,
    )
