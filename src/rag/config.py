import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    collection_name: str = "pdf_chunks"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    groq_model_name: str = "openai/gpt-oss-120b"

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


def create_groq_chat_model(settings: Settings):
    from langchain_groq import ChatGroq

    load_dotenv(settings.project_root / ".env")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to the project .env file; "
            "do not put the key in source code or a notebook."
        )

    return ChatGroq(
        groq_api_key=api_key,
        model_name=settings.groq_model_name,
        temperature=0.2,
        max_tokens=1024,
    )
