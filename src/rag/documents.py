from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(
    source_dirs: Iterable[Path],
    project_root: Path,
) -> list[Document]:
    documents: list[Document] = []

    for source_dir in source_dirs:
        # Skip dirs that don't exist or are empty — don't crash
        if not source_dir.is_dir():
            continue

        supported_files = [
            path
            for path in source_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pdf", ".txt"}
        ]

        for path in sorted(supported_files, key=lambda p: p.as_posix().casefold()):
            try:
                if path.suffix.lower() == ".pdf":
                    loaded = PyMuPDFLoader(str(path)).load()
                else:
                    loaded = TextLoader(str(path), encoding="utf-8").load()
            except Exception as exc:
                raise RuntimeError(f"Could not load document: {path}") from exc

            try:
                source = path.resolve().relative_to(project_root.resolve()).as_posix()
            except ValueError:
                source = path.resolve().as_posix()

            for document in loaded:
                document.metadata["source"] = source
                documents.append(document)

    if not documents:
        raise ValueError(
            "No PDF or TXT documents found. "
            "Add files to data/pdf or data/text_file and re-index."
        )

    return documents


def split_documents(
    documents: Iterable[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Document]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(list(documents))

    unique_chunks: list[Document] = []
    seen_content: set[str] = set()
    for chunk in chunks:
        if chunk.page_content in seen_content:
            continue
        seen_content.add(chunk.page_content)
        unique_chunks.append(chunk)

    if not unique_chunks:
        raise ValueError("Document splitting produced no non-empty chunks.")

    return unique_chunks
