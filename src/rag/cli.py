import argparse
from typing import Sequence

from .chain import create_rag_chain
from .config import create_groq_chat_model, load_settings
from .embeddings import SentenceTransformerEmbeddings
from .indexing import index_corpus
from .vector_store import ChromaVectorStore


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask questions about local documents.")
    parser.add_argument("--question", help="Question to ask the document collection.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of relevant chunks to retrieve (default: 3).",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Sync PDF and TXT documents before asking.",
    )
    args = parser.parse_args(argv)

    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero.")
    if not args.question and not args.index:
        parser.error("Provide --question, --index, or both.")

    settings = load_settings()
    vector_store = ChromaVectorStore(
        settings.vector_store_dir,
        collection_name=settings.collection_name,
    )

    if args.index or (args.question and vector_store.count == 0):
        embedding_model = SentenceTransformerEmbeddings(
            settings.embedding_model_name
        )
        indexed_count = index_corpus(settings, vector_store, embedding_model)
        print(f"Indexed {indexed_count} unique chunks.")
    elif not args.question:
        print(f"Indexed collection already contains {vector_store.count} chunks.")
        return 0
    else:
        embedding_model = SentenceTransformerEmbeddings(
            settings.embedding_model_name
        )

    if not args.question:
        return 0

    llm = create_groq_chat_model(settings)
    chain = create_rag_chain(
        vector_store=vector_store,
        embedding_model=embedding_model,
        llm=llm,
        top_k=args.top_k,
    )
    result = chain.invoke({"query": args.question})

    print(f"\nAnswer:\n{result['answer']}")
    if result["sources"]:
        print("\nSources:")
        for source in result["sources"]:
            page = f", page {source['page']}" if source["page"] is not None else ""
            print(f"- {source['source']}{page}")
    return 0
