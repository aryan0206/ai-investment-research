from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import settings


class RAGEngine:
    def __init__(self):
        # Runs 100% locally — downloads ~80MB model on first use, cached after that
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},       # CPU is fine on ThinkPad
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = Chroma(
            persist_directory=settings.CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings,
        )

    def ingest_documents(self, documents: List[Document]) -> None:
        """Embed and store documents in ChromaDB.

        No rate limits since embeddings run locally.
        Still batch to avoid RAM spikes on a 4GB machine.
        """
        batch_size = 50
        total = len(documents)
        total_batches = (total + batch_size - 1) // batch_size

        print(f"[RAGEngine] Ingesting {total} pages in {total_batches} batches...")

        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            batch_num = i // batch_size + 1
            self.vectorstore.add_documents(batch)
            print(f"[RAGEngine] Batch {batch_num}/{total_batches} done ({min(i + batch_size, total)}/{total} pages)")

        print(f"\n[RAGEngine] ✓ Ingestion complete: {total} pages embedded")

    def retrieve(self, query: str, k: int = 6) -> List[Document]:
        """Retrieve top-k semantically relevant chunks for a query."""
        results = self.vectorstore.similarity_search(query, k=k)
        print(f"[RAGEngine] Retrieved {len(results)} chunks for: '{query[:60]}...'")
        return results

    def clear(self) -> None:
        """Wipe all vectors and re-initialize the collection."""
        self.vectorstore.delete_collection()
        self.vectorstore = Chroma(
            persist_directory=settings.CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings,
        )
        print("[RAGEngine] Vector store cleared.")
