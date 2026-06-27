"""
Run this script once to parse annual report PDFs and embed them into ChromaDB.

Usage (from the backend/ directory):
    python ingest.py           # incremental — appends to existing vectors
    python ingest.py --fresh   # wipes existing vectors first
"""
import sys
import os

# Ensure app modules resolve correctly when run from backend/
sys.path.insert(0, os.path.dirname(__file__))

from app.services.pdf_parser import PDFParser
from app.services.rag_engine import RAGEngine


def main(fresh: bool = False):
    print("=" * 50)
    print("AI Research Copilot — Ingestion Pipeline")
    print("=" * 50)

    parser = PDFParser()
    engine = RAGEngine()

    if fresh:
        print("\n[!] Fresh mode: clearing existing vector store...")
        engine.clear()

    print("\n[1/2] Parsing annual reports...")
    docs = parser.parse_all()

    print(f"\n[2/2] Embedding {len(docs)} pages into ChromaDB...")
    engine.ingest_documents(docs)

    print("\n✓ Ingestion complete. You can now run the API.")


if __name__ == "__main__":
    fresh_flag = "--fresh" in sys.argv
    main(fresh=fresh_flag)
