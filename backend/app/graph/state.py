from typing import List, Optional
from langchain_core.documents import Document
from typing_extensions import TypedDict


class ResearchState(TypedDict):
    # Input
    company: str                        # e.g. "Reliance Industries"
    query: str                          # User's question

    # Retrieval
    retrieved_docs: List[Document]      # Chunks from annual report

    # Phase 2 placeholders (wired in later)
    financial_data: Optional[dict]      # yfinance fundamentals
    news_data: Optional[List[dict]]     # NewsAPI results
    signals_data: Optional[dict]        # RSI, MA crossovers

    # Synthesis
    synthesis: Optional[str]            # Synthesized research notes
    conflicts_detected: Optional[str]   # Cross-source conflicts (Phase 2)

    # Report
    final_report: Optional[str]         # Final output

    # Evaluation (Phase 3)
    evaluation_score: Optional[dict]    # Rule-based + LLM-as-judge scores

    # Metadata
    citations: List[str]                # Source pages referenced
