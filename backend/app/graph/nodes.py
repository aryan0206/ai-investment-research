from langchain_openai import ChatOpenAI
from app.config import settings
from app.graph.state import ResearchState
from app.services.rag_engine import RAGEngine

# OpenRouter is OpenAI-API-compatible — just swap the base_url and api_key
llm = ChatOpenAI(
    model=settings.LLM_MODEL,
    openai_api_key=settings.OPENROUTER_API_KEY,
    openai_api_base=settings.OPENROUTER_BASE_URL,
    temperature=0.2,
    max_tokens=2048,
)

rag_engine = RAGEngine()


# ---------------------------------------------------------------------------
# Node 1: Planner
# Phase 1: pass-through. Phase 2: will decompose query into sub-tasks.
# ---------------------------------------------------------------------------
def planner_node(state: ResearchState) -> ResearchState:
    print(f"[Planner] Company: {state['company']}")
    print(f"[Planner] Query:   {state['query']}")
    return state


# ---------------------------------------------------------------------------
# Node 2: Retrieval
# Pulls relevant chunks from ChromaDB for the given company + query.
# ---------------------------------------------------------------------------
def retrieval_node(state: ResearchState) -> ResearchState:
    query = f"{state['company']} {state['query']}"
    docs = rag_engine.retrieve(query, k=6)

    citations = [
        f"{d.metadata['source']} (Page {d.metadata['page']})"
        for d in docs
    ]

    return {
        **state,
        "retrieved_docs": docs,
        "citations": citations,
    }


# ---------------------------------------------------------------------------
# Node 3: Synthesizer
# Reads retrieved chunks, extracts facts, flags internal conflicts.
# ---------------------------------------------------------------------------
def synthesizer_node(state: ResearchState) -> ResearchState:
    docs_text = "\n\n".join([
        f"[Source: {d.metadata['source']}, Page {d.metadata['page']}]\n{d.page_content}"
        for d in state["retrieved_docs"]
    ])

    prompt = f"""You are a senior investment research analyst.

Company: {state['company']}
Analyst Query: {state['query']}

Below are excerpts from the company's annual report. Read them carefully before writing anything.

--- BEGIN EXCERPTS ---
{docs_text}
--- END EXCERPTS ---

Your task:
1. Extract key facts, metrics, and strategic initiatives (cite page numbers).
2. Identify risks explicitly mentioned in the report.
3. Note any forward-looking guidance or management commentary.
4. Flag any contradictions or inconsistencies across the excerpts.
5. Note what information is missing or unavailable from these excerpts.

Be factual. Do not invent numbers. If a number is not in the excerpts, say so.

Research Synthesis:"""

    response = llm.invoke(prompt)

    return {
        **state,
        "synthesis": response.content,
    }


# ---------------------------------------------------------------------------
# Node 4: Report Generator
# Formats the synthesis into a structured investment report.
# ---------------------------------------------------------------------------
def report_generator_node(state: ResearchState) -> ResearchState:
    citations_text = "\n".join(state.get("citations", []))

    prompt = f"""You are a senior analyst at a top-tier investment research firm.

Company: {state['company']}

Based on the following research synthesis, write a professional investment report.

--- RESEARCH SYNTHESIS ---
{state['synthesis']}
--- END SYNTHESIS ---

Report format (follow this exactly):

## Executive Summary
2-3 paragraphs. What does this company do, and what is the headline finding?

## Key Findings
Bullet points with specific data points. Cite page numbers from the annual report.

## Risk Factors
Bullet points. Only include risks actually mentioned in the source material.

## Investment Thesis
Two sub-sections:
- Bull Case: reasons to be optimistic
- Bear Case: reasons to be cautious

## Sources Cited
{citations_text}

Rules:
- Do not fabricate numbers or events not in the synthesis.
- Every claim in Key Findings must have a page citation.
- Keep language professional but direct.

Final Report:"""

    response = llm.invoke(prompt)

    return {
        **state,
        "final_report": response.content,
    }
