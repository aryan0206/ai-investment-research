from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.graph import research_graph
from app.graph.state import ResearchState

router = APIRouter()


class AnalyzeRequest(BaseModel):
    company: str
    query: str = "Analyze the company's financial health, key risks, and growth strategy."


class AnalyzeResponse(BaseModel):
    company: str
    query: str
    report: str
    citations: list[str]
    sources_used: int


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_company(request: AnalyzeRequest):
    try:
        initial_state: ResearchState = {
            "company": request.company,
            "query": request.query,
            "retrieved_docs": [],
            "financial_data": None,
            "news_data": None,
            "signals_data": None,
            "synthesis": None,
            "conflicts_detected": None,
            "final_report": None,
            "evaluation_score": None,
            "citations": [],
        }

        result = research_graph.invoke(initial_state)

        return AnalyzeResponse(
            company=result["company"],
            query=result["query"],
            report=result["final_report"],
            citations=result["citations"],
            sources_used=len(result["retrieved_docs"]),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    return {"status": "healthy", "graph": "loaded", "phase": 1}
