# AI Investment Research Copilot

Multi-stage agentic research system built with LangGraph.

## Phase 1 — Annual Report RAG

### Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # then fill in your GEMINI_API_KEY
```

### Ingest a report

Drop any annual report PDF into `backend/data/reports/`, then:

```bash
cd backend
python ingest.py               # incremental
python ingest.py --fresh       # wipe + re-ingest
```

### Run the API

```bash
cd backend
uvicorn app.main:app --reload
```

### Test

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"company": "Reliance Industries", "query": "What are the key risks?"}'
```

Or open http://localhost:8000/docs for the interactive Swagger UI.

## Architecture

```
User Query
    ↓
Planner Node
    ↓
Retrieval Node  (ChromaDB semantic search over annual report)
    ↓
Synthesizer Node  (extracts facts, flags conflicts)
    ↓
Report Generator Node  (structured investment report)
    ↓
Final Output
```

## Roadmap

- **Phase 2** — Parallel research nodes: yfinance fundamentals + NewsAPI + signals
- **Phase 3** — Evaluation pipeline: rule-based checks + LLM-as-judge scoring
- **Phase 4** — Streamlit / Next.js frontend
