# AI Investment Research Copilot

A multi-agent AI investment research assistant that combines live market data, financial statements, company filings, news retrieval, and Retrieval-Augmented Generation (RAG) to produce structured investment research reports.

This project is being built as an end-to-end Agentic AI system using LangGraph, FastAPI, and modern LLM tooling.

---

## Project Vision

Instead of simply answering questions with an LLM, this system coordinates multiple specialized AI agents that collaborate to perform investment research.

The long-term goal is to build an assistant capable of answering questions such as:

- Why did Reliance fall today?
- Compare TCS vs Infosys.
- Is Apple overvalued?
- Summarize this week's market.
- Explain PE Ratio like I'm a beginner.
- Analyze my investment portfolio.
- Generate a complete bull vs bear investment thesis for any company.

---

# Current Features (Phase 2)

- Live stock lookup
- Automatic ticker resolution
- Financial statement retrieval
- Fundamental ratio analysis
- Live market data
- News retrieval
- Annual Report RAG
- Structured report generation
- FastAPI backend
- LangGraph workflow foundation

---

# Technology Stack

### Backend

- Python
- FastAPI
- LangGraph

### AI

- OpenRouter
- Google Gemma 4
- Retrieval-Augmented Generation (RAG)

### Data Sources

- yfinance
- NewsAPI
- Company Annual Reports

### Vector Database

- ChromaDB

### Frontend (Planned)

- Next.js
- Tailwind CSS

### Database (Planned)

- PostgreSQL

---

# Current Architecture

```
                    User
                      │
                      ▼
                FastAPI Backend
                      │
                      ▼
                 LangGraph Flow
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
 Financial Agent   News Agent   RAG Agent
        │             │             │
        └─────────────┼─────────────┘
                      ▼
           Research Synthesizer
                      ▼
          Structured Research Report
```

---

# Roadmap

## Phase 1 ✅
- Annual Report RAG
- PDF ingestion
- Vector search
- Report generation

## Phase 2 ✅
- Financial Agent
- News Agent
- Live market data
- Ticker resolver

## Phase 3 🚧
- Multi-agent orchestration
- Planner Agent
- Parallel execution
- Shared state

## Phase 4
- Research synthesizer
- Bull/Bear thesis generation
- Risk analysis
- Investment scoring

## Phase 5
- Company comparison
- Portfolio analysis
- Market education assistant

## Phase 6
- Evaluation pipeline
- LLM-as-a-Judge
- Automated quality scoring

## Phase 7
- Next.js frontend
- Interactive dashboard
- Report viewer

## Phase 8
- Production-ready AI Investment Research Copilot

---

# Project Structure

```
backend/
│
├── app/
│   ├── api/
│   ├── graph/
│   ├── services/
│   │    ├── financial_agent.py
│   │    ├── news_agent.py
│   │    ├── rag_engine.py
│   │    ├── formatting.py
│   │    └── schema.py
│   │
│   └── ticker_resolver.py
│
├── ingest.py
├── test_live.py
├── test_news_agent.py
└── requirements.txt
```

---

# Running the Project

## 1. Clone

```bash
git clone https://github.com/aryan0206/ai-investment-research.git
cd ai-investment-research/backend
```

## 2. Create Virtual Environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

macOS/Linux

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment

Create a `.env` file from `.env.example`.

Add your API keys.

---

## 5. Ingest Annual Reports

```bash
python ingest.py --fresh
```

---

## 6. Run FastAPI

```bash
uvicorn app.main:app --reload
```

Swagger UI:

```
http://localhost:8000/docs
```

---

# Example Queries

- Compare Apple and Microsoft.
- Analyze Reliance Industries.
- What are the biggest risks facing Tesla?
- Explain PE Ratio.
- Summarize today's news for Infosys.
- Generate a bull and bear thesis for NVIDIA.

---

# Learning Goals

This project is an exploration of:

- Agentic AI Systems
- Multi-Agent Workflows
- LangGraph
- Retrieval-Augmented Generation
- Financial AI
- FastAPI
- Vector Databases
- LLM Evaluation
- Modern AI System Design

---

## Status

**Current Progress:** Phase 2 of 8

Building toward a production-style AI Investment Research Copilot capable of autonomous financial research using multiple collaborating AI agents.
