# AI Investment Research Copilot

> An AI-powered investment research assistant that automates equity research using **Retrieval-Augmented Generation (RAG)**, **LangGraph**, and **Large Language Models (LLMs)**.

## Overview

AI Investment Research Copilot is a multi-stage agentic research system designed to assist investors and analysts in understanding public companies.

The current implementation focuses on retrieving information from annual reports, synthesizing relevant insights, and generating structured investment-style reports with citations.

This project is being developed in multiple phases, with future support planned for financial data, news analysis, multi-agent workflows, evaluation pipelines, and a web-based user interface.

---

## Current Features (Phase 1)

* PDF ingestion pipeline for annual reports
* Automatic document chunking and embedding
* ChromaDB vector database for semantic search
* Retrieval-Augmented Generation (RAG)
* LangGraph-based multi-stage workflow
* Structured investment research reports
* Source citations with page references
* FastAPI REST API
* Interactive Swagger API documentation

---

## Architecture

```
                      User Query
                           │
                           ▼
                    FastAPI REST API
                           │
                           ▼
                  LangGraph Workflow
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
      Planner Node                 Query Analysis
            │
            ▼
      RAG Retrieval Engine
      (ChromaDB Semantic Search)
            │
            ▼
      Relevant Document Chunks
            │
            ▼
      LLM Report Generation
            │
            ▼
 Structured Investment Report
            │
            ▼
  Executive Summary • Risks • Findings • Citations
```

---

## Tech Stack

### Backend

* Python
* FastAPI
* LangGraph
* LangChain
* ChromaDB

### AI

* OpenRouter
* Google Gemma 4 (31B Instruct)

### Document Processing

* PyPDF
* Vector Embeddings
* Retrieval-Augmented Generation (RAG)

### Planned Frontend

* Next.js
* React

---

## Project Structure

```
backend/
│
├── app/
│   ├── api/
│   ├── graph/
│   ├── services/
│   ├── config.py
│   └── main.py
│
├── data/
│   └── reports/
│
├── ingest.py
├── requirements.txt
└── .env.example
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/aryan0206/ai-investment-research.git
cd ai-investment-research/backend
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Add your OpenRouter API key inside the `.env` file.

---

## Ingest an Annual Report

Place an annual report PDF inside

```
backend/data/reports/
```

Then run

```bash
python ingest.py
```

or

```bash
python ingest.py --fresh
```

to rebuild the vector database from scratch.

---

## Run the API

```bash
uvicorn app.main:app --reload
```

Open

```
http://127.0.0.1:8000/docs
```

to access the interactive Swagger UI.

---

## Example Request

```json
{
  "company": "Reliance Industries",
  "query": "Analyze the company's financial health, major risks, and growth strategy."
}
```

---

## Example Output

The API generates a structured investment research report containing:

* Executive Summary
* Key Findings
* Bull Case
* Bear Case
* Risk Factors
* Supporting Evidence
* Source Citations

---

## Roadmap

### Phase 1 ✅

* Annual Report RAG
* LangGraph workflow
* FastAPI backend
* Structured report generation
* Citation support

### Phase 2 🚧

* Financial fundamentals (yfinance)
* News retrieval
* Parallel LangGraph agents
* Company comparison

### Phase 3

* Evaluation pipeline
* LLM-as-a-Judge
* Quality scoring
* Automated testing

### Phase 4

* Next.js frontend
* Interactive dashboard
* Authentication
* Portfolio management

---

## Future Improvements

* SEC filing support
* Earnings call transcript analysis
* Financial ratio analysis
* Stock price visualization
* Investment recommendation dashboard
* Multi-company comparative reports

---

## Disclaimer

This project is intended for educational and research purposes only. It should not be considered financial or investment advice.

---
