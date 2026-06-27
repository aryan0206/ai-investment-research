from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.routes import router

# Validate env vars at startup — fail fast
settings.validate()

app = FastAPI(
    title="AI Investment Research Copilot",
    description="Multi-stage agentic research system — Phase 1",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "message": "AI Investment Research Copilot",
        "phase": 1,
        "endpoints": {
            "analyze": "POST /api/v1/analyze",
            "health":  "GET  /api/v1/health",
        },
    }
