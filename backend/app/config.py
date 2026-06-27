import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # OpenRouter — LLM only (completions)
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # Free model on OpenRouter — swap to any other free model if this hits limits
    LLM_MODEL: str = "google/gemma-4-31b-it:free"

    # Local embeddings — no API, no rate limits, no cost
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Storage
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    REPORTS_DIR: str = "./data/reports"

    def validate(self):
        if not self.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is not set in .env")

settings = Settings()
