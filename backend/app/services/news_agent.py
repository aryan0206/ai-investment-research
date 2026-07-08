"""
News Agent — Phase 2.

Public entrypoint: get_news_data(company: str) -> NewsData

Fetches recent news headlines and article snippets for a given company,
deduplicates them, and runs a single LLM pass to produce a structured
NewsData object containing:
  - raw articles (title, source, published_at, url, snippet)
  - an LLM-generated sentiment label (bullish / bearish / neutral / mixed)
  - a concise summary of the key themes across all articles
  - a list of the top 3-5 material events extracted from the articles

Design principles (mirrors financial_agent.py):
  - Intentionally standalone — no LangGraph dependency.
  - Never raises — always returns a NewsData with success=False on error.
  - Graceful degradation: if NewsAPI is unconfigured, falls back to
    yfinance's built-in news endpoint (no extra API key required).
  - LLM summarisation is optional: if the LLM call fails the raw articles
    are still returned with success=True and a warning.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import requests
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class NewsArticle(BaseModel):
    """A single news article as returned by the agent."""

    title: str
    source: Optional[str] = None
    published_at: Optional[str] = Field(
        None, description="ISO-8601 datetime string, e.g. '2025-06-01T14:30:00Z'"
    )
    url: Optional[str] = None
    snippet: Optional[str] = Field(
        None, description="First ~400 chars of article description/content"
    )


class NewsData(BaseModel):
    """Top-level output of the News Agent.

    Passed directly into the LangGraph ResearchState['news_data'] field and
    consumed by the Evidence Synthesizer in the next step.
    """

    company: str
    articles: list[NewsArticle] = Field(default_factory=list)

    # LLM-derived fields — None when LLM summarisation was skipped/failed
    sentiment: Optional[str] = Field(
        None,
        description="Overall market sentiment: 'bullish' | 'bearish' | 'neutral' | 'mixed'",
    )
    summary: Optional[str] = Field(
        None, description="2-3 sentence synthesis of key news themes"
    )
    key_events: list[str] = Field(
        default_factory=list,
        description="Top 3-5 material events extracted from the news",
    )

    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data_warnings: list[str] = Field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SNIPPET_MAX_LEN = 400


def _truncate(text: Optional[str], max_len: int = _SNIPPET_MAX_LEN) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    return text[:max_len] + "…" if len(text) > max_len else text


def _fetch_via_newsapi(company: str, api_key: str, max_articles: int = 10) -> list[NewsArticle]:
    """Call the NewsAPI /everything endpoint for the company name."""
    url = "https://newsapi.org/v2/everything"
    params = {
        # Wrap in quotes for exact-phrase matching — without this, "Reliance"
        # would match any article containing the word "reliance" anywhere in
        # the body, giving completely off-topic results.
        "q": f'"{company}"',
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": max_articles,
        "apiKey": api_key,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    articles = []
    for item in data.get("articles", []):
        snippet = _truncate(item.get("description") or item.get("content"))
        articles.append(
            NewsArticle(
                title=item.get("title") or "",
                source=item.get("source", {}).get("name"),
                published_at=item.get("publishedAt"),
                url=item.get("url"),
                snippet=snippet,
            )
        )
    return articles


def _fetch_via_yfinance(company: str, max_articles: int = 10) -> list[NewsArticle]:
    """Fallback: use yfinance's built-in news endpoint.

    yfinance.Ticker.news returns a list of dicts. The company name is used
    as a bare ticker-like search string — imperfect but good enough as a
    fallback when NewsAPI is unavailable.
    """
    import yfinance as yf  # lazy import — not everyone has yfinance installed

    # yfinance news is keyed by ticker; use the company name as a best-effort
    # search token. For common companies (Apple, Reliance, etc.) this works well.
    ticker = yf.Ticker(company)
    raw_news = getattr(ticker, "news", None) or []
    articles = []
    for item in raw_news[:max_articles]:
        content = item.get("content", {})
        title = content.get("title", item.get("title", ""))
        snippet = _truncate(content.get("summary") or content.get("description"))
        pub_date = content.get("pubDate") or item.get("providerPublishTime")

        # pubDate can be a unix timestamp (int) or ISO string
        if isinstance(pub_date, (int, float)):
            pub_date = datetime.fromtimestamp(pub_date, tz=timezone.utc).isoformat()

        provider = content.get("provider", {})
        source = (
            provider.get("displayName")
            or item.get("publisher")
            or item.get("source")
        )
        url = content.get("canonicalUrl", {}).get("url") or item.get("link") or item.get("url")

        articles.append(
            NewsArticle(
                title=title,
                source=source,
                published_at=str(pub_date) if pub_date else None,
                url=url,
                snippet=snippet,
            )
        )
    return articles


def _deduplicate(articles: list[NewsArticle]) -> list[NewsArticle]:
    """Remove articles with identical titles (case-insensitive)."""
    seen: set[str] = set()
    unique: list[NewsArticle] = []
    for a in articles:
        key = a.title.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(a)
    return unique


def _filter_relevant(company: str, articles: list[NewsArticle]) -> list[NewsArticle]:
    """Remove articles where no keyword from the company name appears in the
    title. Title-only (not snippet) because NewsAPI's description/content
    fields often mention a company incidentally — e.g. "Share Markets Close
    Higher, Auto Stocks Rally" with a snippet that says "...Reliance
    Industries was among the top gainers..." is not really an article about
    Reliance, even though the keyword appears in the body text. The title is
    the reliable signal of what an article is actually about.

    Falls back to returning all articles unchanged if the filter would remove
    everything (prevents returning an empty list for niche companies).
    """
    # Only meaningful for multi-word company names; single words like 'Apple'
    # are too ambiguous to filter aggressively.
    keywords = [w.lower() for w in company.split() if len(w) > 3]
    if not keywords or len(company.split()) == 1:
        return articles

    relevant = [
        a for a in articles
        if any(kw in a.title.lower() for kw in keywords)
    ]
    return relevant if relevant else articles  # never return empty


# ---------------------------------------------------------------------------
# LLM Summarisation
# ---------------------------------------------------------------------------

def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base=settings.OPENROUTER_BASE_URL,
        temperature=0.1,
        max_tokens=512,
    )


def _llm_summarise(
    company: str,
    articles: list[NewsArticle],
    max_retries: int = 3,
) -> tuple[str, str, list[str]]:
    """Run a single LLM call over the headlines and return (sentiment, summary, key_events).

    Retries up to max_retries times with exponential back-off on 429 rate-limit
    errors (common on the free OpenRouter tier).  Raises on other errors or
    when retries are exhausted.
    """
    headlines_block = "\n".join(
        f"- [{a.published_at or 'unknown date'}] {a.title}"
        + (f" -- {a.snippet}" if a.snippet else "")
        for a in articles
    )

    prompt = f"""You are a senior equity research analyst reviewing recent news for {company}.

Below are the latest news headlines and snippets:

{headlines_block}

Your task (respond in exactly this format, no extra text):

SENTIMENT: <one of: bullish | bearish | neutral | mixed>

SUMMARY: <2-3 sentences synthesising the main themes across all articles>

KEY_EVENTS:
- <event 1>
- <event 2>
- <event 3>
(3 to 5 bullet points; only material, company-specific events — no generic market noise)

Rules:
- Base your analysis only on the articles above.
- SENTIMENT must be a single word from the allowed set.
- Do not fabricate events not mentioned in the articles.
"""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            llm = _build_llm()
            response = llm.invoke(prompt)
            text = response.content.strip()
            break  # success — exit retry loop
        except Exception as exc:
            last_exc = exc
            if "429" in str(exc) and attempt < max_retries - 1:
                wait = (attempt + 1) * 8  # 8 s, then 16 s
                logger.warning(
                    "[NewsAgent] Rate limited (attempt %d/%d), retrying in %ds...",
                    attempt + 1, max_retries, wait,
                )
                time.sleep(wait)
            else:
                raise
    else:
        raise last_exc  # type: ignore[misc]

    # --- Parse the fixed-format response (section-tracking state machine) ---
    # Track which section the current line belongs to so multi-line summaries
    # and stray bullets don't get misclassified.
    sentiment = "neutral"
    summary_parts: list[str] = []
    key_events: list[str] = []
    current_section: str = ""

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        upper = line.upper()
        if upper.startswith("SENTIMENT:"):
            raw = line.split(":", 1)[1].strip().lower()
            if raw in ("bullish", "bearish", "neutral", "mixed"):
                sentiment = raw
            current_section = "sentiment"

        elif upper.startswith("SUMMARY:"):
            first_chunk = line.split(":", 1)[1].strip()
            if first_chunk:
                summary_parts.append(first_chunk)
            current_section = "summary"

        elif upper.startswith("KEY_EVENTS:") or upper.startswith("KEY EVENTS:"):
            current_section = "key_events"

        elif current_section == "summary" and not line.startswith("-"):
            # Continuation line of a multi-line summary
            summary_parts.append(line)

        elif current_section == "key_events" and line.startswith("-"):
            event = line.lstrip("-").strip()
            if event:
                key_events.append(event)

    summary = " ".join(summary_parts).strip()
    return sentiment, summary, key_events


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def get_news_data(company: str, max_articles: int = 10) -> NewsData:
    """Main entrypoint.  Never raises — always returns a NewsData object,
    with success=False and error_message set if something went wrong fatally.

    Fetch priority:
      1. NewsAPI (if NEWS_API_KEY is set in .env)
      2. yfinance built-in news (free, no key required, limited)

    LLM summarisation is attempted after fetching; if it fails the raw
    articles are still returned with success=True.
    """
    company = company.strip()
    if not company:
        return NewsData(
            company=company,
            success=False,
            error_message="Company name must not be empty.",
        )

    warnings: list[str] = []
    articles: list[NewsArticle] = []

    # --- Fetch ---
    news_api_key = os.getenv("NEWS_API_KEY", "")
    if news_api_key:
        try:
            articles = _fetch_via_newsapi(company, news_api_key, max_articles)
            logger.info("[NewsAgent] Fetched %d articles via NewsAPI for '%s'", len(articles), company)
        except Exception as exc:
            warnings.append(f"NewsAPI fetch failed ({exc}); falling back to yfinance news.")
            logger.warning("[NewsAgent] NewsAPI failed for '%s': %s", company, exc)

    if not articles:
        try:
            articles = _fetch_via_yfinance(company, max_articles)
            logger.info("[NewsAgent] Fetched %d articles via yfinance for '%s'", len(articles), company)
        except Exception as exc:
            warnings.append(f"yfinance news fetch failed: {exc}")
            logger.error("[NewsAgent] yfinance news failed for '%s': %s", company, exc)

    if not articles:
        return NewsData(
            company=company,
            success=False,
            error_message=f"Could not retrieve any news for '{company}' from any source.",
            data_warnings=warnings,
        )

    # Some sources (notably yfinance's aggregated feed, and occasionally
    # NewsAPI) return items with a missing/blank title — e.g. a syndicated
    # piece from hackernoon.com with no headline set. These are useless
    # downstream (nothing to display, nothing for the LLM to summarise) and
    # must be dropped before dedup/relevance filtering, both of which key off
    # the title.
    articles = [a for a in articles if a.title and a.title.strip()]

    if not articles:
        return NewsData(
            company=company,
            success=False,
            error_message=f"Fetched articles for '{company}' but all had empty titles.",
            data_warnings=warnings,
        )

    articles = _deduplicate(articles)
    articles = _filter_relevant(company, articles)

    # --- LLM Summarisation (optional — gracefully degraded) ---
    sentiment: Optional[str] = None
    summary: Optional[str] = None
    key_events: list[str] = []

    if settings.OPENROUTER_API_KEY:
        try:
            sentiment, summary, key_events = _llm_summarise(company, articles)
            logger.info("[NewsAgent] LLM summarisation complete for '%s' — sentiment: %s", company, sentiment)
        except Exception as exc:
            warnings.append(f"LLM summarisation failed ({exc}); returning raw articles only.")
            logger.warning("[NewsAgent] LLM summarisation failed for '%s': %s", company, exc)
    else:
        warnings.append("OPENROUTER_API_KEY not set; skipping LLM summarisation.")

    return NewsData(
        company=company,
        articles=articles,
        sentiment=sentiment,
        summary=summary,
        key_events=key_events,
        data_warnings=warnings,
        success=True,
    )
