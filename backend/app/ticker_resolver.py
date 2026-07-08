"""
Ticker resolution: turn free-text ("Reliance", "Apple", "Infosys") into a
yfinance-compatible symbol, using yfinance's own Search endpoint rather than
a hardcoded lookup table — works for any global exchange yfinance covers.

Note: yf.Search hits Yahoo's live search endpoint, so it requires network
access and can occasionally be flaky/rate-limited like any other yfinance
call. We wrap it defensively and never raise out of this module.

RANKING NOTE (important): yf.Search returns results ordered by Yahoo's own
text-relevance scoring, NOT by company size or significance. This causes bad
picks for short/ambiguous queries — e.g. "Reliance" was matching "RS" (a tiny
"Reliance Inc.") ahead of RELIANCE.NS (Reliance Industries, a >$200B company)
purely because of string-match relevance. We fix this by re-ranking
candidates using market cap (the strongest available proxy for "the company
a person actually means") with a secondary preference for primary listings
over ADRs/foreign listings when the query gives no exchange hint.
"""

from __future__ import annotations

import logging
import re

import yfinance as yf

from app.services.schema import TickerResolution

logger = logging.getLogger(__name__)

# If the input already looks like a valid ticker (all caps, optional
# exchange suffix, no spaces), skip search and try it directly first —
# saves a network round-trip for the common case of "AAPL" or "RELIANCE.NS".
_LIKELY_TICKER_RE = re.compile(r"^[A-Z0-9]{1,10}(\.[A-Z]{1,4})?$")

# ADR / foreign-listing markers in yfinance's `quoteType`/`exchange` fields.
# Used to deprioritize ADRs in favor of the primary listing when a query is
# just a bare company name with no explicit exchange/ticker given.
_ADR_EXCHANGE_HINTS = {"NYQ", "NMS", "NGM", "NCM", "PNK"}  # common US exchange codes


def _looks_like_ticker(query: str) -> bool:
    return bool(_LIKELY_TICKER_RE.match(query.strip()))


def _try_direct(symbol: str) -> TickerResolution | None:
    """Attempt to treat the query as a ticker directly. Validates by checking
    that yfinance returns a non-empty info dict with a recognizable name."""
    try:
        t = yf.Ticker(symbol)
        info = t.info
        if not info or info.get("symbol") is None:
            return None
        name = info.get("longName") or info.get("shortName")
        if not name:
            return None
        return TickerResolution(
            query=symbol,
            resolved_symbol=info.get("symbol", symbol),
            resolved_name=name,
            exchange=info.get("exchange"),
            confidence="exact",
            candidates=[],
        )
    except Exception as exc:
        logger.debug("Direct ticker lookup failed for %s: %s", symbol, exc)
        return None


def _fetch_market_cap(symbol: str) -> float:
    """Best-effort market cap lookup for ranking purposes. Returns 0.0 (lowest
    rank) on any failure rather than raising — ranking is a "nice to have",
    a failed lookup shouldn't break resolution entirely."""
    try:
        info = yf.Ticker(symbol).fast_info
        cap = getattr(info, "market_cap", None)
        return float(cap) if cap else 0.0
    except Exception as exc:
        logger.debug("Market cap lookup failed for %s during ranking: %s", symbol, exc)
        return 0.0


def _is_primary_listing(quote: dict) -> bool:
    """Heuristic: a symbol with no exchange suffix dot and not flagged as an
    ADR-style US-listed depositary receipt is more likely the company's home
    listing. This is imperfect (genuinely US-domiciled companies have no
    suffix either) but works as a tie-breaker, not a hard filter."""
    symbol = quote.get("symbol", "")
    quote_type = quote.get("quoteType", "")
    # yfinance/Yahoo doesn't reliably expose an "is ADR" flag, so we use the
    # symbol shape as a weak signal: foreign companies' ADRs typically trade
    # as plain US tickers (no exchange suffix) even though the company's
    # primary listing has a country suffix like .NS, .L, .HK, etc.
    return quote_type == "EQUITY" and "." in symbol


def _rank_candidates(query: str, equities: list[dict]) -> list[dict]:
    """Re-rank Search results by market cap (descending), with a bonus for
    primary (suffixed) listings when the query doesn't itself contain an
    exchange hint. This directly fixes cases like 'Reliance' -> 'RS' instead
    of 'Reliance' -> 'RELIANCE.NS'.

    Cost note: this issues one fast_info lookup per candidate. We cap to the
    top 5 raw Search results (Yahoo's relevance ordering is decent enough
    that the right answer is virtually always in this set) to bound latency
    and avoid hammering yfinance with 8 calls for one resolution."""
    query_has_exchange_hint = "." in query or query.upper() == query
    capped = equities[:5]

    scored = []
    for q in capped:
        symbol = q.get("symbol")
        if not symbol:
            continue
        cap = _fetch_market_cap(symbol)
        primary_bonus = 1.0
        if not query_has_exchange_hint and _is_primary_listing(q):
            # Modest bonus, not a hard override — a primary listing of a
            # small company still shouldn't outrank a mega-cap ADR. This
            # only matters as a tie-breaker between comparably-sized options.
            primary_bonus = 1.15
        scored.append((cap * primary_bonus, q))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [q for _, q in scored] if scored else equities


def resolve_ticker(query: str) -> TickerResolution:
    """Resolve free-text company name or ticker symbol to a yfinance symbol.

    Strategy:
    1. If the input already looks like a ticker, try it directly (cheap, fast).
    2. Otherwise (or if step 1 fails), use yf.Search to get candidate matches,
       then re-rank them by market cap (see module docstring for why raw
       Search order is unreliable for this).
    3. If nothing usable comes back, return an 'unresolved' result — callers
       must check `.confidence` / `.resolved_symbol is None` before proceeding.
    """
    query = query.strip()
    if not query:
        return TickerResolution(query=query, confidence="unresolved")

    if _looks_like_ticker(query):
        direct = _try_direct(query.upper())
        if direct is not None:
            return direct

    try:
        search = yf.Search(query, max_results=8)
        quotes = getattr(search, "quotes", None) or []
    except Exception as exc:
        logger.warning("yf.Search failed for query '%s': %s", query, exc)
        return TickerResolution(query=query, confidence="unresolved")

    if not quotes:
        return TickerResolution(query=query, confidence="unresolved")

    # Prefer equity results over other instrument types (ETFs, indices, etc.)
    equities = [q for q in quotes if q.get("quoteType") == "EQUITY"] or quotes

    # Re-rank by market cap instead of trusting Yahoo's text-relevance order.
    ranked = _rank_candidates(query, equities)

    best = ranked[0]
    candidates = [q.get("symbol") for q in ranked[1:5] if q.get("symbol")]

    resolved_symbol = best.get("symbol")
    resolved_name = best.get("longname") or best.get("shortname")

    if not resolved_symbol:
        return TickerResolution(query=query, confidence="unresolved", candidates=candidates)

    return TickerResolution(
        query=query,
        resolved_symbol=resolved_symbol,
        resolved_name=resolved_name,
        exchange=best.get("exchange"),
        confidence="exact" if len(ranked) == 1 else "best_guess",
        candidates=candidates,
    )
