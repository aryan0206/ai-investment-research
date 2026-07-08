"""
Pydantic schema for the Financial Agent's output.

Design principle: every numeric field is Optional. yfinance's data
availability varies wildly by exchange/ticker (US large-caps have everything,
small NSE/BSE names often have gaps). We never fabricate a value — if it's
not available and can't be computed from the statements we have, it's None.
Downstream consumers (Synthesizer, Report Generator) must handle None
gracefully rather than assuming completeness.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TickerResolution(BaseModel):
    """Result of resolving a free-text company name/ticker into a yfinance symbol."""

    query: str = Field(..., description="Original user input, e.g. 'Reliance' or 'AAPL'")
    resolved_symbol: Optional[str] = Field(
        None, description="yfinance-compatible ticker, e.g. 'RELIANCE.NS', 'AAPL'"
    )
    resolved_name: Optional[str] = Field(None, description="Full company name as reported by Yahoo Finance")
    exchange: Optional[str] = Field(None, description="Exchange code, e.g. 'NSI', 'NMS'")
    confidence: str = Field(
        "unresolved", description="One of: 'exact', 'best_guess', 'unresolved'"
    )
    candidates: list[str] = Field(
        default_factory=list, description="Other plausible symbols, for disambiguation/logging"
    )


class MarketData(BaseModel):
    """Real-time / near-real-time market snapshot."""

    current_price: Optional[float] = None
    currency: Optional[str] = None
    previous_close: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None
    market_cap: Optional[float] = None
    volume: Optional[int] = None
    avg_volume_10d: Optional[int] = None
    beta: Optional[float] = None


class FinancialRatios(BaseModel):
    """Valuation, profitability, and solvency ratios.

    Each field notes its source priority in the docstring-equivalent comment
    below — handled in code, not here, but the contract is:
    1. Try yfinance `.info` dict key directly.
    2. If missing, compute manually from financial statements.
    3. If still unavailable, None.
    """

    # Valuation
    pe_ratio_trailing: Optional[float] = None
    pe_ratio_forward: Optional[float] = None
    pb_ratio: Optional[float] = None
    ev_to_ebitda: Optional[float] = None
    price_to_sales: Optional[float] = None

    # Profitability
    roe: Optional[float] = Field(None, description="Return on Equity, as a decimal e.g. 0.18 = 18%")
    roce: Optional[float] = Field(None, description="Return on Capital Employed, decimal")
    net_profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    gross_margin: Optional[float] = None

    # Solvency / liquidity
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    interest_coverage: Optional[float] = None

    # Per-share
    eps_trailing: Optional[float] = None
    eps_forward: Optional[float] = None
    book_value_per_share: Optional[float] = None
    dividend_yield: Optional[float] = None


class StatementLineItems(BaseModel):
    """A trimmed set of the most analytically useful line items per statement,
    rather than dumping the entire raw yfinance DataFrame. Keeps the schema
    stable even if yfinance's full line-item set shifts between versions.
    All values in the company's reporting currency, most recent period."""

    period_end: Optional[str] = Field(None, description="ISO date string of the period this data covers")

    # Income statement
    total_revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    ebitda: Optional[float] = None

    # Balance sheet
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    total_equity: Optional[float] = None
    total_debt: Optional[float] = None
    cash_and_equivalents: Optional[float] = None

    # Cash flow
    operating_cash_flow: Optional[float] = None
    free_cash_flow: Optional[float] = None
    capital_expenditures: Optional[float] = None


class FinancialData(BaseModel):
    """Top-level output of the Financial Agent. This is the object that gets
    passed into the LangGraph state and consumed by the Evidence Aggregator
    in Week 4."""

    symbol: str
    company_name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None

    market_data: MarketData = Field(default_factory=MarketData)
    ratios: FinancialRatios = Field(default_factory=FinancialRatios)
    annual: Optional[StatementLineItems] = Field(
        None, description="Most recent annual (FY) statement line items"
    )
    ttm: Optional[StatementLineItems] = Field(
        None, description="Trailing-twelve-month statement line items, where available"
    )

    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    data_warnings: list[str] = Field(
        default_factory=list,
        description="Human-readable notes on what couldn't be fetched/computed and why",
    )
    success: bool = Field(True, description="False if the agent could not resolve/fetch a usable dataset")
    error_message: Optional[str] = None
