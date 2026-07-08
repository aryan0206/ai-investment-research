"""
Financial Agent — Week 2.

Public entrypoint: get_financial_data(query: str) -> FinancialData

Given free-text input ("Reliance", "AAPL", "Infosys"), resolves the ticker,
pulls market data + financial statements from yfinance, computes any ratios
missing from `.info`, and returns a fully-typed FinancialData object.

This module is intentionally standalone (no LangGraph dependency) so it can
be unit-tested in isolation and wired into the graph in Week 4 as a node
that simply calls get_financial_data() and writes the result into state.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

from app.services.schema import FinancialData, FinancialRatios, MarketData, StatementLineItems
from app.services.ticker_resolver import resolve_ticker

logger = logging.getLogger(__name__)


def _safe_float(value) -> Optional[float]:
    """yfinance often returns numpy types or NaN; normalize to plain float or None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    return f


def _safe_int(value) -> Optional[int]:
    f = _safe_float(value)
    return int(f) if f is not None else None


def _get_row(df: pd.DataFrame, *possible_labels: str) -> Optional[pd.Series]:
    """yfinance statement DataFrames are indexed by line-item name, but exact
    labels shift between versions/companies (e.g. 'Net Income' vs
    'Net Income Common Stockholders'). Try each candidate label in order."""
    if df is None or df.empty:
        return None
    for label in possible_labels:
        if label in df.index:
            return df.loc[label]
    return None


def _latest_value(df: pd.DataFrame, *possible_labels: str) -> Optional[float]:
    row = _get_row(df, *possible_labels)
    if row is None or row.empty:
        return None
    return _safe_float(row.iloc[0])


def _extract_statement(
    income_stmt: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cash_flow: pd.DataFrame,
) -> Optional[StatementLineItems]:
    if income_stmt is None and balance_sheet is None and cash_flow is None:
        return None

    period_end = None
    for df in (income_stmt, balance_sheet, cash_flow):
        if df is not None and not df.empty:
            period_end = str(df.columns[0].date()) if hasattr(df.columns[0], "date") else str(df.columns[0])
            break

    total_revenue = _latest_value(income_stmt, "Total Revenue", "Revenue")
    gross_profit = _latest_value(income_stmt, "Gross Profit")
    operating_income = _latest_value(income_stmt, "Operating Income", "Operating Income Loss")
    net_income = _latest_value(income_stmt, "Net Income", "Net Income Common Stockholders")
    ebitda = _latest_value(income_stmt, "EBITDA", "Normalized EBITDA")

    total_assets = _latest_value(balance_sheet, "Total Assets")
    total_liabilities = _latest_value(
        balance_sheet, "Total Liabilities Net Minority Interest", "Total Liab"
    )
    total_equity = _latest_value(
        balance_sheet, "Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"
    )
    total_debt = _latest_value(balance_sheet, "Total Debt")
    cash = _latest_value(
        balance_sheet, "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"
    )

    operating_cf = _latest_value(cash_flow, "Operating Cash Flow", "Total Cash From Operating Activities")
    free_cf = _latest_value(cash_flow, "Free Cash Flow")
    capex = _latest_value(cash_flow, "Capital Expenditure", "Capital Expenditures")

    # Compute free cash flow manually if yfinance didn't provide it directly
    if free_cf is None and operating_cf is not None and capex is not None:
        free_cf = operating_cf + capex  # capex is typically reported as negative

    return StatementLineItems(
        period_end=period_end,
        total_revenue=total_revenue,
        gross_profit=gross_profit,
        operating_income=operating_income,
        net_income=net_income,
        ebitda=ebitda,
        total_assets=total_assets,
        total_liabilities=total_liabilities,
        total_equity=total_equity,
        total_debt=total_debt,
        cash_and_equivalents=cash,
        operating_cash_flow=operating_cf,
        free_cash_flow=free_cf,
        capital_expenditures=capex,
    )


def _build_market_data(info: dict, fast_info) -> MarketData:
    def fi(key):
        try:
            return getattr(fast_info, key)
        except Exception:
            return None

    return MarketData(
        current_price=_safe_float(info.get("currentPrice") or fi("last_price")),
        currency=info.get("currency"),
        previous_close=_safe_float(info.get("previousClose") or fi("previous_close")),
        day_high=_safe_float(info.get("dayHigh") or fi("day_high")),
        day_low=_safe_float(info.get("dayLow") or fi("day_low")),
        week_52_high=_safe_float(info.get("fiftyTwoWeekHigh") or fi("year_high")),
        week_52_low=_safe_float(info.get("fiftyTwoWeekLow") or fi("year_low")),
        market_cap=_safe_float(info.get("marketCap") or fi("market_cap")),
        volume=_safe_int(info.get("volume") or fi("last_volume")),
        avg_volume_10d=_safe_int(info.get("averageDailyVolume10Day")),
        beta=_safe_float(info.get("beta")),
    )


def _build_ratios(info: dict, annual: Optional[StatementLineItems]) -> FinancialRatios:
    ratios = FinancialRatios(
        pe_ratio_trailing=_safe_float(info.get("trailingPE")),
        pe_ratio_forward=_safe_float(info.get("forwardPE")),
        pb_ratio=_safe_float(info.get("priceToBook")),
        ev_to_ebitda=_safe_float(info.get("enterpriseToEbitda")),
        price_to_sales=_safe_float(info.get("priceToSalesTrailing12Months")),
        roe=_safe_float(info.get("returnOnEquity")),
        net_profit_margin=_safe_float(info.get("profitMargins")),
        operating_margin=_safe_float(info.get("operatingMargins")),
        gross_margin=_safe_float(info.get("grossMargins")),
        debt_to_equity=_safe_float(info.get("debtToEquity")),
        current_ratio=_safe_float(info.get("currentRatio")),
        quick_ratio=_safe_float(info.get("quickRatio")),
        eps_trailing=_safe_float(info.get("trailingEps")),
        eps_forward=_safe_float(info.get("forwardEps")),
        book_value_per_share=_safe_float(info.get("bookValue")),
        dividend_yield=_safe_float(info.get("dividendYield")),
    )

    # yfinance reports debtToEquity as a percentage (e.g. 45.2 = 45.2%), not a
    # raw ratio. Normalize to decimal form for consistency with our other
    # ratios. This applies regardless of whether statement data is available,
    # since it's correcting the `.info` value directly.
    if ratios.debt_to_equity is not None and ratios.debt_to_equity > 5:
        ratios.debt_to_equity = ratios.debt_to_equity / 100

    # Manual fallbacks computed from statements when `.info` didn't have them.
    if annual is not None:
        if ratios.gross_margin is None and annual.total_revenue and annual.gross_profit is not None:
            ratios.gross_margin = annual.gross_profit / annual.total_revenue

        if ratios.operating_margin is None and annual.total_revenue and annual.operating_income is not None:
            ratios.operating_margin = annual.operating_income / annual.total_revenue

        if ratios.net_profit_margin is None and annual.total_revenue and annual.net_income is not None:
            ratios.net_profit_margin = annual.net_income / annual.total_revenue

        if ratios.roe is None and annual.net_income is not None and annual.total_equity:
            ratios.roe = annual.net_income / annual.total_equity

        if (
            ratios.roce is None
            and annual.operating_income is not None
            and annual.total_assets is not None
            and annual.total_liabilities is not None
        ):
            capital_employed = annual.total_assets - (annual.total_liabilities - (annual.total_debt or 0))
            if capital_employed:
                ratios.roce = annual.operating_income / capital_employed

        if (
            ratios.debt_to_equity is None
            and annual.total_debt is not None
            and annual.total_equity
        ):
            ratios.debt_to_equity = annual.total_debt / annual.total_equity

    return ratios


def get_financial_data(query: str) -> FinancialData:
    """Main entrypoint. Never raises — always returns a FinancialData object,
    with `success=False` and `error_message` set if something went wrong."""

    resolution = resolve_ticker(query)
    if not resolution.resolved_symbol:
        return FinancialData(
            symbol=query,
            success=False,
            error_message=f"Could not resolve '{query}' to a valid ticker symbol.",
            data_warnings=["Ticker resolution failed; no data fetched."],
        )

    symbol = resolution.resolved_symbol
    warnings: list[str] = []
    if resolution.confidence == "best_guess":
        warnings.append(
            f"Ticker resolution was ambiguous; picked '{symbol}' "
            f"({resolution.resolved_name}). Other candidates: {resolution.candidates}"
        )

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
    except Exception as exc:
        logger.error("Failed to fetch info for %s: %s", symbol, exc)
        return FinancialData(
            symbol=symbol,
            company_name=resolution.resolved_name,
            success=False,
            error_message=f"Failed to fetch data for '{symbol}': {exc}",
            data_warnings=warnings,
        )

    try:
        fast_info = ticker.fast_info
    except Exception:
        fast_info = None

    # Statements — each wrapped individually since yfinance can partially
    # fail (e.g. annual statements available but TTM not, or vice versa).
    annual_income, annual_balance, annual_cf = None, None, None
    ttm_income, ttm_cf = None, None

    try:
        annual_income = ticker.income_stmt
    except Exception as exc:
        warnings.append(f"Annual income statement unavailable: {exc}")
    try:
        annual_balance = ticker.balance_sheet
    except Exception as exc:
        warnings.append(f"Balance sheet unavailable: {exc}")
    try:
        annual_cf = ticker.cash_flow
    except Exception as exc:
        warnings.append(f"Cash flow statement unavailable: {exc}")
    try:
        ttm_income = ticker.ttm_income_stmt
    except Exception:
        pass  # TTM not available for all tickers (common for non-US exchanges); not worth warning about
    try:
        ttm_cf = ticker.ttm_cash_flow
    except Exception:
        pass

    annual = _extract_statement(annual_income, annual_balance, annual_cf)
    # TTM balance sheet doesn't really exist conceptually (balance sheet is a
    # snapshot, not a flow) — reuse the latest annual balance sheet for TTM
    # context if income/cash flow TTM data is present.
    ttm = _extract_statement(ttm_income, annual_balance, ttm_cf) if (ttm_income is not None or ttm_cf is not None) else None

    if annual is None:
        warnings.append("No annual financial statements available for this ticker.")

    market_data = _build_market_data(info, fast_info)
    ratios = _build_ratios(info, annual)

    return FinancialData(
        symbol=symbol,
        company_name=resolution.resolved_name or info.get("longName") or info.get("shortName"),
        sector=info.get("sector"),
        industry=info.get("industry"),
        market_data=market_data,
        ratios=ratios,
        annual=annual,
        ttm=ttm,
        data_warnings=warnings,
        success=True,
    )
