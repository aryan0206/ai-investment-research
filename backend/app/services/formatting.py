"""
Presentation-layer formatting for financial numbers.

Principle: FinancialData (schema.py) always carries raw float values —
17509666258944.0, not "₹17.5 Trillion". Precision matters for any downstream
computation (ratio math, the LLM doing analysis, evaluation scoring in
Week 6). Formatting is purely a last-mile concern for whatever renders a
report to a human — the Report Generator (Week 5) or an API response layer.

This module is intentionally currency-aware: market cap formatting differs
for INR (Lakh/Crore convention is more natural for Indian users than
Trillion/Billion, though we default to international units unless told
otherwise) vs USD/other currencies.
"""

from __future__ import annotations

from typing import Optional

# Currency symbol lookup for common currencies yfinance reports. Falls back
# to the raw currency code (e.g. "JPY") if not in this map.
_CURRENCY_SYMBOLS = {
    "USD": "$",
    "INR": "₹",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
}


def _symbol_for(currency: Optional[str]) -> str:
    if not currency:
        return ""
    return _CURRENCY_SYMBOLS.get(currency.upper(), currency.upper() + " ")


def format_large_number(
    value: Optional[float],
    currency: Optional[str] = None,
    decimals: int = 2,
) -> str:
    """Format a large raw number into a human-readable string with magnitude
    suffix, e.g. 17509666258944.0 -> '₹17.51 Trillion'.

    Uses international short-scale units (Thousand/Million/Billion/Trillion)
    regardless of currency. If you want Indian Lakh/Crore convention for INR
    values specifically, use format_inr_large_number instead.
    """
    if value is None:
        return "N/A"

    sym = _symbol_for(currency)
    abs_value = abs(value)
    sign = "-" if value < 0 else ""

    if abs_value >= 1_000_000_000_000:
        return f"{sign}{sym}{abs_value / 1_000_000_000_000:.{decimals}f} Trillion"
    if abs_value >= 1_000_000_000:
        return f"{sign}{sym}{abs_value / 1_000_000_000:.{decimals}f} Billion"
    if abs_value >= 1_000_000:
        return f"{sign}{sym}{abs_value / 1_000_000:.{decimals}f} Million"
    if abs_value >= 1_000:
        return f"{sign}{sym}{abs_value / 1_000:.{decimals}f} Thousand"
    return f"{sign}{sym}{abs_value:.{decimals}f}"


def format_inr_large_number(value: Optional[float], decimals: int = 2) -> str:
    """Format using Indian numbering convention: Lakh (10^5), Crore (10^7),
    and Lakh Crore (10^12, used for mega-cap company valuations — e.g. a
    company worth ₹18,00,000 Crore is conventionally read as '₹18 Lakh
    Crore', not as a seven-digit Crore figure). More natural than
    Trillion/Billion for an Indian audience analyzing NSE/BSE-listed
    companies."""
    if value is None:
        return "N/A"

    abs_value = abs(value)
    sign = "-" if value < 0 else ""
    sym = _CURRENCY_SYMBOLS["INR"]

    one_lakh = 100_000
    one_crore = 10_000_000
    one_lakh_crore = 1_000_000_000_000  # 1 lakh * 1 crore

    if abs_value >= one_lakh_crore:
        return f"{sign}{sym}{abs_value / one_lakh_crore:.{decimals}f} Lakh Crore"
    if abs_value >= one_crore:
        return f"{sign}{sym}{abs_value / one_crore:.{decimals}f} Crore"
    if abs_value >= one_lakh:
        return f"{sign}{sym}{abs_value / one_lakh:.{decimals}f} Lakh"
    return f"{sign}{sym}{abs_value:.{decimals}f}"


def format_percent(value: Optional[float], decimals: int = 2) -> str:
    """Format a decimal ratio as a percentage string, e.g. 0.182 -> '18.20%'.
    Our schema stores ratios like ROE/margins as decimals (0.18 = 18%), not
    pre-multiplied — this is the single place that does the *100 conversion."""
    if value is None:
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_ratio(value: Optional[float], decimals: int = 2, suffix: str = "x") -> str:
    """Format a multiple-style ratio, e.g. P/E of 22.567 -> '22.57x'."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}{suffix}"


def format_price(value: Optional[float], currency: Optional[str] = None, decimals: int = 2) -> str:
    """Format a per-share price value, e.g. 142.337 -> '$142.34'."""
    if value is None:
        return "N/A"
    sym = _symbol_for(currency)
    return f"{sym}{value:,.{decimals}f}"
