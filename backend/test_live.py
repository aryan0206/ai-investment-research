"""
Manual/live test for the Financial Agent — run this on your machine where
you have working internet access. This sandbox's network is locked to
package registries only, so this script was NOT executed end-to-end here;
all unit-level logic (statement parsing, ratio fallback math) was tested
separately with mocked data and passed.

Run from the `backend/` directory (so the `app` package resolves):
    python test_live.py
"""

from app.services.financial_agent import get_financial_data

TEST_QUERIES = [
    "Reliance",        # free-text, Indian (NSE) — fuzzy resolution via Search
    "RELIANCE.NS",      # exact NSE ticker — should skip Search, go direct
    "Apple",            # free-text, US
    "AAPL",              # exact US ticker — direct path
    "Infosys",          # free-text, Indian, second NSE name
    "asdkjqwlekqjwe",   # garbage input — should return success=False cleanly
]

if __name__ == "__main__":
    for query in TEST_QUERIES:
        print("=" * 70)
        print(f"QUERY: {query!r}")
        print("=" * 70)

        data = get_financial_data(query)

        print(f"success: {data.success}")
        if not data.success:
            print(f"error_message: {data.error_message}")
            print()
            continue

        print(f"symbol: {data.symbol}")
        print(f"company_name: {data.company_name}")
        print(f"sector: {data.sector} / {data.industry}")
        print()
        print("-- Market Data --")
        print(data.market_data.model_dump_json(indent=2))
        print()
        print("-- Ratios --")
        print(data.ratios.model_dump_json(indent=2))
        print()
        print("-- Annual Statement --")
        print(data.annual.model_dump_json(indent=2) if data.annual else "None")
        print()
        if data.data_warnings:
            print("-- Warnings --")
            for w in data.data_warnings:
                print(f"  - {w}")
        print()
