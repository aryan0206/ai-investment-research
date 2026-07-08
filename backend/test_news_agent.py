"""
Smoke test for the News Agent.

Run from the `backend/` directory:

    cd backend
    python test_news_agent.py

What this tests
---------------
1. A diverse set of companies (Indian large-caps, US mega-caps, mid-cap) to
   confirm NewsAPI exact-phrase search returns on-topic articles for all of them.
2. Hard assertions on every result so failures are immediately obvious.
3. Graceful-failure path — garbage ticker must return success=False cleanly.
4. A final PASS / FAIL summary line so you can read the result at a glance.

Requires: NEWS_API_KEY and OPENROUTER_API_KEY set in backend/.env
"""
import os
import sys
import time

# Force UTF-8 stdout so any stray non-ASCII from article titles doesn't crash
# on Windows terminals that default to cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.services.news_agent import NewsData, get_news_data

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
# Each tuple: (query, expect_success)
TEST_CASES = [
    # Indian large-caps
    ("Reliance Industries", True),
    ("Tata Consultancy Services", True),
    ("HDFC Bank", True),
    ("Infosys", True),
    # US mega-caps
    ("Apple", True),
    ("Microsoft", True),
    ("Tesla", True),
    # Garbage — must fail gracefully
    ("xyzzy_invalid_co_123", False),
]

VALID_SENTIMENTS = {"bullish", "bearish", "neutral", "mixed", None}
LLM_DELAY_SECONDS = int(os.getenv("LLM_DELAY_SECONDS", "10"))  # override without editing code

# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def assert_success(data: NewsData, query: str):
    assert data.success is True, f"[{query}] expected success=True, got False -- {data.error_message}"
    assert len(data.articles) > 0, f"[{query}] expected at least 1 article, got 0"
    assert all(a.title for a in data.articles), f"[{query}] some articles have empty titles"
    assert data.sentiment in VALID_SENTIMENTS, (
        f"[{query}] invalid sentiment value: {data.sentiment!r}"
    )
    # Relevance check: for multi-word company names, at least 80% of articles
    # should contain a company keyword in the title. Title-only to mirror
    # _filter_relevant()'s actual matching logic — checking title+snippet
    # here would let the assertion pass even if the agent regressed back to
    # matching on snippet text (which is how the original off-topic-article
    # bug slipped through).
    keywords = [w.lower() for w in query.split() if len(w) > 3]
    if len(query.split()) > 1 and keywords:
        hits = sum(
            1 for a in data.articles
            if any(kw in a.title.lower() for kw in keywords)
        )
        threshold = int(len(data.articles) * 0.8)
        assert hits >= threshold, (
            f"[{query}] relevance check failed: only {hits}/{len(data.articles)} articles "
            f"contain a company keyword in the title. Expected >= {threshold}.\n"
            + "\n".join(f"  - {a.title}" for a in data.articles)
        )


def assert_failure(data: NewsData, query: str):
    assert data.success is False, f"[{query}] expected success=False, got True"
    assert data.error_message, f"[{query}] success=False but error_message is empty"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run():
    passed = 0
    failed = 0
    results: list[tuple[str, bool, str]] = []  # (query, passed, note)

    for query, expect_success in TEST_CASES:
        print(f"\n{'=' * 68}")
        print(f"  QUERY: {query!r}  (expect success={expect_success})")
        print(f"{'=' * 68}")

        data = get_news_data(query)

        # --- Print summary ---
        print(f"  success   : {data.success}")
        if not data.success:
            print(f"  error     : {data.error_message}")
        else:
            print(f"  articles  : {len(data.articles)}")
            print(f"  sentiment : {data.sentiment}")
            if data.summary:
                print(f"  summary   : {data.summary[:180]}{'...' if len(data.summary) > 180 else ''}")
            if data.key_events:
                print(f"  key events:")
                for ev in data.key_events:
                    print(f"    * {ev}")
            print(f"  headlines :")
            for i, a in enumerate(data.articles, 1):
                print(f"    [{i}] {a.source or 'unknown':20s} {a.title[:70]}")
        if data.data_warnings:
            for w in data.data_warnings:
                print(f"  [!] {w[:120]}")

        # --- Assert ---
        try:
            if expect_success:
                assert_success(data, query)
            else:
                assert_failure(data, query)
            print("\n  [PASS]")
            passed += 1
            results.append((query, True, ""))
        except AssertionError as e:
            print(f"\n  [FAIL] -- {e}")
            failed += 1
            results.append((query, False, str(e)))

        # Space out LLM calls to avoid free-tier 429s
        if expect_success:
            time.sleep(LLM_DELAY_SECONDS)

    # --- Final summary ---
    total = passed + failed
    print(f"\n{'=' * 68}")
    print(f"  RESULTS: {passed}/{total} passed")
    print(f"{'=' * 68}")
    for query, ok, note in results:
        icon = "[PASS]" if ok else "[FAIL]"
        print(f"  {icon}  {query}")
        if note:
            print(f"      +-- {note[:100]}")

    if failed:
        raise SystemExit(f"\n{failed} test(s) failed.")
    else:
        print("\n  All tests passed.")


if __name__ == "__main__":
    run()
