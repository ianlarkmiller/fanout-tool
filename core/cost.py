"""Cost: provider prices + a pre-run estimate for the UI.

PRICES are the per-token sheet (used later for exact post-run accounting). The pre-run `estimate()`
uses MEASURED per-run averages (real token counts from real runs) priced at this sheet — more honest
than guessing token counts before anything has run. It is an average, not a ceiling: a single run can
cost noticeably more or less depending on the query (see the per-constant notes).
"""
from __future__ import annotations

# USD per 1,000,000 tokens, plus per-call fees. Verified vs vendor pricing 2026-06-23.
PRICES = {
    "openai":    {"in": 5.00, "out": 30.00, "per_search": 0.010},        # gpt-5.5; web search $10/1k
    "anthropic": {"in": 3.00, "out": 15.00, "per_search": 0.010},        # Claude Sonnet 4.6; web search ~$10/1k
    "gemini":    {"in": 1.50, "out": 9.00,  "per_grounded_call": 0.035}, # Gemini 3.5 Flash; grounding ~$35/1k
    "embedding": 0.15,                                                    # gemini-embedding-001, per 1M tokens
}

# Measured per-RUN averages, re-derived 2026-10-24. The earlier figures (openai 0.106, gemini 0.038,
# modeled 0.008/0.011, briefs 0.05) were priced on an older, cheaper rate sheet and left out Gemini's
# thinking tokens, so real runs came in ~1.2-2.9x over the estimate. Sources:
# - openai: the 100 saved finance-set runs (10 queries x 10) re-priced at this sheet ($0.139 of tokens)
#   + ~2.4 web-search calls/run x $0.01 (measured 2026-10-24). Varies ~$0.06-0.27 by query.
# - anthropic: the same 100 runs; already priced at this sheet (2026-10-24 runs came in lower, ~$0.06).
# - gemini elicited + modeled: 2026-10-24 real runs with thinking tokens counted as output (15 elicited,
#   15 base, 8 persona runs). The saved finance-set captures can't be re-used here: they never recorded
#   thinking tokens.
ELICITED_PER_RUN = {"openai": 0.16, "gemini": 0.054, "anthropic": 0.096}
MODELED_BASE_PER_RUN = 0.017      # base, no persona
MODELED_PERSONA_PER_RUN = 0.024   # each persona

# Per-QUERY analysis costs (pooled across whatever sources were selected).
PATTERNS_PER_QUERY = 0.001        # cold embeddings; ~free once cached within a session
BRIEFS_PER_QUERY = 0.12           # mean of 107 real briefs at this sheet (98 article + 9 on 2026-10-24); ~$0.01-0.29

# Rough per-step DURATIONS in seconds (measured 2026-06-25) — used to pace the progress bar so it
# tracks the real work. elicit_run = one engine, one run; model_run = one modeled run.
STEP_SECONDS = {"elicit_run": 33.0, "model_run": 10.0, "patterns": 3.0, "brief": 50.0}


def estimate(
    *,
    n_queries: int,
    runs: int,
    elicited_engines: list[str],
    modeled_base: bool,
    n_personas: int,
    do_patterns: bool,
    do_briefs: bool,
) -> dict:
    """Rough pre-run cost estimate, broken out by stage. All values USD.

    elicited_engines: subset of {"openai","gemini","anthropic"} to elicit from (empty = none).
    modeled_base:     include the no-persona modeled pass.
    n_personas:       number of buyer personas for the with-personas modeled pass.
    do_patterns/do_briefs: whether each analysis pass runs.
    """
    elicited = sum(ELICITED_PER_RUN.get(e, 0.0) for e in elicited_engines) * runs * n_queries

    modeled = 0.0
    if modeled_base:
        modeled += MODELED_BASE_PER_RUN * runs * n_queries
    modeled += MODELED_PERSONA_PER_RUN * n_personas * runs * n_queries

    patterns = PATTERNS_PER_QUERY * n_queries if do_patterns else 0.0
    briefs = BRIEFS_PER_QUERY * n_queries if do_briefs else 0.0

    total = elicited + modeled + patterns + briefs
    return {
        "elicited": round(elicited, 4),
        "modeled": round(modeled, 4),
        "patterns": round(patterns, 4),
        "briefs": round(briefs, 4),
        "total": round(total, 4),
    }
