"""Cost: provider prices + a pre-run estimate for the UI.

PRICES are the per-token sheet (used later for exact post-run accounting). The pre-run `estimate()`
uses MEASURED per-run averages from the source project's run-costs.md — more honest than guessing
token counts before anything has run.
"""
from __future__ import annotations

# USD per 1,000,000 tokens, plus per-call fees. Verified vs vendor pricing 2026-06-23.
PRICES = {
    "openai":    {"in": 5.00, "out": 30.00, "per_search": 0.010},        # gpt-5.5; web search $10/1k
    "anthropic": {"in": 3.00, "out": 15.00, "per_search": 0.010},        # Claude Sonnet 4.6; web search ~$10/1k
    "gemini":    {"in": 1.50, "out": 9.00,  "per_grounded_call": 0.035}, # Gemini 3.5 Flash; grounding ~$35/1k
    "embedding": 0.15,                                                    # gemini-embedding-001, per 1M tokens
}

# Measured per-RUN averages (run-costs.md, the 10-finance-query set, 10 runs each).
ELICITED_PER_RUN = {"openai": 0.106, "gemini": 0.038, "anthropic": 0.096}
MODELED_BASE_PER_RUN = 0.008      # base, no persona
MODELED_PERSONA_PER_RUN = 0.011   # each persona

# Per-QUERY analysis costs (pooled across whatever sources were selected).
PATTERNS_PER_QUERY = 0.001        # cold embeddings; ~free once cached within a session
BRIEFS_PER_QUERY = 0.05           # typical; the widest pooled pages run up to ~0.15


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
