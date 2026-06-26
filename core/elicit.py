"""Elicited (real) query fan-outs (ported from fanout.py).

Runs a question through each engine with web search enabled and extracts the actual sub-queries it
searched (OpenAI web_search_call actions; Anthropic web_search tool_use blocks; Gemini
grounding_metadata.web_search_queries). Refactor vs. the script: API keys are arguments, no argparse,
no stdout, no file writes. `elicit()` returns the capture dict shape the analyzer consumes.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

OPENAI_MODEL = "gpt-5.5"
ANTHROPIC_MODEL = "claude-sonnet-4-6"
ANTHROPIC_WEB_SEARCH_TOOL = "web_search_20250305"
ANTHROPIC_MAX_USES = 20
GEMINI_MODEL = "gemini-3.5-flash"

MODELS = {"openai": OPENAI_MODEL, "anthropic": ANTHROPIC_MODEL, "gemini": GEMINI_MODEL}

PRICES = {
    "openai":    {"in": 5.00, "out": 30.00, "per_search": 0.010},
    "anthropic": {"in": 3.00, "out": 15.00, "per_search": 0.010},
    "gemini":    {"in": 1.50, "out": 9.00,  "per_grounded_call": 0.035},
}

SEARCH_INSTRUCTION = (
    "You are answering a user's web-search query. Always search the web to "
    "answer it -- issue whatever searches you need -- and base your answer on "
    "current web results, not on prior knowledge alone."
)


def fanout_openai(prompt: str, api_key: str) -> tuple[list[str], dict]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, timeout=180, max_retries=1)
    resp = client.responses.create(
        model=OPENAI_MODEL,
        tools=[{"type": "web_search"}],
        instructions=SEARCH_INSTRUCTION,
        input=prompt,
    )
    queries: list[str] = []
    for item in resp.output:
        if getattr(item, "type", None) != "web_search_call":
            continue
        action = getattr(item, "action", None)
        if action is None or getattr(action, "type", None) != "search":
            continue
        action_qs: list[str] = []
        q = getattr(action, "query", None)
        if q:
            action_qs.append(q)
        for q in getattr(action, "queries", None) or []:
            if q:
                action_qs.append(q)
        queries.extend(dict.fromkeys(action_qs))
    return queries, _usage(getattr(resp, "usage", None), "input_tokens", "output_tokens", len(queries))


def fanout_anthropic(prompt: str, api_key: str) -> tuple[list[str], dict]:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key, timeout=180, max_retries=1)
    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=4096,
        system=SEARCH_INSTRUCTION,
        tools=[{"type": ANTHROPIC_WEB_SEARCH_TOOL, "name": "web_search", "max_uses": ANTHROPIC_MAX_USES}],
        messages=[{"role": "user", "content": prompt}],
    )
    queries: list[str] = []
    for block in msg.content:
        if getattr(block, "type", None) == "server_tool_use" and getattr(block, "name", None) == "web_search":
            q = (block.input or {}).get("query")
            if q:
                queries.append(q)
    usage = _usage(msg.usage, "input_tokens", "output_tokens", len(queries))
    if len(queries) >= ANTHROPIC_MAX_USES:
        usage["truncated"] = True
    return queries, usage


def fanout_gemini(prompt: str, api_key: str) -> tuple[list[str], dict]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=180_000))
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            system_instruction=SEARCH_INSTRUCTION,
        ),
    )
    queries: list[str] = []
    for cand in resp.candidates or []:
        gm = getattr(cand, "grounding_metadata", None)
        if gm and getattr(gm, "web_search_queries", None):
            queries.extend(q for q in gm.web_search_queries if q and q.strip())
    um = resp.usage_metadata
    return queries, _usage(um, "prompt_token_count", "candidates_token_count", len(queries))


def _usage(obj, in_attr: str, out_attr: str, searches: int) -> dict:
    tin = getattr(obj, in_attr, None) or getattr(obj, "prompt_tokens", 0) or 0
    tout = getattr(obj, out_attr, None) or getattr(obj, "completion_tokens", 0) or 0
    return {"in": int(tin), "out": int(tout), "searches": searches}


def call_cost(key: str, usage: dict) -> float:
    p = PRICES[key]
    c = usage["in"] / 1e6 * p["in"] + usage["out"] / 1e6 * p["out"]
    c += usage["searches"] * p.get("per_search", 0.0)
    c += p.get("per_grounded_call", 0.0)
    return c


ENGINES = {
    "openai": ("OpenAI", fanout_openai),
    "anthropic": ("Claude", fanout_anthropic),
    "gemini": ("Gemini", fanout_gemini),
}


def _one_run(key: str, prompt: str, api_key: str, i: int) -> dict:
    """One engine, one run. Returns a per-run record (errors recorded, never raised)."""
    fn = ENGINES[key][1]
    try:
        qs, usage = fn(prompt, api_key)
    except Exception as exc:  # noqa: BLE001 — record and continue
        return {"run": i, "error": f"{type(exc).__name__}: {exc}"}
    c = call_cost(key, usage)
    return {"run": i, "queries": qs, **usage, "cost_usd": round(c, 6)}


def elicit(prompt: str, engines: list[str], runs: int, keys: dict[str, str]) -> dict:
    """Elicit live fan-outs from each requested engine that has a key. ALL (engine × run) calls run
    concurrently. `keys`: {engine: api_key}. Returns {prompt, engines: {engine: record}}."""
    jobs = [(e, i) for e in engines if keys.get(e) for i in range(1, runs + 1)]
    out: dict = {"prompt": prompt, "engines": {}}
    if not jobs:
        return out
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        recs = list(ex.map(lambda ei: (ei[0], _one_run(ei[0], prompt, keys[ei[0]], ei[1])), jobs))
    by_engine: dict[str, list] = {}
    for e, rec in recs:
        by_engine.setdefault(e, []).append(rec)
    for e in engines:
        if not keys.get(e):
            continue
        runs_list = sorted(by_engine.get(e, []), key=lambda r: r["run"])
        out["engines"][e] = {"label": ENGINES[e][0], "model": MODELS[e], "runs": runs_list,
                             "total_cost_usd": round(sum(r.get("cost_usd", 0.0) for r in runs_list), 6)}
    return out
