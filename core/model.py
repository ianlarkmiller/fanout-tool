"""Modeled (guessed) query fan-outs (ported from model_fanout.py).

Asks an LLM to predict the sub-queries an AI search engine would fan a question into, optionally
conditioned on a buyer persona. Refactor vs. the script: API keys are arguments, the prompt is read
from the bundled core/prompts/modeled-fanout.md, no .env loading, no file writes, no stdout, no
spend ledger. `model_one()` returns the same capture dict shape the analyzer consumes.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

MODELS = {
    "gemini": "gemini-3.5-flash",
    "openai": "gpt-5.5",
    "anthropic": "claude-sonnet-4-6",
}
TEMPERATURE = 0.7

PRICES = {  # USD per 1M tokens; modeling has no web-search surcharge
    "gemini":    {"in": 1.50, "out": 9.00},
    "openai":    {"in": 5.00, "out": 30.00},
    "anthropic": {"in": 3.00, "out": 15.00},
}

PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "modeled-fanout.md")


def build_prompt(query: str, persona: str | None = None) -> str:
    """Assemble the modeled-fan-out prompt. Base prompt if persona is None/empty; otherwise inject
    the persona add-on at {persona_block}. HTML comments are stripped; {year} is the current year."""
    with open(PROMPT_PATH, encoding="utf-8") as fh:
        text = re.sub(r"<!--.*?-->", "", fh.read(), flags=re.S)
    base, _, addon = text.partition("===PERSONA-ADDON===")
    base = base.replace("===PROMPT===", "").strip()
    block = ""
    if persona and persona.strip():
        block = "\n" + addon.strip().replace("{persona}", persona.strip()) + "\n"
    year = str(_dt.datetime.now().year)
    return (base.replace("{persona_block}", block)
                .replace("{year}", year)
                .replace("{query}", query.strip()))


def _extract_json(text: str) -> dict:
    """Models occasionally wrap JSON in ```json fences or stray prose — be forgiving."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.M)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def call_gemini(prompt: str, api_key: str) -> tuple[dict, dict]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=120_000))
    resp = client.models.generate_content(
        model=MODELS["gemini"],
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=TEMPERATURE,
        ),
    )
    um = resp.usage_metadata
    usage = {"in": int(getattr(um, "prompt_token_count", 0) or 0),
             "out": int(getattr(um, "candidates_token_count", 0) or 0)}
    return _extract_json(resp.text or ""), usage


def call_openai(prompt: str, api_key: str) -> tuple[dict, dict]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, timeout=120, max_retries=1)
    resp = client.responses.create(
        model=MODELS["openai"],
        input=prompt,
        temperature=TEMPERATURE,
    )
    u = getattr(resp, "usage", None)
    usage = {"in": int(getattr(u, "input_tokens", 0) or 0),
             "out": int(getattr(u, "output_tokens", 0) or 0)}
    return _extract_json(resp.output_text or ""), usage


def call_anthropic(prompt: str, api_key: str) -> tuple[dict, dict]:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key, timeout=120, max_retries=1)
    msg = client.messages.create(
        model=MODELS["anthropic"],
        max_tokens=4096,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    usage = {"in": int(msg.usage.input_tokens), "out": int(msg.usage.output_tokens)}
    return _extract_json(text), usage


ENGINES = {
    "gemini": ("Gemini", call_gemini),
    "openai": ("OpenAI", call_openai),
    "anthropic": ("Claude", call_anthropic),
}


def cost(engine: str, usage: dict) -> float:
    p = PRICES[engine]
    return usage["in"] / 1e6 * p["in"] + usage["out"] / 1e6 * p["out"]


def model_one(query: str, persona: str | None, engine: str, runs: int, api_key: str,
              persona_label: str | None = None) -> dict:
    """Run the modeled fan-out `runs` times for one query. `persona` is the assembled BUYER PERSONA
    text (None/empty = base, no-persona). `persona_label` names the modeled source for the analyzer
    (None -> "base"). Returns a capture dict: {query, persona_file, engine, model, result:{runs:[...]}}."""
    fn = ENGINES[engine][1]
    prompt = build_prompt(query, persona)

    def _one(i: int) -> dict:
        try:
            data, usage = fn(prompt, api_key)
        except Exception as exc:  # noqa: BLE001 — record and continue
            return {"run": i, "error": f"{type(exc).__name__}: {exc}"}
        c = cost(engine, usage)
        subs = data.get("sub_queries", []) if isinstance(data, dict) else []
        return {"run": i, "persona_reading": data.get("persona_reading", ""),
                "sub_queries": subs, **usage, "cost_usd": round(c, 6)}

    with ThreadPoolExecutor(max_workers=max(runs, 1)) as ex:
        runs_list = list(ex.map(_one, range(1, runs + 1)))  # ex.map preserves run order
    rec = {"engine": engine, "model": MODELS[engine], "query": query, "runs": runs_list,
           "total_cost_usd": round(sum(r.get("cost_usd", 0.0) for r in runs_list), 6)}
    return {"query": query, "persona_file": persona_label, "engine": engine,
            "model": MODELS[engine], "result": rec}
