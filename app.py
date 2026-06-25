"""Query fan-out tool — Streamlit front end.

UI only. The work lives in core/ (elicit / model / patterns / brief / cost). Bring-your-own-keys:
keys are read from password inputs, kept in session state, passed into the core functions, and never
persisted. See tool-build-plan.md for the build phases.
"""
import json
import re
import threading
import time

import streamlit as st

from core import brief, cost, elicit, model, patterns
from core.persona_fields import PERSONA_FIELDS, assemble

st.set_page_config(page_title="Query fan-out tool", layout="wide", initial_sidebar_state="expanded")

# ---- session state ----
st.session_state.setdefault("personas", [])
st.session_state.setdefault("results", None)


def _slug(name: str, idx: int) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    if not s or s == "base":
        s = f"persona{idx + 1}"
    return s


ENGINE_DISPLAY = {"openai": "OpenAI", "gemini": "Gemini", "anthropic": "Anthropic"}

# ---------------------------------------------------------------- sidebar ----
with st.sidebar:
    elicited_engines = st.multiselect(
        "Elicit live fan-outs from", ["openai", "gemini", "anthropic"], default=[],
        format_func=lambda e: f"{ENGINE_DISPLAY[e]} ({elicit.MODELS[e]})",
        help="The real sub-queries each engine searches. Slower and pricier; needs that engine's key. "
             "Each provider uses one fixed model (shown in the option).",
    )
    modeled_base = st.checkbox("Modeled fan-out (no persona)", value=True,
                               help="An LLM's prediction of the fan-out for an anonymous searcher.")
    modeled_personas = st.checkbox("Modeled fan-out (with personas)", value=False,
                                   help="Predict the fan-out for specific buyer personas.")
    model_engine = "gemini"  # modeling is fixed to the validated model (the prompt was tuned on Flash)
    st.caption(f"Modeled fan-outs use {model.MODELS['gemini']}.")

    runs = int(st.number_input(
        "Runs per query", min_value=1, max_value=20, value=5,
        help="Each query is run this many times and the results pooled — a single fan-out is noisy. "
             "5–10 is the sweet spot.",
    ))
    do_patterns = not st.checkbox("Skip PATTERNS (free deterministic analysis)", value=False)
    do_briefs = not st.checkbox("Skip BRIEFS (the writer's brief)", value=False)

    # Which keys are actually required given the current selections (drives the labels below).
    _using_modeled = modeled_base or modeled_personas
    need_openai = do_briefs or ("openai" in elicited_engines) or (_using_modeled and model_engine == "openai")
    need_gemini = (do_patterns or do_briefs or ("gemini" in elicited_engines)
                   or (_using_modeled and model_engine == "gemini"))
    need_anthropic = ("anthropic" in elicited_engines) or (_using_modeled and model_engine == "anthropic")

    st.header("API keys")
    st.caption("Used in memory for this run only — never logged or stored.")

    def _klabel(name: str, need: bool) -> str:
        return f"{name} API key" + (" — required" if need else " — not needed for current selections")

    openai_key = st.text_input(_klabel("OpenAI", need_openai), type="password")
    gemini_key = st.text_input(_klabel("Google Gemini", need_gemini), type="password")
    anthropic_key = st.text_input(_klabel("Anthropic", need_anthropic), type="password")

keys = {"openai": openai_key, "gemini": gemini_key, "anthropic": anthropic_key}

# ----------------------------------------------------------------- header ----
st.title("Query fan-out tool")
st.caption(
    'Made by Ian Miller '
    '<a href="https://www.linkedin.com/in/ian-l-miller" target="_blank" title="Ian Miller on LinkedIn">'
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="#0A66C2" '
    'style="vertical-align:text-bottom;"><path d="M20.45 20.45h-3.56v-5.57c0-1.33-.02-3.04-1.85-3.04-1.85 '
    '0-2.13 1.45-2.13 2.94v5.67H9.35V9h3.42v1.56h.05c.48-.9 1.64-1.85 3.37-1.85 3.6 0 4.27 2.37 4.27 '
    '5.45v6.29zM5.34 7.43a2.07 2.07 0 1 1 0-4.14 2.07 2.07 0 0 1 0 4.14zM7.12 20.45H3.56V9h3.56v11.45zM22.22 '
    '0H1.77C.79 0 0 .77 0 1.73v20.54C0 23.23.79 24 1.77 24h20.45c.98 0 1.78-.77 1.78-1.73V1.73C24 .77 23.2 '
    '0 22.22 0z"/></svg></a>',
    unsafe_allow_html=True,
)
st.caption(
    "Model the sub-queries an AI search engine fans a question into (optionally per buyer persona), "
    "pool repeated runs, and get a deterministic entity analysis + a writer's brief — every angle and "
    "source tagged by where it came from."
)

# ---------------------------------------------------------------- queries ----
st.caption("⚙️ Your API keys and run options are in the sidebar — if it isn't visible, tap the ›› at the "
           "top-left to open it.")

queries_text = st.text_area("Queries (one per line)", height=110,
                            placeholder="what's the best way to get out of credit card debt?")
queries = [q.strip() for q in queries_text.splitlines() if q.strip()]

# --------------------------------------------------------------- personas ----
if modeled_personas:
    st.subheader("Buyer personas")
    for idx, p in enumerate(st.session_state.personas):
        with st.expander(f"Persona {idx + 1}: {p.get('name') or 'unnamed'}", expanded=True):
            p["name"] = st.text_input("Persona name", value=p.get("name", ""), key=f"pname{idx}")
            p.setdefault("fields", {})
            for f in PERSONA_FIELDS:
                p["fields"][f["key"]] = st.text_input(
                    f["label"], value=p["fields"].get(f["key"], ""), help=f["help"],
                    placeholder=f.get("placeholder", ""), key=f"p{idx}_{f['key']}",
                )
            if st.button("Remove this persona", key=f"prm{idx}"):
                st.session_state.personas.pop(idx)
                st.rerun()
    if st.button("Add another persona", type="primary", icon=":material/add:"):
        st.session_state.personas.append({"name": "", "fields": {}})
        st.rerun()

personas = st.session_state.personas if modeled_personas else []

# ----------------------------------------------------------- cost estimate ----
n_personas = len(personas)
est = cost.estimate(
    n_queries=max(len(queries), 1), runs=runs, elicited_engines=elicited_engines,
    modeled_base=modeled_base, n_personas=n_personas, do_patterns=do_patterns, do_briefs=do_briefs,
)
st.markdown(
    f'<div style="background:#f7f7f4; border-left:3px solid #557c63; padding:0.7rem 1rem; '
    f'border-radius:4px; margin:0.5rem 0; font-size:0.92rem;">'
    f'<strong>Estimated API cost: ~&#36;{est["total"]:.2f}</strong> for {max(len(queries), 1)} '
    f'quer{"y" if max(len(queries), 1) == 1 else "ies"} &times; {runs} runs. '
    f'This is a rough estimate of what the AI providers (OpenAI, Anthropic, Google) will charge to '
    f'<strong>your own API keys</strong> for this run — the tool itself is free and never charges you anything.'
    f'</div>',
    unsafe_allow_html=True,
)


def _validate() -> list[str]:
    errs = []
    if not queries:
        errs.append("Enter at least one query.")
    if not (elicited_engines or modeled_base or modeled_personas):
        errs.append("Select at least one thing to run (elicited engines and/or modeled).")
    for e in elicited_engines:
        if not keys.get(e):
            errs.append(f"Elicited '{e}' needs the {e} API key.")
    if (modeled_base or modeled_personas) and not keys.get(model_engine):
        errs.append(f"Modeling with '{model_engine}' needs the {model_engine} API key.")
    if modeled_personas and not any((p.get("fields") and assemble(p["fields"])) for p in personas):
        errs.append("Add at least one persona with some fields filled in, or uncheck personas.")
    if (do_patterns or do_briefs) and not keys.get("gemini"):
        errs.append("PATTERNS/BRIEFS need the Gemini key (for embeddings). Add it or skip both.")
    if do_briefs and not keys.get("openai"):
        errs.append("BRIEFS needs the OpenAI key (for the writer's brief). Add it or skip BRIEFS.")
    return errs


def _friendly(err) -> str:
    """Turn an exception or error string into a short, user-facing reason."""
    s = str(err).lower()
    if any(k in s for k in ("401", "unauthorized", "invalid api key", "invalid_api_key", "api key not valid",
                            "incorrect api key", "authentication", "permission denied")):
        return "the API key looks invalid or unauthorized — check you pasted the right key in the right box"
    if any(k in s for k in ("429", "quota", "rate limit", "rate_limit", "insufficient_quota",
                            "insufficient credit", "billing", "out of credit")):
        return "the account hit a rate limit or is out of credits/quota"
    if "timeout" in s or "timed out" in s:
        return "the request timed out — the provider may be slow; try fewer runs or engines"
    if any(k in s for k in ("connection", "network", "503", "502", "service unavailable",
                            "overloaded", "econnreset", "ssl")):
        return "couldn't reach the provider (network/service issue) — try again in a moment"
    return str(err)


def _run_step(fn, prog, start, end):
    """Run blocking fn() in a worker thread while creeping the progress bar from `start` toward `end`,
    so the bar never stalls during a long call. Returns fn()'s result; re-raises any error it threw."""
    holder = {}

    def work():
        try:
            holder["result"] = fn()
        except Exception as exc:  # noqa: BLE001 — surfaced on the main thread
            holder["error"] = exc

    t = threading.Thread(target=work, daemon=True)
    t.start()
    pct = start
    ceiling = max(start, end - 0.01)  # don't reach the segment end until the work actually finishes
    while t.is_alive():
        time.sleep(0.25)
        pct = min(pct + (ceiling - pct) * 0.08 + 0.004, ceiling)  # ease toward ceiling, always inch forward
        prog.progress(min(pct, 1.0))
    t.join()
    prog.progress(min(end, 1.0))
    if "error" in holder:
        raise holder["error"]
    return holder.get("result")


# -------------------------------------------------------------------- run ----
if st.button("▶ Run", type="primary"):
    errs = _validate()
    if errs:
        for e in errs:
            st.error(e)
    else:
        results = []
        cache: dict = {}
        n_pers_run = sum(1 for p in personas if assemble(p.get("fields", {})))
        steps_per_q = ((1 if elicited_engines else 0) + (1 if modeled_base else 0) + n_pers_run
                       + (1 if do_patterns else 0) + (1 if do_briefs else 0))
        total = max(steps_per_q * len(queries), 1)
        done = 0
        prog = st.progress(0.0)
        with st.status("Running… this can take a while — keep this tab open.", expanded=True) as status:
            for q in queries:
                st.write(f"**{q}**")
                caps = []
                try:
                    if elicited_engines:
                        st.write(f"  · eliciting from {', '.join(elicited_engines)} — this is the slow part, "
                                 f"give it a moment ({runs} runs each)…")
                        ecap = _run_step(lambda: elicit.elicit(q, elicited_engines, runs, keys),
                                         prog, done / total, (done + 1) / total)
                        done += 1
                        caps.append(ecap)
                        for eng, rec in ecap["engines"].items():
                            if not any("error" not in r for r in rec["runs"]):
                                first = next((r["error"] for r in rec["runs"] if "error" in r), "")
                                st.warning(f"⚠️ {ENGINE_DISPLAY.get(eng, eng)} elicitation failed — {_friendly(first)}")
                    if modeled_base:
                        st.write("  · modeling (no persona)…")
                        mcap = _run_step(
                            lambda: model.model_one(q, None, model_engine, runs, keys[model_engine], None),
                            prog, done / total, (done + 1) / total)
                        done += 1
                        caps.append(mcap)
                        if not any("error" not in r for r in mcap["result"]["runs"]):
                            first = next((r["error"] for r in mcap["result"]["runs"] if "error" in r), "")
                            st.warning(f"⚠️ Modeled (no persona) failed — {_friendly(first)}")
                    if modeled_personas:
                        for idx, p in enumerate(personas):
                            ptext = assemble(p.get("fields", {}))
                            if not ptext:
                                continue
                            nm = p.get("name") or idx + 1
                            st.write(f"  · modeling persona '{nm}'…")
                            label = _slug(p.get("name", ""), idx)
                            pcap = _run_step(
                                lambda ptext=ptext, label=label: model.model_one(
                                    q, ptext, model_engine, runs, keys[model_engine], label),
                                prog, done / total, (done + 1) / total)
                            done += 1
                            caps.append(pcap)
                            if not any("error" not in r for r in pcap["result"]["runs"]):
                                first = next((r["error"] for r in pcap["result"]["runs"] if "error" in r), "")
                                st.warning(f"⚠️ Persona '{nm}' modeling failed — {_friendly(first)}")
                    pat = brf = None
                    has_data = any(
                        ("engines" in c and any(r.get("queries") for rec in c["engines"].values()
                                                for r in rec["runs"]))
                        or ("result" in c and any(r.get("sub_queries") for r in c["result"]["runs"]))
                        for c in caps)
                    if (do_patterns or do_briefs) and not has_data:
                        st.error(f"❌ No fan-outs were captured for “{q}” — every selected source failed "
                                 f"(see the warnings above), so there's nothing to analyze. Check your keys/credits.")
                    if has_data and do_patterns:
                        try:
                            pat = _run_step(lambda: patterns.patterns_md(q, caps, keys["gemini"], cache),
                                            prog, done / total, (done + 1) / total)
                            st.write("  · PATTERNS done")
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"❌ PATTERNS failed for “{q}” — {_friendly(exc)}")
                        done += 1
                    if has_data and do_briefs:
                        try:
                            brf = _run_step(lambda: brief.brief_md(q, caps, keys["gemini"], keys["openai"], cache),
                                            prog, done / total, (done + 1) / total)
                            st.write("  · BRIEFS done")
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"❌ BRIEFS failed for “{q}” — {_friendly(exc)}")
                        done += 1
                    results.append({"query": q, "captures": caps, "patterns": pat, "brief": brf})
                except Exception as exc:  # noqa: BLE001 — backstop so one query can't kill the whole run
                    st.error(f"❌ “{q}” failed — {_friendly(exc)}")
            prog.progress(1.0)
            status.update(label="Done", state="complete")
        st.session_state.results = results

# ---------------------------------------------------------------- results ----
results = st.session_state.results
if results:
    st.divider()
    st.header("Results")
    for ri, r in enumerate(results):
        st.subheader(r["query"])
        tabs = st.tabs(["Brief", "Patterns", "Raw elicited", "Raw modeled"])
        with tabs[0]:
            if r["brief"]:
                st.markdown(r["brief"])
                st.download_button("Download brief (.md)", r["brief"],
                                   file_name="brief.md", key=f"dlb_{ri}")
            else:
                st.caption("BRIEFS was skipped.")
        with tabs[1]:
            if r["patterns"]:
                st.markdown(r["patterns"])
                st.download_button("Download patterns (.md)", r["patterns"],
                                   file_name="patterns.md", key=f"dlp_{ri}")
            else:
                st.caption("PATTERNS was skipped.")
        elicited_caps = [c for c in r["captures"] if "engines" in c]
        modeled_caps = [c for c in r["captures"] if "result" in c]
        with tabs[2]:
            if elicited_caps:
                for c in elicited_caps:
                    for eng, rec in c["engines"].items():
                        st.markdown(f"**{eng}** — {rec.get('model','')}")
                        for run in rec["runs"]:
                            if "queries" in run:
                                st.markdown(f"*run {run['run']}:* " + "; ".join(run["queries"]))
                st.download_button("Download raw elicited (.json)",
                                   json.dumps(elicited_caps, indent=2, ensure_ascii=False),
                                   file_name="elicited.json", key=f"dle_{ri}")
            else:
                st.caption("No elicited fan-outs were run.")
        with tabs[3]:
            if modeled_caps:
                for c in modeled_caps:
                    label = c.get("persona_file") or "base (no persona)"
                    st.markdown(f"**{label}** — {c.get('model','')}")
                    for run in c["result"]["runs"]:
                        subs = [s.get("sub_query", "") for s in run.get("sub_queries", [])]
                        if subs:
                            st.markdown(f"*run {run['run']}:* " + "; ".join(subs))
                st.download_button("Download raw modeled (.json)",
                                   json.dumps(modeled_caps, indent=2, ensure_ascii=False),
                                   file_name="modeled.json", key=f"dlm_{ri}")
            else:
                st.caption("No modeled fan-outs were run.")
