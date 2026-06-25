"""Query fan-out tool — Streamlit front end.

UI only. The work lives in core/ (elicit / model / patterns / brief / cost). Bring-your-own-keys:
keys are read from password inputs, kept in session state, passed into the core functions, and never
persisted. See tool-build-plan.md for the build phases.
"""
import json
import re

import streamlit as st

from core import brief, cost, elicit, model, patterns
from core.persona_fields import PERSONA_FIELDS, assemble

st.set_page_config(page_title="Query fan-out tool", layout="wide")

# ---- session state ----
st.session_state.setdefault("personas", [])
st.session_state.setdefault("results", None)


def _slug(name: str, idx: int) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    if not s or s == "base":
        s = f"persona{idx + 1}"
    return s


# ---------------------------------------------------------------- sidebar ----
with st.sidebar:
    st.header("API keys")
    st.caption("Used in memory for this run only — never logged or stored. Bring your own.")
    openai_key = st.text_input("OpenAI API key", type="password")
    gemini_key = st.text_input("Google Gemini API key", type="password")
    anthropic_key = st.text_input("Anthropic API key (optional)", type="password")

    st.header("What to run")
    elicited_engines = st.multiselect(
        "Elicit live fan-outs from", ["openai", "gemini", "anthropic"], default=[],
        help="The real sub-queries each engine searches. Slower and pricier; needs that engine's key.",
    )
    modeled_base = st.checkbox("Modeled fan-out (no persona)", value=True,
                               help="An LLM's prediction of the fan-out for an anonymous searcher.")
    modeled_personas = st.checkbox("Modeled fan-out (with personas)", value=False,
                                   help="Predict the fan-out for specific buyer personas (defined below).")
    model_engine = st.selectbox("Model with", ["gemini", "openai", "anthropic"], index=0,
                                help="Which LLM generates the modeled fan-outs.")

    runs = int(st.number_input(
        "Runs per query", min_value=1, max_value=20, value=5,
        help="Each query is run this many times and the results pooled — a single fan-out is noisy. "
             "5–10 is the sweet spot.",
    ))
    do_patterns = not st.checkbox("Skip PATTERNS (free deterministic analysis)", value=False)
    do_briefs = not st.checkbox("Skip BRIEFS (the writer's brief)", value=False)

keys = {"openai": openai_key, "gemini": gemini_key, "anthropic": anthropic_key}

# ----------------------------------------------------------------- header ----
st.title("Query fan-out tool")
st.caption(
    "Model the sub-queries an AI search engine fans a question into (optionally per buyer persona), "
    "pool repeated runs, and get a deterministic entity analysis + a writer's brief — every angle and "
    "source tagged by where it came from."
)

# ---------------------------------------------------------------- queries ----
queries_text = st.text_area("Queries (one per line)", height=110,
                            placeholder="what's the best way to get out of credit card debt?")
queries = [q.strip() for q in queries_text.splitlines() if q.strip()]

# --------------------------------------------------------------- personas ----
if modeled_personas:
    st.subheader("Buyer personas")
    st.caption("Same six fields each time — that's what keeps personas comparable.")
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
    if st.button("➕ Add another persona"):
        st.session_state.personas.append({"name": "", "fields": {}})
        st.rerun()

personas = st.session_state.personas if modeled_personas else []

# ----------------------------------------------------------- cost estimate ----
n_personas = len(personas)
est = cost.estimate(
    n_queries=max(len(queries), 1), runs=runs, elicited_engines=elicited_engines,
    modeled_base=modeled_base, n_personas=n_personas, do_patterns=do_patterns, do_briefs=do_briefs,
)
st.info(
    f"**Rough cost estimate:** ~${est['total']:.2f} "
    f"(elicited ${est['elicited']:.2f} · modeled ${est['modeled']:.2f} · "
    f"PATTERNS ${est['patterns']:.3f} · BRIEFS ${est['briefs']:.2f}) "
    f"for {max(len(queries),1)} quer{'y' if len(queries)==1 else 'ies'} × {runs} runs. "
    f"You pay your own providers; this estimate uses measured per-run averages."
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


# -------------------------------------------------------------------- run ----
if st.button("▶ Run", type="primary"):
    errs = _validate()
    if errs:
        for e in errs:
            st.error(e)
    else:
        results = []
        cache: dict = {}
        prog = st.progress(0.0)
        with st.status("Running…", expanded=True) as status:
            for qi, q in enumerate(queries):
                st.write(f"**{q}**")
                caps = []
                if elicited_engines:
                    st.write(f"  · eliciting from {', '.join(elicited_engines)} ({runs} runs each)…")
                    caps.append(elicit.elicit(q, elicited_engines, runs, keys))
                if modeled_base:
                    st.write("  · modeling (no persona)…")
                    caps.append(model.model_one(q, None, model_engine, runs, keys[model_engine], None))
                if modeled_personas:
                    for idx, p in enumerate(personas):
                        ptext = assemble(p.get("fields", {}))
                        if not ptext:
                            continue
                        st.write(f"  · modeling persona '{p.get('name') or idx + 1}'…")
                        caps.append(model.model_one(q, ptext, model_engine, runs,
                                                    keys[model_engine], _slug(p.get("name", ""), idx)))
                pat = patterns.patterns_md(q, caps, keys["gemini"], cache) if do_patterns else None
                if do_patterns:
                    st.write("  · PATTERNS done")
                brf = brief.brief_md(q, caps, keys["gemini"], keys["openai"], cache) if do_briefs else None
                if do_briefs:
                    st.write("  · BRIEFS done")
                results.append({"query": q, "captures": caps, "patterns": pat, "brief": brf})
                prog.progress((qi + 1) / len(queries))
            status.update(label="Done", state="complete")
        st.session_state.results = results

# ---------------------------------------------------------------- results ----
results = st.session_state.results
if results:
    st.divider()
    st.header("Results")
    for r in results:
        st.subheader(r["query"])
        tabs = st.tabs(["Brief", "Patterns", "Raw elicited", "Raw modeled"])
        with tabs[0]:
            if r["brief"]:
                st.markdown(r["brief"])
                st.download_button("Download brief (.md)", r["brief"],
                                   file_name="brief.md", key=f"dlb_{r['query'][:20]}")
            else:
                st.caption("BRIEFS was skipped.")
        with tabs[1]:
            if r["patterns"]:
                st.markdown(r["patterns"])
                st.download_button("Download patterns (.md)", r["patterns"],
                                   file_name="patterns.md", key=f"dlp_{r['query'][:20]}")
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
                                   file_name="elicited.json", key=f"dle_{r['query'][:20]}")
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
                                   file_name="modeled.json", key=f"dlm_{r['query'][:20]}")
            else:
                st.caption("No modeled fan-outs were run.")
