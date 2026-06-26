"""Query fan-out tool — Streamlit front end.

UI only. The work lives in core/ (elicit / model / patterns / brief / cost). Bring-your-own-keys:
keys are read from password inputs, kept in session state, passed into the core functions, and never
persisted. See tool-build-plan.md for the build phases.
"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait

import streamlit as st

from core import brief, cost, elicit, keycheck, model, patterns
from core.persona_fields import PERSONA_FIELDS, assemble

st.set_page_config(page_title="Query fan-out tool", layout="wide", initial_sidebar_state="expanded")

# Hide Streamlit's per-widget "Press Enter to submit form" hint — our forms commit on the buttons,
# so the hint is misleading (and it's the very behavior the form design removes).
st.markdown('<style>[data-testid="InputInstructions"]{display:none;}</style>', unsafe_allow_html=True)

# ---- session state ----
st.session_state.setdefault("results", None)
st.session_state.setdefault("personas", [])  # list of {"name","fields"} dicts, edited in the main area
# Persist the last-submitted query in the URL so it survives a mobile tab reload (no extra deps).
_restored_q = st.query_params.get("q", "")


def _slug(name: str, idx: int) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    if not s or s == "base":
        s = f"persona{idx + 1}"
    return s


ENGINE_DISPLAY = {"openai": "OpenAI", "gemini": "Gemini", "anthropic": "Anthropic"}
model_engine = "gemini"  # modeling is fixed to the validated model (the prompt was tuned on Flash)

def _klabel(name: str, need: bool) -> str:
    return f"{name} API key" + (" — required" if need else " — not needed for current selections")


# ----------------------------------------- sidebar: run options + API keys ----
with st.sidebar:
    st.header("Run options")
    elicited_engines = st.multiselect(
        "Elicit live fan-outs from", ["openai", "gemini", "anthropic"], default=[],
        format_func=lambda e: f"{ENGINE_DISPLAY[e]} ({elicit.MODELS[e]})",
        help="The real sub-queries each engine searches. Slower and pricier; needs that engine's key.",
    )
    modeled_base = st.checkbox("Modeled fan-out (no persona)", value=True,
                               help="An LLM's prediction of the fan-out for an anonymous searcher.")
    modeled_personas = st.checkbox("Modeled fan-out (with personas)", value=False,
                                   help="Predict the fan-out for specific buyer personas.")
    st.caption(f"Modeled fan-outs use {model.MODELS['gemini']}.")
    runs = int(st.number_input(
        "Runs per query", min_value=1, max_value=20, value=5,
        help="Each query is run this many times and pooled — a single fan-out is noisy. 5–10 is the sweet spot.",
    ))
    do_patterns = not st.checkbox("Skip PATTERNS (free deterministic analysis)", value=False)
    do_briefs = not st.checkbox("Skip BRIEFS (the writer's brief)", value=False)

    _using_modeled = modeled_base or modeled_personas
    need_openai = do_briefs or ("openai" in elicited_engines) or (_using_modeled and model_engine == "openai")
    need_gemini = (do_patterns or do_briefs or ("gemini" in elicited_engines)
                   or (_using_modeled and model_engine == "gemini"))
    need_anthropic = ("anthropic" in elicited_engines) or (_using_modeled and model_engine == "anthropic")

    st.header("API keys")
    st.caption("Used in memory for this run only — never logged or stored.")
    with st.form("keys_form"):
        st.text_input(_klabel("OpenAI", need_openai), type="password", key="openai_key")
        st.text_input(_klabel("Google Gemini", need_gemini), type="password", key="gemini_key")
        st.text_input(_klabel("Anthropic", need_anthropic), type="password", key="anthropic_key")
        _saved = st.form_submit_button("Save & check keys")
    _raw_keys = {p: st.session_state.get(f"{p}_key", "") for p in ("openai", "gemini", "anthropic")}
    keys = {p: v.strip() for p, v in _raw_keys.items()}
    if _saved:
        with st.spinner("Checking keys…"):
            st.session_state["key_status"] = keycheck.check_keys(_raw_keys)
    _kstatus = st.session_state.get("key_status", {})
    for _p in ("openai", "gemini", "anthropic"):
        if not _raw_keys[_p]:
            continue
        _ok, _msg = _kstatus.get(_p, (None, ""))
        if _ok is True:
            st.caption(f"✓ {ENGINE_DISPLAY[_p]} key valid" + (f" — {_msg}" if _msg else ""))
        elif _ok is False:
            st.caption(f"✗ {ENGINE_DISPLAY[_p]} key — {_msg}")
        else:
            st.caption(f"… {ENGINE_DISPLAY[_p]} key entered — tap **Save & check keys**")


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
st.caption("⚙️ Your API keys and run options are in the sidebar — if it isn't visible, tap the ›› at the top-left to open it.")

# ----------------------------------------------------------- cost estimate ----
n_personas = len(st.session_state.personas) if modeled_personas else 0
est = cost.estimate(
    n_queries=1, runs=runs, elicited_engines=elicited_engines,
    modeled_base=modeled_base, n_personas=n_personas, do_patterns=do_patterns, do_briefs=do_briefs,
)
st.markdown(
    f'<div style="background:#f7f7f4; border-left:3px solid #557c63; padding:0.7rem 1rem; '
    f'border-radius:4px; margin:0.5rem 0; font-size:0.92rem;">'
    f'<strong>Estimated API cost: ~&#36;{est["total"]:.2f} per query</strong> (× however many queries '
    f'you enter), at {runs} runs each. This is a rough estimate of what the AI providers (OpenAI, '
    f'Anthropic, Google) will charge to <strong>your own API keys</strong> — the tool itself is free '
    f'and never charges you anything.'
    f'</div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------- inputs ----
queries_text = st.text_area("Queries (one per line)", value=_restored_q, key="query_box", height=110,
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


def _validate() -> list[str]:
    errs = []
    if not queries:
        errs.append("Enter at least one query.")
    if not (elicited_engines or modeled_base or modeled_personas):
        errs.append("Select at least one thing to run (elicited engines and/or modeled).")
    for e in elicited_engines:
        if not keys.get(e):
            errs.append(f"Elicited '{e}' needs the {e} API key.")
    if _using_modeled and not keys.get(model_engine):
        errs.append(f"Modeling needs the {model_engine} API key.")
    if modeled_personas and not any(assemble(p["fields"]) for p in personas):
        errs.append("Add at least one persona with some fields filled in, or uncheck 'Modeled fan-out (with personas)'.")
    if (do_patterns or do_briefs) and not keys.get("gemini"):
        errs.append("PATTERNS/BRIEFS need the Gemini key (for embeddings). Add it or skip both.")
    if do_briefs and not keys.get("openai"):
        errs.append("BRIEFS needs the OpenAI key (for the writer's brief). Add it or skip BRIEFS.")
    _ks = st.session_state.get("key_status", {})
    for _p in ("openai", "gemini", "anthropic"):
        if keys.get(_p) and _ks.get(_p, (None, ""))[0] is False:
            errs.append(f"{ENGINE_DISPLAY[_p]} key failed the check ({_ks[_p][1]}) — fix it in the sidebar "
                        f"and tap “Save & check keys”.")
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


def _run_warnings(label: str, runs_list: list) -> None:
    """Surface a warning if ANY run errored (not only when all of them did)."""
    errs = [r["error"] for r in runs_list if "error" in r]
    if not errs:
        return
    total = len(runs_list)
    ok = total - len(errs)
    if ok == 0:
        st.warning(f"⚠️ {label} failed — all {total} run(s) errored — {_friendly(errs[0])}")
    else:
        st.warning(f"⚠️ {label}: {len(errs)} of {total} run(s) failed — {_friendly(errs[0])} "
                   f"(kept the {ok} that worked)")


def _parallel(tasks, prog, start, end, expected_s):
    """Run tasks (list of (key, callable)) ALL concurrently while advancing the bar from `start` to
    `end` over ~expected_s; snap to `end` when all finish. Returns (results{key:val}, errors{key:exc})."""
    results, errors = {}, {}
    if not tasks:
        prog.progress(min(end, 1.0))
        return results, errors
    ex = ThreadPoolExecutor(max_workers=len(tasks))
    futs = {ex.submit(fn): key for key, fn in tasks}
    t0 = time.time()
    ceiling = max(start, end - 0.005)
    pending = set(futs)
    pct = start
    while pending:
        done_now, pending = wait(pending, timeout=0.3)
        for f in done_now:
            key = futs[f]
            try:
                results[key] = f.result()
            except Exception as exc:  # noqa: BLE001
                errors[key] = exc
        el = time.time() - t0
        if el <= expected_s:
            pct = max(pct, start + (ceiling - start) * (el / expected_s))
        else:  # past the estimate — keep inching toward the ceiling so it never looks frozen
            pct = min(pct + (ceiling - pct) * 0.05 + 0.0015, ceiling)
        prog.progress(min(pct, ceiling))
    ex.shutdown(wait=True)
    prog.progress(min(end, 1.0))
    return results, errors


# -------------------------------------------------------------------- run ----
if st.button("▶ Run", type="primary"):
    # remember the submitted query in the URL (survives a reload); skip if too long for a URL
    _qjoined = "\n".join(queries)
    if queries and len(_qjoined) <= 1500:
        st.query_params["q"] = _qjoined
    errs = _validate()
    if errs:
        for e in errs:
            st.error(e)
    else:
        results = []
        n_pers_run = sum(1 for p in personas if assemble(p.get("fields", {})))
        S = cost.STEP_SECONDS
        # Wall-clock per query with everything parallel: capture (elicited + modeled, all at once) =
        # the slowest of the two; analysis (PATTERNS + BRIEFS at once) = the slower of the two.
        capture_wall = max(S["elicit_run"] if elicited_engines else 0.0,
                           S["model_run"] if (modeled_base or n_pers_run) else 0.0)
        analysis_wall = max(S["patterns"] if do_patterns else 0.0, S["brief"] if do_briefs else 0.0)
        per_q_wall = max(capture_wall + analysis_wall, 1.0)
        cap_frac = capture_wall / per_q_wall   # share of a query's bar span used by the capture phase
        nq = len(queries)
        prog = st.progress(0.0)
        with st.status("Running… this can take a while — keep this tab open.", expanded=True) as status:
            for qi, q in enumerate(queries):
                base = qi / nq
                cap_end = base + cap_frac / nq
                q_end = (qi + 1) / nq
                st.write(f"**{q}**")
                try:
                    # ---- capture phase: elicited + all modeled runs, ALL concurrent ----
                    ctasks = []
                    if elicited_engines:
                        ctasks.append(("elicited", lambda q=q: elicit.elicit(q, elicited_engines, runs, keys)))
                    if modeled_base:
                        ctasks.append(("modeled (no persona)",
                                       lambda q=q: model.model_one(q, None, model_engine, runs,
                                                                   keys[model_engine], None)))
                    if modeled_personas:
                        for idx, p in enumerate(personas):
                            ptext = assemble(p.get("fields", {}))
                            if not ptext:
                                continue
                            nm = p.get("name") or idx + 1
                            label = _slug(p.get("name", ""), idx)
                            ctasks.append((f"persona '{nm}'",
                                           lambda q=q, ptext=ptext, label=label: model.model_one(
                                               q, ptext, model_engine, runs, keys[model_engine], label)))
                    if ctasks:
                        st.write(f"  · running {len(ctasks)} source(s) × {runs} runs in parallel…")
                    cres, cerr = _parallel(ctasks, prog, base, cap_end, capture_wall)
                    caps = []
                    for key, _fn in ctasks:
                        if key in cerr:
                            st.warning(f"⚠️ {key} failed — {_friendly(cerr[key])}")
                            continue
                        cap = cres[key]
                        caps.append(cap)
                        if key == "elicited":
                            for eng, rec in cap["engines"].items():
                                _run_warnings(f"{ENGINE_DISPLAY.get(eng, eng)} elicitation", rec["runs"])
                                if any(r.get("truncated") for r in rec["runs"]):
                                    st.warning(f"⚠️ {ENGINE_DISPLAY.get(eng, eng)} hit its search cap on some "
                                               f"runs — fan-out may be truncated.")
                        else:
                            _run_warnings(key, cap["result"]["runs"])
                    # ---- analysis phase: PATTERNS + BRIEFS concurrent (each its own embed cache) ----
                    has_data = any(
                        ("engines" in c and any(r.get("queries") for rec in c["engines"].values()
                                                for r in rec["runs"]))
                        or ("result" in c and any(r.get("sub_queries") for r in c["result"]["runs"]))
                        for c in caps)
                    if (do_patterns or do_briefs) and not has_data:
                        st.error(f"❌ No fan-outs were captured for “{q}” — every selected source failed "
                                 f"(see the warnings above), so there's nothing to analyze. Check your keys/credits.")
                    atasks = []
                    if has_data and do_patterns:
                        atasks.append(("patterns",
                                       lambda q=q, caps=caps: patterns.patterns_md(q, caps, keys["gemini"], {})))
                    if has_data and do_briefs:
                        atasks.append(("brief",
                                       lambda q=q, caps=caps: brief.brief_md(q, caps, keys["gemini"],
                                                                             keys["openai"], {})))
                    if atasks:
                        st.write("  · generating PATTERNS + BRIEFS in parallel…")
                    ares, aerr = _parallel(atasks, prog, cap_end, q_end, analysis_wall)
                    if "patterns" in aerr:
                        st.error(f"❌ PATTERNS failed for “{q}” — {_friendly(aerr['patterns'])}")
                    if "brief" in aerr:
                        st.error(f"❌ BRIEFS failed for “{q}” — {_friendly(aerr['brief'])}")
                    results.append({"query": q, "captures": caps,
                                    "patterns": ares.get("patterns"), "brief": ares.get("brief")})
                except Exception as exc:  # noqa: BLE001 — backstop so one query can't kill the whole run
                    st.error(f"❌ “{q}” failed — {_friendly(exc)}")
                prog.progress(q_end)  # always sync the bar to the query boundary, whatever happened
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
