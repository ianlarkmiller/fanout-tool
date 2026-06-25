"""Query fan-out tool — Streamlit front end.

UI only. All the work lives in the `core` package (model / elicit / patterns / brief / cost).
Bring-your-own-keys: API keys are read from password inputs, kept in session state, passed into
the core functions, and never persisted. See tool-build-plan.md for the build phases.
"""
import streamlit as st

from core import cost
from core.persona_fields import PERSONA_FIELDS

st.set_page_config(page_title="Query fan-out tool", layout="wide")

st.title("Query fan-out tool")
st.caption(
    "Model the sub-queries an AI search engine fans a question into (optionally per buyer persona), "
    "pool repeated runs, and get a deterministic entity analysis + a writer's brief — every angle and "
    "source tagged by where it came from. Bring your own API keys."
)

st.info(
    "🚧 Under construction. Phase 0 scaffold is in place; the core extraction (Phase 1) and the "
    "input/output UI (Phases 2+) are next. See tool-build-plan.md."
)

# Smoke check that the core package imports and the new modules are wired up.
with st.expander("dev: core wiring check", expanded=False):
    st.write(f"persona template fields: {[f['key'] for f in PERSONA_FIELDS]}")
    demo = cost.estimate(
        n_queries=1, runs=10, elicited_engines=[], modeled_base=True,
        n_personas=1, do_patterns=True, do_briefs=True,
    )
    st.write("example cost estimate (1 query, 10 runs, base + 1 persona, PATTERNS + BRIEFS):", demo)
