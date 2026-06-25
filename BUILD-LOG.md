# Build log

Running record of what's built and validated. Strategy/why lives in the job-search repo
(`projects/article-pipeline/real-vs-guessed-fanouts/web-tool-plan.md`); the engineering plan is in
that repo's `tool-build-plan.md`.

## 2026-06-25 — Phases 0–3 in one autonomous session

**Phase 0 — scaffold (done).** Repo skeleton: `app.py`, `core/` package, `requirements.txt`,
`.streamlit/config.toml`, `.gitignore`, MIT `LICENSE`, `README.md`. New modules written from scratch:
- `core/cost.py` — provider PRICES + a pre-run `estimate()` using measured per-run averages.
- `core/persona_fields.py` — the standardized six-field buyer-persona template + `assemble()`.

**Phase 1 — core extraction (done + validated).** The four scripts from the job-search repo were
ported into an importable, in-memory core. Behaviour-critical logic (entity extraction,
canonicalization, Wilson tiering, provenance, clustering, brief assembly) is unchanged; the only
edits are: API keys passed as arguments, an in-memory embedding cache (no disk `.npz`), the modeled
prompt bundled at `core/prompts/modeled-fanout.md`, in-memory `pool_captures()` entry points, and no
stdout/file-I/O/env-loading/spend-ledger.
- `core/elicit.py`  ← `fanout.py`        (live engine fan-outs; OpenAI/Anthropic/Gemini extraction)
- `core/model.py`   ← `model_fanout.py`  (modeled fan-outs ± persona; `build_prompt`, `model_one`)
- `core/cluster.py` ← `cluster_patterns.py` (Gemini embeddings + sklearn HDBSCAN; `embed(texts, key, cache)`)
- `core/patterns.py`← `unified_fanout.py` (PATTERNS; `patterns_md(prompt, captures, gemini_key)`)
- `core/brief.py`   ← `unified_brief.py`  (BRIEFS; `brief_md(prompt, captures, gemini_key, openai_key)`)

Validation:
- **Offline parity** — pooled the committed q01 captures (3 elicited engines + base/david/maria) and
  ran the ported `entity_spine`: Must tier = `[CFPB, FTC, NFCC]`, exactly matching the committed
  `combined/PATTERNS.md`. `pool_captures` (the in-memory path the app uses) gives identical results to
  the file-based loader.
- **End-to-end smoke** — ran `patterns_md` (real Gemini embeddings + clustering) and `brief_md` (real
  gpt-5.5 brief) on q01. PATTERNS reproduced the committed per-source counts
  (`FTC — openai 10/10✓ · base 10/10✓ · david 1/10 · maria 10/10✓`); the 454-string clustering ran;
  BRIEFS produced a correct page-type/angles/cite-warning brief with the current provenance wording.
  Spend ≈ 5¢.

**Phases 2–3 — Streamlit app (built, not yet live-smoke-tested).** `app.py` wires the full flow:
- Sidebar: BYO API keys (password inputs, in-memory only), elicited-engine multiselect, modeled
  base/persona toggles, model-engine choice, run count (5–10 guidance), skip-PATTERNS / skip-BRIEFS.
- Main: multi-query input (one per line); dynamic personas (add/remove, the six template fields);
  a live cost estimate; a validated Run button with progress.
- Results: per-query tabs — Brief, Patterns, Raw elicited (by engine), Raw modeled (by persona/base)
  — each with a download button.

## What's validated vs. not

- ✅ Core logic ported faithfully (offline parity + end-to-end smoke on real APIs).
- ✅ Modeled + PATTERNS + BRIEFS pipeline runs on real keys.
- ✅ The Streamlit app **boots clean** under Streamlit's headless `AppTest` (no exceptions; title +
  all widgets — Run button, checkboxes, query area, key inputs — build). Fixed a real bug found in
  review: results `download_button` keys now use the result index (were `query[:20]`, which would
  collide and crash Streamlit on similar queries). A full interactive click-through (enter keys +
  click Run + inspect output) is the remaining manual check.
- ⚠️ The **elicited** path is ported and unit-consistent with the captured data, but a live elicited
  run (real engine web-search calls) hasn't been triggered from the app.

## Next

1. `pip install -r requirements.txt` and `streamlit run app.py`; click through one modeled run.
2. Error-handling polish (empty/invalid keys, API failures, quota) surfaced in the UI.
3. Deploy to Streamlit Community Cloud or Hugging Face Spaces (free).
4. Then drop the live link into the article where the tool is teased.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```
Paste your own OpenAI + Gemini keys in the sidebar (Anthropic optional). Keys are used in memory for
the run only.
