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
- ✅ The Streamlit app is **validated headlessly end-to-end** via Streamlit's `AppTest`: it boots
  clean (all widgets build), and driving the real Run path (keys → query → Run) on a *fresh* query
  ("is debt settlement worth it?", not pre-captured) renders both a PATTERNS entity spine and a
  BRIEFS brief with provenance tags — no exceptions. So the modeled → PATTERNS → BRIEFS flow works on
  arbitrary queries through the actual UI code. Also fixed a real bug found in review: results
  `download_button` keys now use the result index (were `query[:20]`, which would collide and crash
  Streamlit on similar queries).
- ✅ The **elicited** path is now validated live too: via AppTest, a Gemini elicited run on a fresh
  query ("best budget mechanical keyboard") captured 7 real web-search sub-queries and rendered
  pooled PATTERNS + BRIEFS with no exceptions. All three paths — modeled, elicited, pooled
  analysis/brief — are exercised on real APIs.

## Status: DEPLOYED (2026-06-25)

Live (public) at **https://fanout-tool.streamlit.app** — Streamlit Community Cloud, deploying from the
now-public repo on `main`. The repo was flipped public to deploy (it holds only code/docs/config —
no secrets; BYO-keys).

Remaining:
1. Ian's real-browser click-through on the hosted app with his own keys.
2. Optional cosmetic polish from the spec review: surface the run-count recommendation inline (it's
   currently a hover tooltip); rename the "Patterns" output tab → "Analysis" to match the spec wording;
   note the engine picker is a multiselect, not a single dropdown.
3. Error-handling polish (invalid keys, API failures, quota) surfaced in the UI.
4. Drop the live link into the article where the tool is teased.

## 2026-10-24 — Bug fixes from the headless test

A headless test (report: job-search repo, `projects/article-pipeline/real-vs-guessed-fanouts/tool-test-2026-10-24.md`)
found persona, labeling and cost-estimate bugs. Fixed:
- **Personas whose names share a last word merged into one source** ("First-time buyer" + "Repeat buyer" →
  `buyer`). `patterns.persona_of` now keeps the app's in-memory label whole (the last-word split is only for
  file-based captures). Two personas with the *same* name now get unique labels (`buyer`, `buyer-2`).
- **"Remove this persona" deleted the wrong persona's data** (it removed the last one). Persona widgets are now
  keyed by a stable per-persona id instead of list position.
- **PATTERNS counted the no-persona base run as a persona** ("· 1 persona(s) ⚑1-persona" on base-only runs).
  The count now excludes base, and ⚑1-persona only fires when one persona, and not base, supports the entity.
- **Cost estimate ran ~1.2–2.9× low.** The measured averages had been priced on an older, cheaper rate sheet,
  and Gemini's thinking tokens were never counted. Gemini usage now counts thinking as output, so the per-run
  `cost_usd` in the raw downloads is correct too. The per-run averages are re-derived at the current sheet
  (see `core/cost.py`), and empty personas are no longer charged.
- Wording: a failed PATTERNS/BRIEFS now says "failed — reason" (not "was skipped"); the status line names
  only the stages that run; PATTERNS' single-engine warning fixed for the one-engine and zero-entity cases; brief
  fixes (empty "Angles to cover", persona checklist item with no personas); a non-object model reply is a clean
  per-run error; an engine that never searched gets its own warning; "1 run" grammar.

Validated offline with fake providers (25 bug checks that failed before the fix and pass after, 6 control
checks that pass both times, plus the 28-case regression suite) and one real end-to-end run (~$0.45). **Committed locally; not yet pushed or deployed.**

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```
Paste your own OpenAI + Gemini keys in the sidebar (Anthropic optional). Keys are used in memory for
the run only.
