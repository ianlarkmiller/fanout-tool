"""BRIEFS — the LLM writer's brief (ported from unified_brief.py).

Pools any mix of sources for one query and produces a writer-ready brief: page type, angles
(Must/Should/Optional), entities to name/cite, a checklist — every item tagged elicited/modeled.
The LLM recognizes entity vocabulary from the raw strings; everything that drives a TAG stays
deterministic (cluster tiering + entity run-attribution), so provenance can't be hallucinated.

Refactor vs. the script: imports core.patterns/core.cluster; API keys are arguments; the embedding
cache is passed in; no stdout/file I/O/spend ledger. `brief_md()` takes in-memory captures.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

from . import cluster as cp
from . import patterns as uf

LLM_MODEL = "gpt-5.5"
GPT55_IN, GPT55_OUT = 4.43e-6, 12.87e-6
TIER_ORDER = {"Must": 0, "Should": 1, "Optional": 2}
PROV_RANK = {"real+guess": 0, "real": 1, "guess": 2, "weak": 3}
PROV_LABEL = {"real+guess": "elicited+modeled", "real": "elicited", "guess": "modeled", "weak": "weak"}


def provtag(p):
    return PROV_LABEL.get(p, p)


CLUSTER_CAP = 40
GUESS_CAP = 10


def classify(persrc, sruns, has_elicited):
    stat = {s: (len(r), sruns[s], uf.wilson_lb(len(r), sruns[s])) for s, r in persrc.items()}
    reliable = [s for s, (k, n, w) in stat.items() if w >= uf.RELIABLE_W]
    rclasses = {uf.class_of(s) for s in reliable}
    if len(rclasses) >= 2:
        tier = "Must"
    elif len(rclasses) == 1:
        only = next(iter(rclasses))
        tier = "Must" if only.startswith("elicited") else ("Optional" if has_elicited else "Must")
    elif any(k / n >= uf.OPT_FLOOR for (k, n, w) in stat.values()):
        tier = "Optional"
    else:
        return None
    present = [s for s, (k, n, w) in stat.items() if k / n >= uf.OPT_FLOOR]
    er = any(s.startswith("elicited") for s in present)
    mr = any(s.startswith("modeled") for s in present)
    return {"tier": tier, "reliable": set(reliable),
            "prov": "real+guess" if er and mr else "real" if er else "guess" if mr else "weak"}


def cluster_rows(group, sruns, has_elicited, gemini_api_key, cache):
    str_runs = defaultdict(lambda: defaultdict(set)); disp = {}
    for s, runs in group.items():
        for rid, strs in runs.items():
            for q in strs:
                k = cp.norm(q); str_runs[k][s].add(rid); disp.setdefault(k, q)
    strings = sorted(str_runs)
    labels = uf.cluster_strings([disp[s] for s in strings], gemini_api_key, cache)
    groups = defaultdict(list)
    for st, lab in zip(strings, labels):
        if lab != -1:
            groups[lab].append(st)
    rows = []
    for members in groups.values():
        persrc = defaultdict(set)
        for st in members:
            for s, r in str_runs[st].items():
                persrc[s] |= r
        t = classify(persrc, sruns, has_elicited)
        if not t:
            continue
        ecov = lambda m: sum(len(r) for s, r in str_runs[m].items() if s.startswith("elicited"))
        cov_m = lambda m: sum(len(r) for r in str_runs[m].values())
        members.sort(key=lambda m: (-ecov(m), -cov_m(m)))
        cov = sum(len(r) for r in persrc.values())
        rows.append({**t, "medoid": disp[members[0]], "samples": [disp[m] for m in members[:5]], "cov": cov})
    rows.sort(key=lambda r: (TIER_ORDER[r["tier"]], PROV_RANK.get(r["prov"], 9), -r["cov"]))
    return rows


def match_back(entities, group):
    pats = []
    for e in entities:
        forms = [e.get("canonical", "")] + list(e.get("aliases", []))
        forms = [re.sub(r"\s*\([^)]*\)\s*$", "", f).strip() for f in forms]
        forms = sorted({f for f in forms if len(f) >= 2}, key=len, reverse=True)
        pats.append(re.compile("|".join(r"(?<!\w)" + re.escape(f) + r"(?!\w)" for f in forms), re.I)
                    if forms else None)
    runsets = [defaultdict(set) for _ in entities]
    for s, runs in group.items():
        for rid, strs in runs.items():
            blob = "  ".join(strs)
            for i, pat in enumerate(pats):
                if pat and pat.search(blob):
                    runsets[i][s].add(rid)
    return runsets


def engine_breadth(group, prompt):
    pw = set(re.findall(r"[a-z0-9]+", prompt.lower()))
    out = {}
    for s, runs in group.items():
        if not s.startswith("elicited"):
            continue
        distinct = set()
        for strs in runs.values():
            for q in strs:
                toks = set(re.findall(r"[a-z0-9]+", q.lower()))
                if len(toks - pw) >= 2:
                    distinct.add(cp.norm(q))
        out[s] = len(distinct)
    return out


SYS = """You read the raw sub-query strings that AI search engines (and a model) generated for ONE user query, and turn them into a content writer's brief. The query can be ANY topic — assume no domain. You are given the QUERY, the full list of SUB-QUERY STRINGS, and CLUSTERS of those strings (each a representative search + samples). Return ONE JSON object, nothing else:
- page_type: "informational" (wants an explanation/answer) or "commercial" (choosing among specific named options). Plus a one-sentence page_type_reason.
- cluster_labels: {"<id>": {"label","kind"}}. label = a short plain-language angle a writer would cover (a question/topic, NOT a keyword string). kind = "subject" if the cluster is essentially about one specific named thing, else "general".
- entities: the real-world NAMED ENTITIES that actually appear in the sub-query strings — organizations, authorities/regulators, publishers/review-sites, brands, products, named programs or standards. Recognize them even as plain single words (e.g. "Experian", "Avant"). For each: {"canonical" (the clearest full display name), "aliases" (EVERY distinct surface form that appears in the strings for THIS entity — acronym AND spelled-out AND common variants — as literal substrings to search for, e.g. ["Consumer Financial Protection Bureau","CFPB"]), "bucket" ("source" = a publisher/authority/agency to CITE; "subject" = a specific named thing the query is ABOUT — a product/option/provider/program/standard)}. Merge surface forms of the SAME thing into one entry; keep genuinely DIFFERENT models/SKUs separate (e.g. "AirPods Pro 2" vs "AirPods Pro 3" — list both only if both appear). List each real-world thing ONCE, at the most useful granularity: prefer the specific product/program over the bare maker — do NOT also list a bare brand ("Citi", "Jabra", "Apple") or a shorter form ("Jabra Elite 8 Active") when you're already listing a more specific item that contains it, UNLESS that brand is genuinely searched on its own as a distinct topic. Do NOT invent entities or numbers not present in the strings; skip generic phrases and spec fragments (e.g. "best earbuds", "IP55 battery", "0% APR"). Aliases must be strings that literally occur — never a fragment that would over-match (use "Capital One", never "One").
JSON only."""


def llm_json(user, openai_api_key):
    from openai import OpenAI
    resp = OpenAI(api_key=openai_api_key, timeout=120, max_retries=6).responses.create(
        model=LLM_MODEL, instructions=SYS, input=user)
    txt = resp.output_text or ""
    return json.loads(txt[txt.index("{"): txt.rindex("}") + 1])


def brief(qid, prompt, group, gemini_api_key, openai_api_key, cache):
    sruns = uf.source_runs(group)
    has_e = any(s.startswith("elicited") for s in sruns)
    has_personas = any(s.startswith("modeled:") and s.split(":", 1)[1] != "base" for s in sruns)
    all_clusters = cluster_rows(group, sruns, has_e, gemini_api_key, cache)
    corrob = [c for c in all_clusters if c["prov"] in ("real", "real+guess")]
    guesses = [c for c in all_clusters if c["prov"] not in ("real", "real+guess")]
    keep_guess = max(0, min(CLUSTER_CAP - len(corrob), GUESS_CAP))
    clusters = corrob + guesses[:keep_guess]
    dropped_guess = len(guesses) - keep_guess

    distinct = sorted({q for runs in group.values() for strs in runs.values() for q in strs})
    inp = [f'QUERY: "{prompt}"', "", "SUB-QUERY STRINGS (recognize the named entities that appear here):"]
    inp += [f"- {q}" for q in distinct]
    inp += ["", "CLUSTERS (id: representative search | samples):"]
    for i, c in enumerate(clusters):
        inp.append(f"{i}: {c['medoid']} | " + " / ".join(c["samples"][1:4]))
    try:
        out = llm_json("\n".join(inp), openai_api_key)
    except Exception as exc:  # degrade rather than fail
        return (f"## Brief — \"{prompt}\" ({qid})\n\n*LLM polish failed ({type(exc).__name__}: {exc}); "
                f"use the raw analysis from PATTERNS.*\n")

    cl = {int(k): v for k, v in out.get("cluster_labels", {}).items()}
    commercial = out.get("page_type") == "commercial"
    ents = out.get("entities", [])
    runsets = match_back(ents, group)
    plow = prompt.lower(); pw = set(plow.split())
    cent = []
    for e, rs in zip(ents, runsets):
        if not rs:
            continue
        name = e.get("canonical", ""); nml = name.lower()
        specific = bool(re.search(r"\d", name) or re.search(r"[a-z][A-Z]", name) or len(nml.split()) >= 3)
        if nml and not specific and all(t in pw for t in nml.split()):
            continue
        nclass = len({uf.class_of(s) for s in rs})
        if nclass < 2 and max(len(r) for r in rs.values()) < 2:
            continue
        t = classify(rs, sruns, has_e)
        if t:
            flat = frozenset((s, r) for s, rr in rs.items() for r in rr)
            npers = len({s for s in rs if s.startswith("modeled:") and s.split(":", 1)[1] != "base"})
            cent.append({"name": e.get("canonical", "?"), "bucket": e.get("bucket", "subject"),
                         "flat": flat, "npers": npers, **t})
    cent.sort(key=lambda e: -len(e["name"]))
    kept = []
    for e in cent:
        nl = e["name"].lower()
        if any(nl != k["name"].lower() and nl in k["name"].lower() and e["bucket"] == k["bucket"]
               and e["flat"] <= k["flat"] for k in kept):
            continue
        kept.append(e)
    cent = kept
    cite_breadth = {s: sum(1 for e in cent if sum(1 for src, _ in e["flat"] if src == s) >= 2)
                    for s in sruns if s.startswith("elicited")}
    for e in cent:
        e.pop("flat", None)
    cent.sort(key=lambda e: (TIER_ORDER[e["tier"]], PROV_RANK.get(e["prov"], 9)))

    elic_breadth = engine_breadth(group, prompt)
    elic_n = [n for s, n in sruns.items() if s.startswith("elicited")]

    def src_label(s):
        if s.startswith("elicited") and elic_breadth.get(s, 0) == 0:
            return s + " (anchor-only)"
        return s
    header = f'## Brief — "{prompt}" ({qid})' if qid else f'## Brief — "{prompt}"'
    L = [header,
         f"**Page type:** {out.get('page_type', '?')} — {out.get('page_type_reason', '')}",
         f"**Sources pooled:** {', '.join(src_label(s) for s in sorted(sruns))}"
         + ("" if has_e else "  ⚠ MODELED-ONLY — every angle below is modeled")]
    cite_fanned = [s for s, c in cite_breadth.items() if c >= 2]
    if has_e and any(cite_breadth.values()) and len(cite_fanned) <= 1:
        top_s = max(cite_breadth, key=cite_breadth.get)
        bd = ", ".join(f"{s.split(':', 1)[1]} {c}" for s, c in sorted(cite_breadth.items(), key=lambda kv: -kv[1]))
        L.append(f"> ⚠ The entities to **cite/name** come almost entirely from one engine ({top_s.split(':', 1)[1]}) — "
                 f"citable-entity support per engine: {bd}. Other engines may fan out to angles but name few citable "
                 f"sources, so weight the [elicited]/[elicited+modeled] tags in **Name & cite** as effectively one-engine.")
    if elic_n and min(elic_n) < 5:
        sev = ("the Must/Should/Optional split is essentially noise — don't rely on it" if min(elic_n) <= 3
               else "treat the Must/Should split as indicative, not firm")
        L.append(f"> ⚠ Low run-count elicited capture (min {min(elic_n)} runs); tiers are unstable — {sev}.")
    folded = 0; suppress = set()
    if commercial:
        SUBJ_GENERIC = {"card", "cards", "credit", "the", "co", "inc", "plan"}
        toks = lambda s: frozenset(t for t in re.findall(r"[a-z0-9]+", s.lower())
                                   if t not in SUBJ_GENERIC and len(t) > 2)
        subj_keys = [(e["name"], toks(e["name"])) for e in cent if e["bucket"] != "source"]
        subj_keys = sorted([(n, k) for n, k in subj_keys if k], key=lambda nk: -len(nk[1]))
        seen_subj = set()
        for i, c in enumerate(clusters):
            blobtok = set(re.findall(r"[a-z0-9]+", (c["medoid"] + " " + " ".join(c["samples"])).lower()))
            key = next((n for n, k in subj_keys if k <= blobtok), None)
            if key is None:
                continue
            if key in seen_subj:
                suppress.add(i); folded += 1
            else:
                seen_subj.add(key)
    shown = [(i, c) for i, c in enumerate(clusters) if i not in suppress
             and not (commercial and cl.get(i, {}).get("kind") == "subject" and c["prov"] in ("guess", "weak"))]
    lown = bool(elic_n) and min(elic_n) <= 3
    shown_must = []
    if lown:
        L += ["", "**Angles to cover** (sample too small to rank — listed elicited-corroborated first; "
              "tag = elicited vs modeled):"]
        for i, c in sorted(shown, key=lambda ic: (PROV_RANK.get(ic[1]["prov"], 9), -ic[1]["cov"])):
            L.append(f"- **{cl.get(i, {}).get('label') or c['medoid']}** — [{provtag(c['prov'])}] _(raw: \"{c['medoid']}\")_")
    else:
        L += ["", "**Angles to cover** (tier = how reliably engines probe it; tag = elicited vs modeled):"]
        for tname in ("Must", "Should", "Optional"):
            rows = [(cl.get(i, {}).get("label") or c["medoid"], c) for i, c in shown if c["tier"] == tname]
            if not rows:
                continue
            L.append(f"- *{tname}*")
            for label, c in rows:
                L.append(f"  - **{label}** — [{provtag(c['prov'])}] _(raw: \"{c['medoid']}\")_")
                if tname == "Must":
                    shown_must.append(label)
    if folded > 0:
        L.append(f"- *(+{folded} more angles about products already in the list below — folded to avoid redundant sections)*")
    if dropped_guess > 0:
        L.append(f"- *(+{dropped_guess} lower-signal modeled angles not shown — all engine-corroborated angles are above)*")
    if not lown and not shown_must and not any(c["tier"] == "Should" for c in clusters):
        L.append("- *(no general angles cleared the bar — the brief is the named items below)*")

    def pflag(e):
        n = e.get("npers", 0)
        return f" · {n} persona{'' if n == 1 else 's'}" if n and has_personas and e["prov"] in ("guess", "real+guess") else ""

    def tlabel(e):
        return "" if lown else f" · _{e['tier']}_"
    if lown:
        cent.sort(key=lambda e: PROV_RANK.get(e["prov"], 9))
    src = [e for e in cent if e["bucket"] == "source"]
    subj = [e for e in cent if e["bucket"] != "source"]
    L += ["", "**Name & cite** (tag = elicited vs modeled"
          + ("" if lown else "; tier = how reliably engines probe it") + "):"]
    if src:
        L.append("- *Sources to cite:*")
        if has_e and any(e["prov"] == "guess" for e in src):
            tail = ("; where one is persona-flagged, weigh it by whether that buyer is yours."
                    if has_personas else ".")
            L.append("  - ⚠ The **[modeled]** sources below came only from the model — the live engines weren't "
                     f"seen searching them. Treat them as predictions, not confirmed engine behavior{tail}")
        L += [f"  - {e['name']} — [{provtag(e['prov'])}]{tlabel(e)}{pflag(e)}" for e in src]
    if subj:
        L.append("- *Subjects to name:*")
        if has_e and any(e["prov"] == "guess" for e in subj):
            tip = ("Where one is persona-flagged, it's that buyer's hypothesis — a single-persona signal "
                   "(one persona) is the weakest, so name those only if that buyer is yours."
                   if has_personas else "Cover them only if they fit the page.")
            L.append("  - ⚠ The **[modeled]** subjects below came only from the model — the live engines weren't "
                     f"seen searching them. {tip}")
        L += [f"  - {e['name']} — [{provtag(e['prov'])}]{tlabel(e)}{pflag(e)}" for e in subj]

    L += ["", "**Checklist:**"]
    for m in shown_must:
        L.append(f"- [ ] A clear section answering: {m}")
    if commercial:
        L.append("- [ ] Build the comparison the engine is reconstructing (each item × the attributes it checks)")
    L += ["- [ ] Name every entity above; cite the **[elicited]** / **[elicited+modeled]** ones to their source",
          "- [ ] Cover persona-flagged predictions only if they fit your buyer",
          "- [ ] Verify every claim against its source before publishing", ""]
    return "\n".join(L)


def brief_md(prompt: str, captures: list[dict], gemini_api_key: str, openai_api_key: str,
             cache: dict | None = None) -> str:
    """Public entry: pool in-memory captures and return the BRIEFS markdown."""
    cache = {} if cache is None else cache
    group = uf.pool_captures(captures)
    return brief("", prompt, group, gemini_api_key, openai_api_key, cache)
