"""PATTERNS — source-agnostic fan-out -> deterministic entity analysis (ported from unified_fanout.py).

Pools any mix of elicited (per engine) + modeled (base/persona) captures for one question and tiers
the recurring named entities into Must/Should/Optional with provenance tags. No generative LLM — just
regex entity extraction + Wilson tiering, plus an embeddings-based clustering *diagnostic*.

Refactor vs. the script: entity/tiering logic is unchanged; embeddings take the Gemini key + an
in-memory cache; capture data comes in as dicts (pool_captures) instead of files; report() returns a
string. `load_group`/`load_file` are kept (file-based) for tests.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict

import numpy as np

from . import cluster as cp

# ---- knobs ----
RELIABLE_W = 0.25
OPT_FLOOR = 0.10
CONF_W = 0.30
Z = 1.96


def wilson_lb(k, n):
    if n == 0:
        return 0.0
    p = k / n
    return max(0.0, (p + Z * Z / (2 * n) - Z * ((p * (1 - p) / n + Z * Z / (4 * n * n)) ** 0.5)) / (1 + Z * Z / n))


def class_of(source: str) -> str:
    """Each elicited engine is its own class; ALL modeled personas collapse to one `modeled`."""
    return "modeled" if source.startswith("modeled") else source


# ---- entity extraction + canonicalization ----
SOURCE_LEXICON = {"wirecutter", "rtings", "tom's guide", "tomsguide", "nerdwallet", "bankrate", "forbes",
    "cnet", "reddit", "consumer reports", "investopedia", "healthline", "webmd", "yelp", "tripadvisor",
    "pcmag", "techradar", "engadget", "the verge", "wikipedia", "quora", "trustpilot"}
MODELNUM = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2}\s+(?:[A-Z]?[1-9]\d{0,2}|X\d+)\b"
                      r"(?!\s*(?:months?|%|percent|days?|years?|hours?|gb|mm|mins?))")
SINGLEWORD = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
SINGLE_STOP = {"best", "top", "how", "what", "when", "where", "why", "which", "who", "should", "could", "would", "does",
    "did", "will", "your", "you", "our", "their", "the", "and", "for", "with", "without", "using", "use", "get", "getting",
    "new", "now", "near", "find", "compare", "comparison", "versus", "guide", "review", "reviews", "tips", "need", "help",
    "explained", "official", "that", "this", "these", "those", "most", "more", "less", "good", "better", "cheap",
    "cheapest", "free", "worth", "real", "step", "steps", "way", "ways", "pros", "cons", "about", "into", "from", "over",
    "english", "spanish", "american", "americans"}
PROPER_FOR = re.compile(r"\b[A-Z][a-zA-Z.&]*(?:\s+(?:of|the|and|&|for)\s+[A-Z][a-zA-Z.&]*|\s+[A-Z][a-zA-Z.&]*)+")
SRC_TOKENS = {s for s in SOURCE_LEXICON if " " not in s}


def is_acronym(disp: str) -> bool:
    return disp.isupper() and disp.isalpha() and 2 <= len(disp) <= 6


def is_contiguous_sub(sub, full) -> bool:
    return any(full[i:i + len(sub)] == sub for i in range(len(full) - len(sub) + 1))


def extract_entities(text: str, plow: str = "") -> set[str]:
    found = set()
    for m in cp.ACRONYM.findall(text):
        if m not in cp.STOP_ACR:
            found.add(m)
    for rx in (cp.PROPER, PROPER_FOR):
        for m in rx.findall(text):
            m = cp.clean_entity(m)
            if len(m) > 3 and " " in m:
                found.add(m)
    for m in cp.CAMEL.findall(text):
        found.add(m)
    low = text.lower()
    for s in SOURCE_LEXICON:
        if s in low:
            found.add(s)
    for m in MODELNUM.findall(text):
        found.add(m.strip())
    mw_tokens = {t for e in found if " " in e for t in e.split()}
    for m in SINGLEWORD.findall(text):
        if m.isupper():
            continue
        ml = m.lower()
        if ml in SINGLE_STOP or ml in GENERIC or m in mw_tokens:
            continue
        found.add(m)
    if plow:
        pnums = set(re.findall(r"\d+", plow))
        if pnums:
            found = {(" ".join(e.split()[:-1]) if len(e.split()) > 1 and e.split()[-1] in pnums else e)
                     for e in found}
        pwords = set(plow.split())
        found = {e for e in found if e and not all(t in pwords for t in e.lower().split())}
    return found


def norm_ent(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+20\d\d$", "", s)
    s = re.sub(r"\s+0$", "", s)
    s = re.sub(r"\s+", " ", s).strip(" .&,")
    return s.lower()


GENERIC = {"apr", "aprs", "terms", "official", "balance", "transfer", "offer", "offers", "period", "intro",
    "introductory", "months", "month", "rate", "rates", "fee", "fees", "card", "cards", "cost", "comparison",
    "loan", "loans", "credit", "debt", "best", "review", "reviews", "program", "plan", "plans", "percent",
    "promotional", "promo", "interest", "payoff", "options", "guide", "vs", "or"}


def is_junk(n: str) -> bool:
    toks = n.split()
    return (len(n) < 3 or " vs " in n or " or " in n or not re.search(r"[a-z]", n)
            or len(toks) > 5
            or all(t in GENERIC or len(t) <= 2 for t in toks)
            or sum(1 for t in toks if t in SRC_TOKENS) >= 2)


def initials(n: str) -> str:
    skip = {"of", "the", "for", "and", "a", "&", "to"}
    return "".join(w[0] for w in n.split() if w and w not in skip and w[0].isalpha())


def build_canonical(raw_displays) -> dict[str, str]:
    norm2disp: dict[str, str] = {}
    for d in raw_displays:
        n = norm_ent(d)
        if is_junk(n):
            continue
        disp = re.sub(r"\s+20\d\d$", "", d).strip()
        if n not in norm2disp or len(disp) > len(norm2disp[n]):
            norm2disp[n] = disp
    keys = set(norm2disp)
    norms = sorted(norm2disp, key=lambda n: -len(n.split()))
    ng = lambda toks: [t for t in toks if t not in GENERIC and len(t) > 2]

    def find_target(n):
        toks = n.split(); disp = norm2disp[n]
        if len(toks) >= 2 and toks[-1] != n and toks[-1] in keys and is_acronym(norm2disp[toks[-1]]):
            return toks[-1]
        if len(toks) >= 2 and initials(n) in keys and initials(n) != n:
            return initials(n)
        if len(toks) >= 2:
            for m in norms:
                if m != n and len(m.split()) > len(toks) and is_contiguous_sub(toks, m.split()):
                    return m
        if len(toks) == 1 and re.search(r"[a-z][A-Z]", disp):
            cont = [m for m in norms if m != n and len(m.split()) > 1 and toks[0] in m.split()]
            if len(cont) == 1:
                return cont[0]
            if len(cont) >= 2:
                return None
        a = set(ng(toks))
        if a:
            cand = [m for m in norms if m != n and a & set(ng(m.split()))
                    and (set(toks) - set(m.split()))
                    and all(t in GENERIC for t in (set(toks) - set(m.split())))
                    and len(ng(m.split())) > len(a)]
            if len(cand) == 1:
                return cand[0]
        return n

    parent = {n: find_target(n) for n in norms}

    def root(n):
        seen = set()
        while True:
            p = parent.get(n, n)
            if p is None:
                return None
            if p == n or n in seen:
                return n
            seen.add(n); n = p
    out: dict[str, str] = {}
    for d in raw_displays:
        n = norm_ent(d)
        if is_junk(n):
            continue
        r = root(n)
        if r is None:
            continue
        out[d] = norm2disp[r]
    return out


# ---- loaders ----
def qid_of(path: str) -> str:
    b = re.sub(r"\.json$", "", os.path.basename(path))
    return re.split(r"-", b)[0]


def persona_of(data: dict) -> str:
    pf = data.get("persona_file")
    if not pf:
        return "base"
    return re.sub(r"\.md$", "", os.path.basename(pf)).split("-")[-1]


def _parse_capture(data: dict):
    """Parse one in-memory capture dict (elicited or modeled) -> {source: {run_id: [strings]}}."""
    out = defaultdict(lambda: defaultdict(list))
    if isinstance(data.get("result"), dict) and "runs" in data["result"]:
        src = "modeled:" + persona_of(data)
        for i, r in enumerate(data["result"]["runs"], 1):
            if "error" in r:
                continue
            out[src][f"{src}#{i}"] = [s["sub_query"].strip()
                                      for s in r.get("sub_queries", []) if s.get("sub_query")]
    elif "engines" in data:
        for ek, er in data["engines"].items():
            src = f"elicited:{ek}"
            for r in er["runs"]:
                if "queries" not in r:
                    continue
                out[src][f"{src}#{r['run']}"] = [q for q in r["queries"] if not cp.OWN_CALC.match(q)]
    else:
        raise ValueError("unrecognized capture schema")
    return out


def pool_captures(captures: list[dict]):
    """Merge in-memory capture dicts (from elicit() / model_one()) into one group."""
    merged = defaultdict(lambda: defaultdict(list))
    for data in captures:
        for s, runs in _parse_capture(data).items():
            for rid, strs in runs.items():
                merged[s][rid] = strs
    return merged


def load_file(path: str):
    data = json.load(open(path, encoding="utf-8"))
    out = _parse_capture(data)
    prompt = data.get("query") or data.get("prompt") or data.get("qid", "")
    return qid_of(path), prompt, out


def load_group(paths):
    merged = defaultdict(lambda: defaultdict(list)); prompt = ""; qid = None
    for p in paths:
        q, pr, srcs = load_file(p)
        qid = qid or q; prompt = prompt or pr
        for s, runs in srcs.items():
            for rid, strs in runs.items():
                merged[s][rid] = strs
    return qid, prompt, merged


def source_runs(group):
    return {s: len(runs) for s, runs in group.items()}


# ---- entity spine ----
def entity_spine(group, sruns, prompt=""):
    has_elicited = any(s.startswith("elicited") for s in sruns)
    plow = prompt.lower()
    raw_hits = defaultdict(lambda: defaultdict(set)); all_disp = set()
    for s, runs in group.items():
        for rid, strs in runs.items():
            per_run = set()
            for q in strs:
                for e in extract_entities(q, plow):
                    per_run.add(e); all_disp.add(e)
            for e in per_run:
                raw_hits[e][s].add(rid)
    canon = build_canonical(all_disp)
    ent = defaultdict(lambda: defaultdict(set))
    for e, persrc in raw_hits.items():
        c = canon.get(e)
        if not c:
            continue
        for s, rids in persrc.items():
            ent[c][s] |= rids
    rows = []
    for c, persrc in ent.items():
        stat = {s: (len(rids), sruns[s], wilson_lb(len(rids), sruns[s])) for s, rids in persrc.items()}
        reliable = [s for s, (k, n, w) in stat.items() if w >= RELIABLE_W]
        rclasses = {class_of(s) for s in reliable}
        if len(rclasses) >= 2:
            tier = "Must"
        elif len(rclasses) == 1:
            only = next(iter(rclasses))
            tier = "Must" if only.startswith("elicited") else ("Optional" if has_elicited else "Must")
        elif any(k / n >= OPT_FLOOR for (k, n, w) in stat.values()):
            tier = "Optional"
        else:
            continue
        best_w = max(w for (_, _, w) in stat.values())
        present = [s for s, (k, n, w) in stat.items() if k / n >= OPT_FLOOR]
        er = any(s.startswith("elicited") for s in present)
        mr = any(s.startswith("modeled") for s in present)
        prov = "real+guess" if er and mr else "real" if er else "guess" if mr else "weak"
        n_pers_rel = sum(1 for s in reliable if s.startswith("modeled"))
        rows.append({"entity": c, "tier": tier, "reliable": set(reliable), "prov": prov,
                     "rclasses": len(rclasses), "stat": stat, "lowconf": best_w < CONF_W,
                     "n_pers_rel": n_pers_rel})
    order = {"Must": 0, "Should": 1, "Optional": 2}
    rows.sort(key=lambda r: (order[r["tier"]], -r["rclasses"], -max(w for (_, _, w) in r["stat"].values())))
    return rows


# ---- clustering diagnostic ----
def cluster_strings(strings, api_key, cache):
    if len(strings) < cp.MIN_CLUSTER_SIZE:
        return np.full(len(strings), -1)
    v = cp.embed(strings, api_key, cache)
    if cp.MEAN_CENTER and len(v) > 2:
        v = v - v.mean(axis=0); v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return cp.cluster(v)


def clustering_diagnostic(group, api_key, cache):
    str_src = defaultdict(set); disp = {}
    for s, runs in group.items():
        for strs in runs.values():
            for q in strs:
                k = cp.norm(q); str_src[k].add(s); disp.setdefault(k, q)
    strings = sorted(str_src)
    labels = cluster_strings([disp[s] for s in strings], api_key, cache)
    pooled_noise = defaultdict(lambda: [0, 0]); cluster_srcs = defaultdict(set)
    for st, lab in zip(strings, labels):
        for s in str_src[st]:
            pooled_noise[s][1] += 1
            if lab == -1:
                pooled_noise[s][0] += 1
        if lab != -1:
            cluster_srcs[lab] |= str_src[st]
    shared = sum(1 for v in str_src.values() if len(v) > 1)
    n_clusters = len({l for l in labels if l != -1})
    mean_sc = (np.mean([len(v) for v in cluster_srcs.values()]) if cluster_srcs else 0)
    perclass = {}
    for s, runs in group.items():
        ss = sorted({cp.norm(q) for strs in runs.values() for q in strs})
        lab = cluster_strings([disp[s2] for s2 in ss], api_key, cache) if ss else np.array([])
        perclass[s] = {"distinct": len(ss), "clusters": len({l for l in lab if l != -1}),
                       "noise": int(sum(1 for l in lab if l == -1))}
    return {"pooled_clusters": n_clusters, "mean_srcclasses_per_cluster": round(float(mean_sc), 2),
            "distinct_total": len(strings), "shared_across_sources": shared,
            "pooled_noise": {s: f"{n}/{t} ({100*n/t:.0f}%)" for s, (n, t) in pooled_noise.items()},
            "perclass": perclass}


# ---- report ----
def fmt_sources(stat, reliable):
    parts = []
    for s, (k, n, w) in sorted(stat.items()):
        tag = "✓" if s in reliable else " "
        parts.append(f"{s.split(':',1)[1]} {k}/{n}{tag}")
    return " · ".join(parts)


PROV_LABEL = {"real+guess": "elicited+modeled", "real": "elicited", "guess": "modeled", "weak": "weak"}


def provtag(p):
    return PROV_LABEL.get(p, p)


def rclass_phrase(n):
    return "no source reliably searches it" if n == 0 else f"reliably searched by {n} source{'' if n == 1 else 's'}"


def report(qid, prompt, group, api_key, cache):
    sruns = source_runs(group)
    classes = sorted(sruns)
    has_e = any(s.startswith("elicited") for s in classes)
    has_m = any(s.startswith("modeled") for s in classes)
    rows = entity_spine(group, sruns, prompt)
    elic_contrib = {s for r in rows for s in r["stat"]
                    if s.startswith("elicited") and r["stat"][s][0] > 0}
    prov = "elicited + modeled" if has_e and has_m else ("elicited-only" if has_e else "modeled-only")
    header = f'# Unified fan-out analysis — {qid}: "{prompt}"' if qid else f'# Unified fan-out analysis — "{prompt}"'
    L = [header, "",
         "## Source-class manifest", f"- provenance: **{prov}**"]
    if has_e and len(elic_contrib) <= 1:
        only = (next(iter(elic_contrib)).split(":")[1] if elic_contrib else "none")
        L.append(f"- ⚠ only **{len(elic_contrib)}** elicited engine fanned out to entities ({only}); the "
                 f"others were anchor-only — so the 'elicited' signal here is effectively single-engine.")
    for s in classes:
        L.append(f"- {s} — {sruns[s]} runs")
    L += ["", "## Entity spine (primary signal; ✓ = reliable in that source, Wilson LB ≥ "
          f"{RELIABLE_W}; the \"reliably searched by N sources\" count = independent sources that reliably "
          "search it — each engine counts once, all modeled personas as one (up to 4 with everything pooled); "
          "⚠ = weak support; ⚑ = single-persona modeled)"]
    for tier in ("Must", "Should", "Optional"):
        ts = [r for r in rows if r["tier"] == tier]
        if not ts:
            continue
        shown = ts[:25]
        L.append(f"\n**{tier}** (showing {len(shown)} of {len(ts)})")
        for r in shown:
            pers = f" · {r['n_pers_rel']} persona(s)" if r["prov"] in ("guess", "real+guess") and r["n_pers_rel"] else ""
            flag = ("  ⚠" if r["lowconf"] else "") + ("  ⚑1-persona" if r["prov"] == "guess" and r["n_pers_rel"] == 1 else "")
            L.append(f"- [{provtag(r['prov'])}] `{r['entity']}` — {rclass_phrase(r['rclasses'])} "
                     f"[{fmt_sources(r['stat'], r['reliable'])}]{pers}{flag}")
    d = clustering_diagnostic(group, api_key, cache)
    L += ["", "## Clustering diagnostic (enrichment / measurement only)",
          f"- {d['distinct_total']} distinct strings; only {d['shared_across_sources']} shared verbatim "
          f"across ≥2 sources → strings are near-disjoint by source, so cross-source clusters come from "
          f"the EMBEDDING merging different wordings, not literal overlap.",
          f"- pooled clusters: {d['pooled_clusters']}; mean source-classes per cluster: "
          f"{d['mean_srcclasses_per_cluster']} (higher = the embedding is mixing sources by intent)",
          "- pooled noise rate per source (high = that source's strings vanish when pooled):"]
    for s, v in sorted(d["pooled_noise"].items()):
        L.append(f"    - {s}: {v}")
    L.append("- per-source-class clustering (can a sparse source even cluster alone?):")
    for s, v in sorted(d["perclass"].items()):
        L.append(f"    - {s}: {v['distinct']} distinct → {v['clusters']} clusters, {v['noise']} noise")
    return "\n".join(L)


def patterns_md(prompt: str, captures: list[dict], gemini_api_key: str, cache: dict | None = None) -> str:
    """Public entry: pool in-memory captures and return the PATTERNS markdown."""
    cache = {} if cache is None else cache
    group = pool_captures(captures)
    return report("", prompt, group, gemini_api_key, cache)
