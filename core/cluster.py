"""Embedding + clustering helpers (ported from cluster_patterns.py).

Only the pieces the analysis layer (patterns.py / brief.py) actually uses. The behaviour-critical
parts — the entity regexes, `norm`, HDBSCAN params — are unchanged. The only refactor: `embed()`
takes the Gemini API key and an in-memory cache dict as arguments (no disk .npz, no env read, no
spend ledger).
"""
from __future__ import annotations

import re

import numpy as np

# ---- knobs (unchanged from cluster_patterns.py) ----
EMBED_MODEL = "gemini-embedding-001"
MIN_CLUSTER_SIZE = 3        # HDBSCAN: smallest group of sub-queries to call a cluster
MIN_SAMPLES = 1             # HDBSCAN: ms=1 and ms=2 are identical on this data
MEAN_CENTER = True          # subtract the per-query mean vector before clustering

OWN_CALC = re.compile(r"^\s*calculator\s*:", re.I)
ACRONYM = re.compile(r"\b[A-Z]{2,6}\b")
PROPER = re.compile(r"\b[A-Z][a-zA-Z.&]*(?:\s+(?:of|the|and|&)\s+[A-Z][a-zA-Z.&]*|\s+[A-Z][a-zA-Z.&]*)+")
CAMEL = re.compile(r"\b[A-Z][a-z]+[A-Z][a-zA-Z]+\b")
STOP_ACR = {"APR", "APRS", "USD", "FAQ", "US", "OR", "AND", "VS", "DIY",
            "AGE", "ONE", "NEW", "PRO", "MAX", "GEN", "PLUS", "MINI", "TOP", "ALL", "NOW"}


def norm(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower()).rstrip("?.")


def clean_entity(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+[A-Z]\.?$", "", s)   # drop dangling single-letter tails ("Federal Reserve G.")
    return s.strip(" .&")


def embed(texts: list[str], api_key: str, cache: dict) -> np.ndarray:
    """Embed via Gemini, returning L2-normalized vectors. `cache` is a mutable {text: np.array}
    dict held by the caller (per session); only cache misses hit the API."""
    miss = [t for t in dict.fromkeys(texts) if t not in cache]
    if miss:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
        got: list[list[float]] = []
        for i in range(0, len(miss), 100):
            resp = client.models.embed_content(
                model=EMBED_MODEL, contents=miss[i:i + 100],
                config=types.EmbedContentConfig(task_type="CLUSTERING"),
            )
            got.extend(e.values for e in resp.embeddings)
        for t, v in zip(miss, got):
            cache[t] = np.array(v, dtype=np.float64)
    arr = np.array([cache[t] for t in texts], dtype=np.float64)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12  # L2-normalize for cosine
    return arr


def cluster(vecs: np.ndarray) -> np.ndarray:
    """HDBSCAN on L2-normalized vectors. Returns a label per point; -1 = noise."""
    from sklearn.cluster import HDBSCAN
    if len(vecs) < MIN_CLUSTER_SIZE:
        return np.full(len(vecs), -1, dtype=int)
    m = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES, metric="euclidean", copy=True)
    return m.fit_predict(vecs)
