"""API-key validation: format/whitespace checks (instant, local) + a minimal real-inference heartbeat.

Order per key: trim & flag stray whitespace → check the expected prefix (and catch a key pasted in the
WRONG provider's box) → make a tiny real API call (≈ a fraction of a cent on the user's own key). The
inference call validates auth AND that the key can actually run — so it also catches out-of-credits /
no-model-access, which a free list-models check would miss.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

_DISPLAY = {"openai": "OpenAI", "gemini": "Gemini", "anthropic": "Anthropic"}
# Accepted key prefixes per provider (Gemini ships both the classic AIza... and newer AQ.... formats).
_PREFIXES = {"openai": ("sk-",), "anthropic": ("sk-ant-",), "gemini": ("AIza", "AQ.")}


def _looks_like(key: str) -> str | None:
    """Which provider a key's prefix matches (to catch wrong-box paste). Order matters: sk-ant- first."""
    if key.startswith("sk-ant-"):
        return "anthropic"
    if key.startswith("sk-"):
        return "openai"
    if key.startswith(_PREFIXES["gemini"]):
        return "gemini"
    return None


def _friendly(exc) -> str:
    s = str(exc).lower()
    if any(k in s for k in ("401", "unauthorized", "invalid", "api key", "api_key",
                            "authentication", "permission")):
        return "invalid or unauthorized"
    if any(k in s for k in ("429", "quota", "insufficient", "credit", "billing", "rate limit")):
        return "out of credits or rate-limited"
    if "timeout" in s or "timed out" in s:
        return "couldn't reach the provider (timeout)"
    return "couldn't validate"


def _heartbeat(provider: str, key: str) -> None:
    """A minimal real call on the tool's actual models — validates auth + that the key can run."""
    if provider == "openai":
        from openai import OpenAI
        OpenAI(api_key=key, timeout=30, max_retries=0).responses.create(
            model="gpt-5.5", input="ping", max_output_tokens=16)
    elif provider == "gemini":
        from core.cluster import embed  # exercise the tool's real embedding path
        embed(["ping"], key, {})
    elif provider == "anthropic":
        import anthropic
        anthropic.Anthropic(api_key=key, timeout=30, max_retries=0).messages.create(
            model="claude-sonnet-4-6", max_tokens=1, messages=[{"role": "user", "content": "ping"}])


def check_key(provider: str, raw_key: str) -> tuple[bool | None, str]:
    """(True, note) valid / (False, reason) bad / (None, "") no key. Takes the RAW (untrimmed) value."""
    if not raw_key:
        return (None, "")
    key = raw_key.strip()
    if not key:
        return (False, "is only whitespace")
    # Wrong box: the key clearly matches a different provider's format.
    looks = _looks_like(key)
    if looks and looks != provider:
        return (False, f"looks like {_DISPLAY[looks]}'s key format — paste it in the {_DISPLAY[looks]} box instead")
    notes = []
    if key != raw_key:
        notes.append("had extra spaces (trimmed)")
    if not key.startswith(_PREFIXES[provider]):
        notes.append("unexpected format (usually starts with "
                     + " or ".join(f"'{p}'" for p in _PREFIXES[provider]) + ")")
    try:
        _heartbeat(provider, key)
    except Exception as exc:  # noqa: BLE001
        reason = _friendly(exc)
        return (False, f"{reason} ({'; '.join(notes)})" if notes else reason)
    return (True, "; ".join(notes))


def check_keys(keys: dict[str, str]) -> dict[str, tuple[bool | None, str]]:
    """Validate all non-empty (raw) keys concurrently. Returns {provider: (ok, note_or_reason)}."""
    provs = [p for p, v in keys.items() if v]
    if not provs:
        return {}
    with ThreadPoolExecutor(max_workers=len(provs)) as ex:
        return dict(zip(provs, ex.map(lambda p: check_key(p, keys[p]), provs)))
