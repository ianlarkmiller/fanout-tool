"""The standardized buyer-persona template — the six fields used to build every persona.

Using the same six fields for every persona is what keeps personas comparable instead of free-form
blurbs that vary in ways you can't account for. The UI renders one input per field; `assemble()`
turns a filled-in dict into the BUYER PERSONA block text the modeled-fan-out prompt expects.
"""
from __future__ import annotations

PERSONA_FIELDS = [
    {
        "key": "role",
        "label": "Role / self-description",
        "help": "One first-person line — who they are.",
        "placeholder": '"I\'m a married homeowner who does my own financial research."',
    },
    {
        "key": "goal",
        "label": "Goal / situation",
        "help": "What they're trying to do, and why now.",
        "placeholder": "Carrying credit-card debt from a renovation; wants the cheapest, fastest way to clear it.",
    },
    {
        "key": "constraints",
        "label": "Hard constraints",
        "help": "The limits they'd state — budget, credit, eligibility, timing.",
        "placeholder": "Excellent credit; qualifies for any product; won't pay more interest than necessary.",
    },
    {
        "key": "expertise",
        "label": "Expertise level",
        "help": "Beginner / intermediate / advanced — what they already know.",
        "placeholder": "Advanced — already knows balance transfers and avalanche-vs-snowball.",
    },
    {
        "key": "preferences",
        "label": "Preferences",
        "help": "What they want, and what they don't.",
        "placeholder": "Self-directed and data-driven; optimize for lowest total cost, not hand-holding.",
    },
    {
        "key": "location",
        "label": "Location",
        "help": "Country or region.",
        "placeholder": "U.S.",
    },
]

# Map field key -> display label, for assembling the persona block.
_LABELS = {f["key"]: f["label"] for f in PERSONA_FIELDS}


def assemble(fields: dict[str, str]) -> str:
    """Turn a {field_key: value} dict into the BUYER PERSONA block text.

    Empty fields are skipped. Output mirrors the persona-file format the prompt was tuned on:
        - Role / self-description: ...
        - Goal / situation: ...
        ...
    """
    lines = []
    for f in PERSONA_FIELDS:
        val = (fields.get(f["key"]) or "").strip()
        if val:
            lines.append(f"- {_LABELS[f['key']]}: {val}")
    return "\n".join(lines)
