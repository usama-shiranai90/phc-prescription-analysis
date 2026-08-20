"""Split a symptom note into its constituent complaints.

2,943 of 14,074 notes are explicitly enumerated (`1. ... 2. ...`) and many
more use commas as complaint separators. The production tokeniser collapses
all of that into one bag of tokens, so a note reading
`"1. pain in both heels and back pain 2. generalized weakness"` becomes
indistinguishable from any other note containing those words in any
arrangement.

Two guards keep the enumeration split honest:

  * the digit markers must form an increasing run starting at 1, so the `2` in
    `"pain in the rt. leg for 2 months"` cannot be mistaken for a list marker;
  * ` and ` is deliberately NOT a separator. `"burning sensation of hands and
    feet"` is one complaint with a coordinated site, and splitting it strands
    `feet` without its head noun.
"""
from __future__ import annotations

import re

# A list marker: an optionally zero-padded 1-2 digit number followed by . or )
# and whitespace. Anchored so decimals ('1.5') and dates cannot match.
_MARKER = re.compile(r"(?:(?<=^)|(?<=[\s,;.\-]))(0?\d{1,2})\s*[.)]\s+(?!\d)")

# Separators that reliably delimit complaints in this corpus.
_SEP = re.compile(r"\s*[;,]\s*|\s+\+\s+|\s*&\s*")

# Trailing duration phrases. Removed only to form the concept-matching key;
# the span itself keeps them.
_DURATION = re.compile(
    r"\b(?:for|since|from|over)\s+(?:the\s+)?(?:last\s+|past\s+)?"
    r"(?:\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|several|few|many|some|"
    r"long|couple\s+of)?\s*"
    r"(?:day|days|week|weeks|month|months|year|years|yrs|hour|hours|"
    r"time|times|while|period)\b(?:\s+(?:back|ago|now))?", re.I)
_DURATION2 = re.compile(r"\bfor\s+(?:several|few|many|some|long|a\s+long)\s+"
                        r"(?:time|while|period)?\b", re.I)
_TRAIL_PUNCT = re.compile(r"^[\s\.,;:\-\)\(]+|[\s\.,;:\-\)\(]+$")

# Notes that record the absence of a complaint. ~5% of the corpus. Treated as
# a distinct state rather than as an empty note, because an empty note also
# arises from a missing field.
_NO_COMPLAINT = re.compile(
    r"^(?:no|nil|none|n/?a|nothing|not\s+any)?\s*"
    r"(?:specific\s+|significant\s+|any\s+)?"
    r"(?:complaint|complaints|complain|complains|symptom|symptoms|problem|"
    r"problems|issue|issues)?\s*$", re.I)
_CHECKUP = re.compile(
    r"^(?:for\s+)?(?:general|routine|regular|annual|yearly|periodic|health)"
    r"[\s\-]*(?:check\s*-?\s*up|checkup|check|screening|examination|exam)\b",
    re.I)


def is_no_complaint(text: str) -> bool:
    """True for notes that assert the absence of a complaint."""
    t = str(text).strip()
    if not t:
        return False
    if _CHECKUP.match(t):
        return True
    core = _TRAIL_PUNCT.sub("", t)
    if not core:
        return False
    if len(core) > 40:
        return False
    return bool(_NO_COMPLAINT.match(core))


def _enumerated_chunks(text: str) -> list[str] | None:
    """Split on list markers, but only when they form a real 1,2,3 run."""
    hits = list(_MARKER.finditer(text))
    if not hits:
        return None
    run, expect = [], 1
    for m in hits:
        if int(m.group(1)) == expect:
            run.append(m)
            expect += 1
    if len(run) < 2:
        return None
    chunks, prev_end = [], None
    lead = text[:run[0].start()].strip()
    if lead:
        chunks.append(lead)
    for i, m in enumerate(run):
        start = m.end()
        end = run[i + 1].start() if i + 1 < len(run) else len(text)
        chunks.append(text[start:end])
        prev_end = end
    return [c for c in chunks if c.strip()]


def strip_duration(span: str) -> str:
    """Remove trailing duration phrases so spans collapse onto a concept key."""
    s = _DURATION.sub(" ", str(span))
    s = _DURATION2.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return _TRAIL_PUNCT.sub("", s).strip()


def segment(text: str, min_chars: int = 3) -> list[str]:
    """Return the complaint spans of a note (empty list for no-complaint)."""
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    if not t or is_no_complaint(t):
        return []
    chunks = _enumerated_chunks(t) or [t]
    spans: list[str] = []
    for chunk in chunks:
        for part in _SEP.split(chunk):
            part = _TRAIL_PUNCT.sub("", part or "")
            part = re.sub(r"\s+", " ", part).strip()
            if not part:
                continue
            if len(part) < min_chars and spans:
                # Fragments like 'rt' or 'b' belong to the preceding span.
                spans[-1] = f"{spans[-1]} {part}"
            elif len(part) >= min_chars:
                spans.append(part)
    return spans


def span_keys(text: str) -> list[str]:
    """Duration-stripped, lower-cased spans -- the concept-matching key."""
    out = []
    for s in segment(text):
        k = strip_duration(s).lower()
        k = re.sub(r"[^a-z0-9/ ]+", " ", k)
        k = re.sub(r"\s+", " ", k).strip()
        if k:
            out.append(k)
    return out
