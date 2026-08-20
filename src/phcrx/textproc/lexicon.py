"""Vocabulary sources and a corpus-driven typo corrector.

A generic English spellchecker is the wrong tool here. `micturition`,
`bodyache`, `plid`, `blso` are either absent from a general dictionary or sit
one edit from a common English word, so a general corrector deletes exactly
the tokens that carry the clinical signal. The corrector below only ever maps
*out of vocabulary* tokens *into* a vocabulary assembled from two evidence
sources:

  Tier A -- clinical: content words of the 1,547 ICD-10 category descriptions
            already cached for retrieval, the 89 pharmacological class names,
            and the right-hand sides of the verified abbreviation glossary.
  Tier B -- corpus: tokens seen at least `min_corpus_freq` times in the TRAIN
            split that are not themselves one edit from a Tier-A term.

The Tier-B exclusion is what handles frequent misspellings. `generalised`
occurs often enough to look like a legitimate type on corpus frequency alone,
but it is one edit from the Tier-A word `generalized`, so it is demoted to a
correction source rather than promoted to a correction target.

Everything is fitted on TRAIN only.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from ..config import PROCESSED
from ..nlp.glossary import GLOSSARY

# Closed-class English that must never be treated as a misspelling, and must
# never be a correction target ('no' -> 'not' would invert a negation).
FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "for",
    "from", "with", "without", "by", "as", "is", "are", "was", "were", "be",
    "been", "has", "have", "had", "do", "does", "did", "not", "no", "yes",
    "he", "she", "it", "his", "her", "him", "they", "them", "their", "this",
    "that", "these", "those", "there", "here", "since", "ago", "last", "next",
    "both", "all", "any", "some", "few", "more", "most", "less", "very",
    "also", "after", "before", "during", "while", "when", "than", "then",
    "up", "down", "out", "off", "over", "under", "about", "per", "each",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "day", "days", "week", "weeks", "month", "months", "year", "years",
    "time", "times", "old", "new", "now", "known", "case", "same", "other",
    "left", "right", "upper", "lower", "mild", "severe", "moderate",
    "occasional", "sometimes", "often", "still", "again", "only", "such",
}

# British/Commonwealth orthography is mixed with American spelling throughout
# the corpus. Collapsing the pair before the corrector runs stops the two
# surface forms competing for one concept, and stops the corrector spending
# its edit budget on a difference that is not an error.
_ORTHO_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^([a-z]{3,})isation$"), r"\1ization"),
    (re.compile(r"^([a-z]{3,})ised$"), r"\1ized"),
    (re.compile(r"^([a-z]{3,})ising$"), r"\1izing"),
    (re.compile(r"^([a-z]{3,})ise$"), r"\1ize"),
    (re.compile(r"^oesophag"), "esophag"),
    (re.compile(r"^haemo"), "hemo"),
    (re.compile(r"^anaem"), "anem"),
    (re.compile(r"^ischaem"), "ischem"),
    (re.compile(r"^diarrhoea"), "diarrhea"),
    (re.compile(r"^gynaec"), "gynec"),
    (re.compile(r"^paediatr"), "pediatr"),
    (re.compile(r"^tumour"), "tumor"),
    (re.compile(r"^foetal"), "fetal"),
]

# 'exercise', 'wise', 'noise' would be mangled by the -ise rule.
_ORTHO_EXEMPT = {"exercise", "wise", "noise", "raise", "praise", "advise",
                 "revise", "supervise", "promise", "surprise", "precise",
                 "concise", "otherwise", "rise", "arise", "else", "these",
                 "those", "expertise", "franchise", "merchandise"}

_WORD_RE = re.compile(r"[a-z]+")


def _ortho_word(w: str) -> str:
    if w in _ORTHO_EXEMPT:
        return w
    for pat, rep in _ORTHO_RULES:
        new = pat.sub(rep, w)
        if new != w:
            return new
    return w


def apply_orthography(text: str) -> str:
    """Collapse British spelling onto the American form used by ICD-10-CM."""
    return _WORD_RE.sub(lambda m: _ortho_word(m.group(0)), text)


# --- vocabulary sources ----------------------------------------------------

def _words(s: str, min_len: int = 3) -> list[str]:
    return [w for w in _WORD_RE.findall(apply_orthography(str(s).lower()))
            if len(w) >= min_len]


def icd_terms() -> Counter:
    """Content words of the ICD-10 category descriptions used for retrieval."""
    path = PROCESSED / "icd_index" / "icd_ref.parquet"
    if not path.exists():
        return Counter()
    c = Counter()
    for d in pd.read_parquet(path)["descr"].astype(str):
        c.update(_words(d))
    return c


def class_terms() -> Counter:
    """Words of the pharmacological class names (the drug2cat targets)."""
    path = PROCESSED / "rxgen_vocab.json"
    if not path.exists():
        return Counter()
    names = json.loads(path.read_text(encoding="utf-8")).get("category_names", {})
    c = Counter()
    for n in names.values():
        c.update(_words(n))
    return c


def glossary_terms() -> Counter:
    """Right-hand sides of the verified abbreviation glossary."""
    c = Counter()
    for v in GLOSSARY.values():
        c.update(_words(v))
    return c


def tier_a() -> set[str]:
    """Clinical vocabulary: always trusted, never corrected away."""
    return set(icd_terms()) | set(class_terms()) | set(glossary_terms())


# --- bounded edit distance -------------------------------------------------

def bounded_levenshtein(a: str, b: str, max_dist: int) -> int:
    """Levenshtein with early exit; returns max_dist + 1 when over budget."""
    la, lb = len(a), len(b)
    if abs(la - lb) > max_dist:
        return max_dist + 1
    if la > lb:
        a, b, la, lb = b, a, lb, la
    prev = list(range(lb + 1))
    big = max_dist + 1
    for i in range(1, la + 1):
        cur = [i] + [big] * lb
        ca = a[i - 1]
        lo = max(1, i - max_dist)
        hi = min(lb, i + max_dist)
        best = big
        for j in range(lo, hi + 1):
            cost = 0 if ca == b[j - 1] else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            cur[j] = v
            if v < best:
                best = v
        if best > max_dist:
            return big
        prev = cur
    return prev[lb] if prev[lb] <= max_dist else big


def edit_budget(tok: str) -> int:
    """Edit budget by length. Short tokens sit in a dense lexical
    neighbourhood ('cold'/'gold'/'bold'), so they get at most one edit."""
    n = len(tok)
    if n < 4:
        return 0
    return 1 if n <= 6 else 2


@dataclass
class TypoCorrector:
    """Maps out-of-vocabulary tokens onto the nearest trusted term.

    Fitted on TRAIN text only. `min_corpus_freq` sets how much corpus evidence
    a non-clinical token needs before it is trusted rather than corrected.
    """
    min_corpus_freq: int = 5
    trusted: set = field(default_factory=set)
    corpus_freq: Counter = field(default_factory=Counter)
    clinical: set = field(default_factory=set)
    mapping: dict = field(default_factory=dict)
    _buckets: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict)

    def fit(self, train_texts) -> "TypoCorrector":
        self.clinical = tier_a()
        self.corpus_freq = Counter()
        for t in train_texts:
            self.corpus_freq.update(_words(t, min_len=1))

        # Index Tier A first so Tier-B candidates can be tested against it.
        self._index(self.clinical)
        demoted = set()
        for w, f in self.corpus_freq.items():
            if w in self.clinical or w in FUNCTION_WORDS:
                continue
            if f >= self.min_corpus_freq and len(w) >= 3:
                if self._nearest(w, None, edit_budget(w)) is not None:
                    demoted.add(w)      # frequent spelling variant

        self.trusted = set(self.clinical) | set(FUNCTION_WORDS) | {
            w for w, f in self.corpus_freq.items()
            if f >= self.min_corpus_freq and len(w) >= 3 and w not in demoted}
        self._index(self.trusted)
        self._cache = {}
        self.mapping = {}
        return self

    def _index(self, vocab) -> None:
        self._buckets = {}
        for w in vocab:
            if w:
                self._buckets.setdefault((w[0], len(w)), []).append(w)

    def _candidates(self, tok: str, budget: int):
        # A misspelling almost never changes the first character, and the
        # length cannot move further than the edit budget.
        for n in range(len(tok) - budget, len(tok) + budget + 1):
            yield from self._buckets.get((tok[0], n), ())

    def _nearest(self, tok: str, restrict, budget: int):
        if budget <= 0:
            return None
        best, best_key = None, None
        for c in self._candidates(tok, budget):
            if c == tok or (restrict is not None and c not in restrict):
                continue
            d = bounded_levenshtein(tok, c, budget)
            if d > budget:
                continue
            # Prefer fewer edits, then the better-attested term; clinical
            # vocabulary outranks corpus frequency at equal distance.
            key = (d, 0 if c in self.clinical else 1,
                   -self.corpus_freq.get(c, 0), c)
            if best_key is None or key < best_key:
                best, best_key = c, key
        return best

    def correct_token(self, tok: str) -> str:
        hit = self._cache.get(tok)
        if hit is not None:
            return hit
        out = tok
        if tok not in self.trusted and tok.isalpha() and len(tok) >= 4:
            cand = self._nearest(tok, self.trusted, edit_budget(tok))
            if cand is not None:
                out = cand
                self.mapping[tok] = cand
        self._cache[tok] = out
        return out

    def correct_text(self, text: str) -> str:
        return _WORD_RE.sub(lambda m: self.correct_token(m.group(0)), text)
