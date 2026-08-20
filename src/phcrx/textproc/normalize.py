"""The normalisation pipeline, as a set of switchable stages.

Order matters and is not arbitrary:

  1. encoding repair   -- `preprocess.norm_text`: mojibake, NFKC, Bengali
                          digits, stray HTML, whitespace. Must be first;
                          every later stage assumes well-formed text.
  2. glossary expansion -- `nlp.glossary.expand`. Must precede typo
                          correction: `d/m` and `h/o` are not misspellings,
                          and a corrector let loose on them first would either
                          leave them opaque or map them onto something wrong.
  3. orthography        -- British -> American, matching the ICD-10-CM
                          reference the clinical lexicon is drawn from. Must
                          precede correction so the corrector does not spend
                          its edit budget on a non-error.
  4. typo correction    -- `lexicon.TypoCorrector`, fitted on TRAIN only.

Each stage is individually switchable so the ablation can attribute a metric
change to exactly one of them.

Glossary expansion additionally has a *surface form* choice, which the first
ablation showed is not a detail. Replacing `d/m` with `diabetes mellitus` cost
micro-F1 rather than adding it, and the suspected mechanism is fragmentation:
to a bag-of-words model `d/m` is one highly distinctive type, while its
expansion is two commoner types that also occur in unrelated notes.
`glossary_mode` therefore selects the surface form of the expansion:

    "spaced"      d/m -> "diabetes mellitus"        (the original behaviour)
    "joined"      d/m -> "diabetes_mellitus"        (one token, semantics kept)
    "augmented"   d/m -> "d/m diabetes_mellitus"    (surface form also kept)
    "canonical"   d/m -> "diabetes_mellitus", and the spelled-out phrase
                  "diabetes mellitus" is rewritten to the same token, so the
                  two surface forms share one feature
    "canonical_augmented"
                  as "canonical", but nothing is removed -- the joined token is
                  added alongside the abbreviation and the spelled-out words

Only "spaced" is used by `concepts`, `build_features` and `diagnostics`, and it
delegates to `nlp.glossary.expand` unchanged, so no consumer outside the
ablation changes behaviour.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..nlp.glossary import GLOSSARY, expand as glossary_expand
from ..preprocess import norm_text, tokenize
from .lexicon import TypoCorrector, apply_orthography

# Stage names in pipeline order; used by the ablation to build cumulative arms.
STAGES = ("glossary", "orthography", "typo")

GLOSSARY_MODES = ("spaced", "joined", "augmented", "canonical",
                  "canonical_augmented")

# --- glossary surface forms ------------------------------------------------

JOIN = "_"
_ALNUM = re.compile(r"[a-z0-9]+")


def _joined(value: str) -> str:
    """'Post-prandial blood sugar' -> 'post_prandial_blood_sugar'."""
    return JOIN.join(_ALNUM.findall(str(value).lower()))


_JOINED_FORM = {a: _joined(v) for a, v in GLOSSARY.items()}

# One pass, longest alternative first, so an expansion can never be rewritten
# a second time by a shorter key that happens to occur inside it.
_ABBR_RE = re.compile(
    r"(?<![a-z0-9])(" +
    "|".join(re.escape(a) for a in sorted(GLOSSARY, key=len, reverse=True)) +
    r")(?![a-z0-9])")

# The spelled-out forms of the multi-word glossary values. These are not rare:
# the corpus writes 'low back pain' in 528 notes and 'lbp' in 333, and a model
# that cannot tell the two are the same thing fits one concept twice.
_PHRASES: list[tuple[str, str]] = []
for _abbr, _value in GLOSSARY.items():
    _w = _ALNUM.findall(str(_value).lower())
    if len(_w) > 1:
        # The word separator must exclude JOIN, or the pattern matches the
        # joined token this module has just produced and emits it twice.
        _PHRASES.append((r"(?<![a-z0-9_])"
                         + r"[^a-z0-9_]+".join(re.escape(x) for x in _w)
                         + r"(?![a-z0-9_])", JOIN.join(_w)))
# Longest first, so 'upper respiratory tract infection' is not consumed by the
# shorter 'urinary tract infection' pattern overlapping it.
_PHRASES.sort(key=lambda p: -len(p[0]))
_PHRASE_RE = re.compile("|".join(f"({p})" for p, _ in _PHRASES))
_PHRASE_TARGET = [t for _, t in _PHRASES]


def _phrase_sub(m: "re.Match", augment: bool) -> str:
    target = _PHRASE_TARGET[(m.lastindex or 1) - 1]
    return f"{m.group(0)} {target}" if augment else target


def expand_abbrev(text, mode: str = "spaced") -> str:
    """Expand local abbreviations under one of `GLOSSARY_MODES`."""
    if mode not in GLOSSARY_MODES:
        raise ValueError(f"unknown glossary_mode {mode!r}")
    if mode == "spaced":
        # Delegate, so every other consumer of this package is untouched.
        return glossary_expand(text)

    s = str(text).lower()
    if mode in ("joined", "canonical"):
        s = _ABBR_RE.sub(lambda m: _JOINED_FORM[m.group(1)], s)
    else:
        s = _ABBR_RE.sub(lambda m: f"{m.group(1)} {_JOINED_FORM[m.group(1)]}", s)
    if mode == "canonical":
        s = _PHRASE_RE.sub(lambda m: _phrase_sub(m, False), s)
    elif mode == "canonical_augmented":
        s = _PHRASE_RE.sub(lambda m: _phrase_sub(m, True), s)
    return re.sub(r"\s+", " ", s).strip()


# The production token regex plus underscore-joined runs, so a joined expansion
# survives tokenisation as the single type it was built to be. Everything else
# -- slashed abbreviations, bare words, numbers -- matches exactly as
# `preprocess._TOKEN_RE` matches it.
_TOKEN_RE_JOINED = re.compile(
    r"[a-z]+(?:_[a-z]+)+|[a-z]+(?:[/.][a-z]+)+|[a-z]+|\d+(?:\.\d+)?")


def tokens_joined(text) -> list[str]:
    return _TOKEN_RE_JOINED.findall(norm_text(text).lower())


@dataclass
class Normalizer:
    """Composable text normalisation. Fit on TRAIN, apply anywhere."""
    glossary: bool = True
    orthography: bool = True
    typo: bool = True
    glossary_mode: str = "spaced"
    min_corpus_freq: int = 5
    corrector: TypoCorrector | None = None

    def fit(self, train_texts) -> "Normalizer":
        if not self.typo:
            return self
        # The corrector must see text at the stage it will actually run at,
        # otherwise its corpus counts are for a different token distribution.
        prepared = [self._upto_typo(t) for t in train_texts]
        self.corrector = TypoCorrector(min_corpus_freq=self.min_corpus_freq).fit(prepared)
        return self

    def _upto_typo(self, text) -> str:
        s = norm_text(text).lower()
        if self.glossary:
            s = expand_abbrev(s, self.glossary_mode).lower()
        if self.orthography:
            s = apply_orthography(s)
        return s

    def __call__(self, text) -> str:
        s = self._upto_typo(text)
        if self.typo and self.corrector is not None:
            s = self.corrector.correct_text(s)
        return s

    def transform(self, texts) -> list[str]:
        return [self(t) for t in texts]

    def tokenize(self, text) -> list[str]:
        """Tokenise *already-normalised* text the way this arm requires.

        Identical to `preprocess.tokenize` unless the arm produced joined
        expansions, which the production regex would split back apart on the
        underscore -- undoing the only thing the arm changes.
        """
        if self.glossary and self.glossary_mode != "spaced":
            return tokens_joined(text)
        return tokenize(text)


def baseline_tokens(text) -> list[str]:
    """Exactly what `preprocess.tokenize` produces today -- the baseline arm."""
    return tokenize(text)


def tokens(text) -> list[str]:
    """Tokenise already-normalised text with the production token regex."""
    return tokenize(text)
