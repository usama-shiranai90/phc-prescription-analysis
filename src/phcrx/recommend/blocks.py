"""Feature blocks and estimators, as scikit-learn transformers.

Every block is a `TransformerMixin` whose `fit` sees the TRAIN rows and
nothing else, so a `Pipeline` built out of them cannot leak by construction:
the TF-IDF vocabulary, the typo corrector, the imputation medians, the
standardisation constants and the one-hot levels are all learned inside
`Pipeline.fit`, which is only ever called on the train slice.

Eight feature sets are assembled from four blocks. Which one a component gets
is decided on VALIDATION, per component -- the feature-importance workstream
measured that dropping temporal/prescriber/site *raised* held-out AUROC for
drug-presence targets while the multi-label class-set task preferred the full
matrix, so the choice is target-dependent and has to be measured rather than
assumed.

    text        TF-IDF over the symptom note                          [D]
    raw         age/sex/smoking/glucose-type + 13 vitals + miss bits   [D]
    text_raw    both of the above                                      [D]
    text_raw_sem  + induced concepts and SapBERT sentence embeddings   [D]
    clinical    engineered matrix minus temporal/prescriber/site
    text_clinical  TF-IDF + that
    all         TF-IDF + the whole engineered matrix
    all_sem     + concepts and SapBERT

[D] marks the sets that are **deployable from a raw encounter alone**. The
others need `features.parquet`, whose 309 columns include a TF-IDF/SVD basis,
ICD codes and leave-one-out site/prescriber target encodings that were fitted
by a separate offline job; a walk-in encounter cannot produce them without
that job in the loop. `evaluate` measures what restricting to [D] costs.

Text representation is fixed in advance from the text-processing ablation
(`results/rx_generation/textproc/ablation.csv`) rather than re-searched here:
glossary expansion in `augmented` mode -- the abbreviation is kept *and* a
joined concept token is added -- plus orthography and typo correction, then
unigrams and bigrams. Replacing `d/m` with `diabetes mellitus` costs micro-F1;
augmenting with it is what produced the significant macro (+0.0253
[+0.011, +0.040]) and tail-macro (+0.0423 [+0.016, +0.069]) gains.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import SEED
from .corpus import (NON_CLINICAL_FAMILIES, RAW_CATEGORICAL, RAW_NUMERIC,
                     RAW_VITALS)

BIGRAM_SEP = "|"
GLOSSARY_MODE = "augmented"
C_DEFAULT = 4.0                 # the constant used by every earlier linear arm
N_JOBS = min(12, (os.cpu_count() or 2))

# Fitted state is memoised per (block, params, train-content hash) so that
# eight feature sets x six components do not refit the same vectoriser 48
# times. The key includes a digest of the exact training rows, so a memo hit
# is only ever a hit on identical input -- it changes nothing about what is
# fitted or on what.
_MEMO: dict = {}


def _identity(x):
    """Module-level so a fitted vectoriser stays picklable."""
    return x


def _digest(parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()


def _texts(X) -> list[str]:
    return X["symptom_text"].fillna("").astype(str).tolist()


# ---------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------
class TextTfidf(TransformerMixin, BaseEstimator):
    """TF-IDF over unigrams + bigrams of the normalised symptom note."""

    def __init__(self, min_df: int = 2, glossary_mode: str = GLOSSARY_MODE):
        self.min_df = min_df
        self.glossary_mode = glossary_mode

    def _docs(self, texts, nz):
        out = []
        for t in texts:
            toks = nz.tokenize(nz(t))
            out.append(toks + [f"{a}{BIGRAM_SEP}{b}"
                               for a, b in zip(toks, toks[1:])])
        return out

    def fit(self, X, y=None):
        from ..textproc.normalize import Normalizer
        texts = _texts(X)
        key = ("TextTfidf", self.min_df, self.glossary_mode, _digest(texts))
        if key in _MEMO:
            self.normalizer_, self.vec_ = _MEMO[key]
            return self
        self.normalizer_ = Normalizer(
            glossary=True, orthography=True, typo=True,
            glossary_mode=self.glossary_mode).fit(texts)
        self.vec_ = TfidfVectorizer(analyzer=_identity, min_df=self.min_df,
                                    sublinear_tf=True)
        self.vec_.fit(self._docs(texts, self.normalizer_))
        _MEMO[key] = (self.normalizer_, self.vec_)
        return self

    def transform(self, X):
        return self.vec_.transform(self._docs(_texts(X), self.normalizer_))


# ---------------------------------------------------------------------------
# semantics: induced concepts + SapBERT
# ---------------------------------------------------------------------------
_ENCODER = None


def _encoder():
    """SapBERT, loaded once per process. Pretrained and frozen: nothing is fitted."""
    global _ENCODER
    if _ENCODER is None:
        from ..nlp.icd_index import SapBertEncoder
        _ENCODER = SapBertEncoder()
    return _ENCODER


class _SemCache:
    """Disk cache of the semantic row for a *normalised* note.

    Keyed by the normalised text, so a cache hit is only ever a hit on the
    identical string. That makes the cache a pure speed-up: it can never make
    the model see a row it would not otherwise have computed.
    """

    def __init__(self, path):
        self.path = path
        self.keys: dict[str, int] = {}
        self.mat: np.ndarray | None = None
        if path.exists():
            blob = np.load(path, allow_pickle=False)
            self.mat = blob["mat"]
            self.keys = {k: i for i, k in enumerate(blob["keys"].tolist())}

    def get(self, norm_texts, expanded, cv):
        want = [_digest([t]) for t in norm_texts]
        missing = sorted({k for k in want if k not in self.keys})
        if missing:
            pos = {k: i for i, k in enumerate(want)}
            todo_norm = [norm_texts[pos[k]] for k in missing]
            todo_exp = [expanded[pos[k]] for k in missing]
            enc = _encoder()
            M, flag = cv.assign(todo_norm, enc)
            E = enc.encode(todo_exp)
            block = np.hstack([M.astype(np.float32),
                               flag.astype(np.float32)[:, None],
                               E.astype(np.float32)])
            base = len(self.keys)
            self.mat = block if self.mat is None else np.vstack([self.mat, block])
            for i, k in enumerate(missing):
                self.keys[k] = base + i
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Written to a sibling then renamed: two fitting runs can share the
            # cache without a reader ever seeing a half-written archive.
            tmp = self.path.with_suffix(f".{os.getpid()}.tmp.npz")
            np.savez_compressed(
                tmp, mat=self.mat,
                keys=np.array(list(self.keys.keys()), dtype=object).astype("U40"))
            os.replace(tmp, self.path)
        return self.mat[[self.keys[k] for k in want]]


_SEM_CACHE = None


class SemanticBlock(TransformerMixin, BaseEstimator):
    """Induced concept multi-hot + SapBERT sentence embedding of the note.

    The concept vocabulary was induced on TRAIN spans by
    `src.phcrx.textproc.concepts` and is loaded frozen; SapBERT is pretrained
    and never updated. Two things *are* fitted, both on train rows only: the
    typo corrector inside the normalisers, and a truncated SVD that compresses
    the 768-dimensional sentence embedding.

    The SVD is a compute decision, not a modelling one. A dense 768-column
    block turns an otherwise sparse design matrix into ~8.6M stored values and
    makes the one-vs-rest fits an order of magnitude slower; 128 components
    retain the great majority of the variance at a fifth of the cost. It is
    the same device `features/build_features.py` already uses for its
    character n-grams.
    """

    def __init__(self, n_svd: int = 128):
        self.n_svd = n_svd

    def _raw(self, X):
        global _SEM_CACHE
        from . import OUT
        if _SEM_CACHE is None:
            _SEM_CACHE = _SemCache(OUT / "sem_cache.npz")
        texts = _texts(X)
        norm = self.nz_norm_.transform(texts)
        expanded = [self.nz_exp_(t) for t in texts]
        return _SEM_CACHE.get(norm, expanded, self.cv_)

    def fit(self, X, y=None):
        from sklearn.decomposition import TruncatedSVD
        from ..textproc.concepts import ConceptVocab
        from ..textproc.normalize import Normalizer
        texts = _texts(X)
        key = ("SemanticBlock", self.n_svd, _digest(texts))
        if key in _MEMO:
            self.nz_norm_, self.nz_exp_, self.cv_, self.svd_ = _MEMO[key]
            self.n_concepts_ = len(self.cv_)
            return self
        self.nz_norm_ = Normalizer(glossary=True, orthography=True,
                                   typo=True).fit(texts)
        self.nz_exp_ = Normalizer(glossary=True, orthography=False, typo=False)
        self.cv_ = ConceptVocab.load()
        self.n_concepts_ = len(self.cv_)
        M = self._raw(X)
        self.svd_ = TruncatedSVD(n_components=self.n_svd, random_state=SEED)
        self.svd_.fit(M[:, self.n_concepts_ + 1:])
        _MEMO[key] = (self.nz_norm_, self.nz_exp_, self.cv_, self.svd_)
        return self

    def transform(self, X):
        M = self._raw(X)
        k = self.n_concepts_ + 1
        return sp.csr_matrix(np.hstack([M[:, :k],
                                        self.svd_.transform(M[:, k:])]))


# ---------------------------------------------------------------------------
# tabular
# ---------------------------------------------------------------------------
def raw_frame(X: pd.DataFrame) -> pd.DataFrame:
    """The subset of the record a clinician has in hand at the consultation.

    Demographics, smoking, which glucose assay was run, the thirteen PHC
    vitals and an explicit missingness bit per vital -- *which* vitals were
    measured is itself signal, because it reflects what the operator
    suspected. Nothing here needs a previously fitted offline job, which is
    what makes the `[D]` feature sets deployable.
    """
    F = pd.DataFrame(index=X.index)
    for c in RAW_NUMERIC:
        F[c] = pd.to_numeric(X[c], errors="coerce") if c in X else np.nan
    for v in RAW_VITALS:
        val = pd.to_numeric(X[v], errors="coerce") if v in X else pd.Series(
            np.nan, index=X.index)
        F[v] = val
        F[f"{v}_miss"] = val.isna().astype(np.int8)
    F["n_vitals_measured"] = 13 - F[[f"{v}_miss" for v in RAW_VITALS]].sum(axis=1)
    for c in RAW_CATEGORICAL:
        F[c] = X[c].astype(str) if c in X else "NA"
    return F


RAW_NUM_COLS = (RAW_NUMERIC
                + [c for v in RAW_VITALS for c in (v, f"{v}_miss")]
                + ["n_vitals_measured"])


class TabularBlock(TransformerMixin, BaseEstimator):
    """Median-impute + standardise numerics, one-hot the categoricals.

    `derive="raw"` recomputes its own columns from the raw encounter record;
    otherwise the columns are read straight from the engineered matrix that
    `features/build_features.py` produced (train-fitted, consumed as-is).
    """

    def __init__(self, num_cols=(), cat_cols=(), derive=None):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.derive = derive

    def _frame(self, X):
        return raw_frame(X) if self.derive == "raw" else X

    def fit(self, X, y=None):
        F = self._frame(X)
        num = [c for c in self.num_cols if c in F.columns]
        cat = [c for c in self.cat_cols if c in F.columns]
        self.ct_ = ColumnTransformer(
            [("num", SkPipeline([("imp", SimpleImputer(strategy="median",
                                                       keep_empty_features=True)),
                                 ("sc", StandardScaler())]), num),
             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True),
              cat)],
            remainder="drop", sparse_threshold=1.0)
        self.ct_.fit(F)
        return self

    def transform(self, X):
        out = self.ct_.transform(self._frame(X))
        return out if sp.issparse(out) else sp.csr_matrix(out)


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------
def _fit_one(X, y, C, max_iter):
    m = LogisticRegression(C=C, max_iter=max_iter, solver="liblinear",
                           random_state=SEED)
    m.fit(X, np.ascontiguousarray(np.asarray(y).astype(np.int8)))
    return m


class MultiLabelOvR(BaseEstimator):
    """One-vs-rest L2 logistic regression over a fixed label space.

    Labels with no positive in TRAIN cannot be fitted (a binary solver needs
    two classes) and are given probability 0 rather than being silently
    dropped from the label space -- the macro and tail metrics have to charge
    the model for classes it never learned, otherwise they flatter it.
    """

    def __init__(self, C: float = C_DEFAULT, max_iter: int = 3000,
                 n_jobs: int = N_JOBS):
        self.C = C
        self.max_iter = max_iter
        self.n_jobs = n_jobs

    def fit(self, X, Y):
        from joblib import Parallel, delayed
        Y = np.asarray(Y)
        self.n_labels_ = Y.shape[1]
        self.kept_ = np.where(Y.sum(0) > 0)[0]
        # max_nbytes=None disables joblib's read-only memmapping of large
        # arrays, which is what forces the rest of the project to fit
        # one-vs-rest serially: liblinear needs a writable buffer and a
        # memmapped worker array is read-only.
        self.models_ = Parallel(n_jobs=self.n_jobs, max_nbytes=None)(
            delayed(_fit_one)(X, Y[:, j], self.C, self.max_iter)
            for j in self.kept_)
        return self

    def predict_proba(self, X):
        P = np.zeros((X.shape[0], self.n_labels_))
        for j, m in zip(self.kept_, self.models_):
            P[:, j] = m.predict_proba(X)[:, 1]
        return P


class MulticlassLR(BaseEstimator):
    """Multinomial-free one-vs-rest logistic regression over a fixed class set.

    `n_classes` is pinned so the probability matrix always has the same width
    as the declared class list, even if a class happens to be absent from the
    training rows -- otherwise `argmax` would silently mean a different class.
    """

    def __init__(self, n_classes: int = 2, C: float = C_DEFAULT,
                 max_iter: int = 3000, n_jobs: int = N_JOBS):
        self.n_classes = n_classes
        self.C = C
        self.max_iter = max_iter
        self.n_jobs = n_jobs

    def fit(self, X, y):
        # liblinear is binary-only from sklearn 1.9, so multiclass is fitted
        # one-vs-rest over the same binary estimator every other linear arm in
        # the project uses, then renormalised.
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        inner = MultiLabelOvR(C=self.C, max_iter=self.max_iter,
                              n_jobs=self.n_jobs)
        Y = np.stack([(y == c).astype(np.int8) for c in self.classes_], axis=1)
        self.est_ = inner.fit(X, Y)
        return self

    def predict_proba(self, X):
        raw = self.est_.predict_proba(X)
        raw = raw / np.maximum(raw.sum(1, keepdims=True), 1e-12)
        P = np.zeros((X.shape[0], self.n_classes))
        P[:, np.asarray(self.classes_, dtype=int)] = raw
        return P

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


# ---------------------------------------------------------------------------
# feature-set registry
# ---------------------------------------------------------------------------
def _blocks(FF: dict) -> dict:
    fam = FF["families"]
    eng = FF["engineered_columns"]
    cat = [c for c in FF["categorical_features"] if c in eng]
    clin = [c for c in eng if fam.get(c) not in NON_CLINICAL_FAMILIES]
    return {
        "text": ("text", TextTfidf()),
        "sem": ("sem", SemanticBlock()),
        "raw": ("raw", TabularBlock(num_cols=RAW_NUM_COLS,
                                    cat_cols=RAW_CATEGORICAL, derive="raw")),
        "tab_all": ("tab_all", TabularBlock(
            num_cols=[c for c in eng if c not in cat], cat_cols=cat)),
        "tab_clinical": ("tab_clinical", TabularBlock(
            num_cols=[c for c in clin if c not in cat],
            cat_cols=[c for c in cat if c in clin])),
    }


# name -> (blocks, deployable-from-a-raw-encounter?)
FEATURE_SETS: dict[str, tuple[tuple[str, ...], bool]] = {
    "text": (("text",), True),
    "raw": (("raw",), True),
    "text_raw": (("text", "raw"), True),
    "text_raw_sem": (("text", "raw", "sem"), True),
    "clinical": (("tab_clinical",), False),
    "text_clinical": (("text", "tab_clinical"), False),
    "all": (("text", "tab_all"), False),
    "all_sem": (("text", "tab_all", "sem"), False),
}
DEPLOYABLE = {k for k, (_, d) in FEATURE_SETS.items() if d}


def make_features(name: str, FF: dict) -> FeatureUnion:
    bank = _blocks(FF)
    parts = [bank[b] for b in FEATURE_SETS[name][0]]
    return FeatureUnion([(n, clone(t)) for n, t in parts])


def make_pipeline(feature_set: str, estimator, FF: dict) -> SkPipeline:
    return SkPipeline([("features", make_features(feature_set, FF)),
                       ("est", estimator)])
