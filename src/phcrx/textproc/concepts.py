"""A concept vocabulary induced from the corpus, not written by hand.

A hand-authored symptom list encodes what the author expects a primary-care
cohort to complain about, which is exactly the assumption the rest of this
project has been trying to avoid making. Instead:

  1. segment every TRAIN note into complaint spans and strip duration tails,
     giving a distribution over complaint *phrases*;
  2. keep phrases attested at least `min_phrase_freq` times;
  3. embed them with SapBERT (the same encoder already used for ICD
     retrieval, trained for biomedical entity linking) and cluster with
     average-linkage agglomerative clustering under a cosine threshold;
  4. keep clusters whose total corpus support clears `min_support`, and name
     each one after its most frequent member phrase.

Assignment at inference is exact-phrase first, then nearest centroid above a
cosine floor, so an unseen paraphrase still lands on a concept.

Induction uses TRAIN spans only. Val/test notes are only ever *assigned*.

    python -m src.phcrx.textproc.concepts
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from ..config import PROCESSED, RESULTS
from ..nlp.icd_index import SapBertEncoder
from .normalize import Normalizer
from .segment import is_no_complaint, span_keys

OUT = RESULTS / "textproc"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = PROCESSED / "textproc"
CACHE.mkdir(parents=True, exist_ok=True)

# A span must carry at least one of these to be a plausible complaint phrase;
# spans that are pure administrative filler ('follow up', 'seen by') would
# otherwise found their own clusters.
_MIN_CHARS = 3


class ConceptVocab:
    """Induced concept clusters plus the assignment rule."""

    def __init__(self, labels: list[str], centroids: np.ndarray,
                 phrase2concept: dict[str, int], support: list[int],
                 members: list[list[str]], tau: float):
        self.labels = labels
        self.centroids = centroids
        self.phrase2concept = phrase2concept
        self.support = support
        self.members = members
        self.tau = tau

    def __len__(self) -> int:
        return len(self.labels)

    # -- induction ---------------------------------------------------------
    @classmethod
    def induce(cls, train_norm_texts, encoder: SapBertEncoder,
               min_phrase_freq: int = 2, min_support: int = 25,
               distance_threshold: float = 0.30, tau: float = 0.72,
               max_phrases: int = 4000) -> "ConceptVocab":
        from sklearn.cluster import AgglomerativeClustering

        freq = Counter()
        for t in train_norm_texts:
            for k in span_keys(t):
                if len(k) >= _MIN_CHARS:
                    freq[k] += 1
        phrases = [p for p, f in freq.most_common(max_phrases)
                   if f >= min_phrase_freq]
        if not phrases:
            raise RuntimeError("no candidate phrases -- segmentation produced nothing")
        emb = encoder.encode(phrases)

        cl = AgglomerativeClustering(
            n_clusters=None, distance_threshold=distance_threshold,
            metric="cosine", linkage="average").fit(emb)

        groups: dict[int, list[int]] = defaultdict(list)
        for i, c in enumerate(cl.labels_):
            groups[int(c)].append(i)

        labels, cents, sup, members = [], [], [], []
        for _, idx in sorted(groups.items()):
            s = int(sum(freq[phrases[i]] for i in idx))
            if s < min_support:
                continue
            best = max(idx, key=lambda i: freq[phrases[i]])
            w = np.array([freq[phrases[i]] for i in idx], dtype=np.float32)
            v = (emb[idx] * w[:, None]).sum(0)
            n = np.linalg.norm(v) or 1.0
            labels.append(phrases[best])
            cents.append(v / n)
            sup.append(s)
            members.append([phrases[i] for i in idx])

        order = np.argsort(-np.asarray(sup))
        labels = [labels[i] for i in order]
        members = [members[i] for i in order]
        sup = [sup[i] for i in order]
        cents = np.stack([cents[i] for i in order]).astype(np.float32)

        p2c = {}
        for ci, ms in enumerate(members):
            for m in ms:
                p2c[m] = ci
        return cls(labels, cents, p2c, sup, members, tau)

    # -- assignment --------------------------------------------------------
    def assign(self, norm_texts, encoder: SapBertEncoder) -> tuple[np.ndarray, np.ndarray]:
        """Multi-hot concept matrix (n_notes x n_concepts) + no-complaint flag."""
        spans_per_note = [span_keys(t) for t in norm_texts]
        flag = np.array([1 if (str(t).strip() and is_no_complaint(str(t))) else 0
                         for t in norm_texts], dtype=np.int8)

        need, owner = [], []
        for i, spans in enumerate(spans_per_note):
            for s in spans:
                if s not in self.phrase2concept and len(s) >= _MIN_CHARS:
                    need.append(s)
                    owner.append(i)
        resolved: dict[str, int] = {}
        if need:
            uniq = sorted(set(need))
            emb = encoder.encode(uniq)
            sims = emb @ self.centroids.T
            best = sims.argmax(1)
            for j, p in enumerate(uniq):
                if sims[j, best[j]] >= self.tau:
                    resolved[p] = int(best[j])

        M = np.zeros((len(spans_per_note), len(self.labels)), dtype=np.int8)
        for i, spans in enumerate(spans_per_note):
            for s in spans:
                c = self.phrase2concept.get(s, resolved.get(s))
                if c is not None:
                    M[i, c] = 1
        return M, flag

    # -- persistence -------------------------------------------------------
    def save(self, stem=None) -> None:
        stem = stem or (CACHE / "concepts")
        np.save(f"{stem}_centroids.npy", self.centroids)
        payload = {"tau": self.tau, "labels": self.labels,
                   "support": self.support, "members": self.members}
        with open(f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1, ensure_ascii=False)

    @classmethod
    def load(cls, stem=None) -> "ConceptVocab":
        stem = stem or (CACHE / "concepts")
        cents = np.load(f"{stem}_centroids.npy")
        with open(f"{stem}.json", encoding="utf-8") as f:
            p = json.load(f)
        p2c = {m: i for i, ms in enumerate(p["members"]) for m in ms}
        return cls(p["labels"], cents, p2c, p["support"], p["members"], p["tau"])


def build(min_support: int = 25, distance_threshold: float = 0.30,
          tau: float = 0.72) -> ConceptVocab:
    enc = pd.read_parquet(PROCESSED / "rxgen_encounters.parquet")
    enc["symptom_text"] = enc["symptom_text"].fillna("").astype(str)
    tr = enc.loc[enc.split == "train", "symptom_text"].tolist()

    nz = Normalizer().fit(tr)
    tr_norm = nz.transform(tr)
    encoder = SapBertEncoder()
    cv = ConceptVocab.induce(tr_norm, encoder, min_support=min_support,
                             distance_threshold=distance_threshold, tau=tau)
    cv.save()

    all_norm = nz.transform(enc["symptom_text"].tolist())
    M, flag = cv.assign(all_norm, encoder)
    cov = (M.sum(1) > 0)
    with_text = np.array([bool(str(t).strip()) for t in enc["symptom_text"]])
    print(f"concepts={len(cv)}  "
          f"coverage: {cov[with_text].mean():.1%} of notes with text carry "
          f"at least one concept; mean concepts/note={M[with_text].sum(1).mean():.2f}")
    print(f"no-complaint flag set on {int(flag.sum())} notes")
    print("\nTop 40 induced concepts (label = most frequent member phrase):")
    for i in range(min(40, len(cv))):
        ex = ", ".join(cv.members[i][1:4])
        print(f"  {cv.support[i]:5d}  {cv.labels[i][:38]:38s} | {ex[:60]}")

    (OUT / "concepts_summary.json").write_text(json.dumps({
        "n_concepts": len(cv),
        "note_coverage": float(cov[with_text].mean()),
        "mean_concepts_per_note": float(M[with_text].sum(1).mean()),
        "concepts": [{"label": cv.labels[i], "support": cv.support[i],
                      "n_members": len(cv.members[i]),
                      "members": cv.members[i][:12]} for i in range(len(cv))],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nwrote", OUT / "concepts_summary.json")
    return cv


if __name__ == "__main__":
    build()
