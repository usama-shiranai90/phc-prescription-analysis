"""Metrics, bootstrap machinery and probability calibration.

Set metrics reproduce the definitions already in use so that numbers here are
comparable to the ones in `docs/model_comparison.md` and
`results/rx_generation/textproc/ablation.csv`:

    micro_f1        2*sum(tp) / (sum|pred| + sum|gold|)
    macro_f1        unweighted mean per-label F1 over labels with >=1 gold
                    positive in the evaluated rows (head_to_head convention,
                    the one the published macro numbers use)
    macro_f1_all    the same over every label in the space, absent labels
                    scoring 0 (textproc convention)
    tail_macro_f1   macro over the half of the label space with the fewest
                    TRAIN positives -- fixed before anything is fitted, so it
                    cannot be chosen to flatter a result
    jaccard         mean per-row |int| / |union|, with the empty/empty row
                    scoring 1
    exact_match     mean per-row set equality

Bootstrapping is exact and cheap rather than approximate and slow. Every set
metric is a function of per-row-per-label (tp, pred, gold) counts, so one
resample is a weighted column sum; B resamples are a single (B x n) @ (n x L)
matmul. The same resample indices are reused across every system, which makes
each comparison *paired* -- systems are scored on the same rows and an
unpaired interval would badly overstate the uncertainty of a difference.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-9


# ---------------------------------------------------------------------------
# label-space helpers
# ---------------------------------------------------------------------------
def tail_labels(Ytrain: np.ndarray) -> np.ndarray:
    """Half of the label space with the fewest train positives."""
    order = np.argsort(np.asarray(Ytrain).sum(0), kind="stable")
    return np.sort(order[: max(1, len(order) // 2)])


# ---------------------------------------------------------------------------
# set metrics
# ---------------------------------------------------------------------------
def count_matrices(Y: np.ndarray, P: np.ndarray):
    Y, P = np.asarray(Y, bool), np.asarray(P, bool)
    return ((Y & P).astype(np.float32), P.astype(np.float32),
            Y.astype(np.float32))


def _from_sums(tps, prs, gds, tail, jac, exact):
    micro = 2 * tps.sum(-1) / np.maximum(prs.sum(-1) + gds.sum(-1), EPS)
    f1 = 2 * tps / np.maximum(prs + gds, EPS)
    present = gds > 0
    macro = np.where(present.sum(-1) > 0,
                     (f1 * present).sum(-1) / np.maximum(present.sum(-1), EPS),
                     0.0)
    macro_all = f1.mean(-1)
    tail_macro = f1[..., tail].mean(-1)
    prec = tps.sum(-1) / np.maximum(prs.sum(-1), EPS)
    rec = tps.sum(-1) / np.maximum(gds.sum(-1), EPS)
    return {"micro_f1": micro, "macro_f1": macro, "macro_f1_all": macro_all,
            "tail_macro_f1": tail_macro, "micro_precision": prec,
            "micro_recall": rec, "jaccard": jac, "exact_match": exact,
            "mean_pred_size": prs.sum(-1), "mean_gold_size": gds.sum(-1)}


def row_scalars(Y: np.ndarray, P: np.ndarray):
    Y, P = np.asarray(Y, bool), np.asarray(P, bool)
    tp = (Y & P).sum(1).astype(np.float64)
    union = (Y | P).sum(1).astype(np.float64)
    jac = np.where(union == 0, 1.0, tp / np.maximum(union, EPS))
    return jac, (Y == P).all(1).astype(np.float64)


def set_metrics(Y: np.ndarray, P: np.ndarray, tail: np.ndarray) -> dict:
    TP, PR, GD = count_matrices(Y, P)
    jac, exact = row_scalars(Y, P)
    n = max(len(Y), 1)
    m = _from_sums(TP.sum(0), PR.sum(0), GD.sum(0), tail,
                   float(jac.mean()), float(exact.mean()))
    m["mean_pred_size"] = float(m["mean_pred_size"]) / n
    m["mean_gold_size"] = float(m["mean_gold_size"]) / n
    m["empty_pred_rate"] = float((PR.sum(1) == 0).mean())
    return {k: float(v) for k, v in m.items()}


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
class Resampler:
    """One fixed set of B row resamples, shared by every system scored."""

    def __init__(self, n: int, n_boot: int = 2000, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n = n
        self.idx = rng.integers(0, n, (n_boot, n))
        W = np.zeros((n_boot, n), dtype=np.float32)
        for b in range(n_boot):
            W[b] = np.bincount(self.idx[b], minlength=n)
        self.W = W

    def set_metrics(self, Y, P, tail) -> dict:
        TP, PR, GD = count_matrices(Y, P)
        jac, exact = row_scalars(Y, P)
        d = _from_sums(self.W @ TP, self.W @ PR, self.W @ GD, tail,
                       (self.W @ jac) / self.n, (self.W @ exact) / self.n)
        d["mean_pred_size"] = d["mean_pred_size"] / self.n
        d["mean_gold_size"] = d["mean_gold_size"] / self.n
        return d

    def row_mean(self, v) -> np.ndarray:
        return (self.W @ np.asarray(v, dtype=np.float64)) / self.n

    def curve_metric(self, y, p, fn) -> np.ndarray:
        out = np.empty(len(self.idx))
        y, p = np.asarray(y), np.asarray(p)
        for b, i in enumerate(self.idx):
            yb = y[i]
            out[b] = fn(yb, p[i]) if 0 < yb.sum() < len(yb) else np.nan
        return out


def ci(v, lo=2.5, hi=97.5) -> tuple[float, float]:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan"), float("nan")
    a, b = np.percentile(v, [lo, hi])
    return float(a), float(b)


def delta_stats(arm, ref) -> dict:
    d = np.asarray(arm, dtype=float) - np.asarray(ref, dtype=float)
    d = d[np.isfinite(d)]
    lo, hi = ci(d)
    return {"delta": float(d.mean()), "delta_lo95": lo, "delta_hi95": hi,
            "p_better": float((d > 0).mean())}


# ---------------------------------------------------------------------------
# scalar metrics
# ---------------------------------------------------------------------------
def auroc(y, p) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def avg_precision(y, p) -> float:
    y = np.asarray(y)
    return float(average_precision_score(y, p)) if y.sum() else float("nan")


def ece(p, y, n_bins: int = 10) -> float:
    """Expected calibration error, equal-width confidence bins."""
    p = np.asarray(p, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if not len(p):
        return float("nan")
    b = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    tot = 0.0
    for k in range(n_bins):
        m = b == k
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def ece_macro(P, Y, n_bins: int = 10) -> float:
    P, Y = np.asarray(P, dtype=float), np.asarray(Y, dtype=float)
    if P.ndim == 1:
        return ece(P, Y, n_bins)
    vals = [ece(P[:, j], Y[:, j], n_bins) for j in range(P.shape[1])
            if Y[:, j].sum() > 0]
    return float(np.mean(vals)) if vals else float("nan")


def brier(P, Y) -> float:
    return float(np.mean((np.asarray(P, dtype=float)
                          - np.asarray(Y, dtype=float)) ** 2))


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
MIN_CALIB_POSITIVES = 10


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _as2d(A) -> np.ndarray:
    """(n,) -> (n, 1); (n, L) unchanged. Never (1, n)."""
    A = np.asarray(A, dtype=float)
    return A.reshape(-1, 1) if A.ndim == 1 else A


class ProbCalibrator:
    """Per-label isotonic or Platt scaling, fitted on VALIDATION only.

    A label with fewer than `MIN_CALIB_POSITIVES` positives in the calibration
    set keeps its raw probability: an isotonic fit on four positives is noise
    dressed as a correction.
    """

    def __init__(self, method: str = "none"):
        self.method = method

    def fit(self, P, Y):
        P, Y = _as2d(P), _as2d(Y)
        if P.shape != Y.shape:
            raise ValueError(f"P{P.shape} and Y{Y.shape} disagree")
        self.models_ = []
        for j in range(P.shape[1]):
            y, p = Y[:, j], P[:, j]
            if self.method == "none" or y.sum() < MIN_CALIB_POSITIVES or \
                    y.sum() == len(y):
                self.models_.append(None)
            elif self.method == "isotonic":
                m = IsotonicRegression(y_min=0.0, y_max=1.0,
                                       out_of_bounds="clip")
                m.fit(p, y)
                self.models_.append(("iso", m))
            elif self.method == "sigmoid":
                m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
                m.fit(_logit(p).reshape(-1, 1), y.astype(int))
                self.models_.append(("sig", m))
            else:
                raise ValueError(self.method)
        return self

    def transform(self, P):
        flat = np.asarray(P).ndim == 1
        P = _as2d(P)
        out = P.copy()
        for j, m in enumerate(self.models_):
            if m is None:
                continue
            kind, mod = m
            out[:, j] = (mod.predict(P[:, j]) if kind == "iso"
                         else mod.predict_proba(_logit(P[:, j]).reshape(-1, 1))[:, 1])
        return np.clip(out.ravel() if flat else out, 0.0, 1.0)


def select_calibrator(P, Y, methods=("none", "sigmoid", "isotonic"),
                      n_splits: int = 5, seed: int = 0):
    """Choose a calibration method by K-fold CV *inside* the validation split.

    Fitting a calibrator on val and then choosing the method by its fit on the
    same val rows would pick whichever method overfits val hardest. Folding
    inside val estimates out-of-sample calibrated ECE honestly; the winner is
    then refitted on all of val.
    """
    from sklearn.model_selection import KFold
    P2, Y2 = _as2d(P), _as2d(Y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = {}
    for meth in methods:
        vals = []
        for tr, te in kf.split(P2):
            c = ProbCalibrator(meth).fit(P2[tr], Y2[tr])
            vals.append(ece(c.transform(P2[te]), Y2[te]))
        scores[meth] = float(np.mean(vals))
    best = min(scores, key=scores.get)
    cal = ProbCalibrator(best).fit(P2, Y2)
    return best, cal, scores


# ---------------------------------------------------------------------------
# operating point
# ---------------------------------------------------------------------------
THR_GRID = np.arange(0.05, 0.71, 0.01)


def tune_threshold(Y, P, grid=THR_GRID) -> float:
    Y = np.asarray(Y, bool)
    best, bf = float(grid[0]), -1.0
    gold = Y.sum()
    for t in grid:
        B = P >= t
        f = 2 * float((Y & B).sum()) / max(float(B.sum() + gold), EPS)
        if f > bf:
            best, bf = float(t), f
    return best


def precision_recall_at_k(Y, P, ks=(1, 3, 5)) -> dict:
    """Ranking quality, which is what a suggestion list actually shows."""
    Y = np.asarray(Y, bool)
    order = np.argsort(-np.asarray(P), axis=1)
    out = {}
    gold = Y.sum(1)
    for k in ks:
        top = order[:, :k]
        hit = np.take_along_axis(Y, top, axis=1).sum(1).astype(float)
        out[f"precision_at_{k}"] = float((hit / k).mean())
        has = gold > 0
        out[f"recall_at_{k}"] = float((hit[has] / gold[has]).mean()) if has.any() \
            else float("nan")
    return out
