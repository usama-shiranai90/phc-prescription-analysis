"""Feature importance for each prescription component.

    python -m src.phcrx.features.importance

For every prescription component -- will any drug be given, how many, which of
the top-20 pharmacological classes, the structured attributes (dose / duration /
instruction / form), the advice set and the test-order set -- fit a
HistGradientBoosting model on the TRAIN split, evaluate on the patient-disjoint
TEST split, and report importance two ways:

* **permutation importance on TEST** -- the honest measure: what the fitted
  model actually leans on when it is scored on unseen patients. Computed both
  per feature and per *family* (the whole family's columns are permuted with a
  single shared row order, which keeps within-family correlation intact and so
  does not undercount redundant families the way summing single-feature
  importances does).
* **univariate mutual information on TRAIN** -- marginal association, ignoring
  the model. A feature can score high here and zero on permutation because a
  correlated feature already carries the signal.

Three model variants are fitted for every target so the confounding can be
read off directly:

    full            all 309 features
    no_prescriber   prescriber family dropped
    clinical_only   prescriber *and* site families dropped -- patient signal only

Outputs land in `results/rx_generation/features/`.
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss, f1_score,
                             mean_absolute_error, mutual_info_score, r2_score,
                             roc_auc_score, root_mean_squared_error)

from ..config import PROCESSED, RESULTS

warnings.filterwarnings("ignore")

OUT = RESULTS / "features"
OUT.mkdir(parents=True, exist_ok=True)
DOCS = PROCESSED.parents[1] / "docs"

SEED = 0
N_REPEATS_FEATURE = 4     # per-feature permutation repeats
N_REPEATS_FAMILY = 10     # per-family permutation repeats (cheap, 10 families)
MI_BINS = 16

VARIANTS = {
    "full": (),
    "no_prescriber": ("prescriber",),
    "clinical_only": ("prescriber", "site"),
}

# Targets shown in the figures and called out in the report.
HEADLINE = [
    "y_any_drug", "y_n_drugs", "y_class_ppi_9",
    "y_class_oral_hypo_glycenic_drug_5", "y_class_arb_acei_1",
    "y_class_paracetamol_12", "y_class_vitamin_18", "y_class_nsaids_11",
    "y_class_bromazepum_13", "y_duration_mode", "y_any_advice", "y_any_test",
]


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def _predict(model, X, kind: str) -> np.ndarray:
    if kind == "binary":
        return model.predict_proba(X)[:, 1]
    if kind == "multiclass":
        return model.predict(X)
    return model.predict(X)


def _score(y, pred, kind: str) -> float:
    """Higher is better, for every target kind."""
    if kind == "binary":
        return float(roc_auc_score(y, pred))
    if kind == "multiclass":
        return float(accuracy_score(y, pred))
    return float(-mean_absolute_error(y, pred))       # count -> negative MAE


def _full_metrics(y, model, X, kind: str) -> dict:
    if kind == "binary":
        p = model.predict_proba(X)[:, 1]
        prev = float(np.mean(y))
        ap = float(average_precision_score(y, p))
        return {"prevalence": prev, "auroc": float(roc_auc_score(y, p)),
                "avg_precision": ap, "ap_lift": ap / prev if prev else np.nan,
                "brier": float(brier_score_loss(y, p))}
    if kind == "multiclass":
        yh = model.predict(X)
        vals, cnts = np.unique(y, return_counts=True)
        return {"accuracy": float(accuracy_score(y, yh)),
                "majority_baseline": float(cnts.max() / cnts.sum()),
                "balanced_accuracy": float(balanced_accuracy_score(y, yh)),
                "macro_f1": float(f1_score(y, yh, average="macro")),
                "n_classes": int(len(vals))}
    yh = model.predict(X)
    return {"mae": float(mean_absolute_error(y, yh)),
            "rmse": float(root_mean_squared_error(y, yh)),
            "r2": float(r2_score(y, yh)),
            "mean_target": float(np.mean(y))}


# ---------------------------------------------------------------------------
# importance
# ---------------------------------------------------------------------------
def permutation_importance_groups(model, X: pd.DataFrame, y, kind: str,
                                  groups: dict[str, list[str]],
                                  n_repeats: int, rng: np.random.Generator
                                  ) -> pd.DataFrame:
    """Drop in test score when a group of columns is jointly shuffled.

    All columns of a group share one row permutation, so the group's internal
    correlation structure survives and only its link to the row is broken. For
    single-column groups this reduces to ordinary permutation importance.
    """
    base = _score(y, _predict(model, X, kind), kind)
    rows = []
    Xw = X.copy()
    n = len(X)
    for name, cols in groups.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        orig = X[cols].to_numpy(copy=True)
        drops = []
        for _ in range(n_repeats):
            perm = rng.permutation(n)
            Xw.loc[:, cols] = orig[perm]
            drops.append(base - _score(y, _predict(model, Xw, kind), kind))
        Xw.loc[:, cols] = orig
        rows.append({"group": name, "n_cols": len(cols),
                     "importance": float(np.mean(drops)),
                     "importance_std": float(np.std(drops)),
                     "baseline_score": base})
    return pd.DataFrame(rows)


def _bin_matrix(X: pd.DataFrame, edges: dict) -> np.ndarray:
    """Quantile-bin every column; NaN gets its own top bin."""
    out = np.empty(X.shape, dtype=np.int16)
    for j, c in enumerate(X.columns):
        v = X[c].to_numpy(dtype=float)
        e = edges[c]
        b = np.digitize(v, e) if len(e) else np.zeros(len(v), dtype=int)
        out[:, j] = np.where(np.isnan(v), MI_BINS + 1, b)
    return out


def _mi_edges(Xtr: pd.DataFrame) -> dict:
    edges = {}
    for c in Xtr.columns:
        v = Xtr[c].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        if v.size == 0:
            edges[c] = np.array([])
            continue
        u = np.unique(v)
        edges[c] = (u[:-1] + np.diff(u) / 2 if u.size <= MI_BINS else
                    np.unique(np.quantile(v, np.linspace(0, 1, MI_BINS + 1)[1:-1])))
    return edges


def mutual_information(Xb: np.ndarray, cols: list[str], y) -> pd.Series:
    """Univariate MI (nats) between each binned feature and the target."""
    yb = pd.Series(y).astype("category").cat.codes.to_numpy()
    return pd.Series([mutual_info_score(Xb[:, j], yb) for j in range(Xb.shape[1])],
                     index=cols)


# ---------------------------------------------------------------------------
# modelling
# ---------------------------------------------------------------------------
def _make_model(kind: str, cat_cols: list[str]):
    common = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                  min_samples_leaf=25, l2_regularization=1.0,
                  early_stopping=True, validation_fraction=0.15,
                  n_iter_no_change=20, random_state=SEED,
                  categorical_features=cat_cols or None)
    if kind == "count":
        return HistGradientBoostingRegressor(loss="poisson", **common)
    return HistGradientBoostingClassifier(**common)


def _target_spec(df: pd.DataFrame, meta: dict) -> list[dict]:
    spec = [
        {"name": "y_any_drug", "kind": "binary", "component": "1_any_drug",
         "label": "Any drug prescribed"},
        {"name": "y_n_drugs", "kind": "count", "component": "2_n_drugs",
         "label": "Number of drugs"},
    ]
    for c in meta["top_classes"]:
        spec.append({"name": f"y_class_{c}", "kind": "binary",
                     "component": "3_drug_class", "label": c})
    spec.append({"name": "y_class_unclassified", "kind": "binary",
                 "component": "3_drug_class", "label": "unclassified (no class label)"})
    for n, lbl in (("y_dose_mode", "Modal dose pattern"),
                   ("y_duration_mode", "Modal duration bucket"),
                   ("y_instruction_mode", "Modal instruction"),
                   ("y_type_mode", "Modal formulation")):
        spec.append({"name": n, "kind": "multiclass", "component": "4_attributes",
                     "label": lbl, "subset": "has_drug"})
    spec.append({"name": "y_chronic_therapy", "kind": "binary",
                 "component": "4_attributes", "label": "Any >90d/continuous order"})
    spec += [
        {"name": "y_any_advice", "kind": "binary", "component": "5_advice",
         "label": "Any advice given"},
        {"name": "y_n_advice", "kind": "count", "component": "5_advice",
         "label": "Number of advice items"},
    ]
    for n, lbl in meta["top_advice"].items():
        spec.append({"name": n, "kind": "binary", "component": "5_advice", "label": lbl})
    spec += [
        {"name": "y_any_test", "kind": "binary", "component": "6_tests",
         "label": "Any test ordered"},
        {"name": "y_n_tests", "kind": "count", "component": "6_tests",
         "label": "Number of tests"},
    ]
    for n, lbl in meta["top_tests"].items():
        spec.append({"name": n, "kind": "binary", "component": "6_tests", "label": lbl})
    return [s for s in spec if s["name"] in df.columns]


# ---------------------------------------------------------------------------
def main() -> None:
    t_start = time.time()
    df = pd.read_parquet(PROCESSED / "features.parquet")
    famjson = json.loads((PROCESSED / "feature_families.json").read_text(encoding="utf-8"))
    fam = famjson["families"]
    feats = famjson["feature_columns"]
    cat_all = list(famjson["categorical_features"])
    meta = famjson["meta"]

    families = sorted(set(fam.values()))
    fam_cols = {f: [c for c in feats if fam[c] == f] for f in families}
    spec = _target_spec(df, meta)
    print(f"{len(feats)} features / {len(families)} families / {len(spec)} targets")

    is_tr = (df["split"] == "train").to_numpy()
    is_te = (df["split"] == "test").to_numpy()
    has_drug = (df["y_n_drugs"] > 0).to_numpy()

    perf_rows, famimp_rows, featimp_rows = [], [], []
    edges_cache: dict[tuple, tuple] = {}

    for si, s in enumerate(spec, 1):
        name, kind = s["name"], s["kind"]
        sub = has_drug if s.get("subset") == "has_drug" else np.ones(len(df), bool)
        y_all = df[name]
        ok = sub & y_all.notna().to_numpy()
        tr, te = is_tr & ok, is_te & ok
        ytr, yte = y_all[tr].to_numpy(), y_all[te].to_numpy()
        if kind == "binary":
            ytr, yte = ytr.astype(int), yte.astype(int)
            if ytr.sum() < 30 or yte.sum() < 10:
                print(f"  [{si}/{len(spec)}] {name}: too few positives, skipped")
                continue
        elif kind == "multiclass" and len(np.unique(ytr)) < 2:
            continue

        t0 = time.time()
        for variant, drop in VARIANTS.items():
            cols = [c for c in feats if fam[c] not in drop]
            cats = [c for c in cat_all if c in cols]
            Xtr, Xte = df.loc[tr, cols], df.loc[te, cols]
            model = _make_model(kind, cats).fit(Xtr, ytr)

            m = _full_metrics(yte, model, Xte, kind)
            m.update({"target": name, "variant": variant, "kind": kind,
                      "component": s["component"], "label": s["label"],
                      "n_train": int(tr.sum()), "n_test": int(te.sum()),
                      "n_features": len(cols), "score": _score(yte, _predict(model, Xte, kind), kind)})
            perf_rows.append(m)

            rng = np.random.default_rng(SEED)
            gi = permutation_importance_groups(
                model, Xte, yte, kind,
                {f: c for f, c in fam_cols.items() if f not in drop},
                N_REPEATS_FAMILY, rng)
            gi["target"], gi["variant"] = name, variant
            gi["kind"], gi["component"] = kind, s["component"]
            famimp_rows.append(gi)

            if variant != "full":
                continue

            # per-feature permutation, full model only
            rng = np.random.default_rng(SEED + 1)
            fi = permutation_importance_groups(
                model, Xte, yte, kind, {c: [c] for c in cols},
                N_REPEATS_FEATURE, rng)
            fi = fi.rename(columns={"group": "feature"}).drop(columns=["n_cols"])
            fi["family"] = fi["feature"].map(fam)

            key = (s.get("subset") or "all",)
            if key not in edges_cache:
                e = _mi_edges(df.loc[is_tr & sub, feats])
                edges_cache[key] = (e, _bin_matrix(df.loc[is_tr & sub, feats], e))
            _, Xb_all = edges_cache[key]
            mask = (tr[is_tr & sub] if False else
                    np.ones(int((is_tr & sub).sum()), bool))
            fi["mutual_info"] = fi["feature"].map(
                mutual_information(Xb_all[mask], feats, y_all[is_tr & sub].to_numpy()))
            fi["target"], fi["component"] = name, s["component"]
            featimp_rows.append(fi)

        base = [r for r in perf_rows if r["target"] == name]
        head = (f"auroc={base[0]['auroc']:.3f}" if kind == "binary" else
                f"acc={base[0]['accuracy']:.3f}" if kind == "multiclass" else
                f"mae={base[0]['mae']:.3f}")
        print(f"  [{si}/{len(spec)}] {name:<38} {head}  ({time.time()-t0:.1f}s)")

    perf = pd.DataFrame(perf_rows)
    famimp = pd.concat(famimp_rows, ignore_index=True).rename(columns={"group": "family"})
    featimp = pd.concat(featimp_rows, ignore_index=True)

    perf.to_csv(OUT / "target_performance.csv", index=False)
    famimp.to_csv(OUT / "family_importance.csv", index=False)
    featimp.to_csv(OUT / "feature_importance.csv", index=False)

    # with / without prescriber (and without site) comparison
    piv = perf.pivot_table(index=["target", "kind", "component", "label"],
                           columns="variant", values="score").reset_index()
    piv["delta_prescriber"] = piv["full"] - piv["no_prescriber"]
    piv["delta_site_and_prescriber"] = piv["full"] - piv["clinical_only"]
    piv.to_csv(OUT / "variant_comparison.csv", index=False)

    _figures(perf, famimp, featimp, piv, families)

    summary = {
        "n_features": len(feats), "n_targets": int(perf["target"].nunique()),
        "family_counts": famjson["family_counts"],
        "runtime_sec": round(time.time() - t_start, 1),
        "mean_family_importance_full": famimp[famimp.variant == "full"]
            .groupby("family")["importance"].mean().sort_values(ascending=False).round(5).to_dict(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print("MEAN FAMILY PERMUTATION IMPORTANCE (full model, averaged over targets)")
    for k, v in summary["mean_family_importance_full"].items():
        print(f"  {k:<12} {v:+.4f}")
    print(f"\nwrote {OUT}")
    print(f"total runtime {summary['runtime_sec']}s")


# ---------------------------------------------------------------------------
def _figures(perf, famimp, featimp, piv, families) -> None:
    plt.rcParams.update({"figure.dpi": 130, "font.size": 8,
                         "axes.grid": True, "grid.alpha": 0.25})

    # --- fig 1: family x target heatmap, share of total importance -----------
    f = famimp[(famimp.variant == "full") & (famimp.target.isin(HEADLINE))]
    M = f.pivot_table(index="family", columns="target", values="importance")
    M = M.reindex(columns=[t for t in HEADLINE if t in M.columns])
    Mn = M.clip(lower=0)
    Mn = Mn / Mn.sum(axis=0).replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    im = ax.imshow(Mn.to_numpy(), aspect="auto", cmap="magma_r", vmin=0, vmax=0.6)
    ax.set_xticks(range(Mn.shape[1]))
    ax.set_xticklabels([c.replace("y_class_", "").replace("y_", "") for c in Mn.columns],
                       rotation=40, ha="right")
    ax.set_yticks(range(Mn.shape[0]))
    ax.set_yticklabels(Mn.index)
    ax.grid(False)
    for i in range(Mn.shape[0]):
        for j in range(Mn.shape[1]):
            v = Mn.iat[i, j]
            if np.isfinite(v) and v >= 0.04:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.35 else "black")
    fig.colorbar(im, ax=ax, label="share of positive family importance")
    ax.set_title("Which feature family drives which prescription component\n"
                 "(grouped permutation importance on the held-out test split, full model)")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_family_importance.png")
    plt.close(fig)

    # --- fig 2: confounder ablation -----------------------------------------
    b = piv[piv.kind == "binary"].copy()
    b = b[b.target.isin(perf.target.unique())].sort_values("full", ascending=False).head(24)
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    yy = np.arange(len(b))
    ax.barh(yy + 0.26, b["full"], height=0.25, label="full", color="#2b6cb0")
    ax.barh(yy, b["no_prescriber"], height=0.25, label="no prescriber", color="#dd6b20")
    ax.barh(yy - 0.26, b["clinical_only"], height=0.25, label="no prescriber + no site",
            color="#718096")
    ax.set_yticks(yy)
    ax.set_yticklabels([t.replace("y_class_", "").replace("y_", "") for t in b["target"]],
                       fontsize=6.5)
    ax.axvline(0.5, color="k", ls=":", lw=0.8)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("test AUROC")
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("How much of prescribing is explained by who prescribed and where\n"
                 "(binary targets, test AUROC by feature set)")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_confounder_ablation.png")
    plt.close(fig)

    # --- fig 3: top individual features for six headline targets ------------
    picks = [t for t in ["y_any_drug", "y_class_ppi_9",
                         "y_class_oral_hypo_glycenic_drug_5", "y_class_arb_acei_1",
                         "y_class_paracetamol_12", "y_any_test"]
             if t in set(featimp.target)]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2))
    palette = {f: c for f, c in zip(
        sorted(families),
        plt.cm.tab10(np.linspace(0, 1, max(len(families), 10)))[:len(families)])}
    for ax, t in zip(axes.ravel(), picks):
        d = featimp[featimp.target == t].nlargest(12, "importance")[::-1]
        ax.barh(range(len(d)), d["importance"],
                color=[palette[f] for f in d["family"]])
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels(d["feature"], fontsize=6)
        ax.set_title(t.replace("y_class_", "").replace("y_", ""), fontsize=8)
        ax.set_xlabel("AUROC drop when permuted", fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[f]) for f in sorted(families)]
    fig.legend(handles, sorted(families), loc="lower center", ncol=len(families),
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Top individual features per prescription component "
                 "(permutation importance, test split, full model)", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(OUT / "fig3_top_features.png")
    plt.close(fig)

    # --- fig 4: performance vs prevalence -----------------------------------
    p = perf[(perf.variant == "full") & (perf.kind == "binary")]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    comps = sorted(p["component"].unique())
    cmap = dict(zip(comps, plt.cm.Dark2(np.linspace(0, 1, 8))[:len(comps)]))
    for c in comps:
        d = p[p.component == c]
        axes[0].scatter(d["prevalence"], d["auroc"], s=22, label=c, color=cmap[c])
        axes[1].scatter(d["prevalence"], d["ap_lift"], s=22, label=c, color=cmap[c])
    axes[0].axhline(0.5, color="k", ls=":", lw=0.8)
    axes[0].set_xscale("log"); axes[0].set_xlabel("test prevalence")
    axes[0].set_ylabel("test AUROC"); axes[0].set_ylim(0.45, 1.0)
    axes[1].axhline(1.0, color="k", ls=":", lw=0.8)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("test prevalence")
    axes[1].set_ylabel("average precision / prevalence")
    axes[0].legend(fontsize=6, loc="lower right")
    fig.suptitle("Per-target discrimination against how rare the component is", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_performance_vs_prevalence.png")
    plt.close(fig)
    print(f"figures -> {OUT}")


if __name__ == "__main__":
    main()
