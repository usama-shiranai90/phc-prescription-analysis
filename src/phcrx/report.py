"""Aggregate ablation + baseline results into paper-ready tables and figures.

Run after train.py and baselines.py:
    python -m src.phcrx.report
"""
from __future__ import annotations

import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import RESULTS

# Validated categorical palette (light surface). Assigned in fixed order.
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_YELLOW, C_MAGENTA = "#eda100", "#e87ba4"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8983"
SURFACE, GRID = "#fcfcfb", "#e5e4df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
    "axes.edgecolor": INK_3, "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.frameon": False, "figure.dpi": 150,
})

VARIANT_LABEL = {
    "full": "PHC-RxGen (full)",
    "no_char_cnn": "− char-CNN",
    "no_text_rnn": "− text BiLSTM",
    "no_vitals": "− vitals",
    "no_history": "− history GRU",
    "no_fusion": "− transformer fusion",
    "gru_decoder": "GRU decoder",
    "text_only": "text only",
    "tabular_only": "tabular only (no text)",
    "prior_only": "prior only (demo+geo)",
}

MODALITY_LABEL = {"text": "symptom text", "vitals": "vitals", "demo": "demographics",
                  "history": "prior visits", "all": "all inputs"}


def _style(ax, xlabel=""):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel)


def load():
    abl = json.loads((RESULTS / "rxgen_ablations.json").read_text())
    # Corrected re-runs supersede same-named variants from the first grid.
    fixed_path = RESULTS / "rxgen_fixed_ablations.json"
    if fixed_path.exists():
        fixed = json.loads(fixed_path.read_text())
        superseded = {r["variant"] for r in fixed}
        abl = [r for r in abl if r["variant"] not in superseded] + fixed
    base_path = RESULTS / "baselines.json"
    base = json.loads(base_path.read_text()) if base_path.exists() else []
    diag_path = RESULTS / "diagnostics.json"
    diag = json.loads(diag_path.read_text()) if diag_path.exists() else {}
    tmp_path = RESULTS / "rxgen_temporal.json"
    temporal = json.loads(tmp_path.read_text()) if tmp_path.exists() else []
    era_path = RESULTS / "era_shift.json"
    era = json.loads(era_path.read_text()) if era_path.exists() else {}
    return abl, base, diag, temporal, era


def agg_variants(runs):
    """mean/std across seeds for each variant."""
    by = defaultdict(list)
    for r in runs:
        by[r["variant"]].append(r)
    out = {}
    for v, rs in by.items():
        def ms(path):
            vals = []
            for r in rs:
                d = r["test"]
                for p in path.split("."):
                    d = d[p]
                vals.append(float(d))
            a = np.array(vals)
            return a.mean(), a.std(), len(a)
        out[v] = {
            "micro_f1": ms("micro_f1"), "jaccard": ms("jaccard"),
            "macro_f1": ms("macro_f1"), "exact": ms("exact_match"),
            "cat_f1": ms("category_level.cat_micro_f1"),
            "empty_f1": ms("empty_f1"),
            "n_params": rs[0]["n_params"],
            "runs": rs,
        }
    return out


# --- Figure 1: model vs baselines, brand vs category level ----------------
def fig_baselines(agg, base, path):
    rows = []
    for b in base:
        t = b["test"]
        rows.append((b["model"], t["micro_f1"], t["category_level"]["cat_micro_f1"]))
    if "full" in agg:
        a = agg["full"]
        rows.append(("PHC-RxGen (full)", a["micro_f1"][0], a["cat_f1"][0]))
    rows.sort(key=lambda r: r[1])

    labels = [r[0] for r in rows]
    brand = np.array([r[1] for r in rows])
    cat = np.array([r[2] for r in rows])
    y = np.arange(len(rows))
    h = 0.38
    gap = 0.02   # 2px-equivalent surface gap between adjacent fills

    fig, ax = plt.subplots(figsize=(7.2, 0.52 * len(rows) + 1.5))
    ax.barh(y + (h + gap) / 2, brand, h, color=C_BLUE, zorder=3, label="Brand level (719 labels)")
    ax.barh(y - (h + gap) / 2, cat, h, color=C_ORANGE, zorder=3, label="Category level (89 classes)")
    for yi, v in zip(y + (h + gap) / 2, brand):
        ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=7.5, color=INK_2)
    for yi, v in zip(y - (h + gap) / 2, cat):
        ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=7.5, color=INK_2)

    ax.set_yticks(y, labels)
    ax.set_xlim(0, max(cat.max(), brand.max()) * 1.18)
    _style(ax, "Micro-F1 on held-out patients")
    ax.set_title("Prescription generation vs. non-neural baselines", loc="left", pad=12)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --- Figure 2: ablation contributions -------------------------------------
def fig_ablations(agg, path):
    order = [v for v in VARIANT_LABEL if v in agg]
    if not order:
        return
    means = np.array([agg[v]["micro_f1"][0] for v in order])
    stds = np.array([agg[v]["micro_f1"][1] for v in order])
    idx = np.argsort(means)
    order = [order[i] for i in idx]
    means, stds = means[idx], stds[idx]
    labels = [VARIANT_LABEL[v] for v in order]
    colors = [C_BLUE if v == "full" else C_AQUA for v in order]

    fig, ax = plt.subplots(figsize=(7.0, 0.46 * len(order) + 1.4))
    y = np.arange(len(order))
    ax.barh(y, means, 0.62, color=colors, zorder=3,
            xerr=stds, error_kw=dict(ecolor=INK_3, elinewidth=1.0, capsize=2.5))
    for yi, m, s in zip(y, means, stds):
        ax.text(m + s + 0.004, yi, f"{m:.3f}", va="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, (means + stds).max() * 1.18)
    _style(ax, "Micro-F1 (mean ± s.d. over 3 seeds)")
    ax.set_title("Ablation: contribution of each encoder / decoder component",
                 loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --- Figure 3: long-tail behaviour ----------------------------------------
def fig_strata(agg, path):
    if "full" not in agg:
        return
    bands = ["head", "mid", "tail"]
    per = {b: [] for b in bands}
    for r in agg["full"]["runs"]:
        st = r["test"].get("by_stratum", {})
        for b in bands:
            per[b].append(st.get(b, {}).get("recall", 0.0))
    means = [float(np.mean(per[b])) for b in bands]
    stds = [float(np.std(per[b])) for b in bands]
    support = [agg["full"]["runs"][0]["test"].get("by_stratum", {}).get(b, {}).get("support", 0)
               for b in bands]

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    x = np.arange(len(bands))
    ax.bar(x, means, 0.55, color=[C_BLUE, C_AQUA, C_MAGENTA], zorder=3,
           yerr=stds, error_kw=dict(ecolor=INK_3, elinewidth=1.0, capsize=3))
    for xi, m, s, n in zip(x, means, stds, support):
        ax.text(xi, m + s + 0.012, f"{m:.3f}", ha="center", fontsize=8, color=INK_2)
        ax.text(xi, -0.045, f"n={n}", ha="center", fontsize=7, color=INK_3)
    ax.set_xticks(x, ["Head\n(≥100 train)", "Mid\n(10–99)", "Tail\n(<10)"])
    ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel("Recall")
    ax.set_title("Recall by drug-frequency band", loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --- Figure 4: input-permutation test (the decisive conditioning check) ---
def fig_permutation(diag, path):
    perm = diag.get("permutation")
    if not perm:
        return
    mods = [m for m in ("text", "vitals", "demo", "history") if m in perm]
    deltas = [perm[m]["delta_micro_f1"] for m in mods]
    order = np.argsort(deltas)
    mods = [mods[i] for i in order]
    deltas = [deltas[i] for i in order]
    intact = diag["baseline"]["micro_f1"]
    # Red where shuffling clearly hurts (input is used), grey where it does not.
    colors = ["#e34948" if d < -0.01 else INK_3 for d in deltas]

    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(mods) + 1.6))
    y = np.arange(len(mods))
    ax.barh(y, deltas, 0.6, color=colors, zorder=3)
    for yi, d in zip(y, deltas):
        ax.text(d - 0.004, yi, f"{d:+.3f}", va="center", ha="right",
                fontsize=8, color=INK_2)
    ax.axvline(0, color=INK_3, linewidth=0.9)
    ax.set_yticks(y, [MODALITY_LABEL[m] for m in mods])
    ax.set_xlim(min(deltas) * 1.35, 0.012)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Δ micro-F1 when this input is shuffled across patients")
    ax.set_title(f"Which inputs the model actually uses\n"
                 f"(intact micro-F1 = {intact:.3f}; more negative = more relied upon)",
                 loc="left", pad=12, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def markdown_report(agg, base, diag, temporal, era) -> str:
    L = ["# PHC-RxGen — Prescription Generation Results", ""]
    L += ["Held-out **patient-level** split (no patient appears in both train and test).",
          "Brand level = 719 marketed products; category level = 89 pharmacological",
          "classes (16 PPI brands, 14 paracetamol brands collapse to one class each).", ""]

    if diag.get("constant_modal_train_set"):
        c = diag["constant_modal_train_set"]
        L += ["> **Read Jaccard against its floor.** The modal training "
              f"prescription is the *empty* set, and predicting it for every "
              f"encounter already scores Jaccard **{c['jaccard']:.3f}** "
              f"(micro-F1 {c['micro_f1']:.3f}). Jaccard is inflated by "
              "empty-set agreement on the 21.5% of encounters with no "
              "pharmacotherapy; micro-F1 is the honest headline.", ""]

    L += ["## Baselines", "",
          "| Baseline | Micro-F1 | Jaccard | Macro-F1 | Exact | Category F1 |",
          "|---|---|---|---|---|---|"]
    for b in base:
        t = b["test"]
        L.append(f"| {b['model']} | {t['micro_f1']:.4f} | {t['jaccard']:.4f} | "
                 f"{t['macro_f1']:.4f} | {t['exact_match']:.4f} | "
                 f"{t['category_level']['cat_micro_f1']:.4f} |")
    if diag.get("constant_modal_train_set"):
        c = diag["constant_modal_train_set"]
        L.append(f"| _always-empty (floor)_ | {c['micro_f1']:.4f} | {c['jaccard']:.4f} "
                 f"| 0.0000 | {c['exact_match']:.4f} | {c['cat_micro_f1']:.4f} |")

    L += ["", "## Ablations (mean ± s.d. over 3 seeds)", "",
          "| Variant | Params | Micro-F1 | Jaccard | Macro-F1 | Exact | Category F1 | Empty-Rx F1 |",
          "|---|---|---|---|---|---|---|---|"]
    for v in sorted(agg, key=lambda k: -agg[k]["micro_f1"][0]):
        a = agg[v]
        f = lambda k: f"{a[k][0]:.4f} ± {a[k][1]:.4f}"
        L.append(f"| {VARIANT_LABEL.get(v, v)} | {a['n_params']/1e6:.2f}M | {f('micro_f1')} | "
                 f"{f('jaccard')} | {f('macro_f1')} | {f('exact')} | {f('cat_f1')} | "
                 f"{f('empty_f1')} |")

    if diag.get("permutation"):
        b = diag["baseline"]["micro_f1"]
        L += ["", "## Input-permutation test", "",
              "Each modality is shuffled across patients at inference on the",
              "*trained full model*. A metric that does not move means that input",
              "was not being used. This separates real conditioning from a",
              "well-fit prescribing prior.", "",
              f"Intact micro-F1: **{b:.4f}**", "",
              "| Shuffled input | Micro-F1 | Δ | Uses it? |", "|---|---|---|---|"]
        for m, s in diag["permutation"].items():
            used = "**yes**" if s["delta_micro_f1"] < -0.01 else "no"
            L.append(f"| {MODALITY_LABEL.get(m, m)} | {s['micro_f1']:.4f} | "
                     f"{s['delta_micro_f1']:+.4f} | {used} |")

    if diag.get("diversity"):
        d = diag["diversity"]
        L += ["", "## Output diversity", "",
              f"- Distinct prescriptions generated: **{d['distinct_pred_sets']}** "
              f"over {d['n_test']} test encounters (gold: {d['distinct_gold_sets']})",
              f"- Modal generated prescription covers {d['modal_pred_set_share']:.1%} "
              f"of encounters (gold modal: {d['modal_gold_set_share']:.1%})",
              f"- Modal generated prescription: "
              f"`{d['modal_pred_set'] or '(empty — no pharmacotherapy)'}`"]

    if temporal:
        tagg = agg_variants(temporal)
        L += ["", "## Temporal shift — train ≤2015, test ≥2017", "",
              "**This is the headline result for deployability.** The patient-level",
              "split overstates forward-in-time performance by roughly 3×.", "",
              "| Variant | Brand micro-F1 | vs patient split | Category micro-F1 | vs patient split |",
              "|---|---|---|---|---|"]
        for v in sorted(tagg, key=lambda k: -tagg[k]["micro_f1"][0]):
            t, p = tagg[v], agg.get(v)
            bt, ct = t["micro_f1"][0], t["cat_f1"][0]
            bd = f"{100*(bt-p['micro_f1'][0])/p['micro_f1'][0]:+.0f}%" if p else "—"
            cd = f"{100*(ct-p['cat_f1'][0])/p['cat_f1'][0]:+.0f}%" if p else "—"
            L.append(f"| {VARIANT_LABEL.get(v, v)} | {bt:.4f} ± {t['micro_f1'][1]:.4f} | "
                     f"{bd} | {ct:.4f} ± {t['cat_f1'][1]:.4f} | {cd} |")
        L += ["", "At brand level the model falls to roughly the global-frequency-prior "
              "baseline (0.069): **no useful transfer**. Category-level scores degrade "
              "less (−47% vs −66%), so the pharmacological abstraction is more stable "
              "than brand identity — but it does not rescue transfer either."]

    if era:
        d = era["drug_vocabulary"]; p = era["prescribers"]; s = era["sites"]
        L += ["", "### Why it collapses", "",
              f"- Only **{era['top10_overlap']}/10** of the most-prescribed brands are "
              f"shared between eras.",
              f"  - ≤2015: {', '.join(era['top10_early_names'][:6])} …",
              f"  - ≥2017: {', '.join(era['top10_late_names'][:6])} …",
              f"- Drug-vocabulary Jaccard between eras: **{d['jaccard_of_drug_vocabularies']:.3f}**; "
              f"**{d['pct_late_orders_for_unseen_brands']:.1f}%** of later orders are for brands "
              f"never seen in training (structurally unpredictable).",
              f"- **{p['pct_late_encounters_by_new_prescriber']:.0f}%** of later encounters were "
              f"written by a prescriber absent from training ({p['new_in_late']} new prescribers).",
              f"- **{s['pct_late_encounters_at_new_site']:.0f}%** occurred at a site absent from "
              f"training ({s['new_in_late']} new sites).",
              f"- The task itself did **not** drift: empty-prescription rate "
              f"{era['empty_rx_rate']['early']:.1%} → {era['empty_rx_rate']['late']:.1%}.",
              "",
              "The corpus is non-stationary in *formulary and staffing*, not in clinical "
              "presentation. That is what breaks brand-level transfer."]

    if "full" in agg:
        r0 = agg["full"]["runs"][0]["test"]
        L += ["", "## Full model — detail (seed 0)", ""]
        st = r0.get("by_stratum", {})
        if st:
            L += ["| Drug band | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
            for b in ("head", "mid", "tail", "unseen"):
                if b in st:
                    s = st[b]
                    L.append(f"| {b} | {s['precision']:.4f} | {s['recall']:.4f} | "
                             f"{s['f1']:.4f} | {s['support']} |")
        attrs = {k: v for k, v in r0.items() if k.startswith("attr_") and k.endswith("_acc")}
        if attrs:
            L += ["", "**Structured attribute accuracy** (on correctly predicted drugs): "
                  + ", ".join(f"{k.replace('attr_','').replace('_acc','')} "
                              f"{v:.3f}" for k, v in attrs.items())]
        rank = {k: v for k, v in r0.items() if "@" in k}
        if rank:
            L += ["", "**Ranking**: " + ", ".join(f"{k} {v:.3f}" for k, v in sorted(rank.items()))]
        for aux in ("advice", "test"):
            if aux in r0:
                a = r0[aux]
                L += ["", f"**Auxiliary {aux} head**: micro-F1 {a['micro_f1']:.3f} "
                      f"(P {a['micro_precision']:.3f} / R {a['micro_recall']:.3f}), "
                      f"ECE {a['ece']:.3f}"]
    return "\n".join(L) + "\n"


def fig_temporal(agg, temporal, path):
    """Patient-split vs temporal-split, at both label granularities."""
    if not temporal:
        return
    tagg = agg_variants(temporal)
    vs = [v for v in ("full", "no_fusion") if v in tagg and v in agg]
    if not vs:
        return
    labels, groups = [], []
    for v in vs:
        for lvl, key in (("brand", "micro_f1"), ("category", "cat_f1")):
            labels.append(f"{VARIANT_LABEL.get(v, v)}\n{lvl} level")
            groups.append((agg[v][key][0], tagg[v][key][0]))
    pat = np.array([g[0] for g in groups])
    tmp = np.array([g[1] for g in groups])

    x = np.arange(len(labels))
    w, gap = 0.36, 0.02
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.bar(x - (w + gap) / 2, pat, w, color=C_BLUE, zorder=3, label="Patient-level split")
    ax.bar(x + (w + gap) / 2, tmp, w, color=C_ORANGE, zorder=3, label="Temporal split (≥2017)")
    for xi, a, b in zip(x, pat, tmp):
        ax.text(xi - (w + gap) / 2, a + 0.008, f"{a:.3f}", ha="center", fontsize=7.5, color=INK_2)
        ax.text(xi + (w + gap) / 2, b + 0.008, f"{b:.3f}", ha="center", fontsize=7.5, color=INK_2)
        ax.text(xi, max(a, b) + 0.045, f"{100*(b-a)/a:+.0f}%", ha="center",
                fontsize=8, color="#e34948")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, max(pat.max(), tmp.max()) * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel("Micro-F1")
    ax.set_title("Performance collapses when tested forward in time", loc="left", pad=26)
    # Legend sits above the axes so it cannot collide with the delta labels.
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    abl, base, diag, temporal, era = load()
    agg = agg_variants(abl)
    fig_baselines(agg, base, RESULTS / "fig1_baselines.png")
    fig_ablations(agg, RESULTS / "fig2_ablations.png")
    fig_strata(agg, RESULTS / "fig3_strata.png")
    fig_permutation(diag, RESULTS / "fig4_permutation.png")
    fig_temporal(agg, temporal, RESULTS / "fig5_temporal.png")
    (RESULTS / "RESULTS.md").write_text(
        markdown_report(agg, base, diag, temporal, era), encoding="utf-8")
    print("wrote figures + RESULTS.md to", RESULTS)
    for v in sorted(agg, key=lambda k: -agg[k]["micro_f1"][0]):
        a = agg[v]
        print(f"  {VARIANT_LABEL.get(v,v):24s} microF1={a['micro_f1'][0]:.4f}±{a['micro_f1'][1]:.4f} "
              f"catF1={a['cat_f1'][0]:.4f}")


if __name__ == "__main__":
    main()
