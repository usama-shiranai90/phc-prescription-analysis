"""Why does performance collapse under a temporal split?

Patient-level micro-F1 is 0.192; train-on-<=2015 / test-on->=2017 is 0.066,
below the global frequency prior. This quantifies what changes between the two
eras so the collapse can be attributed rather than merely reported.

    python -m src.phcrx.era_shift
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from .config import PROCESSED, RESULTS, DataConfig

TRAIN_END, TEST_START = 2015, 2017


def jsd(p: Counter, q: Counter) -> float:
    """Jensen-Shannon divergence between two count distributions (bits)."""
    keys = set(p) | set(q)
    tp, tq = sum(p.values()) or 1, sum(q.values()) or 1
    P = np.array([p.get(k, 0) / tp for k in keys])
    Q = np.array([q.get(k, 0) / tq for k in keys])
    M = 0.5 * (P + Q)
    kl = lambda a, b: np.sum(np.where(a > 0, a * np.log2(a / np.where(b > 0, b, 1e-12)), 0.0))
    return float(0.5 * kl(P, M) + 0.5 * kl(Q, M))


def main() -> None:
    enc = pd.read_parquet(PROCESSED / "rxgen_encounters.parquet")
    orders = pd.read_parquet(PROCESSED / "rxgen_orders.parquet")
    enc["year"] = pd.to_numeric(enc["year"], errors="coerce")

    early_ids = set(enc.loc[enc["year"] <= TRAIN_END, "prescription_id"])
    late_ids = set(enc.loc[enc["year"] >= TEST_START, "prescription_id"])
    e_ord = orders[orders["prescription_id"].isin(early_ids)]
    l_ord = orders[orders["prescription_id"].isin(late_ids)]

    e_drugs = Counter(e_ord["drug_id"].dropna().astype(int))
    l_drugs = Counter(l_ord["drug_id"].dropna().astype(int))
    e_set, l_set = set(e_drugs), set(l_drugs)

    # How many later drug ORDERS are for brands never seen before 2016?
    unseen_orders = sum(c for d, c in l_drugs.items() if d not in e_set)
    total_late = sum(l_drugs.values()) or 1

    e_presc = set(enc.loc[enc["year"] <= TRAIN_END, "prescriber_id"].dropna())
    l_presc_counts = Counter(enc.loc[enc["year"] >= TEST_START, "prescriber_id"].dropna())
    new_presc_enc = sum(c for p, c in l_presc_counts.items() if p not in e_presc)
    total_late_enc = sum(l_presc_counts.values()) or 1

    e_site = set(enc.loc[enc["year"] <= TRAIN_END, "site_id"].dropna())
    l_site_counts = Counter(enc.loc[enc["year"] >= TEST_START, "site_id"].dropna())
    new_site_enc = sum(c for s, c in l_site_counts.items() if s not in e_site)

    # Empty-prescription rate drift
    def empty_rate(ids):
        with_drugs = set(orders.loc[orders["prescription_id"].isin(ids), "prescription_id"])
        return 1 - len(with_drugs) / max(len(ids), 1)

    report = {
        "n_encounters": {"early(<=2015)": len(early_ids), "late(>=2017)": len(late_ids)},
        "drug_vocabulary": {
            "distinct_early": len(e_set), "distinct_late": len(l_set),
            "late_brands_unseen_in_early": len(l_set - e_set),
            "pct_late_orders_for_unseen_brands": 100 * unseen_orders / total_late,
            "jaccard_of_drug_vocabularies": len(e_set & l_set) / max(len(e_set | l_set), 1),
            "jsd_drug_distribution_bits": jsd(e_drugs, l_drugs),
        },
        "prescribers": {
            "distinct_early": len(e_presc), "distinct_late": len(l_presc_counts),
            "new_in_late": len(set(l_presc_counts) - e_presc),
            "pct_late_encounters_by_new_prescriber": 100 * new_presc_enc / total_late_enc,
        },
        "sites": {
            "distinct_early": len(e_site), "distinct_late": len(l_site_counts),
            "new_in_late": len(set(l_site_counts) - e_site),
            "pct_late_encounters_at_new_site": 100 * new_site_enc / total_late_enc,
        },
        "empty_rx_rate": {"early": empty_rate(early_ids), "late": empty_rate(late_ids)},
        "top10_early": [str(d) for d, _ in e_drugs.most_common(10)],
        "top10_late": [str(d) for d, _ in l_drugs.most_common(10)],
    }
    names = dict(zip(orders["drug_id"].astype("Int64"), orders["drug_name"]))
    report["top10_early_names"] = [str(names.get(int(d))) for d, _ in e_drugs.most_common(10)]
    report["top10_late_names"] = [str(names.get(int(d))) for d, _ in l_drugs.most_common(10)]
    top_e = [d for d, _ in e_drugs.most_common(10)]
    top_l = [d for d, _ in l_drugs.most_common(10)]
    report["top10_overlap"] = len(set(top_e) & set(top_l))

    (RESULTS / "era_shift.json").write_text(json.dumps(report, indent=2, default=float))

    d, p, s = report["drug_vocabulary"], report["prescribers"], report["sites"]
    print("=" * 70)
    print(f"Encounters: early(<=2015)={report['n_encounters']['early(<=2015)']}  "
          f"late(>=2017)={report['n_encounters']['late(>=2017)']}")
    print("\nDRUG VOCABULARY SHIFT")
    print(f"  distinct brands early / late      : {d['distinct_early']} / {d['distinct_late']}")
    print(f"  late brands never seen early      : {d['late_brands_unseen_in_early']}")
    print(f"  % late orders for unseen brands   : {d['pct_late_orders_for_unseen_brands']:.1f}%")
    print(f"  vocabulary Jaccard (early vs late): {d['jaccard_of_drug_vocabularies']:.3f}")
    print(f"  JS divergence of drug usage       : {d['jsd_drug_distribution_bits']:.3f} bits")
    print(f"  top-10 drugs shared               : {report['top10_overlap']}/10")
    print(f"    early: {report['top10_early_names']}")
    print(f"    late : {report['top10_late_names']}")
    print("\nPRESCRIBER / SITE TURNOVER")
    print(f"  prescribers early / late          : {p['distinct_early']} / {p['distinct_late']}"
          f"  (new: {p['new_in_late']})")
    print(f"  % late encounters by NEW prescriber: {p['pct_late_encounters_by_new_prescriber']:.1f}%")
    print(f"  sites early / late                : {s['distinct_early']} / {s['distinct_late']}"
          f"  (new: {s['new_in_late']})")
    print(f"  % late encounters at NEW site     : {s['pct_late_encounters_at_new_site']:.1f}%")
    er = report["empty_rx_rate"]
    print(f"\nEMPTY-RX RATE  early={er['early']:.1%}  late={er['late']:.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
