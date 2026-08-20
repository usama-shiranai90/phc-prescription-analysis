"""Merge the MCH prescription corpora into the adult corpus and report totals.

Reads the frozen CSVs produced by `src/sql/extract.sql` (adult) and
`src/sql/extract_mch.sql` (antenatal / postnatal / motherhood), harmonises the
id namespaces, and runs the merged frames through the *existing* cleaning
functions in `src/phcrx/preprocess.py`. Nothing is duplicated: `build_encounters`,
`build_orders` and `bucket_duration` are imported, which pulls in
`harmonise_vitals`, `norm_text`, `tokenize`, `canon_dose` and
`parse_duration_qty` transitively, so the adult and MCH rows are cleaned by
exactly the same code path -- including the mojibake repair, the Bengali-digit
translation, the mmol/L and Celsius unit harmonisation, and the VITAL_RANGES
outlier clipping.

Run:
    python -m src.phcrx.expand.merge_corpora
    python -m src.phcrx.expand.merge_corpora --write     # also emit *_all.csv
    python -m src.phcrx.expand.merge_corpora --parquet   # rxgen_*_expanded.parquet
    python -m src.phcrx.expand.merge_corpora --write --parquet --json
    python -m src.phcrx.expand.merge_corpora --parquet --freeze-adult-split

`--write` produces data/interim/{encounters,rx_orders,rx_advice,rx_tests,rx_cc,
patient_history}_all.csv with the same column shape as the originals plus a
`corpus` column, so a downstream variant of preprocess.py can consume them
without either file being edited.

`--parquet` produces data/processed/rxgen_{encounters,orders}_expanded.parquet
with exactly the same column list as the shipped rxgen_{encounters,orders}
.parquet plus a trailing `corpus` column. The existing files are never touched.

WHY THE IDS ARE OFFSET
----------------------
prescription_id and checkup_id restart at 1 in every MCH corpus and collide
with the adult corpus (adult prescription_id 4..14179, antenatal 1..1400;
adult checkup_id 12..47999, postnatal 1..962). Each corpus therefore gets a
disjoint 10M-wide id block. user_id and site_id are deliberately NOT offset:
they are shared keys (a3m_account_details, eh_project_site) and 66 of the 603
MCH patients also appear in eh_patient_checkup (22 of them hold an adult
prescription too), so leaving them intact is what keeps a patient-level split
from putting the same person on both sides of the train/test boundary.

That offset is the single load-bearing correctness assumption in this module,
so `verify_ids()` runs unconditionally on every invocation and raises rather
than warns. It proves (a) the collision is real in the source data, (b) the
merged id -> source row map is a bijection, (c) no drug order, advice row,
test row, cc row or history row is re-parented onto a different corpus, and
(d) the adult slice of the merged frame is element-wise identical to an
adult-only build. See `verify_ids` for the full list.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd

from ..config import INTERIM, PROCESSED, RESULTS, VITAL_COLS, DataConfig
from ..preprocess import bucket_duration, build_encounters, build_orders

# Disjoint id blocks, one per corpus. Adult keeps its native ids.
CORPUS_OFFSET: dict[str, int] = {
    "adult": 0,
    "antenatal": 10_000_000,
    "postnatal": 20_000_000,
    "motherhood": 30_000_000,
}
BLOCK_WIDTH = 10_000_000
MCH_CORPORA = ("antenatal", "postnatal", "motherhood")

# Which of the 13 pipeline vitals each source table physically carries. Columns
# outside this set are emitted as NULL by extract_mch.sql and are structurally
# (not incidentally) missing -- worth separating in the report.
VITALS_ABSENT_BY_CORPUS: dict[str, tuple[str, ...]] = {
    "adult": (),
    "antenatal": ("waist_hip_ratio", "cholesterol", "uric_acid"),
    "postnatal": ("height", "bmi", "waist_hip_ratio", "cholesterol", "uric_acid"),
    "motherhood": ("height", "bmi", "waist_hip_ratio", "cholesterol", "uric_acid"),
}

# (adult csv, mch csv) per logical table.
PAIRS: dict[str, tuple[str, str]] = {
    "encounters": ("encounters.csv", "encounters_mch.csv"),
    "orders": ("rx_orders.csv", "rx_orders_mch.csv"),
    "advice": ("rx_advice.csv", "rx_advice_mch.csv"),
    "tests": ("rx_tests.csv", "rx_tests_mch.csv"),
    "cc": ("rx_cc.csv", "rx_cc_mch.csv"),
    "hist": ("patient_history.csv", "patient_history_mch.csv"),
}

_ID_COLS = {
    "encounters": ("prescription_id", "checkup_id"),
    "orders": ("prescription_id",),
    "advice": ("prescription_id",),
    "tests": ("prescription_id",),
    "cc": ("prescription_id",),
    "hist": ("checkup_id", "prescription_id"),
}

_read = lambda name: pd.read_csv(
    INTERIM / name, dtype=str, keep_default_na=False, na_values=[""]
)


def _apply_offset(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    """Shift the listed id columns into this row's corpus block, in place."""
    off = df["corpus"].map(CORPUS_OFFSET)
    if off.isna().any():
        bad = sorted(set(df.loc[off.isna(), "corpus"]))
        raise ValueError(f"unknown corpus label(s): {bad}")
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce") + off
        # Back to the string dtype the raw frames use, so the downstream
        # pd.to_numeric in preprocess.build_* behaves identically.
        df[c] = v.map(lambda x: "" if pd.isna(x) else str(int(x))).replace("", np.nan)
    return df


def load_all() -> dict[str, pd.DataFrame]:
    """Load adult + MCH CSVs, tag corpus, namespace the ids, concatenate."""
    out: dict[str, pd.DataFrame] = {}
    for key, (base_name, mch_name) in PAIRS.items():
        base = _read(base_name)
        base["corpus"] = "adult"
        mch = _read(mch_name)
        missing = set(base.columns) - set(mch.columns)
        extra = set(mch.columns) - set(base.columns)
        if missing or extra:
            raise ValueError(
                f"{key}: column shape mismatch. missing from MCH={sorted(missing)} "
                f"extra in MCH={sorted(extra)}"
            )
        mch = mch[base.columns]
        both = pd.concat([base, mch], ignore_index=True)
        out[key] = _apply_offset(both, _ID_COLS[key])
    return out


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 1) if d else 0.0


def _fmt_table(rows: list[dict], cols: list[str]) -> str:
    """Small fixed-width table printer (avoids a pandas display dependency)."""
    widths = {c: max([len(c)] + [len(str(r.get(c, ""))) for r in rows]) for c in cols}
    line = "  ".join(c.rjust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    body = "\n".join(
        "  ".join(str(r.get(c, "")).rjust(widths[c]) for c in cols) for r in rows
    )
    return f"{line}\n{sep}\n{body}"


# ---------------------------------------------------------------------------
# ID-namespace verification. This is the correctness gate for the whole merge.
# ---------------------------------------------------------------------------

class IdCheckFailure(AssertionError):
    """Raised when the id namespacing is not provably sound."""


def _src_ids(key: str, col: str) -> dict[str, pd.Series]:
    """Pre-offset values of `col`, per corpus, straight from the source CSVs.

    Deliberately re-read rather than reconstructed from the merged frame: the
    whole point is to catch a bug in `_apply_offset`, so the reference side of
    the comparison must not have gone through it.
    """
    base_name, mch_name = PAIRS[key]
    out = {"adult": pd.to_numeric(_read(base_name)[col], errors="coerce")}
    mch = _read(mch_name)
    for c, g in mch.groupby("corpus"):
        out[str(c)] = pd.to_numeric(g[col], errors="coerce")
    return out


def verify_ids(raw: dict[str, pd.DataFrame], enc: pd.DataFrame,
               orders: pd.DataFrame) -> dict:
    """Prove the id namespacing is sound. Raises IdCheckFailure otherwise.

    Ten checks, in order of what they rule out:
      1  the collision is real in the source data (so the offset is not
         cargo-culting)
      2  every merged prescription_id is unique -> no merged id maps to two
         different source rows
      3  row count == number of distinct (corpus, source prescription_id)
         pairs -> the map is injective in the other direction too
      4  the recovered source ids reproduce each corpus's source id set exactly
      5  the per-corpus merged id ranges are pairwise disjoint and each corpus's
         raw ids fit inside its 10M block (no wrap into the next block)
      6  same for checkup_id
      7  no drug order / advice / test / cc / history row is re-parented onto a
         different corpus by the merge
      8  the number of orders that resolve to a parent encounter is identical to
         a raw, per-corpus, offset-free join
      9  the adult slice of the merged frames is element-wise identical to an
         adult-only build through the same cleaning functions
     10  the adult prescription_id set is identical to the shipped
         rxgen_encounters.parquet, so the existing benchmark corpus is untouched
    """
    res: dict[str, object] = {}
    fails: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        res[name] = {"pass": bool(ok), "detail": detail}
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f"\n         {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    print("=" * 100)
    print("ID-NAMESPACE VERIFICATION")
    print("=" * 100)

    # --- 1. the collision the offset exists to prevent --------------------
    src_pid = _src_ids("encounters", "prescription_id")
    sets_p = {c: set(v.dropna().astype(int)) for c, v in src_pid.items()}
    occ = Counter(x for s in sets_p.values() for x in s)
    n_multi = sum(1 for _, n in occ.items() if n > 1)
    fused_rows = sum(len(sets_p[c] & sets_p["adult"]) for c in MCH_CORPORA)
    ranges = {c: (min(s), max(s)) for c, s in sets_p.items()}
    check("1_source_collision_is_real", n_multi > 0,
          f"{n_multi} raw prescription_ids occur in >=2 corpora; a naive concat "
          f"would fuse {fused_rows} MCH encounters onto adult ones. raw ranges: "
          + ", ".join(f"{c}={lo}..{hi}" for c, (lo, hi) in ranges.items()))
    res["source_pid_multi_corpus"] = n_multi
    res["naive_concat_fused_rows"] = fused_rows

    # --- 2/3. merged prescription_id is a bijection onto source rows ------
    pid = enc["prescription_id"]
    check("2_merged_prescription_id_unique", bool(pid.is_unique),
          f"{len(pid)} rows, {pid.nunique()} distinct ids, "
          f"{len(pid) - pid.nunique()} duplicated")

    src_recovered = pid - enc["corpus"].map(CORPUS_OFFSET).astype("Int64")
    pairs = set(zip(enc["corpus"], src_recovered))
    n_src_rows = sum(len(v) for v in src_pid.values())
    check("3_bijection_rows_to_pairs",
          len(pairs) == len(enc) == n_src_rows,
          f"rows={len(enc)}  distinct(corpus,src_id) pairs={len(pairs)}  "
          f"source rows={n_src_rows}")

    # --- 4. the inverse map is total and exact ----------------------------
    bad_corpora = []
    for c, s in sets_p.items():
        got = set(src_recovered[enc["corpus"] == c].dropna().astype(int))
        if got != s:
            bad_corpora.append(f"{c}: +{len(got - s)} -{len(s - got)}")
    check("4_source_id_sets_roundtrip", not bad_corpora,
          "; ".join(bad_corpora) if bad_corpora else
          "every corpus's recovered source ids == its CSV's ids")

    # --- 5. blocks are disjoint and wide enough ---------------------------
    merged_rng = {c: (int(pid[enc["corpus"] == c].min()),
                      int(pid[enc["corpus"] == c].max())) for c in sets_p}
    overlaps = []
    items = list(merged_rng.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (ca, (la, ha)), (cb, (lb, hb)) = items[i], items[j]
            if la <= hb and lb <= ha:
                overlaps.append(f"{ca}x{cb}")
    too_wide = [c for c, s in sets_p.items() if max(s) >= BLOCK_WIDTH]
    check("5_id_blocks_disjoint", not overlaps and not too_wide,
          ("overlapping: " + ", ".join(overlaps)) if overlaps else
          ("raw ids exceed the block width: " + ", ".join(too_wide)) if too_wide else
          "merged ranges "
          + ", ".join(f"{c}={lo}..{hi}" for c, (lo, hi) in merged_rng.items()))
    res["merged_pid_ranges"] = {c: list(v) for c, v in merged_rng.items()}

    # --- 6. same story for checkup_id -------------------------------------
    src_cid = _src_ids("encounters", "checkup_id")
    sets_c = {c: set(v.dropna().astype(int)) for c, v in src_cid.items()}
    occ_c = Counter(x for s in sets_c.values() for x in s)
    n_multi_c = sum(1 for _, n in occ_c.items() if n > 1)
    cid = enc["checkup_id"]
    src_cid_rec = cid - enc["corpus"].map(CORPUS_OFFSET).astype("Int64")
    cid_bad = [c for c, s in sets_c.items()
               if set(src_cid_rec[enc["corpus"] == c].dropna().astype(int)) != s]
    # checkup_id need not be unique (one checkup can carry >1 prescription);
    # what must hold is that it never crosses a corpus boundary.
    cid_cross = int(enc.groupby("checkup_id")["corpus"].nunique().gt(1).sum())
    check("6_checkup_id_namespaced", not cid_bad and cid_cross == 0,
          f"{n_multi_c} raw checkup_ids occur in >=2 corpora pre-offset; post-offset "
          f"{cid_cross} merged checkup_ids span >1 corpus; roundtrip failures: "
          f"{cid_bad or 'none'}")
    res["source_cid_multi_corpus"] = n_multi_c

    # --- 7. no child row is re-parented across corpora --------------------
    parent = enc.set_index("prescription_id")["corpus"]
    reparented = {}
    children = {"orders": orders, "advice": raw["advice"], "tests": raw["tests"],
                "cc": raw["cc"], "hist": raw["hist"]}
    for name, df in children.items():
        d = df[["prescription_id", "corpus"]].copy()
        d["prescription_id"] = pd.to_numeric(
            d["prescription_id"], errors="coerce").astype("Int64")
        d = d[d["prescription_id"].notna()]
        got = d["prescription_id"].map(parent)
        reparented[name] = int((got.notna() & (got != d["corpus"])).sum())
    check("7_no_cross_corpus_reparenting", all(v == 0 for v in reparented.values()),
          ", ".join(f"{k}={v}" for k, v in reparented.items()) + " mismatched rows")
    res["reparented_rows"] = reparented

    # --- 8. resolution counts match an offset-free per-corpus join --------
    src_ord = _src_ids("orders", "prescription_id")
    raw_resolved = sum(int(v.dropna().astype(int).isin(sets_p[c]).sum())
                       for c, v in src_ord.items())
    # build_orders drops null-drug rows, so compare on the raw order frame too.
    raw_orders_all = raw["orders"].copy()
    raw_orders_all["prescription_id"] = pd.to_numeric(
        raw_orders_all["prescription_id"], errors="coerce").astype("Int64")
    merged_resolved_all = int(
        raw_orders_all["prescription_id"].isin(set(enc["prescription_id"])).sum())
    merged_resolved = int(
        orders["prescription_id"].isin(set(enc["prescription_id"])).sum())
    check("8_order_resolution_count_unchanged", raw_resolved == merged_resolved_all,
          f"raw per-corpus join={raw_resolved}, merged join={merged_resolved_all} "
          f"({merged_resolved} after build_orders drops null-drug rows)")

    # --- 9. adult slice == adult-only build -------------------------------
    ref_enc = build_encounters(_read("encounters.csv")).reset_index(drop=True)
    ref_ord = build_orders(_read("rx_orders.csv"))
    ref_ord["duration_bucket"] = bucket_duration(ref_ord["duration_days"])
    ref_ord = ref_ord.reset_index(drop=True)
    got_enc = enc[enc["corpus"] == "adult"].drop(columns=["corpus"]).reset_index(drop=True)
    got_ord = orders[orders["corpus"] == "adult"].drop(columns=["corpus"]).reset_index(drop=True)
    diffs = []
    for label, g, r in (("encounters", got_enc, ref_enc), ("orders", got_ord, ref_ord)):
        if len(g) != len(r):
            diffs.append(f"{label}: {len(g)} vs {len(r)} rows")
            continue
        if list(g.columns) != list(r.columns):
            diffs.append(f"{label}: column order differs")
            continue
        for col in r.columns:
            a, b = g[col], r[col]
            if a.map(type).eq(list).any() or b.map(type).eq(list).any():
                a, b = a.map(tuple), b.map(tuple)     # symptom_tokens
            if not a.equals(b):
                n_bad = int((a.astype(object).where(a.notna(), "\x00NA")
                             != b.astype(object).where(b.notna(), "\x00NA")).sum())
                diffs.append(f"{label}.{col}: {n_bad} rows differ")
    check("9_adult_slice_identical_to_adult_only_build", not diffs,
          "; ".join(diffs) if diffs else
          f"encounters {len(got_enc)}=={len(ref_enc)}, orders {len(got_ord)}=={len(ref_ord)}, "
          f"all {len(ref_enc.columns)}+{len(ref_ord.columns)} columns element-wise equal")
    res["adult_encounters_before"] = len(ref_enc)
    res["adult_encounters_after"] = len(got_enc)
    res["adult_orders_before"] = len(ref_ord)
    res["adult_orders_after"] = len(got_ord)
    res["adult_patients_before"] = int(ref_enc["user_id"].nunique())
    res["adult_patients_after"] = int(got_enc["user_id"].nunique())

    # --- 10. the shipped benchmark corpus is untouched --------------------
    shipped = PROCESSED / "rxgen_encounters.parquet"
    if shipped.exists():
        sh = pd.read_parquet(shipped, columns=["prescription_id"])
        same = (set(sh["prescription_id"].astype(int))
                == set(got_enc["prescription_id"].dropna().astype(int)))
        check("10_shipped_parquet_ids_unchanged", same and len(sh) == len(got_enc),
              f"shipped rxgen_encounters.parquet has {len(sh)} rows; the adult slice "
              f"has {len(got_enc)}; id sets {'identical' if same else 'DIFFER'}")
    else:
        check("10_shipped_parquet_ids_unchanged", True,
              "rxgen_encounters.parquet absent -- skipped")

    if fails:
        raise IdCheckFailure(f"id verification failed: {fails}")
    print("\n  all 10 checks passed.")
    return res


# ---------------------------------------------------------------------------
# Splits. preprocess.main() computes them inline, so the logic cannot be
# imported; it is mirrored here verbatim and the mirror is validated against
# the shipped parquet by `check_split_mirror`.
# ---------------------------------------------------------------------------

def patient_split(enc: pd.DataFrame, cfg: DataConfig) -> pd.Series:
    """Patient-level split, identical in construction to preprocess.main()."""
    rng = np.random.default_rng(cfg.seed)
    patients = np.asarray(enc["user_id"].dropna().unique(), dtype=np.int64)
    rng.shuffle(patients)
    n = len(patients)
    n_test = int(n * cfg.test_frac)
    n_val = int(n * cfg.val_frac)
    assign = {}
    for i, p in enumerate(patients):
        assign[p] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    return enc["user_id"].map(assign)


def merged_split(enc: pd.DataFrame, cfg: DataConfig, freeze_adult: bool) -> pd.Series:
    """Split the merged frame, optionally pinning adult patients in place.

    The plain patient split re-shuffles the whole patient pool, so adding 603
    MCH patients moves ~28% of adult encounters to a different split: only half
    the shipped adult test set survives, and the merged run is then not
    comparable with the adult-only numbers already published. With
    `freeze_adult=True` every patient that already has an assignment in
    rxgen_encounters.parquet keeps it -- including the 22 who hold prescriptions
    in both cohorts, which is what stops them leaking -- and only the MCH-only
    patients are drawn, at the same fractions.
    """
    if not freeze_adult:
        return patient_split(enc, cfg)
    p = PROCESSED / "rxgen_encounters.parquet"
    if not p.exists():
        raise FileNotFoundError(f"--freeze-adult-split needs {p}")
    sh = pd.read_parquet(p, columns=["user_id", "split"])
    assign = {int(u): s for u, s in zip(sh["user_id"], sh["split"])}
    rest = np.asarray(
        sorted(set(enc["user_id"].dropna().astype("int64")) - set(assign)),
        dtype=np.int64)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(rest)
    n = len(rest)
    n_test, n_val = int(n * cfg.test_frac), int(n * cfg.val_frac)
    for i, u in enumerate(rest):
        assign[int(u)] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    return enc["user_id"].map(assign)


def adult_split_drift(enc: pd.DataFrame) -> str:
    """How far the merged-frame split moves the adult rows off the shipped one."""
    p = PROCESSED / "rxgen_encounters.parquet"
    if not p.exists():
        return "skipped (no shipped parquet)"
    sh = pd.read_parquet(p, columns=["prescription_id", "split"])
    m = sh.merge(enc.loc[enc["corpus"] == "adult", ["prescription_id", "split"]],
                 on="prescription_id", suffixes=("_shipped", "_now"))
    moved = int((m["split_shipped"] != m["split_now"]).sum())
    kept = len(set(m.loc[m["split_shipped"] == "test", "prescription_id"])
               & set(m.loc[m["split_now"] == "test", "prescription_id"]))
    n_test = int((m["split_shipped"] == "test").sum())
    return (f"{moved}/{len(m)} adult encounters change split ({_pct(moved, len(m))}%); "
            f"{kept}/{n_test} of the shipped adult TEST rows are still in test")


def check_split_mirror(adult_enc: pd.DataFrame, cfg: DataConfig) -> str:
    """Reproduce the shipped adult split to prove the mirrored logic is exact."""
    p = PROCESSED / "rxgen_encounters.parquet"
    if not p.exists():
        return "skipped (no shipped parquet)"
    sh = pd.read_parquet(p, columns=["prescription_id", "split"])
    a = adult_enc.reset_index(drop=True)
    repro = pd.DataFrame({
        "prescription_id": a["prescription_id"].astype("Int64"),
        "split_repro": patient_split(a, cfg).values,
    })
    m = sh.merge(repro, on="prescription_id", how="inner")
    n_bad = int((m["split"] != m["split_repro"]).sum())
    ok = (len(m) == len(sh)) and n_bad == 0
    return (f"EXACT ({len(m)}/{len(sh)} rows matched)" if ok
            else f"MISMATCH ({n_bad} of {len(m)} rows differ)")


# ---------------------------------------------------------------------------
# Expanded model-ready Parquet, same schema as the shipped files + `corpus`.
# ---------------------------------------------------------------------------

ENC_KEEP = (["prescription_id", "checkup_id", "user_id", "site_id", "site_district",
             "checkup_date", "year", "split", "age", "sex", "smoker_flag",
             "glucose_type", "symptom_text", "prescriber_id"]
            + VITAL_COLS + ["symptom_tokens"])
ORDER_KEEP = ["prescription_id", "order_id", "drug_id", "drug_name", "type_name",
              "dose_canon", "duration_bucket", "duration_days", "instruction"]


def write_expanded_parquets(enc: pd.DataFrame, orders: pd.DataFrame) -> None:
    """Emit rxgen_{encounters,orders}_expanded.parquet.

    Column list and order are asserted against the shipped rxgen_*.parquet, so a
    downstream reader can treat the expanded file as a drop-in with one extra
    column. The shipped files are never written to.
    """
    enc_out = enc[ENC_KEEP + ["corpus"]].copy()
    ord_out = orders[ORDER_KEEP + ["corpus"]].copy()
    # year is int32 in the shipped file; keep it there when nothing is missing.
    if enc_out["year"].notna().all():
        enc_out["year"] = enc_out["year"].astype("int32")

    import pyarrow.parquet as pq
    for name, out in (("encounters", enc_out), ("orders", ord_out)):
        ref = PROCESSED / f"rxgen_{name}.parquet"
        if ref.exists():
            want = list(pq.read_schema(ref).names) + ["corpus"]
            assert list(out.columns) == want, (
                f"{name}: schema drift.\n  expanded={list(out.columns)}\n  want={want}")
        path = PROCESSED / f"rxgen_{name}_expanded.parquet"
        out.to_parquet(path, index=False)
        print(f"  wrote {path}  ({len(out)} rows, {len(out.columns)} cols)")
        if ref.exists():
            a, b = pq.read_schema(ref), pq.read_schema(path)
            drift = [f"{f.name}: {a.field(f.name).type} -> {f.type}"
                     for f in b if f.name in a.names and a.field(f.name).type != f.type]
            print(f"       dtype drift vs shipped: {drift or 'none'}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="also emit merged data/interim/*_all.csv")
    ap.add_argument("--parquet", action="store_true",
                    help="emit data/processed/rxgen_{encounters,orders}_expanded.parquet")
    ap.add_argument("--json", action="store_true",
                    help="write the report to results/rx_generation/corpus_merge.json")
    ap.add_argument("--freeze-adult-split", action="store_true",
                    help="pin adult patients to their rxgen_encounters.parquet split "
                         "so the merged run keeps the published adult test set")
    args = ap.parse_args(argv)

    cfg = DataConfig()
    raw = load_all()

    # --- identical cleaning to the adult path -----------------------------
    enc = build_encounters(raw["encounters"])
    orders = build_orders(raw["orders"])
    orders["duration_bucket"] = bucket_duration(orders["duration_days"])
    # build_orders drops rows with a null drug_id; carry corpus through.
    hist = raw["hist"]

    id_report = verify_ids(raw, enc, orders)

    adult_enc = enc[enc["corpus"] == "adult"]
    adult_orders = orders[orders["corpus"] == "adult"]
    adult_drugs = set(adult_orders["drug_id"].dropna().astype(int))

    # --- per-corpus table --------------------------------------------------
    rows = []
    for c in ["adult", *MCH_CORPORA]:
        e = enc[enc["corpus"] == c]
        o = orders[orders["corpus"] == c]
        drugs = set(o["drug_id"].dropna().astype(int))
        absent = set(VITALS_ABSENT_BY_CORPUS[c])
        present_cols = [v for v in VITAL_COLS if v not in absent]
        # Mean per-encounter count of the 13 vitals that survived cleaning.
        vit_present = e[VITAL_COLS].notna().sum(axis=1).mean() if len(e) else 0.0
        rows.append({
            "corpus": c,
            "encounters": len(e),
            "patients": int(e["user_id"].nunique()),
            "orders": len(o),
            "rx_w_drugs": int(o["prescription_id"].nunique()),
            "zero_drug%": _pct(len(e) - o["prescription_id"].nunique(), len(e)),
            "drugs/rx": round(len(o) / max(o["prescription_id"].nunique(), 1), 2),
            "symptom%": _pct(int((e["symptom_text"].str.len() > 0).sum()), len(e)),
            "vital_cols": len(present_cols),
            "vitals/enc": round(float(vit_present), 1),
            "labels": len(drugs),
            "new_labels": len(drugs - adult_drugs) if c != "adult" else 0,
            "sites": int(e["site_id"].nunique()),
            "years": f"{int(e['year'].min())}-{int(e['year'].max())}",
        })

    print("\n" + "=" * 100)
    print("PER-CORPUS TOTALS (after src/phcrx/preprocess.py cleaning)")
    print("=" * 100)
    print(_fmt_table(rows, list(rows[0].keys())))

    # --- combined ----------------------------------------------------------
    mch = enc[enc["corpus"] != "adult"]
    mch_orders = orders[orders["corpus"] != "adult"]
    all_drugs = set(orders["drug_id"].dropna().astype(int))
    new_drugs = all_drugs - adult_drugs
    orders_on_new = int(mch_orders["drug_id"].dropna().astype(int).isin(new_drugs).sum())

    tok_adult = Counter(t for toks in adult_enc["symptom_tokens"] for t in toks)
    tok_all = Counter(t for toks in enc["symptom_tokens"] for t in toks)
    # preprocess builds the real word vocab on the TRAIN split only and prepends
    # 4 specials. This is the whole-corpus equivalent at the same min_word_freq
    # cutoff (2), specials excluded -- an upper bound on the vocabulary growth.
    v_adult = {w for w, n in tok_adult.items() if n >= 2}
    v_all = {w for w, n in tok_all.items() if n >= 2}

    shared_patients = set(adult_enc["user_id"].dropna()) & set(mch["user_id"].dropna())

    combined = {
        "encounters_adult": len(adult_enc),
        "encounters_mch": len(mch),
        "encounters_total": len(enc),
        "encounters_growth_pct": _pct(len(mch), len(adult_enc)),
        "orders_adult": len(adult_orders),
        "orders_mch": len(mch_orders),
        "orders_total": len(orders),
        "orders_growth_pct": _pct(len(mch_orders), len(adult_orders)),
        "patients_adult": int(adult_enc["user_id"].nunique()),
        "patients_total": int(enc["user_id"].nunique()),
        "patients_mch_only": int(mch["user_id"].nunique()),
        "patients_in_both_prescribed": len(shared_patients),
        "drug_labels_adult": len(adult_drugs),
        "drug_labels_total": len(all_drugs),
        "drug_labels_new_from_mch": len(new_drugs),
        "drug_vocab_adult_with_specials": len(adult_drugs) + 4,
        "drug_vocab_merged_with_specials": len(all_drugs) + 4,
        "mch_orders_on_new_labels": orders_on_new,
        "mch_orders_on_new_labels_pct": _pct(orders_on_new, len(mch_orders)),
        "word_vocab_adult_minfreq2": len(v_adult),
        "word_vocab_combined_minfreq2": len(v_all),
        "history_rows_adult": int((hist["corpus"] == "adult").sum()),
        "history_rows_mch": int((hist["corpus"] != "adult").sum()),
    }
    print("\n" + "=" * 100)
    print("COMBINED TOTALS")
    print("=" * 100)
    for k, v in combined.items():
        print(f"  {k:<34} {v}")

    # --- symptom-text coverage --------------------------------------------
    print("\n" + "=" * 100)
    print("SYMPTOM-TEXT COVERAGE (the branch the whole merge depends on)")
    print("=" * 100)
    rx_with_drugs = set(orders["prescription_id"].dropna())
    cov_rows = []
    for label, e in [("adult", adult_enc),
                     ("antenatal", enc[enc["corpus"] == "antenatal"]),
                     ("postnatal", enc[enc["corpus"] == "postnatal"]),
                     ("motherhood", enc[enc["corpus"] == "motherhood"]),
                     ("MCH (all 3)", mch), ("MERGED", enc)]:
        has = e["symptom_text"].str.len() > 0
        ntok = e["symptom_tokens"].map(len)
        drugged = e[e["prescription_id"].isin(rx_with_drugs)]
        cov_rows.append({
            "corpus": label,
            "encounters": len(e),
            "with text": int(has.sum()),
            "text %": _pct(int(has.sum()), len(e)),
            "text % (rx w/ >=1 drug)": _pct(
                int((drugged["symptom_text"].str.len() > 0).sum()), len(drugged)),
            "mean chars": round(float(e["symptom_text"].str.len().mean()), 1),
            "mean tokens": round(float(ntok.mean()), 2),
            "mean tokens|text": round(float(ntok[has].mean()), 2) if int(has.sum()) else 0.0,
            "token vocab": len({t for toks in e["symptom_tokens"] for t in toks}),
        })
    print(_fmt_table(cov_rows, list(cov_rows[0].keys())))
    mch_tok = {t for toks in mch["symptom_tokens"] for t in toks}
    adult_tok = set(tok_adult)
    print(f"\n  MCH symptom tokens absent from the adult corpus: "
          f"{len(mch_tok - adult_tok)} of {len(mch_tok)} "
          f"({_pct(len(mch_tok - adult_tok), len(mch_tok))}%)")
    combined["symptom_pct_adult"] = cov_rows[0]["text %"]
    combined["symptom_pct_mch"] = cov_rows[4]["text %"]
    combined["symptom_pct_merged"] = cov_rows[5]["text %"]
    combined["mch_tokens_unseen_in_adult"] = len(mch_tok - adult_tok)
    combined["mch_tokens_total"] = len(mch_tok)

    # --- distribution shift: the reason to be careful ----------------------
    vd = pd.read_csv(INTERIM / "vocab_drug.csv", dtype=str,
                     keep_default_na=False, na_values=[""])
    vm = pd.read_csv(INTERIM / "vocab_misc.csv", dtype=str,
                     keep_default_na=False, na_values=[""])
    cat_of = dict(zip(pd.to_numeric(vd["drug_id"], errors="coerce"), vd["cat_id"]))
    cat_name = dict(zip(vm.loc[vm["kind"] == "category", "id"],
                        vm.loc[vm["kind"] == "category", "val"]))
    name_of = dict(zip(pd.to_numeric(vd["drug_id"], errors="coerce"), vd["drug_name"]))

    def cat_share(o: pd.DataFrame) -> pd.Series:
        cats = o["drug_id"].astype(float).map(cat_of).map(
            lambda c: cat_name.get(str(c), "(none)") if pd.notna(c) else "(none)")
        return cats.value_counts(normalize=True) * 100

    sa, sm = cat_share(adult_orders), cat_share(mch_orders)
    shift = pd.DataFrame({"adult%": sa, "mch%": sm}).fillna(0.0)
    shift["delta"] = shift["mch%"] - shift["adult%"]
    shift = shift.reindex(shift["delta"].abs().sort_values(ascending=False).index)
    shift.index.name = "category"
    print("\n" + "=" * 100)
    print("PHARMACOLOGICAL-CATEGORY SHIFT (share of drug orders, top 12 by |delta|)")
    print("=" * 100)
    print(shift.head(12).round(1).to_string())

    # --- the MCH-only labels ----------------------------------------------
    print("\n" + "=" * 100)
    print("LABELS THAT EXIST ONLY IN MCH (unreachable reward on an adult test set)")
    print("=" * 100)
    new_counts = Counter(mch_orders.loc[
        mch_orders["drug_id"].dropna().astype(int).isin(new_drugs),
        "drug_id"].astype(int))
    new_rows = [{"drug_id": int(d), "drug_name": name_of.get(float(d), "?"),
                 "mch orders": int(n)} for d, n in new_counts.most_common()]
    print(_fmt_table(new_rows, ["drug_id", "drug_name", "mch orders"]))
    print(f"\n  {len(new_drugs)} labels, {orders_on_new} orders "
          f"({_pct(orders_on_new, len(orders))}% of all merged orders, "
          f"{_pct(orders_on_new, len(mch_orders))}% of MCH orders). "
          f"Output space grows {len(adult_drugs) + 4} -> {len(all_drugs) + 4} "
          f"(+{_pct(len(new_drugs), len(adult_drugs) + 4)}%).")
    combined["output_space_growth_pct"] = _pct(len(new_drugs), len(adult_drugs) + 4)

    # --- where does the extra supervision actually land? -------------------
    # The adult corpus is long-tailed; extra data only helps if it reaches
    # labels that are currently starved. Strata match preprocess.main().
    adult_freq = Counter(adult_orders["drug_id"].dropna().astype(int))

    def stratum(d: int) -> str:
        f = adult_freq.get(d, 0)
        return "head>=100" if f >= 100 else ("mid 10-99" if f >= 10 else
                                             ("tail 1-9" if f >= 1 else "unseen-in-adult"))

    mch_did = mch_orders["drug_id"].dropna().astype(int)
    strat = mch_did.map(stratum).value_counts()
    order = ["head>=100", "mid 10-99", "tail 1-9", "unseen-in-adult"]
    tail_rows = [{
        "adult stratum": s,
        "mch orders": int(strat.get(s, 0)),
        "% of mch orders": _pct(int(strat.get(s, 0)), len(mch_did)),
        "distinct labels": int(mch_did[mch_did.map(stratum) == s].nunique()),
    } for s in order]
    print("\n" + "=" * 100)
    print("WHERE THE MCH SUPERVISION LANDS (MCH orders binned by the label's "
          "ADULT frequency)")
    print("=" * 100)
    print(_fmt_table(tail_rows, list(tail_rows[0].keys())))
    # How many currently-tail adult labels get promoted out of the tail?
    merged_freq = Counter(orders["drug_id"].dropna().astype(int))
    promoted = sum(1 for d, f in adult_freq.items() if f < 10 <= merged_freq[d])
    print(f"\n  adult labels promoted out of the tail (<10 -> >=10 orders) by the "
          f"merge: {promoted}")

    # --- splits and the head/mid/tail distribution before vs after --------
    # preprocess.main() computes drug frequency strata on the TRAIN split only.
    # To isolate the effect of the MCH rows from the effect of a reshuffled
    # split, BOTH sides below use the same merged-frame split: "before" is that
    # split's adult train orders, "after" adds the MCH train orders.
    print("\n" + "=" * 100)
    print(f"SPLIT ON THE MERGED FRAME (patient-level, seed {cfg.seed}, "
          f"test {cfg.test_frac:.0%} / val {cfg.val_frac:.0%})")
    print("=" * 100)
    print(f"  mirrored split logic reproduces the shipped adult split: "
          f"{check_split_mirror(adult_enc, cfg)}")
    enc = enc.reset_index(drop=True)
    enc["split"] = merged_split(enc, cfg, args.freeze_adult_split)
    print(f"  mode: {'FROZEN adult split' if args.freeze_adult_split else 'recomputed on the merged frame'}")
    straddle = int(enc.groupby("user_id")["split"].nunique().gt(1).sum())
    assert straddle == 0, f"{straddle} patients straddle splits on the merged frame"
    print(f"  patients straddling splits: {straddle} (leakage guard)")
    drift = adult_split_drift(enc)
    print(f"  drift vs the shipped adult split: {drift}")
    combined["adult_split_drift"] = drift
    sp_rows = []
    for c in ["adult", *MCH_CORPORA, "MERGED"]:
        e = enc if c == "MERGED" else enc[enc["corpus"] == c]
        vc = e["split"].value_counts()
        sp_rows.append({"corpus": c, "train": int(vc.get("train", 0)),
                        "val": int(vc.get("val", 0)), "test": int(vc.get("test", 0)),
                        "total": len(e)})
    print(_fmt_table(sp_rows, ["corpus", "train", "val", "test", "total"]))

    split_of = dict(zip(enc["prescription_id"], enc["split"]))
    orders = orders.copy()
    orders["split"] = orders["prescription_id"].map(split_of)
    tr = orders[orders["split"] == "train"]
    tr_adult = tr[tr["corpus"] == "adult"]
    f_before = Counter(tr_adult["drug_id"].dropna().astype(int))
    f_after = Counter(tr["drug_id"].dropna().astype(int))

    def _stratum_of(f: int) -> str:
        return "head (>=100)" if f >= 100 else (
            "mid (10-99)" if f >= 10 else ("tail (1-9)" if f >= 1 else "unseen (0)"))

    labels_before = set(
        orders.loc[orders["corpus"] == "adult", "drug_id"].dropna().astype(int))
    labels_after = set(orders["drug_id"].dropna().astype(int))
    strata_order = ["head (>=100)", "mid (10-99)", "tail (1-9)", "unseen (0)"]
    hmt_rows = []
    for s in strata_order:
        lb = [d for d in labels_before if _stratum_of(f_before.get(d, 0)) == s]
        la = [d for d in labels_after if _stratum_of(f_after.get(d, 0)) == s]
        hmt_rows.append({
            "stratum": s,
            "labels before": len(lb),
            "train orders before": sum(f_before.get(d, 0) for d in lb),
            "labels after": len(la),
            "train orders after": sum(f_after.get(d, 0) for d in la),
            "d labels": len(la) - len(lb),
        })
    hmt_rows.append({
        "stratum": "TOTAL", "labels before": len(labels_before),
        "train orders before": sum(f_before.values()),
        "labels after": len(labels_after),
        "train orders after": sum(f_after.values()),
        "d labels": len(labels_after) - len(labels_before)})
    print("\n" + "=" * 100)
    print("HEAD/MID/TAIL DRUG-FREQUENCY DISTRIBUTION, BEFORE vs AFTER "
          "(train-split frequency, same split both sides)")
    print("=" * 100)
    print(_fmt_table(hmt_rows, list(hmt_rows[0].keys())))

    # Per-label movement, restricted to labels the adult corpus already has:
    # this is the number that decides whether the merge helps the long tail.
    moves = Counter((_stratum_of(f_before.get(d, 0)), _stratum_of(f_after.get(d, 0)))
                    for d in labels_before)
    mv_rows = [{"from (adult only)": a, "to (merged)": b, "labels": int(n)}
               for (a, b), n in sorted(moves.items(),
                                       key=lambda kv: (strata_order.index(kv[0][0]),
                                                       strata_order.index(kv[0][1])))
               if a != b]
    print("\n  adult labels that change stratum because of the merge:")
    print("  " + ("\n  ".join(f"{r['from (adult only)']:>13} -> {r['to (merged)']:<13} "
                              f"{r['labels']}" for r in mv_rows) or "none"))
    n_moved = sum(r["labels"] for r in mv_rows)
    print(f"  total moved: {n_moved} of {len(labels_before)} adult labels "
          f"({_pct(n_moved, len(labels_before))}%)")

    # Labels in the merged output space that an adult-only test set can never
    # reward: present in merged TRAIN but with zero adult orders anywhere.
    unreachable = {d for d in labels_after if f_after.get(d, 0) > 0} - labels_before
    print(f"\n  labels in the merged TRAIN set with zero adult orders anywhere "
          f"(unreachable reward on an adult-only test set): {len(unreachable)}")
    combined["unreachable_labels_in_merged_train"] = len(unreachable)

    # --- split-integrity warnings ------------------------------------------
    print("\n" + "=" * 100)
    print("WARNINGS")
    print("=" * 100)
    yr = enc.groupby("corpus")["year"].agg(["min", "max"])
    mch_min = int(yr.loc[list(MCH_CORPORA), "min"].min())
    print(f"  * temporal split: MCH encounters start in {mch_min}. With the default "
          f"DataConfig\n    (temporal_train_end={cfg.temporal_train_end}, "
          f"temporal_val_end={cfg.temporal_val_end}) all "
          f"{len(mch)} MCH rows land in TEST,\n    adding zero training signal and "
          f"changing what the benchmark measures. Merge under\n    the patient split "
          f"only, or re-tune the temporal boundaries.")
    print(f"  * {len(shared_patients)} patients hold prescriptions in BOTH the adult "
          f"and MCH cohorts\n    (and 66 of the 603 MCH patients have an adult checkup "
          f"of any kind, measured in-DB).\n    Splits must be computed on the MERGED "
          f"frame - user_id is deliberately not namespaced -\n    or those patients "
          f"leak across train and test.")
    print(f"  * site concentration: MCH draws on "
          f"{int(mch['site_id'].nunique())} site(s) vs "
          f"{int(adult_enc['site_id'].nunique())} for the adult corpus.")
    sex = mch["sex"].value_counts(dropna=False).to_dict()
    print(f"  * MCH sex distribution: {sex} (structurally female; any sex feature "
          f"becomes\n    a near-perfect corpus indicator).")

    # --- optional artefacts -------------------------------------------------
    if args.write:
        names = {
            "encounters": "encounters_all.csv",
            "orders": "rx_orders_all.csv",
            "advice": "rx_advice_all.csv",
            "tests": "rx_tests_all.csv",
            "cc": "rx_cc_all.csv",
            "hist": "patient_history_all.csv",
        }
        for key, fname in names.items():
            raw[key].to_csv(INTERIM / fname, index=False)
        print(f"\nwrote merged CSVs to {INTERIM}: {', '.join(names.values())}")

    if args.parquet:
        print()
        write_expanded_parquets(enc, orders)

    if args.json:
        payload = {"id_verification": id_report,
                   "per_corpus": rows, "combined": combined,
                   "symptom_coverage": cov_rows,
                   "category_shift": shift.round(2).to_dict(orient="index"),
                   "mch_only_labels": new_rows,
                   "supervision_strata": tail_rows,
                   "tail_labels_promoted": promoted,
                   "split_mode": ("frozen_adult" if args.freeze_adult_split
                                  else "recomputed_on_merged_frame"),
                   "split_counts": sp_rows,
                   "head_mid_tail_before_after": hmt_rows,
                   "stratum_moves": mv_rows}
        path = RESULTS / "corpus_merge.json"
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
