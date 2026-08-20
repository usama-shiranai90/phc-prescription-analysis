"""NCD risk stratification, benchmarked against WHO Global Health Observatory.

Two parts:
  1. Per-encounter NCD flags derived from recorded vitals using WHO diagnostic
     thresholds (plus Asian-specific BMI cut-offs, which matter for a South
     Asian cohort -- WHO's 25/30 thresholds materially understate adiposity
     risk in this population).
  2. A benchmark of the resulting cohort prevalence against WHO GHO indicators
     for Bangladesh, fetched from the public OData API (no key, and no patient
     data is transmitted -- it is a pure download of population statistics).

    python -m src.phcrx.nlp.ncd
"""
from __future__ import annotations

import json
import subprocess

import numpy as np
import pandas as pd

from ..config import PROCESSED, RESULTS
from .ollama_client import _curl

OUT = RESULTS / "nlp"
OUT.mkdir(parents=True, exist_ok=True)

GHO = "https://ghoapi.azureedge.net/api"

# Indicator codes on the GHO. Bangladesh = BGD.
GHO_INDICATORS = {
    "NCD_HYP_PREVALENCE_A": "Hypertension, adults 30-79 (age-standardized)",
    "BP_04": "Raised blood pressure SBP>=140 or DBP>=90 (age-standardized)",
    "NCD_GLUC_04": "Raised fasting blood glucose >=7.0 mmol/L (age-standardized)",
    "NCD_BMI_30C": "Obesity BMI>=30, adults (crude)",
    "NCD_BMI_25C": "Overweight BMI>=25, adults (crude)",
}

# GHO codes the both-sexes stratum as SEX_BTSX (not BTSX).
BOTH_SEXES = {None, "", "BTSX", "SEX_BTSX"}


def gho_fetch(indicator: str, country: str = "BGD") -> list[dict]:
    """Fetch a WHO GHO indicator. Uses Windows curl via WSL interop."""
    try:
        p = subprocess.run(
            [_curl(), "-s", "-m", "40", "-G", f"{GHO}/{indicator}",
             "--data-urlencode", f"$filter=SpatialDim eq '{country}'"],
            capture_output=True)
        if p.returncode != 0:
            return []
        return json.loads(p.stdout.decode("utf-8", "replace")).get("value", [])
    except Exception:
        return []


# --- WHO / Asian-specific diagnostic thresholds ---------------------------
def ncd_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Per-encounter NCD flags. NaN (not False) where the measure is missing."""
    out = pd.DataFrame(index=df.index)
    num = lambda c: pd.to_numeric(df.get(c), errors="coerce")

    sys_, dia = num("bp_sys"), num("bp_dia")
    bp_seen = sys_.notna() | dia.notna()
    out["hypertension_who"] = np.where(
        bp_seen, ((sys_ >= 140) | (dia >= 90)).astype(float), np.nan)
    out["hypertension_stage2"] = np.where(
        bp_seen, ((sys_ >= 160) | (dia >= 100)).astype(float), np.nan)

    # Glucose threshold depends on the assay recorded (FBS vs PBS/RBS).
    glu, gtype = num("blood_glucose"), df.get("glucose_type", pd.Series(index=df.index)).astype(str)
    fbs = gtype.str.upper().str.startswith("FBS")
    thr = np.where(fbs, 126.0, 200.0)
    out["diabetes_range"] = np.where(glu.notna(), (glu >= thr).astype(float), np.nan)
    pre_lo = np.where(fbs, 100.0, 140.0)
    out["prediabetes_range"] = np.where(
        glu.notna(), ((glu >= pre_lo) & (glu < thr)).astype(float), np.nan)

    bmi = num("bmi")
    # Asian-specific cut-offs (WHO 2004 expert consultation).
    out["overweight_asian"] = np.where(bmi.notna(), (bmi >= 23).astype(float), np.nan)
    out["obese_asian"] = np.where(bmi.notna(), (bmi >= 27.5).astype(float), np.nan)
    # WHO thresholds kept alongside so the GHO benchmark is like-for-like.
    out["overweight_who"] = np.where(bmi.notna(), (bmi >= 25).astype(float), np.nan)
    out["obese_who"] = np.where(bmi.notna(), (bmi >= 30).astype(float), np.nan)
    out["underweight"] = np.where(bmi.notna(), (bmi < 18.5).astype(float), np.nan)

    hb, sex = num("blood_hemoglobin"), df.get("sex")
    hb_thr = np.where(pd.Series(sex).eq("M"), 13.0, 12.0)
    out["anaemia_who"] = np.where(hb.notna(), (hb < hb_thr).astype(float), np.nan)

    spo2 = num("oxygen_of_blood")
    out["hypoxaemia"] = np.where(spo2.notna(), (spo2 < 94).astype(float), np.nan)
    pulse = num("pulse_rate")
    out["tachycardia"] = np.where(pulse.notna(), (pulse > 100).astype(float), np.nan)

    # Multimorbidity across the three headline NCDs.
    tri = out[["hypertension_who", "diabetes_range", "obese_asian"]]
    out["ncd_count"] = tri.sum(axis=1, min_count=1)
    out["ncd_multimorbid"] = (out["ncd_count"] >= 2).astype(float)
    return out


def main() -> None:
    enc = pd.read_parquet(PROCESSED / "rxgen_encounters.parquet")
    flags = ncd_flags(enc)
    res = pd.concat([enc[["prescription_id", "user_id", "age", "sex", "year"]],
                     flags], axis=1)
    res.to_parquet(PROCESSED / "ncd_flags.parquet", index=False)

    print("=" * 72)
    print(f"NCD FLAGS  (n={len(res)} encounters)")
    print(f"{'condition':22s} {'measured':>9s} {'cases':>7s} {'prevalence':>11s}")
    print("-" * 72)
    summary = {}
    for c in flags.columns:
        if c in ("ncd_count",):
            continue
        s = flags[c]
        n = int(s.notna().sum())
        k = int(s.fillna(0).sum())
        p = k / n if n else 0.0
        summary[c] = {"measured": n, "cases": k, "prevalence": p}
        print(f"{c:22s} {n:9d} {k:7d} {p:10.1%}")

    # Adults 30-79 subset, to line up with the WHO age-standardised indicator.
    ad = res[(res["age"] >= 30) & (res["age"] <= 79)]
    adult_htn = float(ad["hypertension_who"].mean(skipna=True))
    adult_dm = float(ad["diabetes_range"].mean(skipna=True))
    print(f"\nAdults 30-79 (n={len(ad)}): hypertension {adult_htn:.1%}, "
          f"diabetes-range glucose {adult_dm:.1%}")

    # --- WHO GHO benchmark ------------------------------------------------
    print(f"\n{'-'*72}\nWHO GHO benchmark for Bangladesh (public OData, no patient data sent)")
    bench = {}
    for code, label in GHO_INDICATORS.items():
        vals = gho_fetch(code)
        both = [v for v in vals if v.get("Dim1") in BOTH_SEXES] or vals
        latest = max(both, key=lambda v: v.get("TimeDim") or 0, default=None)
        if latest:
            bench[code] = {"label": label, "year": latest.get("TimeDim"),
                           "value": latest.get("NumericValue"),
                           "display": latest.get("Value"),
                           "sex": latest.get("Dim1"),
                           "age_group": latest.get("Dim2")}
            print(f"  {code:22s} {latest.get('TimeDim')}  "
                  f"{str(latest.get('Value'))[:18]:18s} "
                  f"[{latest.get('Dim1')}] {label[:36]}")
        else:
            bench[code] = {"label": label, "error": "no data returned"}
            print(f"  {code:22s} -- no data returned")

    (OUT / "ncd_summary.json").write_text(json.dumps(
        {"cohort": summary,
         "adults_30_79": {"n": int(len(ad)), "hypertension": adult_htn,
                          "diabetes_range_glucose": adult_dm},
         "who_gho_bangladesh": bench,
         "note": ("Cohort figures are screening-visit prevalence among "
                  "care-seekers, not a population sample; they are expected to "
                  "exceed WHO population estimates and are not directly "
                  "comparable.")}, indent=2, default=float), encoding="utf-8")
    print("\nNOTE: this cohort is care-seekers at screening camps, not a "
          "population sample.\n      Its prevalence should exceed WHO "
          "population figures; the comparison is\n      for orientation, not "
          "validation.")
    print("=" * 72)
    print("wrote", OUT / "ncd_summary.json")


if __name__ == "__main__":
    main()
