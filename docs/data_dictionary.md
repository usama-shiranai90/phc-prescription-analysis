# Portable Health Clinic (PHC) — Data Dictionary & Schema Map

**Source:** `gramweb_ghealth.sql` — phpMyAdmin dump (MySQL 5.7.23), database `gramweb_ghealth`, generated 16 Jan 2023.
**Loaded to:** SQLite `phc.db` — 135 tables, 709,860 rows, 0 insert errors.

## 1. Analysis-relevant subgraph & linkage keys

```
a3m_account_details (account_id) ──1:1── eh_patient (user_id)
        │  fullname, dateofbirth, gender                      │ barcode_id, blood_group, marital_status, address
        │                                                     │
        └──────────────── user_id = account_id ───────────────┘
                                   │
                                   │ user_id
                                   ▼
                        eh_patient_checkup (checkup_id, user_id, site_id)
                          vitals: height, weight, bmi, bp_sys, bp_dia,
                          blood_glucose(+_type), blood_hemoglobin, temperature,
                          oxygen_of_blood, pulse_rate, cholesterol, waist/hip,
                          urinary_*, color_status
                                   │ prescription_id / checkup_id
                                   ▼
                        eh_prescription (prescription_id, checkup_id)
                          │                      │                    │
              eh_prescription_icd10   eh_prescription_cc     rx_prescription_drugs
              (icd_id → tb_data_icd)  (cc_id → eh_prescription_cc_template)  (drug_id → rx_drug_name)

        eh_patient_checkup.site_id = eh_project_site.project_id
             → site_division / site_district / site_upazila (numeric BBS codes)
             → location_bbs2011 (division/district/upazila/unionid, loc_type, loc_name_en)
```

### Confirmed join keys (verified by row overlap)
| From | Key | To | Match rate |
|---|---|---|---|
| eh_patient.user_id | = | a3m_account_details.account_id | 45,338/45,350 (99.97%) |
| eh_patient_checkup.user_id | = | eh_patient.user_id | 46,040/46,056 |
| eh_patient_checkup.site_id | = | eh_project_site.project_id | 45,118/46,056 (98%) |
| eh_prescription.checkup_id | = | eh_patient_checkup.checkup_id | — |
| eh_prescription_icd10.icd_id | = | tb_data_icd.id | ICD-10 |
| eh_prescription_cc.cc_id | = | eh_prescription_cc_template.id | complaints |

## 2. Demographics
- **Sex**: `a3m_account_details.gender` — values `f/m/F/M` (needs case-normalisation). ~52% female.
- **Age**: derived from `dateofbirth`. 45,292/45,338 valid. Many DOBs are year-only (Jan-1 spikes) → **age is year-precision, not day-precision**. Age computed as (checkup_year − birth_year).
- Also present: marital_status, blood_group (in eh_patient), guardian_name, address (free text).

## 3. Vitals — units (inferred from distributions) & clinical reference standards
| Variable | Unit (inferred) | Median | Reference standard for classification |
|---|---|---|---|
| bp_sys / bp_dia | **mmHg** | 120 / 79 | Hypertension: SBP≥140 or DBP≥90 (WHO); also report ACC/AHA stage-1 ≥130/80 |
| blood_glucose | **mg/dL** (98.4%); 1.3% in mmol/L (≤20) → ×18 harmonised & flagged | 104 | Diabetes: FBS≥126, RBS/PBS≥200 mg/dL |
| blood_glucose_type | FBS (fasting), PBS (post-breakfast/prandial, dominant 89%), RBS (random) | — | Type governs threshold |
| bmi | **kg/m²** | 22.5 | Asian cutoffs: overweight≥23, obese≥27.5; WHO: ≥25 / ≥30 |
| height / weight | **cm / kg** | 157 / 55.8 | — |
| blood_hemoglobin | **g/dL** | 11.9 | Anaemia WHO: <13 (M), <12 (non-preg F) |
| temperature | **°F** | 97.2 | Fever ≥100.4°F (38°C) |
| oxygen_of_blood | **% SpO₂** | 98 | Hypoxaemia <94% |
| pulse_rate | **bpm** | 82 | Tachycardia >100, brady <60 |
| cholesterol | **mg/dL** (sparse, n=2,683) | 207 | High ≥240 |

## 4. Triage color_status (eh_patient_checkup)
Values 0–4. PHC uses a logic-based colour triage (Green→Red = healthy→emergency).
`covid_color_wise_advice` legend maps 1–5: 1 Green, 2 Light-Yellow, 3 Yellow, 4 Orange, 5 Red.
**Ambiguity:** checkup uses 0–4, COVID advice uses 1–5 — exact mapping flagged in open questions.

## 5. Diagnoses & complaints
- **ICD-10 (`eh_prescription_icd10`)**: only **207 rows** — very sparse; skews to symptom/screening codes (R-, Z-, Y-). Not a reliable disease-prevalence source on its own.
- **Chief complaints (`eh_prescription_cc`, 2,560 rows)**: richer signal. Top: H/O Hypertension (486), H/O DM (364), General weakness, Low back pain, joint pain. "H/O" = history of (self-reported).
- **Prescribed drugs (`rx_prescription_drugs`, 30,016 rows)**: drug_id → rx_drug_name.

## 6. Geography (Bangladesh BBS-2011 hierarchy)
Division(2) → District/Zila(2) → Upazila(2) → Union(2) → Mouza(3) → Village. `location_bbs2011`: 7 divisions, 64 districts, 544 upazilas, 7,757 unions.
Divisions present in checkup data: **Dhaka (30) 26,257; Chittagong (20) 7,198; Khulna (40) 4,936; Rangpur (55) 4,570; Rajshahi (50) 1,751; Barisal (10) 122.** Site-level granularity; patient home address is free-text (not geocoded).

## 7. Temporal coverage
Checkups 2012–2023 (a few 2010 placeholder dates). Peak 2013 (13,277). Sharp decline after 2016; COVID module 2020–21.

## 8. Other modules (not primary focus)
- COVID screening (`covid_patient_checkup`, self-reported comorbidities), MCH (maternal-child health), eye care (`eye_*`), pathology reports (`pr_*`).
