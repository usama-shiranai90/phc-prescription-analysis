# PHC Study — Open Questions & Data Ambiguities Log

Running log of items needing domain-expert input. Updated as analysis proceeds.

1. **ICD confirmed vs suspected.** `eh_prescription_icd10` is extremely sparse (207 rows for 13,879 prescriptions) and dominated by symptom/screening codes (R03.0 "elevated BP reading without diagnosis of hypertension", Z13.1 "screening for diabetes"). These read as *screening findings/reasons for encounter*, not confirmed diagnoses. **Q: Are ICD codes entered only occasionally, and do they denote confirmed disease or reason-for-visit?**

2. **color_status coding (0–4 vs 1–5).** Checkup triage uses 0–4; the COVID advice legend uses 1–5 (Green→Red). **Q: What is the exact 0–4 → colour/severity mapping for the general telemedicine checkup?** Provisional: treat higher = more severe; 0 likely = not assessed/incomplete.

3. **blood_glucose_type "PBS".** Dominant type (89%). Interpreted as post-breakfast/postprandial sugar. **Q: Confirm PBS definition and the intended diabetes threshold for PBS (using ≥200 mg/dL as for RBS).**

4. **Glucose mixed units.** 1.3% of glucose values ≤20 appear to be mmol/L. Harmonised ×18 to mg/dL and flagged. **Q: Confirm these are unit-entry errors, not genuine severe hypoglycaemia.**

5. **Age precision.** Many DOBs are year-only (Jan-1). Age is year-precision. **Q: acceptable for age-band analysis?**

6. **Repeat visits.** 46,056 checkups / 36,530 patients → some repeat visits. **Q: For prevalence, use first checkup per patient (chosen default) or all visits?** Default here: first-visit-per-patient for prevalence; all visits for volume/temporal.

7. **Selection bias.** Who gets screened is non-random (worksite programs — Epyllion garments/textile sites feature heavily — plus community camps). Prevalence estimates are **program-population**, not general-population. Stated as a limitation throughout.

8. **Physiologic-implausible extremes** (e.g., bp_sys=133187, bmi=187500, height=14861 cm). Treated as data-entry errors and filtered with documented rules; counts reported, not silently dropped.


---

## Advanced-analytics phase — new open questions (prescription + ML/DL)

1. **Antihypertensive care gap (49% at prescribed visits):** Does this reflect genuine under-treatment, deliberate deferral to hospital referral, or a lifestyle-modification-first protocol at the screening stage? Requires visit-level prospective audit with treatment-intent codes to resolve.
2. **Glucose test type per record:** The dominant test appears to be post-breakfast/postprandial ("PBS"), not fasting. Dysglycemia thresholds (and the 4% dysglycemic phenotype) depend on this — needs confirmation of test protocol per record.
3. **UNKNOWN-mapped drugs (20.2% of lines):** Are the low-frequency tail brands clinically meaningful (specialty/referral drugs) or noise/data-entry variants? A pharmacist review of the top-50 UNKNOWN brands would close the coverage gap.
4. **Polypharmacy drivers:** Drug count is unpredictable from vitals (R²=0.015). Linking prescriptions to the symptom/diagnosis tables would test whether presenting complaints explain prescription burden.
5. **Prescription→diagnosis linkage integrity:** Confirm whether every prescription maps to a recorded diagnosis, or whether some are empirical/symptomatic without a coded dx.
6. **Model calibration for deployment:** The predictive models discriminate (AUC 0.78–0.81) but are uncalibrated (class-weighted). Before any triage use they need recalibration on a held-out, prevalence-representative sample.
7. **Longitudinal phenotype transitions:** With only first-visit data the four cardiometabolic phenotypes are static snapshots. Do repeat-visit patients transition between phenotypes, and does prescribing alter trajectory?
