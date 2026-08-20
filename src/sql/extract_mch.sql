-- Maternal & child-health (MCH) corpus extract, harmonised onto the shape of
-- src/sql/extract.sql so the two can be concatenated row-wise.
--
-- Emits UTF-8 CSVs into data/interim/ with a trailing `corpus` column.
-- Run from the project root:
--   psql -h 127.0.0.1 -U postgres -d gphc-fix -f src/sql/extract_mch.sql
--
-- SCOPE. Three of the seven unused corpora are exported. The other four are
-- excluded on measured grounds (see docs/corpus_expansion.md):
--   infant     - 94 encounters, 31 drug orders, 4/13 vitals, 77% zero-drug.
--   childhood  - 10 encounters, 10 drug orders.
--   covid      - 35 encounters; covid_patient_checkup carries only 5 of the 13
--                pipeline vitals and even temperature is populated in 6% of rows.
--   eye        - 45 encounters; extra_symptom populated in 9 rows averaging 2.5
--                characters, and eye_prescription.checkup_id has no vitals-
--                bearing counterpart (the eye_* clinical tables are keyed on
--                preliminary_test_id / report_id and hold only ocular findings).
--
-- ID NAMESPACES. prescription_id and checkup_id restart at 1 in every MCH
-- corpus and therefore COLLIDE with the adult corpus (adult prescription_id
-- 4..14179, antenatal 1..1400). They are emitted verbatim here; the per-corpus
-- offset is applied in src/phcrx/expand/merge_corpora.py. user_id and site_id
-- are NOT namespaced - they are genuinely shared keys (a3m_account_details and
-- eh_project_site), and 134 MCH patients also appear in the adult cohort.
--
-- COLUMN PARITY. mch_antenatal / mch_postnatal / mch_motherhood carry the same
-- physical column names and types as eh_patient_checkup for every vital the
-- pipeline consumes. Only `checkup_for` is absent (it is an eh-only enum) and
-- is emitted as NULL. Columns that exist but are never populated in a given
-- corpus (e.g. height/bmi in postnatal) are emitted as-is and become missing
-- values downstream - the mask bit then carries the signal, as in the adult path.
set search_path to gramweb_ghealth, public;
\set ON_ERROR_STOP on
\pset pager off
set client_encoding to 'UTF8';

-- ---------------------------------------------------------------------------
-- encounters_mch.csv : same 44 columns as encounters.csv, plus `corpus`.
-- ---------------------------------------------------------------------------
\echo '-> encounters_mch.csv'
\copy (with u as ( select 'antenatal'::text as corpus, p.prescription_id, c.checkup_id, c.user_id, c.site_id, c.checkup_date, p.create_user_id as prescriber_id, p.create_date as rx_create_date, p.ref_hospital_id, p.extra_symptom, p.extra_advice, c.height, c.weight, c.bmi, c.waist, c.hip, c.waist_hip_ratio, c.temperature, c.oxygen_of_blood, c.bp_sys, c.bp_dia, c.blood_glucose, c.blood_glucose_type, c.blood_hemoglobin, c.urinary_glucose, c.urinary_protein, c.urinary_urobilinogen, c.urinary_ph, c.pulse_rate, c.arrhythmia, c.cholesterol, c.uric_acid, c.hbsag, c.blood_group as checkup_blood_group, c.smoker, c.color_status from mch_prescription p join mch_antenatal c on c.checkup_id = p.checkup_id union all select 'postnatal'::text, p.prescription_id, c.checkup_id, c.user_id, c.site_id, c.checkup_date, p.create_user_id, p.create_date, p.ref_hospital_id, p.extra_symptom, p.extra_advice, c.height, c.weight, c.bmi, c.waist, c.hip, c.waist_hip_ratio, c.temperature, c.oxygen_of_blood, c.bp_sys, c.bp_dia, c.blood_glucose, c.blood_glucose_type, c.blood_hemoglobin, c.urinary_glucose, c.urinary_protein, c.urinary_urobilinogen, c.urinary_ph, c.pulse_rate, c.arrhythmia, c.cholesterol, c.uric_acid, c.hbsag, c.blood_group, c.smoker, c.color_status from mch_post_prescription p join mch_postnatal c on c.checkup_id = p.checkup_id union all select 'motherhood'::text, p.prescription_id, c.checkup_id, c.user_id, c.site_id, c.checkup_date, p.create_user_id, p.create_date, p.ref_hospital_id, p.extra_symptom, p.extra_advice, c.height, c.weight, c.bmi, c.waist, c.hip, c.waist_hip_ratio, c.temperature, c.oxygen_of_blood, c.bp_sys, c.bp_dia, c.blood_glucose, c.blood_glucose_type, c.blood_hemoglobin, c.urinary_glucose, c.urinary_protein, c.urinary_urobilinogen, c.urinary_ph, c.pulse_rate, c.arrhythmia, c.cholesterol, c.uric_acid, c.hbsag, c.blood_group, c.smoker, c.color_status from mch_motherhood_prescription p join mch_motherhood c on c.checkup_id = p.checkup_id ) select u.prescription_id, u.checkup_id, u.user_id, u.site_id, u.checkup_date, u.prescriber_id, u.rx_create_date, u.ref_hospital_id, u.extra_symptom, u.extra_advice, ad.dateofbirth, upper(btrim(coalesce(ad.gender,''))) as sex, pt.marital_status, pt.blood_group as patient_blood_group, u.height, u.weight, u.bmi, u.waist, u.hip, u.waist_hip_ratio, u.temperature, u.oxygen_of_blood, u.bp_sys, u.bp_dia, u.blood_glucose, u.blood_glucose_type, u.blood_hemoglobin, u.urinary_glucose, u.urinary_protein, u.urinary_urobilinogen, u.urinary_ph, u.pulse_rate, u.arrhythmia, u.cholesterol, u.uric_acid, u.hbsag, u.checkup_blood_group, u.smoker, u.color_status, null::text as checkup_for, s.site_division, s.site_district, s.site_upazila, s.site_type, u.corpus from u left join a3m_account_details ad on ad.account_id = u.user_id left join eh_patient pt on pt.user_id = u.user_id left join eh_project_site s on s.project_id = u.site_id order by u.corpus, u.prescription_id) to 'data/interim/encounters_mch.csv' with (format csv, header true, force_quote *);

-- ---------------------------------------------------------------------------
-- rx_orders_mch.csv : same 17 columns as rx_orders.csv, plus `corpus`.
-- The rx_* dimension tables (drug_name, type, size, doze, instruction,
-- duration) are shared with the adult corpus - all 4104/1415/650 MCH orders
-- resolve in rx_drug_name, so the label space needs no remapping.
-- Orders whose prescription_id has no parent row are dropped by the inner
-- join, exactly as in extract.sql (antenatal 50, postnatal 21, motherhood 5).
-- ---------------------------------------------------------------------------
\echo '-> rx_orders_mch.csv'
\copy (with u as ( select 'antenatal'::text as corpus, d.id, d.prescription_id, d.drug_id, d.type_id, d.drug_size_id, d.doze_id, d.drug_instruction_id, d.drug_duration_id, d.drug_duration, d.special_instruction from rx_prescription_drugs_antenatal d join mch_prescription p on p.prescription_id = d.prescription_id union all select 'postnatal'::text, d.id, d.prescription_id, d.drug_id, d.type_id, d.drug_size_id, d.doze_id, d.drug_instruction_id, d.drug_duration_id, d.drug_duration, d.special_instruction from rx_prescription_drugs_postnatal d join mch_post_prescription p on p.prescription_id = d.prescription_id union all select 'motherhood'::text, d.id, d.prescription_id, d.drug_id, d.type_id, d.drug_size_id, d.doze_id, d.drug_instruction_id, d.drug_duration_id, d.drug_duration, d.special_instruction from rx_prescription_drugs_motherhood d join mch_motherhood_prescription p on p.prescription_id = d.prescription_id ) select u.id as order_id, u.prescription_id, u.drug_id, n.drug_name, n.cat_id as drug_cat_id, u.type_id, t.type_name, u.drug_size_id, sz.drug_size, u.doze_id, z.doze, u.drug_instruction_id, i.drug_instruction, u.drug_duration_id, du.drug_duration as duration_unit, u.drug_duration as duration_qty_raw, u.special_instruction, u.corpus from u left join rx_drug_name n on n.drug_id = u.drug_id left join rx_type t on t.type_id = u.type_id left join rx_size sz on sz.drug_size_id = u.drug_size_id left join rx_doze z on z.doze_id = u.doze_id left join rx_instruction i on i.drug_instruction_id = u.drug_instruction_id left join rx_duration du on du.drug_duration_id = u.drug_duration_id order by u.corpus, u.prescription_id, u.id) to 'data/interim/rx_orders_mch.csv' with (format csv, header true, force_quote *);

-- ---------------------------------------------------------------------------
-- rx_advice_mch.csv : MCH advice ids reuse eh_prescription_advice_template,
-- but 549 of 17160 antenatal rows point at template ids that do not exist
-- there (MCH-only advice text that was never backfilled). Those resolve to an
-- empty advice_en, matching how extract.sql handles unresolvable ids.
-- ---------------------------------------------------------------------------
\echo '-> rx_advice_mch.csv'
\copy (with u as ( select 'antenatal'::text as corpus, a.prescription_id, a.advice_id from mch_prescription_advice a join mch_prescription p on p.prescription_id = a.prescription_id union all select 'postnatal'::text, a.prescription_id, a.advice_id from mch_post_prescription_advice a join mch_post_prescription p on p.prescription_id = a.prescription_id union all select 'motherhood'::text, a.prescription_id, a.advice_id from mch_motherhood_prescription_advice a join mch_motherhood_prescription p on p.prescription_id = a.prescription_id ) select u.prescription_id, u.advice_id, left(btrim(coalesce(tpl.advice_en,'')),200) as advice_en, u.corpus from u left join eh_prescription_advice_template tpl on tpl.id = u.advice_id order by u.corpus, 1, 2) to 'data/interim/rx_advice_mch.csv' with (format csv, header true, force_quote *);

\echo '-> rx_tests_mch.csv'
\copy (with u as ( select 'antenatal'::text as corpus, t.prescription_id, t.test_id from mch_prescription_test t join mch_prescription p on p.prescription_id = t.prescription_id union all select 'postnatal'::text, t.prescription_id, t.test_id from mch_post_prescription_test t join mch_post_prescription p on p.prescription_id = t.prescription_id union all select 'motherhood'::text, t.prescription_id, t.test_id from mch_motherhood_prescription_test t join mch_motherhood_prescription p on p.prescription_id = t.prescription_id ) select u.prescription_id, u.test_id, tpl.test_name, u.corpus from u left join eh_prescription_test_template tpl on tpl.id = u.test_id order by u.corpus, 1, 2) to 'data/interim/rx_tests_mch.csv' with (format csv, header true, force_quote *);

\echo '-> rx_cc_mch.csv'
\copy (with u as ( select 'antenatal'::text as corpus, cc.prescription_id, cc.cc_id from mch_prescription_cc cc join mch_prescription p on p.prescription_id = cc.prescription_id union all select 'postnatal'::text, cc.prescription_id, cc.cc_id from mch_post_prescription_cc cc join mch_post_prescription p on p.prescription_id = cc.prescription_id union all select 'motherhood'::text, cc.prescription_id, cc.cc_id from mch_motherhood_prescription_cc cc join mch_motherhood_prescription p on p.prescription_id = cc.prescription_id ) select u.prescription_id, u.cc_id, tpl.cc_name, tpl.cc_type, tpl.cc_parent, par.cc_name as cc_group, u.corpus from u left join eh_prescription_cc_template tpl on tpl.id = u.cc_id left join eh_prescription_cc_template par on par.id = tpl.cc_parent order by u.corpus, 1, 2) to 'data/interim/rx_cc_mch.csv' with (format csv, header true, force_quote *);

-- ---------------------------------------------------------------------------
-- patient_history_mch.csv : every MCH checkup (prescribed or not) for patients
-- in the MCH cohort, so the history encoder can condition on prior visits.
-- Same columns as patient_history.csv, plus `corpus`.
-- ---------------------------------------------------------------------------
\echo '-> patient_history_mch.csv'
\copy (select c.checkup_id, c.user_id, c.checkup_date, c.site_id, c.prescription_id, c.height, c.weight, c.bmi, c.waist_hip_ratio, c.temperature, c.oxygen_of_blood, c.bp_sys, c.bp_dia, c.blood_glucose, c.blood_glucose_type, c.blood_hemoglobin, c.pulse_rate, c.cholesterol, c.uric_acid, c.color_status, 'antenatal' as corpus from mch_antenatal c where c.user_id in (select c2.user_id from mch_prescription p2 join mch_antenatal c2 on c2.checkup_id = p2.checkup_id) union all select c.checkup_id, c.user_id, c.checkup_date, c.site_id, c.prescription_id, c.height, c.weight, c.bmi, c.waist_hip_ratio, c.temperature, c.oxygen_of_blood, c.bp_sys, c.bp_dia, c.blood_glucose, c.blood_glucose_type, c.blood_hemoglobin, c.pulse_rate, c.cholesterol, c.uric_acid, c.color_status, 'postnatal' from mch_postnatal c where c.user_id in (select c2.user_id from mch_post_prescription p2 join mch_postnatal c2 on c2.checkup_id = p2.checkup_id) union all select c.checkup_id, c.user_id, c.checkup_date, c.site_id, c.prescription_id, c.height, c.weight, c.bmi, c.waist_hip_ratio, c.temperature, c.oxygen_of_blood, c.bp_sys, c.bp_dia, c.blood_glucose, c.blood_glucose_type, c.blood_hemoglobin, c.pulse_rate, c.cholesterol, c.uric_acid, c.color_status, 'motherhood' from mch_motherhood c where c.user_id in (select c2.user_id from mch_motherhood_prescription p2 join mch_motherhood c2 on c2.checkup_id = p2.checkup_id) order by 21, 2, 3, 1) to 'data/interim/patient_history_mch.csv' with (format csv, header true, force_quote *);

\echo 'extract_mch complete.'
