-- Frozen extract for prescription-generation modeling.
-- Emits UTF-8 CSVs into data/interim/. Run from the project root:
--   psql -h 127.0.0.1 -U postgres -d gphc-fix -v out=data/interim -f src/sql/extract.sql
set search_path to gramweb_ghealth, public;
\set ON_ERROR_STOP on
\pset pager off
set client_encoding to 'UTF8';

-- The modelling cohort: every prescription that resolves to a real checkup.
-- Zero-drug prescriptions are KEPT (a documented "no pharmacotherapy" decision).
create temporary view v_cohort as
select p.prescription_id, c.checkup_id, c.user_id, c.site_id, c.checkup_date
from eh_prescription p
join eh_patient_checkup c on c.checkup_id = p.prescription_id * 0 + p.checkup_id;

\echo '-> encounters.csv'
\copy (select p.prescription_id, c.checkup_id, c.user_id, c.site_id, c.checkup_date, p.create_user_id as prescriber_id, p.create_date as rx_create_date, p.ref_hospital_id, p.extra_symptom, p.extra_advice, ad.dateofbirth, upper(btrim(coalesce(ad.gender,''))) as sex, pt.marital_status, pt.blood_group as patient_blood_group, c.height, c.weight, c.bmi, c.waist, c.hip, c.waist_hip_ratio, c.temperature, c.oxygen_of_blood, c.bp_sys, c.bp_dia, c.blood_glucose, c.blood_glucose_type, c.blood_hemoglobin, c.urinary_glucose, c.urinary_protein, c.urinary_urobilinogen, c.urinary_ph, c.pulse_rate, c.arrhythmia, c.cholesterol, c.uric_acid, c.hbsag, c.blood_group as checkup_blood_group, c.smoker, c.color_status, c.checkup_for, s.site_division, s.site_district, s.site_upazila, s.site_type from eh_prescription p join eh_patient_checkup c on c.checkup_id = p.checkup_id left join a3m_account_details ad on ad.account_id = c.user_id left join eh_patient pt on pt.user_id = c.user_id left join eh_project_site s on s.project_id = c.site_id order by p.prescription_id) to 'data/interim/encounters.csv' with (format csv, header true, force_quote *);

\echo '-> rx_orders.csv'
\copy (select d.id as order_id, d.prescription_id, d.drug_id, n.drug_name, n.cat_id as drug_cat_id, d.type_id, t.type_name, d.drug_size_id, sz.drug_size, d.doze_id, z.doze, d.drug_instruction_id, i.drug_instruction, d.drug_duration_id, du.drug_duration as duration_unit, d.drug_duration as duration_qty_raw, d.special_instruction from rx_prescription_drugs d join eh_prescription p on p.prescription_id = d.prescription_id left join rx_drug_name n on n.drug_id = d.drug_id left join rx_type t on t.type_id = d.type_id left join rx_size sz on sz.drug_size_id = d.drug_size_id left join rx_doze z on z.doze_id = d.doze_id left join rx_instruction i on i.drug_instruction_id = d.drug_instruction_id left join rx_duration du on du.drug_duration_id = d.drug_duration_id order by d.prescription_id, d.id) to 'data/interim/rx_orders.csv' with (format csv, header true, force_quote *);

\echo '-> rx_advice.csv'
\copy (select a.prescription_id, a.advice_id, left(btrim(coalesce(tpl.advice_en,'')),200) as advice_en from eh_prescription_advice a join eh_prescription p on p.prescription_id=a.prescription_id left join eh_prescription_advice_template tpl on tpl.id=a.advice_id order by 1,2) to 'data/interim/rx_advice.csv' with (format csv, header true, force_quote *);

\echo '-> rx_tests.csv'
\copy (select t.prescription_id, t.test_id, tpl.test_name from eh_prescription_test t join eh_prescription p on p.prescription_id=t.prescription_id left join eh_prescription_test_template tpl on tpl.id=t.test_id order by 1,2) to 'data/interim/rx_tests.csv' with (format csv, header true, force_quote *);

\echo '-> rx_cc.csv'
\copy (select cc.prescription_id, cc.cc_id, tpl.cc_name, tpl.cc_type, tpl.cc_parent, par.cc_name as cc_group from eh_prescription_cc cc join eh_prescription p on p.prescription_id=cc.prescription_id left join eh_prescription_cc_template tpl on tpl.id=cc.cc_id left join eh_prescription_cc_template par on par.id=tpl.cc_parent order by 1,2) to 'data/interim/rx_cc.csv' with (format csv, header true, force_quote *);

\echo '-> rx_icd10.csv'
-- tb_data_icd uses positional column names; col9 is the ICD description.
\copy (select i.prescription_id, i.icd_id, ic.col9 as icd_name from eh_prescription_icd10 i join eh_prescription p on p.prescription_id=i.prescription_id left join tb_data_icd ic on ic.id=i.icd_id order by 1,2) to 'data/interim/rx_icd10.csv' with (format csv, header true, force_quote *);

-- Full checkup history for every cohort patient (prescribed or not) so the
-- history encoder can condition on prior visits, including un-prescribed ones.
\echo '-> patient_history.csv'
\copy (select c.checkup_id, c.user_id, c.checkup_date, c.site_id, c.prescription_id, c.height, c.weight, c.bmi, c.waist_hip_ratio, c.temperature, c.oxygen_of_blood, c.bp_sys, c.bp_dia, c.blood_glucose, c.blood_glucose_type, c.blood_hemoglobin, c.pulse_rate, c.cholesterol, c.uric_acid, c.color_status from eh_patient_checkup c where c.user_id in (select c2.user_id from eh_prescription p2 join eh_patient_checkup c2 on c2.checkup_id=p2.checkup_id) order by c.user_id, c.checkup_date, c.checkup_id) to 'data/interim/patient_history.csv' with (format csv, header true, force_quote *);

\echo '-> vocab_drug.csv'
\copy (select drug_id, drug_name, cat_id, country_code from rx_drug_name order by drug_id) to 'data/interim/vocab_drug.csv' with (format csv, header true, force_quote *);

\echo '-> vocab_misc.csv'
\copy (select 'type' kind, type_id::text id, type_name val from rx_type union all select 'doze', doze_id::text, doze from rx_doze union all select 'instruction', drug_instruction_id::text, drug_instruction from rx_instruction union all select 'duration_unit', drug_duration_id::text, drug_duration from rx_duration union all select 'size', drug_size_id::text, drug_size from rx_size union all select 'category', cat_id::text, cat_name from rx_category order by 1,2) to 'data/interim/vocab_misc.csv' with (format csv, header true, force_quote *);

\echo 'extract complete.'
