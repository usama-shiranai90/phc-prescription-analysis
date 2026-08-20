-- Profiling for prescription-generation modeling.
-- Run: psql -h 127.0.0.1 -U postgres -d gphc-fix -f src/sql/profile_rx.sql
set search_path to gramweb_ghealth, public;
\pset pager off

\echo '=== 1. Core table volumes ==='
select 'eh_patient_checkup' t, count(*) n from eh_patient_checkup
union all select 'eh_prescription', count(*) from eh_prescription
union all select 'rx_prescription_drugs', count(*) from rx_prescription_drugs
union all select 'eh_prescription_cc', count(*) from eh_prescription_cc
union all select 'eh_prescription_icd10', count(*) from eh_prescription_icd10
union all select 'eh_prescription_advice', count(*) from eh_prescription_advice
union all select 'eh_prescription_test', count(*) from eh_prescription_test
union all select 'eh_patient', count(*) from eh_patient
union all select 'a3m_account_details', count(*) from a3m_account_details
union all select 'rx_drug_name', count(*) from rx_drug_name
union all select 'rx_doze', count(*) from rx_doze
union all select 'rx_duration', count(*) from rx_duration
union all select 'rx_instruction', count(*) from rx_instruction
union all select 'rx_category', count(*) from rx_category
union all select 'rx_type', count(*) from rx_type
union all select 'rx_size', count(*) from rx_size
union all select 'cc_template', count(*) from eh_prescription_cc_template
union all select 'advice_template', count(*) from eh_prescription_advice_template
order by n desc;

\echo '=== 2. Encounter -> prescription linkage ==='
select
  (select count(*) from eh_patient_checkup)                                as checkups,
  (select count(*) from eh_prescription)                                   as prescriptions,
  (select count(*) from eh_prescription p
     join eh_patient_checkup c on c.checkup_id = p.checkup_id)             as rx_valid_checkup,
  (select count(distinct prescription_id) from rx_prescription_drugs)      as rx_with_drugs,
  (select count(distinct prescription_id) from eh_prescription_icd10)      as rx_with_icd10,
  (select count(distinct prescription_id) from eh_prescription_cc)         as rx_with_cc,
  (select count(distinct prescription_id) from eh_prescription_advice)     as rx_with_advice,
  (select count(distinct prescription_id) from eh_prescription_test)       as rx_with_test;

\echo '=== 3. Drugs per prescription ==='
with k as (
  select p.prescription_id, count(d.id) n
  from eh_prescription p
  left join rx_prescription_drugs d on d.prescription_id = p.prescription_id
  group by 1)
select n as n_drugs, count(*) as n_rx,
       round(100.0*count(*)/sum(count(*)) over (),2) as pct
from k group by 1 order by 1;

\echo '=== 3b. Set sizes per prescription ==='
select 'icd10' k, round(avg(n),2) mean, max(n) mx from (select prescription_id, count(*) n from eh_prescription_icd10 group by 1) a
union all select 'cc',     round(avg(n),2), max(n) from (select prescription_id, count(*) n from eh_prescription_cc group by 1) b
union all select 'advice', round(avg(n),2), max(n) from (select prescription_id, count(*) n from eh_prescription_advice group by 1) c
union all select 'test',   round(avg(n),2), max(n) from (select prescription_id, count(*) n from eh_prescription_test group by 1) d;

\echo '=== 4. Output vocabulary sizes (observed) ==='
select 'drug_id' f, count(distinct drug_id) n from rx_prescription_drugs
union all select 'cat_id', count(distinct cat_id) from rx_prescription_drugs
union all select 'type_id', count(distinct type_id) from rx_prescription_drugs
union all select 'drug_size_id', count(distinct drug_size_id) from rx_prescription_drugs
union all select 'doze_id', count(distinct doze_id) from rx_prescription_drugs
union all select 'instruction_id', count(distinct drug_instruction_id) from rx_prescription_drugs
union all select 'duration_id', count(distinct drug_duration_id) from rx_prescription_drugs
union all select 'icd_id', count(distinct icd_id) from eh_prescription_icd10
union all select 'cc_id', count(distinct cc_id) from eh_prescription_cc
union all select 'advice_id', count(distinct advice_id) from eh_prescription_advice
union all select 'test_id', count(distinct test_id) from eh_prescription_test
order by n desc;

\echo '=== 5. Top 25 drugs ==='
select d.drug_id, n.drug_name, count(*) c,
       round(100.0*count(*)/sum(count(*)) over (),2) pct
from rx_prescription_drugs d
left join rx_drug_name n on n.drug_id = d.drug_id
group by 1,2 order by c desc limit 25;

\echo '=== 5b. Label concentration ==='
with f as (select drug_id, count(*) n from rx_prescription_drugs group by 1),
r as (select drug_id, n, row_number() over (order by n desc) rk, sum(n) over () tot from f)
select rk as top_k, round(100.0*sum(n) over (order by rk)/max(tot) over (),2) as cum_pct
from r where rk in (5,10,20,50,100,200,300,400,500) order by rk;

\echo '=== 6. Vitals completeness on prescribed encounters ==='
select count(*) n_enc,
  round(100.0*count(c.bmi)/count(*),1)              pct_bmi,
  round(100.0*count(c.bp_sys)/count(*),1)           pct_bp,
  round(100.0*count(c.blood_glucose)/count(*),1)    pct_glu,
  round(100.0*count(c.temperature)/count(*),1)      pct_temp,
  round(100.0*count(c.pulse_rate)/count(*),1)       pct_pulse,
  round(100.0*count(c.oxygen_of_blood)/count(*),1)  pct_spo2,
  round(100.0*count(c.blood_hemoglobin)/count(*),1) pct_hgb,
  round(100.0*count(c.cholesterol)/count(*),1)      pct_chol,
  round(100.0*count(c.uric_acid)/count(*),1)        pct_uric,
  round(100.0*count(c.waist_hip_ratio)/count(*),1)  pct_whr
from eh_prescription p join eh_patient_checkup c on c.checkup_id = p.checkup_id;

\echo '=== 7. Prescribed encounters per patient (history depth) ==='
with v as (select c.user_id, count(*) n
           from eh_prescription p join eh_patient_checkup c on c.checkup_id=p.checkup_id
           group by 1)
select case when n=1 then '1' when n=2 then '2' when n=3 then '3'
            when n between 4 and 5 then '4-5' when n between 6 and 10 then '6-10'
            else '11+' end as visits,
       count(*) n_patients, sum(n) n_encounters
from v group by 1 order by min(n);

\echo '=== 8. Free-text availability ==='
select count(*) n_rx,
  count(nullif(btrim(coalesce(extra_symptom,'')),'')) n_symptom_txt,
  count(nullif(btrim(coalesce(extra_advice,'')),''))  n_advice_txt,
  (select count(nullif(btrim(coalesce(special_instruction,'')),'')) from rx_prescription_drugs) n_special_instr,
  (select count(nullif(btrim(coalesce(drug_duration,'')),'')) from rx_prescription_drugs) n_dur_txt
from eh_prescription;

\echo '=== 9. Temporal span ==='
select extract(year from c.checkup_date)::int yr, count(*) n
from eh_prescription p join eh_patient_checkup c on c.checkup_id=p.checkup_id
group by 1 order by 1;

\echo '=== 10. Prescriber concentration ==='
with d as (select create_user_id u, count(*) n from eh_prescription group by 1)
select count(*) n_prescribers, max(n) mx, round(avg(n),1) mean,
       percentile_cont(0.5) within group (order by n) med
from d;
