set search_path to gramweb_ghealth, public;
\pset pager off

\echo '=== A. Sample extra_symptom (input free text) ==='
select prescription_id, left(regexp_replace(extra_symptom, '\s+', ' ', 'g'), 140) as symptom
from eh_prescription
where btrim(coalesce(extra_symptom,'')) <> ''
order by prescription_id limit 25;

\echo '=== B. Symptom text length distribution ==='
with t as (select length(btrim(extra_symptom)) L from eh_prescription
           where btrim(coalesce(extra_symptom,'')) <> '')
select count(*) n, min(L) mn, round(avg(L)) mean,
       percentile_cont(0.5) within group (order by L) p50,
       percentile_cont(0.9) within group (order by L) p90,
       percentile_cont(0.99) within group (order by L) p99, max(L) mx
from t;

\echo '=== C. Script/markup checks on symptom text ==='
select
  count(*) n,
  count(*) filter (where extra_symptom ~ '[ঀ-৿]') as has_bengali,
  count(*) filter (where extra_symptom ~ '[A-Za-z]')        as has_latin,
  count(*) filter (where extra_symptom ~ '<[a-zA-Z/]')      as has_html,
  count(*) filter (where extra_symptom ~ '&[a-z]+;')        as has_entity
from eh_prescription where btrim(coalesce(extra_symptom,'')) <> '';

\echo '=== D. Sample chief-complaint template vocabulary ==='
select id, cc_name, cc_type, cc_parent from eh_prescription_cc_template order by id limit 40;

\echo '=== E. Dose / duration / instruction vocabularies (structured Rx attributes) ==='
select 'doze' src, doze_id::text id, doze val from rx_doze order by doze_id limit 50;
\echo '--- duration ---'
select drug_duration_id, drug_duration from rx_duration order by 1;
\echo '--- instruction ---'
select drug_instruction_id, drug_instruction from rx_instruction order by 1;
\echo '--- type ---'
select type_id, type_name from rx_type order by 1;

\echo '=== F. Corrected drug label concentration ==='
with f as (select drug_id, count(*) n from rx_prescription_drugs group by 1),
r as (select drug_id, n, row_number() over (order by n desc) rk,
             sum(n) over (order by n desc rows between unbounded preceding and current row) cum,
             sum(n) over () tot from f)
select rk as top_k, round(100.0*cum/tot,2) as cum_pct_of_orders
from r where rk in (5,10,20,50,100,150,200,300,400,500,600,719) order by rk;

\echo '=== G. Advice + test vocabularies (secondary targets) ==='
select id, left(coalesce(advice_en,''),70) advice_en from eh_prescription_advice_template order by id limit 20;
\echo '--- top tests ---'
select t.test_id, tt.test_name, count(*) c
from eh_prescription_test t left join eh_prescription_test_template tt on tt.id=t.test_id
group by 1,2 order by c desc limit 15;

\echo '=== H. Do prescriptions with 0 drugs still carry advice/tests? ==='
with d as (select p.prescription_id, count(x.id) nd
           from eh_prescription p left join rx_prescription_drugs x on x.prescription_id=p.prescription_id
           group by 1)
select case when nd=0 then 'no drugs' else 'has drugs' end grp, count(*) n_rx,
  count(*) filter (where exists (select 1 from eh_prescription_advice a where a.prescription_id=d.prescription_id)) with_advice,
  count(*) filter (where exists (select 1 from eh_prescription_test t where t.prescription_id=d.prescription_id)) with_test
from d group by 1;

\echo '=== I. Repeat-visit patients: is the previous Rx predictive (drug overlap)? ==='
with e as (
  select c.user_id, p.prescription_id, c.checkup_date,
         row_number() over (partition by c.user_id order by c.checkup_date) rn
  from eh_prescription p join eh_patient_checkup c on c.checkup_id=p.checkup_id),
s as (select e.user_id, e.rn, array_agg(distinct d.drug_id) drugs
      from e join rx_prescription_drugs d on d.prescription_id=e.prescription_id
      group by 1,2)
select count(*) n_pairs,
       round(avg(cardinality(array(select unnest(a.drugs) intersect select unnest(b.drugs)))::numeric
             / nullif(cardinality(array(select unnest(a.drugs) union select unnest(b.drugs))),0)),3) as mean_jaccard_prev_vs_next
from s a join s b on a.user_id=b.user_id and b.rn=a.rn+1;
