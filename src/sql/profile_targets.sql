set search_path to gramweb_ghealth, public;
\pset pager off

\echo '=== H. Do 0-drug prescriptions still carry advice/tests? ==='
with d as (select p.prescription_id, count(x.id) nd
           from eh_prescription p left join rx_prescription_drugs x on x.prescription_id=p.prescription_id
           group by 1)
select case when nd=0 then 'no drugs' else 'has drugs' end grp, count(*) n_rx,
  count(*) filter (where exists (select 1 from eh_prescription_advice a where a.prescription_id=d.prescription_id)) with_advice,
  count(*) filter (where exists (select 1 from eh_prescription_test t where t.prescription_id=d.prescription_id)) with_test,
  count(*) filter (where exists (select 1 from eh_prescription e where e.prescription_id=d.prescription_id
                                 and btrim(coalesce(e.extra_symptom,''))<>'')) with_symptom_txt
from d group by 1;

\echo '=== I. Repeat visits: drug-set carryover (prev -> next Jaccard) ==='
with e as (
  select c.user_id, p.prescription_id, c.checkup_date,
         row_number() over (partition by c.user_id order by c.checkup_date) rn
  from eh_prescription p join eh_patient_checkup c on c.checkup_id=p.checkup_id),
s as (select e.user_id, e.rn, array_agg(distinct d.drug_id) drugs
      from e join rx_prescription_drugs d on d.prescription_id=e.prescription_id
      group by 1,2)
select count(*) n_pairs,
       round(avg( cardinality(array(select unnest(a.drugs) intersect select unnest(b.drugs)))::numeric
                / nullif(cardinality(array(select unnest(a.drugs) union select unnest(b.drugs))),0) ),3) as mean_jaccard
from s a join s b on a.user_id=b.user_id and b.rn=a.rn+1;

\echo '=== J. drug_duration free text (numeric quantity paired with unit) ==='
select drug_duration, count(*) c from rx_prescription_drugs
where btrim(coalesce(drug_duration,''))<>'' group by 1 order by c desc limit 20;

\echo '=== K. doze canonicalisation: duplicate strings across ids ==='
select count(*) n_doze_ids, count(distinct btrim(doze)) n_distinct_strings from rx_doze;
select btrim(d.doze) doze_norm, count(distinct d.doze_id) n_ids, count(*) n_orders
from rx_prescription_drugs x join rx_doze d on d.doze_id=x.doze_id
group by 1 order by n_orders desc limit 20;

\echo '=== L. Missing/zero foreign keys in drug orders (data quality) ==='
select count(*) n_orders,
  count(*) filter (where drug_id is null or drug_id=0)                       null_drug,
  count(*) filter (where type_id is null or type_id=0)                       null_type,
  count(*) filter (where doze_id is null or doze_id=0)                       null_doze,
  count(*) filter (where drug_instruction_id is null or drug_instruction_id=0) null_instr,
  count(*) filter (where drug_duration_id is null or drug_duration_id=0)     null_durunit,
  count(*) filter (where drug_size_id is null or drug_size_id=0)             null_size,
  count(*) filter (where drug_id not in (select drug_id from rx_drug_name))  orphan_drug
from rx_prescription_drugs;

\echo '=== M. Age / sex availability on prescribed encounters ==='
select count(*) n,
  count(ad.dateofbirth) with_dob,
  count(*) filter (where upper(btrim(coalesce(ad.gender,''))) in ('M','F')) with_sex
from eh_prescription p
join eh_patient_checkup c on c.checkup_id=p.checkup_id
left join a3m_account_details ad on ad.account_id = c.user_id;

\echo '=== N. Site / geography coverage ==='
select count(*) n, count(distinct c.site_id) n_sites,
       count(*) filter (where s.project_id is not null) matched_site,
       count(distinct s.site_district) n_districts
from eh_prescription p
join eh_patient_checkup c on c.checkup_id=p.checkup_id
left join eh_project_site s on s.project_id = c.site_id;

\echo '=== O. Top co-prescribed drug pairs (structure the decoder must learn) ==='
with dd as (select distinct prescription_id, drug_id from rx_prescription_drugs)
select a.drug_id d1, na.drug_name n1, b.drug_id d2, nb.drug_name n2, count(*) c
from dd a join dd b on a.prescription_id=b.prescription_id and a.drug_id<b.drug_id
left join rx_drug_name na on na.drug_id=a.drug_id
left join rx_drug_name nb on nb.drug_id=b.drug_id
group by 1,2,3,4 order by c desc limit 15;
