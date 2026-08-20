# Corpus expansion: the unused MCH / COVID / eye prescription tables

**Question.** The pipeline trains on one corpus — `eh_prescription` joined to
`eh_patient_checkup`, with drug orders from `rx_prescription_drugs`. Six further
prescription corpora sit unused in the same database. How much of that is real
training data?

**Answer.** Three of the six are worth merging (antenatal, postnatal, motherhood):
+2,343 encounters (+16.6%) and +6,093 drug orders (+20.1%). Three are not
(infant, childhood, covid, eye — four tables, since infant and childhood are
separate) and are excluded on measured grounds. Whether the merge *helps* is a
genuinely open question that turns on one thing: the MCH prescribing
distribution is almost disjoint from the adult one (total-variation distance
0.74 vs a 0.084 within-adult baseline), but it is strongly identifiable from
the free-text symptom field. **Merge only with the text branch enabled, and only
under the patient split.** Details and the honest downside are in the last section.

All numbers below were measured against `gphc-fix`, schema `gramweb_ghealth`,
on 2026-08-18. Nothing here is estimated.

---

## 1. How the corpora are wired

Every parent prescription table has the identical column set to `eh_prescription`
(`prescription_id, checkup_id, extra_symptom, extra_advice, ref_hospital_id,
create_user_id, create_date, update_user_id, update_date`), and every drug-order
table has the identical column set to `rx_prescription_drugs`. Each parent joins
to its own clinical table on `checkup_id`:

| corpus | parent rx table | clinical table | drug orders table |
|---|---|---|---|
| antenatal | `mch_prescription` | `mch_antenatal` | `rx_prescription_drugs_antenatal` |
| postnatal | `mch_post_prescription` | `mch_postnatal` | `rx_prescription_drugs_postnatal` |
| motherhood | `mch_motherhood_prescription` | `mch_motherhood` | `rx_prescription_drugs_motherhood` |
| infant | `mch_infant_prescription` | `mch_infant` | `rx_prescription_drugs_infant` |
| childhood | `mch_childhood_prescription` | `mch_childhood` | `rx_prescription_drugs_childhood` |
| covid | `covid_prescription` | `covid_patient_checkup` | `covid_rx_prescription_drugs` |
| eye | `eye_prescription` | *none* | `eye_prescription_drugs` |

`mch_general` is **not** an encounter table — it is one row per mother holding
obstetric background (parity, miscarriages, husband's occupation, LMP/EDD). It
has no `checkup_id` and does not join to a prescription. Not used.

Two structural facts matter for the merge:

- **The dimension tables are shared.** All 6,169 MCH drug orders resolve in
  `rx_drug_name`; `rx_type`, `rx_size`, `rx_doze`, `rx_instruction` and
  `rx_duration` are likewise shared. The label space needs no remapping.
- **The id namespaces are not shared.** `prescription_id` and `checkup_id`
  restart at 1 in every MCH corpus and collide head-on with the adult corpus
  (adult `prescription_id` 4–14,179 vs antenatal 1–1,400; adult `checkup_id`
  12–47,999 vs postnatal 1–962). Naive concatenation silently merges unrelated
  encounters. `user_id` and `site_id`, by contrast, *are* shared keys
  (`a3m_account_details`, `eh_project_site`) and must be left alone.

---

## 2. Per-corpus measurements

Raw counts, straight from the database:

| corpus | parent rx | resolves to a checkup | drug orders | orphan orders | rx with ≥1 drug | distinct drugs | has symptom text | mean symptom len |
|---|---|---|---|---|---|---|---|---|
| adult | 14,093 | 14,074 (99.9%) | 30,326 | 87 | 11,099 | 719 | 12,138 (86%) | 38.9 |
| antenatal | 1,380 | 1,380 (100%) | 4,104 | 50 | 1,328 | 135 | 1,295 (94%) | 49.4 |
| postnatal | 584 | 581 (99.5%) | 1,415 | 21 | 517 | 136 | 551 (94%) | 37.4 |
| motherhood | 384 | 382 (99.5%) | 650 | 5 | 332 | 88 | 378 (98%) | 38.8 |
| infant | 96 | 94 (97.9%) | 31 | 1 | 23 | 12 | 84 (88%) | 33.7 |
| childhood | 10 | 10 (100%) | 10 | 6 | 6 | 10 | 6 (60%) | 17.9 |
| covid | 35 | 35 (100%) | 85 | 0 | 25 | 38 | 35 (100%) | 64.7 |
| eye | 45 | — (no vitals table) | 70 | 0 | 32 | 41 | 9 (20%) | **2.5** |

Vitals coverage, as % non-null on prescribed encounters, for the 13 columns the
pipeline consumes. A dash means the column does not exist in that table:

| corpus | height | weight | bmi | whr | temp | spo2 | bp_sys | bp_dia | glucose | hb | pulse | chol | uric |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adult | 90 | 92 | 92 | 86 | 84 | 85 | 97 | 97 | 91 | 26 | 96 | 10 | 17 |
| antenatal | 96 | 96 | 95 | – | 97 | 96 | 96 | 96 | 96 | **95** | 96 | – | – |
| postnatal | 0 | 92 | 0 | – | 93 | 92 | 92 | 92 | 92 | 91 | 92 | – | – |
| motherhood | 0 | 92 | 0 | – | 93 | 92 | 92 | 92 | 92 | 91 | 92 | – | – |
| infant | 97 | 97 | 66 | – | 97 | 18 | 0 | 0 | 0 | 0 | 0 | – | – |
| childhood | 70 | 70 | 0 | – | 70 | 70 | 0 | 0 | 0 | 0 | 0 | – | – |
| covid | – | – | – | – | **6** | 0 | 0 | 0 | 0 | – | – | – | – |
| eye | – | – | – | – | – | – | – | – | – | – | – | – | – |

Antenatal is the standout: it covers 10 of the 13 vitals at 95–97%, and its
haemoglobin coverage (95%) is nearly four times the adult corpus's (26%).
Structured Rx attributes are also *better* filled than adult — instruction
resolves for 95–96% of MCH orders vs 75% for adult, duration-unit 94–96% vs 89%.
The one regression is free-text duration quantity: 29% for antenatal vs 74% adult.

---

## 3. Decisions

### Merged

**`antenatal`** — 1,380 encounters, 4,054 in-corpus orders, 94% symptom text,
10/13 vitals at ~96%, only 4.9% zero-drug prescriptions. This is the single best
unused corpus in the database and by itself is 78% of the expansion.

**`postnatal`** — 581 encounters, 1,394 orders, 94% symptom text, 8/13 vitals at
~92%. Height and BMI columns exist but are never populated; they pass through as
missing and the pipeline's existing mask bit carries that.

**`motherhood`** — 382 encounters, 645 orders, 98% symptom text, same 8/13 vitals.
The smallest of the three, but structurally identical to postnatal and cheap to
include once postnatal is in.

### Excluded

**`infant` — excluded.** 94 resolved encounters but only **31 drug orders**
total: 74 of 96 prescriptions (77%) order no drug at all. Only 4 of the 13 vitals
exist (height, weight, bmi, temperature), SpO2 is 18% filled, and there is no
blood pressure, glucose, haemoglobin or pulse whatsoever. Worse, the ones that do
exist are *unusable as coded*: `VITAL_RANGES` in `config.py` clips weight to
10–250 kg and height to 60–220 cm, so a healthy 3 kg / 50 cm neonate has every
vital nulled out by the shared cleaning path. Making infant work would require
age-conditional vital ranges — a change to `config.py`/`preprocess.py`, which
this workstream does not own, for 31 orders. Not worth it.

**`childhood` — excluded.** 10 prescriptions, 6 of which order a drug, 10 orders
total. Below the noise floor.

**`covid` — excluded.** The symptom text is excellent (35/35 populated,
64.7 chars mean, the longest of any corpus) but `covid_patient_checkup` shares
only 5 of the 13 pipeline vitals, and **even those are empty**: temperature is
populated in 6% of prescribed rows and SpO2, BP and glucose in 0%. What the table
does carry is a different representation entirely — boolean symptom flags
(`fever`, `cough`, `dyspnea`, `loss_of_taste`) with per-symptom day counts, which
have no counterpart in the adult schema and would have to be encoded as new
features. For 35 encounters and 85 orders (0.25% of the adult corpus), drawn from
a distinct 2020–21 pandemic formulary, that is not a trade worth making.

**`eye` — excluded.** This is exactly the "no symptom text and no overlapping
vitals" case named in the brief. `extra_symptom` is populated in 9 of 45 rows and
averages **2.5 characters** — the actual values are `headache`, `itching`,
`FOLLOW UP OF CATARACT SX` and 36 empty strings. And there is no encounter to
join to: `eye_prescription.checkup_id` has no counterpart table. The eye clinical
tables (`eye_preliminary_test`, `eye_vision_test`, `eye_refraction`,
`eye_final_examination`) are keyed on `preliminary_test_id` / `report_id`, not
`checkup_id`, and hold only ocular findings (`cornea_re`, `anterior_chamber_le`,
`pupil_re`) — zero of the 13 vitals. 45 encounters with no text and no vitals
would be 45 rows of pure label noise. Excluded.

---

## 4. Combined totals

Produced by `python -m src.phcrx.expand.merge_corpora`, after running both
extracts and the shared cleaning in `src/phcrx/preprocess.py`:

| corpus | encounters | patients | orders | rx w/ drugs | zero-drug % | drugs/rx | symptom % | vital cols | vitals/enc | labels | new labels | sites | years |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adult | 14,074 | 10,938 | 30,239 | 11,065 | 21.4 | 2.73 | 86.2 | 13 | 9.6 | 717 | – | 78 | 2012–2025 |
| antenatal | 1,380 | 596 | 4,054 | 1,313 | 4.9 | 3.09 | 93.8 | 10 | 9.6 | 132 | 16 | 3 | 2019–2025 |
| postnatal | 581 | 335 | 1,394 | 508 | 12.6 | 2.74 | 94.3 | 8 | 7.3 | 134 | 13 | 1 | 2019–2025 |
| motherhood | 382 | 273 | 645 | 329 | 13.9 | 1.96 | 98.4 | 8 | 7.4 | 88 | 7 | 1 | 2021–2025 |

| | adult only | + MCH | change |
|---|---|---|---|
| encounters | 14,074 | **16,417** | +2,343 (+16.6%) |
| drug orders | 30,239 | **36,332** | +6,093 (+20.1%) |
| distinct patients | 10,938 | **11,519** | +581 (603 MCH patients, 22 of whom already had an adult prescription) |
| distinct drug labels | 717 | **747** | +30 (drug vocab 721 → 751 with the 4 specials) |
| word vocab (min_freq 2) | 1,666 | **1,797** | +131 |
| patient-history rows | 16,856 | 19,389 | +2,533 |
| advice / test / cc rows | 21,113 / 4,656 / 2,609 | 46,304 / 6,642 / 2,930 | +25,191 / +1,986 / +321 |

The advice tables grow disproportionately (+119%): MCH prescriptions carry a mean
of ~11 advice items each vs ~1.5 for adult. 549 of the 17,160 antenatal advice
rows point at template ids absent from `eh_prescription_advice_template` and
resolve to empty text; test ids and cc ids resolve 100%.

---

## 5. Will this actually help?

Here is the case against, stated first, because it is the stronger half.

**The prescribing distribution is close to disjoint.** Total-variation distance
between the adult and MCH brand distributions is **0.74**. For calibration, the
TV distance between two random halves of the adult corpus is **0.084**. At the
pharmacological-category level it is 0.56. Concretely:

| category | adult % of orders | MCH % | delta |
|---|---|---|---|
| Calcium+Vit+D | 5.0 | 21.9 | **+16.9** |
| Vitamin | 7.0 | 20.7 | **+13.7** |
| Vitamin+Minaral | 7.0 | 17.6 | **+10.6** |
| FERROUS ASCORBATE | 0.04 | 4.3 | +4.2 |
| Oral Hypo Glycenic Drug | 4.5 | 0.0 | −4.5 |
| H2-Blocker | 4.4 | 0.1 | −4.3 |
| Beta Blocker | 3.0 | 0.0 | −3.0 |
| TCA/SSRI | 2.9 | 0.0 | −2.9 |
| ARB/ACEI | 3.1 | 0.2 | −2.9 |

MCH is 60% supplements. The adult corpus's antihypertensives, antidiabetics and
psychotropics are essentially absent from it. Merging shifts the unconditional
prior toward calcium and iron on a test set where that prior is wrong.

**The extra supervision lands mostly where it is least needed.** Binning MCH
orders by the label's frequency *in the adult corpus*:

| adult stratum | MCH orders | % of MCH orders | distinct labels |
|---|---|---|---|
| head (≥100 adult orders) | 4,330 | 71.1 | 54 |
| mid (10–99) | 1,254 | 20.6 | 79 |
| tail (1–9) | 268 | 4.4 | 44 |
| unseen in adult | 241 | 4.0 | 30 |

71% of the new supervision reinforces 54 labels that already have ≥100 examples.
Only 13 adult labels are promoted out of the tail (<10 → ≥10 orders). The
long-tail problem, which is where a prescription-generation model actually loses
micro-F1, is barely touched.

**30 labels are added that can never be correct on an adult test set.** They are
by construction absent from the adult corpus, so on an adult-only evaluation they
are 4% of output-space mass with zero recoverable reward — pure precision risk.

**Two hard constraints, not preferences:**

1. **Temporal split is incompatible.** MCH data runs 2019–2025. With the default
   `DataConfig(temporal_train_end=2015, temporal_val_end=2016)`, *all 2,343 MCH
   rows land in test.* The merge would add zero training signal and instead
   change what the benchmark measures. Under the temporal split, do not merge.
2. **Splits must be recomputed on the merged frame.** `user_id` is a shared
   namespace: 66 of the 603 MCH patients also appear in `eh_patient_checkup`, and
   22 hold prescriptions in both cohorts. Splitting the corpora independently
   puts the same person on both sides of the train/test boundary.

Also note the covariate skew: MCH draws on 3 sites vs 78 for adult, and is 100%
female (2,343/2,343). Any model given a sex feature gets a near-perfect corpus
indicator for free.

### The case for

The shift above would be fatal if it were *unconditional*. It is not — it is
recoverable from the symptom text, which is the pipeline's strongest branch.
MCH symptom text is lexically marked to a degree that makes the two regimes
nearly separable: 303 of the 811 MCH symptom tokens never occur in the adult
text at all. `primigravida` and `lscs` have adult frequency 0; `gravida` has 4.
The top antenatal tokens are `weeks` (1,125), `amenorrhoea` (670), `amenorrhea`
(437), `gravida` (384), `para` (142). A text-conditioned model does not have to
average the two formularies — it can learn P(drug | pregnancy context) as a
distinct mode, and the shared 87.5% of the drug vocabulary means the two regimes
still share representation where they should (paracetamol, PPIs, calcium).

On that reading the merge buys: +20% orders, a haemoglobin channel that is 95%
filled instead of 26%, 131 new vocabulary words, and 13 tail labels promoted —
at the cost of a prior that must be corrected by the text branch.

### Recommendation

**Merge antenatal + postnatal + motherhood, under the patient split, as an
ablation — not as the new default — and gate the decision on stratified metrics.**

- Do **not** merge under the temporal split. It is not a tuning question; the
  data is entirely in the test era.
- Do **not** merge into any `use_text=False` configuration. The whole argument
  for safety is that the text branch separates the regimes; without it the model
  sees a 16.6% injection of encounters whose label distribution is at TV 0.74
  from the target, and the supplement prior will bleed straight into adult
  predictions. Expect that ablation to get worse, and report it if you run it.
- Evaluate on the **adult test split only**, and report head/mid/tail micro-F1
  separately. A gain concentrated in the head is the merge shifting the prior
  toward common supplements, not the model learning more medicine.
- Consider a `corpus` indicator feature or a per-corpus loss weight if the naive
  merge degrades adult tail performance. Both are cheap; neither was tried here.

My honest prior: **roughly even odds, skewed slightly positive on head/mid F1 and
slightly negative on adult tail F1.** The 20% order growth is real and the text
signal is genuinely separable, but 71% of it reinforces already-saturated labels.
If the ablation comes back flat, the correct conclusion is that antenatal is
worth keeping as a *separate* evaluation cohort — it is a clean, well-measured,
1,380-encounter corpus in its own right — rather than as adult training data.

---

## 6. Artifacts and how to reproduce

```bash
# 1. extract (Windows side; Postgres is not reachable from WSL)
psql -h 127.0.0.1 -U postgres -d gphc-fix -f src/sql/extract_mch.sql

# 2. merge + report (WSL side; Python lives there)
python -m src.phcrx.expand.merge_corpora                    # report only
python -m src.phcrx.expand.merge_corpora --write --parquet --json
python -m src.phcrx.expand.merge_corpora --parquet --freeze-adult-split
```

`src/sql/extract_mch.sql` writes to `data/interim/`, each file in the same column
shape as its adult counterpart plus a trailing `corpus` column:
`encounters_mch.csv` (2,343), `rx_orders_mch.csv` (6,093), `rx_advice_mch.csv`
(25,191), `rx_tests_mch.csv` (1,986), `rx_cc_mch.csv` (321),
`patient_history_mch.csv` (2,533).

`src/phcrx/expand/merge_corpora.py` imports `build_encounters`, `build_orders`
and `bucket_duration` from `src/phcrx/preprocess.py` — no cleaning logic is
duplicated, so the mojibake repair, Bengali-digit translation, mmol/L and Celsius
harmonisation and `VITAL_RANGES` clipping apply identically to MCH rows. It
applies a disjoint 10M-wide id offset per corpus (`CORPUS_OFFSET`), runs the ten
id-namespace assertions in `verify_ids()` on every invocation, and with `--write`
emits `data/interim/*_all.csv`. `--parquet` emits
`data/processed/rxgen_{encounters,orders}_expanded.parquet`; `--json` writes
`results/rx_generation/corpus_merge.json`.

Nothing in `preprocess.py`, `data.py`, `model.py` or `train.py` was modified, and
no model was retrained.

---

## 7. Execution results

Everything above was analysis. This section is the record of actually running it,
on 2026-08-19, against the same `gphc-fix` / `gramweb_ghealth` database. The
extract was re-run from scratch and produced byte-identical CSVs to the earlier
run (2,343 / 6,093 / 25,191 / 1,986 / 321 / 2,533 rows), so the numbers in §2–§5
are reproducible rather than merely recorded.

### 7.1 The id collision, verified

This was the one thing that could silently destroy the corpus, so it is now
checked by `verify_ids()` on every invocation, which **raises** rather than warns.
Ten checks, all passing:

| # | check | result |
|---|---|---|
| 1 | the collision is real in the source data | **1,390** raw `prescription_id`s occur in ≥2 corpora; a naive concat would fuse **2,322 of the 2,343** MCH encounters (99.1%) onto an unrelated adult encounter |
| 2 | merged `prescription_id` unique | 16,417 rows, 16,417 distinct ids, **0 duplicated** |
| 3 | merged id ↔ source row is a bijection | rows = 16,417 = distinct `(corpus, source_id)` pairs = source rows |
| 4 | inverse map is total and exact | every corpus's recovered source ids equal its CSV's id set exactly |
| 5 | id blocks disjoint | adult 5–14,179 · antenatal 10,000,001–10,001,400 · postnatal 20,000,001–20,000,594 · motherhood 30,000,001–30,000,392; max raw id 14,179 ≪ the 10M block width |
| 6 | `checkup_id` namespaced | **767** raw `checkup_id`s collide pre-offset; post-offset **0** merged `checkup_id` spans more than one corpus |
| 7 | no child row re-parented | orders 0, advice 0, tests 0, cc 0, history 0 rows attached to a different corpus |
| 8 | order resolution unchanged | 36,299 of 36,332 orders find a parent — identical to an offset-free, per-corpus join |
| 9 | adult slice == adult-only build | 14,074 == 14,074 encounters, 30,239 == 30,239 orders, all 50 encounter and 22 order columns **element-wise equal** (including list-valued `symptom_tokens`) |
| 10 | shipped benchmark untouched | adult `prescription_id` set identical to `rxgen_encounters.parquet` (14,074 rows) |

Check 1 is the important one: it measures the damage the offset prevents. It is
not a marginal 1,390-row problem — **99.1% of MCH encounters would have been
fused onto an adult encounter** by a naive `pd.concat`, because both id spaces
are dense from 1.

Check 9 is the "adult counts unchanged" assertion, done as a frame comparison
rather than a row count: adult encounters 14,074 → 14,074, adult orders
30,239 → 30,239, adult distinct patients 10,938 → 10,938, and every cell equal.
The 33 orders that do not resolve to a parent (check 8) are the 26 adult orphans
already present in the shipped pipeline plus 7 postnatal ones — the merge adds no
new orphaning.

Two small corrections to §1 from the measured ranges: the adult
`prescription_id` range *in `encounters.csv`* is 5–14,179, not 4–14,179
(prescription 4 exists in `eh_prescription` but its checkup does not resolve, so
it is one of the 19 dropped rows), and postnatal `prescription_id` runs 1–594
(1–962 in §1 is the postnatal `checkup_id` range).

### 7.2 Combined totals

Measured, and identical to the projections in §4:

| | adult only | + MCH | change |
|---|---|---|---|
| encounters | 14,074 | **16,417** | +2,343 (+16.6%) |
| drug orders | 30,239 | **36,332** | +6,093 (+20.1%) |
| distinct patients | 10,938 | **11,519** | +581 (603 MCH patients, 22 already prescribed in adult) |
| distinct drug labels | 717 | **747** | +30 |
| drug vocab (with the 4 specials) | 721 | **751** | +4.2% output space |
| word vocab (min_freq 2, whole corpus) | 1,666 | **1,797** | +131 |
| history rows | 16,856 | 19,389 | +2,533 |

### 7.3 Drugs that exist only in MCH

**30 labels**, carrying 241 orders — 4.0% of MCH orders, 0.7% of all merged
orders. On an adult-only test set these are 4.2% of output-space mass with zero
recoverable reward. They are also thin: **17 of the 30 are ordered exactly once**
in the entire MCH corpus. The only ones with real mass are FullCare (66),
Ferisen (52), D sefa (41) and Alneed Plus (34) — 193 of the 241 orders between
them, all iron/calcium/multivitamin supplements. Under the merged patient split,
25 of the 30 land in train (26 under the frozen split); the rest are in val/test
and are unlearnable as well as unrewardable.

### 7.4 Symptom-text coverage

| corpus | encounters | with text | text % | text % (rx with ≥1 drug) | mean chars | mean tokens \| text | token vocab |
|---|---|---|---|---|---|---|---|
| adult | 14,074 | 12,129 | 86.2 | 90.1 | 38.4 | 7.53 | 3,443 |
| antenatal | 1,380 | 1,295 | 93.8 | 94.1 | 49.3 | 9.13 | 596 |
| postnatal | 581 | 548 | 94.3 | 95.2 | 37.3 | 8.18 | 373 |
| motherhood | 382 | 376 | 98.4 | 98.5 | 38.6 | 8.02 | 254 |
| MCH (all 3) | 2,343 | 2,219 | **94.7** | 95.0 | 44.6 | 8.70 | 811 |
| **merged** | 16,417 | 14,348 | **87.4** | 90.9 | 39.3 | 7.71 | 3,746 |

The merge *raises* text coverage (86.2% → 87.4%) and lengthens the mean symptom
string. **303 of the 811 MCH symptom tokens (37.4%) never occur in the adult
text**, confirming §5's separability argument on the merged frame.

### 7.5 Head/mid/tail, before vs after

Strata are the ones `preprocess.main()` uses, computed on train-split frequency.
Both columns below use the *same* split, so the delta is the MCH rows alone and
not a reshuffle artefact. "before" is that split's adult train orders; "after"
adds the MCH train orders.

| stratum | labels before | train orders before | labels after | train orders after | Δ labels |
|---|---|---|---|---|---|
| head (≥100) | 58 | 11,987 | 65 | 15,642 | +7 |
| mid (10–99) | 241 | 8,045 | 248 | 8,469 | +7 |
| tail (1–9) | 375 | 1,159 | 391 | 1,211 | +16 |
| unseen (0) | 43 | 0 | 43 | 0 | 0 |
| **total** | **717** | **21,191** | **747** | **25,322** | **+30** |

This is the sharpest negative result in the whole workstream. The merge adds
**4,131 training orders, and 3,655 of them (88.5%) land in the head stratum.**
The tail gains **52 orders across 391 labels**. Only **21 of the 717 adult labels
change stratum at all** (2.9%): 6 mid→head, 1 tail→head, 9 tail→mid, 5
unseen→tail. Under the frozen-split variant the picture is the same: +4,314 train
orders, 3,687 (85.5%) into the head, 21 adult labels moved.

§5 estimated 71% of MCH supervision reinforcing already-saturated labels by
binning on *adult* frequency. Measured on the actual train split with post-merge
strata, it is worse than that: **88.5% of the new supervision mass goes to labels
that already have ≥100 training examples.**

### 7.6 A split hazard that §5 did not anticipate

`preprocess.main()` shuffles the whole patient pool. Adding 603 MCH patients
therefore **re-shuffles the adult ones too**: 3,942 of 14,074 adult encounters
(28.0%) change split, and **only 1,377 of the 2,780 shipped adult test
encounters are still in test.** A merged-vs-unmerged comparison run that way is
confounded — half the adult test set is different, so the measured delta mixes
"did MCH help" with "is this a different test sample".

`--freeze-adult-split` fixes it: every patient already assigned in
`rxgen_encounters.parquet` keeps that assignment (including the 22 who hold
prescriptions in both cohorts, which is what keeps them from leaking), and only
the MCH-only patients are drawn. Measured: 0/14,074 adult encounters change
split, 2,780/2,780 shipped adult test rows preserved, 0 patients straddling.

| split mode | adult train/val/test | MCH train/val/test | adult drift |
|---|---|---|---|
| recomputed (default) | 9,887 / 1,401 / 2,786 | 1,598 / 235 / 510 | 28.0% of rows move |
| `--freeze-adult-split` | 9,900 / 1,394 / 2,780 | 1,657 / 261 / 425 | 0.0% |

**Any head-to-head ablation must use `--freeze-adult-split`.**

### 7.7 Artifacts produced

- `data/interim/*_mch.csv` — six files, re-extracted and byte-identical to the prior run.
- `data/interim/*_all.csv` — six merged files, adult + MCH, ids namespaced, `corpus` column appended.
- `data/processed/rxgen_encounters_expanded.parquet` — 16,417 rows × 29 cols.
- `data/processed/rxgen_orders_expanded.parquet` — 36,332 rows × 10 cols.
- `results/rx_generation/corpus_merge.json` — the full report including the ten id checks.

Both parquets carry **exactly** the shipped `rxgen_*.parquet` column list, in the
same order, with the same Arrow types (asserted at write time; dtype drift
reported as `none`), plus a trailing `corpus` column ∈ {adult, antenatal,
postnatal, motherhood}. The shipped `rxgen_encounters.parquet` and
`rxgen_orders.parquet` were not written to.

### 7.8 Recommendation

**Do not adopt the merge as the default. Run it once as a gated ablation; adopt
only if it clears the gate.** The execution did not change the §5 verdict — it
moved it slightly more negative.

What the numbers now say, plainly:

- The merge is **structurally sound**. All ten id checks pass, the adult corpus
  is provably bit-identical after merging, and the cleaning path is shared. There
  is no correctness reason to avoid it.
- The merge is **unlikely to help where the model actually loses**. 88.5% of the
  new supervision reinforces labels with ≥100 examples; 2.9% of adult labels
  change stratum; the tail gains 52 orders. A +20.1% order count that is 88.5%
  head-directed is not a 20% improvement in effective supervision.
- The merge has a **real, quantified cost**: the output space grows 4.2% with
  labels that an adult test set can never reward, on top of a prescribing
  distribution at TV 0.74 from the target.

Merge **only** under all four conditions:

1. **Patient split, `--freeze-adult-split`.** Not the temporal split — all 2,343
   MCH rows are 2019+ and land in test, contributing zero training signal. Not
   the default recomputed split either, or 28% of the adult rows move and the
   comparison is confounded.
2. **`use_text=True`.** The separability argument is the entire safety case, and
   §7.4 confirms it holds on the merged frame (37.4% of MCH tokens unseen in
   adult). A `use_text=False` merged run should be expected to get worse; report
   it if run.
3. **Evaluate on the adult test split only**, with head/mid/tail micro-F1
   reported separately. A gain that is head-only is the supplement prior
   shifting, not the model learning more medicine.
4. **Gate on the tail.** Adopt only if adult tail micro-F1 is non-inferior and
   head/mid improves. If adult tail degrades, try the `corpus` indicator feature
   or a per-corpus loss weight (the `corpus` column is in both parquets for
   exactly this) before concluding the merge is unusable.

My prior after execution: **slightly negative overall — likely a small head/mid
gain and a flat-to-negative tail and macro-F1.** If the ablation comes back flat,
the correct conclusion is unchanged from §5: antenatal is a clean, well-measured,
1,380-encounter corpus with 93.8% symptom coverage and 95%-filled haemoglobin,
and is worth more as a *separate evaluation cohort* — a genuine
distribution-shift test set the pipeline currently lacks — than as diluted adult
training data.
