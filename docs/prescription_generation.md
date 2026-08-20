# PHC-RxGen — Deep Prescription Generation on the Portable Health Clinic Corpus

Task: given a telemedicine encounter (symptom free-text, vitals, demographics,
geography, prior visits), **generate the prescription** — an ordered set of drug
orders with structured attributes, plus the advice and diagnostic-test sets.

## Summary of findings

> **⚠ SUPERSEDED IN PART.** Finding 3 below ("transformer fusion hurts") holds
> only for the **autoregressive decoder** measured in this document. Under a
> multi-label formulation, fusion **helps**: 0.4190 vs 0.4038 (restricted47) and
> 0.4445 vs 0.4186 (norm46). See `docs/neural_mlc.md`. The fusion encoder was
> not inherently harmful — it was harmful in combination with a decoder this
> corpus could not support.

1. **The multi-modal encoder is justified.** Micro-F1 rises 0.059 (demographics
   + geography only) → 0.105 (+ vitals) → 0.192 (+ symptom text), i.e. 3.3× the
   conditional-prior floor and above every non-neural baseline, including the
   prescriber-identity confounder at 0.147.
2. **Symptom free-text carries most of the signal.** Shuffling it across
   patients costs −0.125 micro-F1; vitals −0.024; prior visits −0.005.
   The history GRU is **not** justified by this corpus.
3. **Transformer fusion hurts at this scale.** Removing it gains +0.016 (~4
   s.d.). 9,900 training encounters cannot support the cross-modal encoder.
4. **Forward-in-time performance degrades — how far depends on the target.**
   Brand-level with the decoder: micro-F1 0.066, the frequency-prior floor.
   **Class-level with the deployed linear model: 0.4615 → 0.3547, −23.1%**
   [−26.3, −20.0], well above its floor (`temporal_class_level.md`). A random
   split overstates forward micro-F1 by ~1.3× at class level, ~3× at brand
   level. Set-level agreement collapses below the always-empty floor in both
   cases. Cause: 74% prescriber turnover, cross-era brand Jaccard 0.462.
5. **Brand-level scoring understates clinical agreement** by roughly a third
   (category micro-F1 0.348 vs brand 0.208); the category abstraction is also
   the more temporally stable target (−47% vs −66%).
6. **Whole-prescription reproduction is not achieved.** Exact-set match
   (0.174–0.179) is *below* the always-empty predictor (0.210). The defensible
   framing is retrieval — recall@10 = 0.462 — i.e. a suggestion aid, not an
   autonomous prescriber.

---

## 1. What the data actually supports

Profiling the live `gphc-fix` / `gramweb_ghealth` database before modelling
changed the task design substantially. Findings (`src/sql/profile_*.sql`):

| Signal | Coverage | Verdict |
|---|---|---|
| `eh_prescription` | 14,093 (14,074 with a valid checkup) | cohort |
| Drug orders | 30,239 orders over 11,065 prescriptions | **primary target** |
| Symptom free-text | 12,138 / 14,093 (86%) | **richest input** |
| Vitals | BP 97%, pulse 96%, BMI 92%, glucose 91% | **primary tabular input** |
| Advice set | 4,148 rx, 76 labels | auxiliary target |
| Test set | 2,211 rx, 84 labels | auxiliary target |
| Chief-complaint codes | 1,568 rx (11%) | too sparse to rely on |
| **ICD-10** | **195 rx, 30 codes** | **unusable — dropped** |

Consequences for the design:

1. **ICD-10 cannot be an input or a target.** Most published medication-
   recommendation work (GAMENet, SafeDrug on MIMIC) conditions on diagnosis and
   procedure codes. Here that signal is absent, so the model must work from
   free text and physiology instead. Results are therefore *not* comparable to
   MIMIC numbers, and should not be presented as if they were.
2. **The label space is brand names, not molecules.** 606 of 717 prescribed
   brands map to 90 pharmacological categories — PPI spans 16 brands,
   paracetamol 14, vitamins 33. Choosing "Maxpro" over "Seclo" is a formulary
   decision, not a clinical one. Everything is therefore scored at **both**
   levels, and the category level is the clinically meaningful one.
3. **Zero-drug prescriptions are kept** (3,028 = 21.5%). They carry advice,
   tests and symptom text, so they are genuine "no pharmacotherapy" decisions.
   Dropping them would delete the model's ability to *withhold* treatment, and
   would inflate every metric. `empty_f1` reports this explicitly.
4. **Prescriber identity is a strong confounder.** 67 prescribers; the
   `prescriber_prior` baseline alone reaches micro-F1 0.147. Any claim about
   learned clinical structure must clear that bar.
5. **History is shallow.** 8,567 patients have exactly one visit; only ~430
   have 3+. Prev→next drug-set Jaccard is 0.126, so history is informative but
   far from a copy operation.

### Data defects repaired in `preprocess.py`

- **Triple-encoded durations**: `drug_duration` mixes Bengali digits (`১৫`),
  ASCII (`15`), and mojibake (`à§§` — UTF-8 bytes read as Latin-1). Repaired by
  a guarded re-decode, then digit transliteration.
- **Attribute sparsity**: instruction absent on 25% of orders, size 29%,
  duration-unit 11%. Modelled as an explicit `<na>` class, never dropped.
- **Dose aliasing**: 68 `doze_id`s collapse to 48 distinct strings
  (`200mg` appears twice, `?+?+?` four times). Canonicalised by normalised
  string, not id.
- **Unit drift**: glucose partly mmol/L (×18 → mg/dL), temperature partly °C
  (→ °F); values outside physiological ranges set to missing so the mask bit
  carries the signal rather than a fabricated number.

---

## 2. Architecture

```
                    ┌───────────── symptom free-text ─────────────┐
                    │  char-CNN (k=2..5)  +  word emb  →  BiLSTM  │
                    └──────────────────────┬──────────────────────┘
 vitals (13 + mask) → per-vital tokens ────┤
 demographics       → dense token ─────────┼──→ Transformer fusion encoder
 district / geo     → embedding token ─────┤     (pre-norm, 3 layers)
 prior visits       → GRU over history ────┘              │
                                                          ▼
                              Transformer decoder (causal, cross-attends)
                                                          │
              ┌───────────────┬───────────────┬───────────┴───────────┐
          drug id         category         attributes            advice / test
        (721, tied)      (89 classes)   type·dose·duration·instr    (multi-label)
```

**Why each component is there — not decoration:**

- **Char-CNN.** `extra_symptom` is clinician shorthand: `D/M`, `H/O`, `TAH with
  BLSO`, plus typos (`Loos motion`, `Incresed`, `Rtio`). A word-level lookup
  alone hits OOV constantly; character convolutions degrade gracefully on
  misspellings and share structure across abbreviations. The `no_char_cnn`
  ablation quantifies this.
- **BiLSTM.** Symptom phrases are short (median 37 chars) and order-bearing
  ("pain in the rt. leg for 2 months") — sequential composition over the
  char+word representation.
- **Per-vital tokens.** Vitals enter as one token each (channel embedding gated
  by value + missingness) rather than one dense vector, so fusion attention can
  weight individual physiology against the text.
- **History GRU.** Recurrent pass over up to 4 prior encounters, each carrying
  vitals, a mean-pooled bag of previously prescribed drugs (sharing the decoder
  embedding table), and log time-gap.
- **Transformer fusion + decoder.** Cross-modal fusion over the token sequence,
  then autoregressive decoding of the drug sequence. Drugs are emitted in a
  canonical descending-frequency order: the target is a *set*, and fixing an
  order removes ordering ambiguity that would otherwise be irreducible loss.
- **Hierarchical category head.** Predicts the pharmacological class alongside
  the brand, giving supervision that is robust to brand substitution.

---

## 3. Evaluation

- **Split**: patient-level (default; asserted zero patient straddling) or
  `--split temporal` (train ≤2015 / val 2016 / test ≥2017) for distribution
  shift. All models share one frozen split.
  The temporal split deliberately allows the 107 patients with visits in both
  eras to appear on both sides — it isolates *temporal* shift, not patient
  novelty, and the two splits should be read as answering different questions.
  Vocabularies for the temporal run are refit on the pre-2016 era only, which
  is why its label and attribute vocabularies are smaller.
- **Vocabularies and normalisation statistics are fit on train only.**
- **Metrics**: set Jaccard, micro/macro-F1, exact-set match, precision@k /
  recall@k, per-drug-band recall (head ≥100 / mid 10–99 / tail <10 train
  occurrences), empty-prescription F1, structured-attribute accuracy
  (conditioned on correctly predicted drugs), advice/test micro-F1 + ECE.
- **3 seeds per variant**, reported as mean ± s.d. — mandatory on an 11k-sample
  corpus where seed variance is comparable to ablation effects.

Macro-F1 is reported precisely because it is low: with 719 labels and a tail of
drugs seen fewer than 10 times in training, a micro-average alone would hide
near-total failure on rare drugs.

### Jaccard has a high floor — do not read it as headline accuracy

The modal prescription in this corpus is the **empty set**. Predicting "no
drugs" for every single test encounter already scores **Jaccard 0.210** (and
micro-F1 0.000), because it agrees perfectly on the 21.5% of encounters with no
pharmacotherapy. A model at Jaccard 0.25 is therefore *not* 25% correct.
**Micro-F1 is the honest headline** here; Jaccard is reported alongside its
floor or not at all.

### Input-permutation test

Ablations retrain the model, so a modality can look unnecessary simply because
the network compensates with a correlated one. To test what a *trained* model
actually uses, each modality is shuffled across patients at inference time. If
the metric does not move, that input was not being read.

This is the experiment that distinguishes genuine patient-conditioning from a
well-fit prescribing prior, and on this corpus it changed the conclusion — see
§5.

---

## 4. Reproducing

```bash
conda activate /home/syedu/anaconda3/envs/collective-research
```

```bash
psql -h 127.0.0.1 -U postgres -d gphc-fix -f src/sql/extract.sql
```

```bash
python -m src.phcrx.preprocess
```

```bash
python -m src.phcrx.baselines
```

```bash
python -m src.phcrx.train --all --seeds 0 1 2 --out rxgen_ablations.json
```

```bash
python -m src.phcrx.report
```

```bash
python -m src.phcrx.diagnose --checkpoint models/rxgen_full_patient_seed0.pt
```

```bash
python -m src.phcrx.predict --checkpoint models/rxgen_full_patient_seed0.pt --n 12
```

Temporal-shift run (refits vocabularies on the pre-2016 era, so it writes its
own `*_temporal_*` checkpoints; re-run `preprocess` without `--split` afterwards
to restore the patient-split artifacts):

```bash
python -m src.phcrx.preprocess --split temporal && python -m src.phcrx.train --variant full no_fusion --seeds 0 1 2 --out rxgen_temporal.json && python -m src.phcrx.era_shift
```

> The Postgres server runs on the **Windows host**; `pg_hba.conf` permits
> loopback only, so the extract step is run from Windows and WSL consumes the
> frozen CSV/Parquet. This is also what makes the experiment reproducible —
> the modelling corpus is a fixed snapshot, not a live query.

---

## 5. What the experiments showed

Held-out **patient-level** split, 3 seeds, mean ± s.d. Full tables and figures
in `results/rx_generation/RESULTS.md`.

### The multi-modal encoder earns its place

| Model | Inputs | Micro-F1 |
|---|---|---|
| always-empty (floor) | — | 0.000 |
| global drug prior @5 | — | 0.069 |
| tf-idf kNN on symptom text | text | 0.138 |
| prescriber prior @3 | prescriber id | 0.146 |
| **prior_only** | demo + geo | **0.059 ± 0.003** |
| **tabular_only** | + vitals, history | **0.105 ± 0.006** |
| **full** | + symptom text | **0.192 ± 0.008** |
| **no_fusion** | same, no fusion self-attention | **0.208 ± 0.004** |

Performance rises 0.059 → 0.105 → 0.192 as physiology and then text are added:
**3.3× the conditional-prior floor**, and clear of every non-neural baseline
including the prescriber-identity confounder.

### Symptom text carries most of the signal

The permutation test on the trained full model:

| Shuffled input | Δ micro-F1 |
|---|---|
| symptom text | **−0.125** |
| vitals | −0.024 |
| demographics | −0.018 |
| prior visits | −0.005 (unused) |

Text dominates. Prior visits are effectively unused, which matches the corpus:
8,567 of 10,938 patients have a single visit and prev→next drug-set Jaccard is
only 0.126. **The history GRU is not justified by this data** and should be
dropped unless the corpus grows longitudinally.

Note the ablation and permutation results are consistent but not
interchangeable: `text_only` (0.188) nearly matches `full` (0.192) while
shuffling vitals still costs 0.024. Vitals carry real information that is
largely *redundant* with the symptom text — the network uses them when trained
to, but can recover almost the same performance without them.

### Transformer fusion hurts at this corpus size

> **Correction.** This section's conclusion is decoder-specific. `docs/neural_mlc.md`
> shows that with a multi-label head the same fusion encoder **improves**
> micro-F1 (+0.015 restricted47, +0.026 norm46). Read the finding below as
> "fusion hurts *when paired with an autoregressive decoder at this corpus
> size*", not as a general result.


Removing the fusion self-attention **improves** micro-F1 by 0.016 (~4 s.d.),
and it is the best configuration tested. With 9,900 training encounters, the
3-layer cross-modal encoder overfits; simple concatenation of modality tokens
into the decoder's cross-attention generalises better. This is a negative
result about architecture scale, not about attention in general, and the
practical recommendation for this corpus is the `no_fusion` variant.

### Brand-level scores understate clinical agreement

Category-level micro-F1 is 0.348 against 0.208 at brand level. A concrete case
from `qualitative_examples.md` — symptom *"itching"*: the clinician prescribed
**Alatrol** (cetirizine), the model generated **Fexo** (fexofenadine). Both are
oral antihistamines for the same indication; brand-level Jaccard scores this
0.00. Roughly a third of the apparent error is brand substitution within class.

### No model beats the trivial predictor on exact-set match

Exact whole-prescription match: full 0.174, no_fusion 0.179 — **below the
always-empty predictor at 0.210**. The models over-generate (they emit a
non-empty prescription for 71% of encounters against a true rate of 79%
non-empty, but distribute those drugs across the wrong items). Reproducing an
entire prescription exactly is not achieved here, and should not be claimed.

The practically meaningful framing is retrieval, not generation:
**recall@10 = 0.462** — surfacing ten candidate drugs captures 46% of what the
clinician actually prescribed. That supports a *suggestion* interface, not an
autonomous one.

### Auxiliary heads are precise but insensitive

Advice head: precision 0.601 at recall 0.082. Test head: precision 0.889 at
recall 0.026. At a 0.5 threshold both fire only on the most confident labels.
ECE is very low (0.004 / 0.001) but that mostly reflects the sparsity of the
label matrix, not good discrimination. These need per-label threshold tuning
before they mean anything; they are currently useful as encoder regularisers,
which is why they were included.

### Forward-in-time performance collapses — the headline result

> **⚠ CORRECTED — the −66% figure below is superseded.**
> It was measured at **brand level (719 labels) with the autoregressive neural
> decoder**, both of which are superseded. Re-measured on the deployed linear
> recommender at class level (`docs/temporal_class_level.md`):
>
> | configuration | patient | temporal | drop |
> |---|---|---|---|
> | brand + autoregressive decoder *(the figure below)* | 0.1915 | 0.0657 | −66% |
> | brand + linear | 0.2317 | 0.0990 | −57.2% [−60.7, −53.7] |
> | **class46 + linear (deployed)** | **0.4615** | **0.3547** | **−23.1%** [−26.3, −20.0] |
>
> Roughly **four fifths of the correction comes from the target** (brand → class)
> and one fifth from the estimator. A random split overstates forward-in-time
> micro-F1 by **~1.3×, not ~3×**, and the temporal result stays clearly above the
> frequency-prior floor (+0.1245), where the brand-level number had fallen *to*
> it.
>
> **Two qualifications are load-bearing.** (1) The *feature set* matters more
> than the split: the `all` arm falls twice as far (−48.1%) and lands almost on
> the prior, so feature sets must be selected on a **temporal** validation split,
> not a random one. (2) **Set-level agreement still collapses** for both arms —
> Jaccard and exact-match fall *below* the always-empty floor and tail-macro-F1
> drops −83.7%. Class-level prescribing transfers forward in time **as a ranked
> suggestion list, not as a whole-prescription draft.**


Training on ≤2015 and testing on ≥2017:

| Level | Patient split | Temporal split | Change |
|---|---|---|---|
| Brand (719) | 0.192 | **0.066 ± 0.002** | **−66%** |
| Category (89) | 0.333 | **0.175 ± 0.005** | **−47%** |

At brand level the model falls to the global-frequency-prior baseline (0.069):
**no useful transfer forward in time.** A random patient-level split overstates
deployable performance by roughly 3×.

The cause is measurable non-stationarity in *formulary and staffing*, not in
clinical presentation (`era_shift.json`):

- Only **2 of the top-10** prescribed brands are shared between eras
  (≤2015: Ostocal D, Napa, Maxpro, Ranitid, Ferocit, Neuro B …;
  ≥2017: Calbo D, Zif-CI, Maxpro, Rivotril, Neuro B, Pantid …).
- Drug-vocabulary Jaccard between eras is **0.462**, and **12.2%** of later
  orders are for brands never seen in training — structurally unpredictable.
- **74%** of later encounters were written by a prescriber absent from
  training (33 new prescribers); **58%** occurred at a new site.
- The task itself did not drift: empty-prescription rate 22.1% → 23.4%.

Category-level scores degrade less (−47% vs −66%), confirming that the
pharmacological abstraction is the more stable target — but it does not rescue
transfer either. **Practical implication:** model at category level, expect
frequent retraining, and never quote a random-split number as an expected
deployment figure.

### The long tail is not learned

Recall by training frequency: head (≥100 occurrences) is the only band with
usable recall; tail drugs (<10 occurrences) are essentially never generated,
and macro-F1 sits at ~0.05. With 719 brands over 9,900 training encounters this
is expected, and it is why macro-F1 is reported rather than hidden.

## 6. Honest limitations

- **No causal or outcome signal.** The model learns to imitate what clinicians
  prescribed, not what helped. Agreement with a historical prescription is not
  evidence of clinical correctness, and the corpus contains no outcome variable
  with which to check.
- **Prescriber style is partly what is being modelled.** With 67 prescribers
  and a prescriber-prior baseline at micro-F1 0.147, a share of the signal is
  "who was on duty", not "what the patient needed".
- **Not deployable as a decision aid.** No drug–drug-interaction checking, no
  contraindication model, no dose-by-weight safety logic, no renal/hepatic
  adjustment. Attribute heads predict the *modal recorded* dose string, which
  is not a dosing recommendation.
- **Corpus is small and non-stationary.** 11k prescriptions with drugs; 52% of
  encounters fall in 2012–2014. The temporal-split result (brand micro-F1
  0.066, at the frequency-prior floor) is the realistic estimate for future
  deployment — the patient-split figure of 0.192 is not.
- **Much of the measured performance is era-specific.** With 74% prescriber
  turnover and a drug-vocabulary Jaccard of 0.462 across eras, a large share of
  what the model learns is which formulary and which clinicians were active in
  the training window.
- **Age is year-precision** (many DOBs are Jan-1 placeholders), so paediatric
  dosing logic cannot be learned reliably.
