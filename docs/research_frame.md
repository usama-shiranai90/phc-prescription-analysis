# Research frame — PHC-RxGen

What is being implemented, what is being tested, what the evidence supports,
and — equally important — what it does not.

---

## 1. Problem statement

The Portable Health Clinic (PHC) programme runs unattended screening camps in
rural Bangladesh. A health worker records vitals and a short free-text symptom
note; a remote physician then writes a prescription. Physician time is the
binding constraint on how many people the programme can serve.

The obvious response is decision support: learn to generate the prescription
from the encounter. But the data that such a system must learn from has three
properties that most published medication-recommendation work does not face:

1. **No diagnosis codes.** 195 of 14,074 encounters carry an ICD code, and most
   of those are not diagnoses (`Y440` "Iron preparations", `R030` "elevated
   blood-pressure reading"). Work on MIMIC (GAMENet, SafeDrug, and successors)
   conditions on diagnosis and procedure codes. That input does not exist here.
2. **The label space is commercial brands, not molecules.** 719 marketed
   products, of which 606 collapse to 90 pharmacological classes — 16 PPI
   brands, 14 paracetamol brands. Brand choice is largely formulary, not
   clinical.
3. **The corpus is non-stationary.** Across 2012–2025 the formulary and the
   clinical staff turn over almost completely.

So the research problem is not "build a prescription generator." It is:

> **Under what conditions, and to what granularity, is clinician prescribing
> behaviour learnable from low-resource community telemedicine records — and
> what evaluation protocol is required to establish that claim honestly?**

The second clause is not decoration. Over the course of this work, five
separate measurements initially produced *plausible but wrong* numbers, four of
which flattered the proposed model. A study of this kind is as much a
measurement-methodology problem as a modelling problem.

---

## 2. Objective

1. Build a reproducible pipeline from the live PHC database to a trained
   prescription-generation model, with all data defects documented and repaired.
2. Establish whether encounter context predicts prescribing above the relevant
   null models, and attribute the predictive signal to specific inputs.
3. Establish the deployment-realistic performance ceiling (forward in time),
   not the split-flattering one.
4. Determine the granularity — brand vs pharmacological class — at which
   prescribing is predictable and temporally stable.
5. Determine whether a purpose-trained model is warranted, given that prompted
   medical LLMs are freely available.
6. Deliver an evaluation protocol that detects the failure modes encountered.

---

## 3. Research questions

| | Question |
|---|---|
| **RQ1** | Does multimodal encounter context (symptom text, vitals, demographics, geography, visit history) predict the prescribed drug set better than prescribing priors — including the prescriber-identity confounder? |
| **RQ2** | Which input modalities actually carry the signal, and does the model use them at inference or merely have access to them? |
| **RQ3** | Does predictive performance transfer forward in time, and if not, what is the mechanism? |
| **RQ4** | At what label granularity — brand or pharmacological class — is prescribing predictable and temporally stable? |
| **RQ5** | Does a purpose-trained model outperform a prompted general/medical LLM constrained to the same formulary? |
| **RQ6** | Can the missing diagnosis layer be reconstructed from free text well enough to be usable, and how would we know? |

---

## 4. Hypotheses and verdicts

Stated in advance of the experiment where possible; verdicts are the measured
outcome, including where the hypothesis was wrong.

| | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| **H1** | Multimodal deep encoding beats prescribing priors | **Supported** | micro-F1 0.059 (demographics+geo) → 0.105 (+vitals) → 0.192 (+text); prescriber prior 0.147, tf-idf kNN 0.138, global prior 0.069 |
| **H2** | Each architectural component contributes | **Rejected in part, then partly reinstated** | With the autoregressive decoder, removing fusion *improves* micro-F1 by +0.016 (~4 s.d.); under a multi-label head fusion **helps** (+0.015 to +0.026, `neural_mlc.md`). The history GRU is noise either way (+0.0002 ablation, −0.004 permutation). |
| **H3** | The model conditions on the patient, not a prescribing prior | **Supported** | Permutation: shuffling symptom text costs −0.125 micro-F1 (−66% relative); shuffling all inputs → 0.032 |
| **H4** | Performance is stable across time | **Rejected, but less severely than first measured** | Brand-level with the decoder: 0.192 → 0.066, at the prior floor. Class-level with the deployed linear model: **0.4615 → 0.3547, −23.1%** [−26.3, −20.0], clearly above its 0.2303 floor. Set-level agreement still falls below the always-empty floor. `temporal_class_level.md` |
| **H5** | Pharmacological class is more predictable and more temporally stable than brand | **Supported** | Category micro-F1 0.309 vs brand 0.187; temporal degradation −47% vs −66% |
| **H6** | Whole-prescription reproduction is achievable | **Rejected** | Exact-set match 0.174–0.179, *below* the always-empty predictor at 0.210 |
| **H7** | A prompted LLM is competitive | **Rejected on accuracy, narrowly** | gpt-oss 20B (12-shot) reaches 0.112 micro-F1 / 0.252 category vs PHC-RxGen 0.187 / 0.309 — a 1.7×/1.23× gap, not a rout. medgemma 4B is far weaker (0.050) and cannot calibrate volume at all |
| **H8** | Missing ICD codes can be reconstructed from symptom text | **Partially supported** | 31% of notes coded at usable confidence; the remaining 69% withheld rather than guessed |

---

## 5. What is actually implemented

### 5.1 Data layer
- Frozen extract from live Postgres (`gramweb_ghealth`) → CSV → Parquet, so the
  modelling corpus is a fixed, reproducible snapshot.
- Documented repairs: triple-encoded duration fields (Bengali digits, ASCII,
  and Latin-1 mojibake `à§§`), dose aliasing (68 ids → 48 canonical strings),
  unit drift (mmol/L glucose, °C temperature), physiologically impossible
  values → missing with an explicit mask bit.
- Split computed once and shared by every model; patient-level split asserts
  zero patient straddling.

### 5.2 Model — PHC-RxGen

```
              ┌──────────── symptom free-text ────────────┐
              │  char-CNN (k=2..5) + word emb → BiLSTM    │   ← CNN + RNN
              └────────────────────┬─────────────────────-┘
 vitals (13 + mask) → one token per vital ──┤
 demographics       → dense token ──────────┼─→ Transformer fusion encoder
 district / geo     → embedding token ──────┤     (pre-norm, 3 layers)
 prior visits       → GRU over history ─────┘              │      ← RNN
                                                           ▼
                        Transformer decoder (causal, cross-attends to context)
                                                           │
        ┌────────────────┬────────────────┬────────────────┴──────────┐
    drug id          category         attributes                advice / test
   (721, tied)      (89 classes)  type·dose·duration·instr      (multi-label)
```

Design rationale, each tied to a measured property of the data:

- **char-CNN** — the symptom field is clinician shorthand with abbreviations
  (`D/M`, `H/O`) and typos (`Loos motion`, `Incresed`). Word lookup alone hits
  OOV constantly.
- **BiLSTM** — short order-bearing phrases (median 37 characters).
- **Per-vital tokens** — lets fusion attention weight individual physiology
  against the text, rather than compressing vitals into one dense vector.
- **History GRU** — prior encounters with a mean-pooled bag of previously
  prescribed drugs, sharing the decoder's embedding table.
- **Hierarchical category head** — supervises the clinically meaningful level,
  robust to brand substitution.
- **Canonical descending-frequency drug ordering** — the target is a *set*;
  fixing an order removes ordering ambiguity that would be irreducible loss.

### 5.3 Clinical NLP layer
- Verified site glossary (46 entries), checked against physiology rather than
  asserted.
- ICD-10 reconstruction: CMS ICD-10-CM reference (Hugging Face) + SapBERT
  entity linking + character n-gram TF-IDF, with tiered confidence gating and
  local-LLM adjudication as the authoritative coder.
- WHO NCD risk stratification with Asian-specific BMI cut-offs, benchmarked
  against the WHO GHO API.

---

## 6. What is actually tested — the experimental apparatus

This is the part that constitutes the methodological contribution.

| Instrument | What it detects | What it caught here |
|---|---|---|
| **Null-model floors** (always-empty, global prior, prescriber prior) | Metrics that look good but beat nothing | Jaccard has a **0.210 floor**; exact-match 0.210 — no model beats it |
| **Ablation grid** (10 variants × 3 seeds) | Components that do not earn their place | Transformer fusion *hurts* **with the decoder** (reversed under multi-label framing); seed s.d. is comparable to ablation effects |
| **Input-permutation test** | Whether a trained model *uses* an input, vs merely having access | Text carries the signal (−0.125); history is unused (−0.005) |
| **Temporal split + shift attribution** | Split-flattered performance | −66%; caused by 74% prescriber turnover, 0.462 vocabulary Jaccard |
| **Stratified reporting** (head/mid/tail) | Long-tail failure hidden by micro-averaging | Tail recall ≈ 0.006; macro-F1 ≈ 0.05 |
| **Two-coder agreement** (retrieval vs LLM) | Automated-coder consistency where no ground truth exists | 57% exact / 91% chapter; retrieval has a systematic lexical bias |
| **LLM behaviour auditing** (mean prediction size, refusal rate, constraint violations, parse failures) | LLM baselines that measure the prompt or the harness | 4 of 5 LLM runs measured my own prompt/client, not the model |

### The measurement failures this apparatus caught

Recorded because they are the transferable result, and because each initially
produced a *plausible* number:

1. **Ablation gate bug.** `tabular_only` left word embeddings active → "text is
   unnecessary" (0.194). With text truly removed: 0.105. Opposite conclusion.
2. **Checkpoint collision.** Temporal run overwrote patient-split checkpoints
   with incompatible vocabularies. Fixed with split-qualified names + a hard
   size-mismatch guard.
3. **Unbounded retrieval.** ICD retrieval assigned `I96 Gangrene` to notes
   reading `"NO"` — 192 patients. Fixed with no-complaint detection and
   confidence tiering.
4. **Lexical bias.** `I1A "Other hypertension"` outranked `I10 "Essential
   hypertension"` on the largest category in the corpus. Excluding I1A merely
   relocated the bias to `I15`. Structural, not fixable by curation.
5. **Prompt-induced refusal.** A system prompt instructing restraint produced
   96% non-prescribing — including a glucose-410 mg/dL encounter the model
   itself described as diabetic. Measuring refusal the prompt induced.
6. **Silent harness failure.** gpt-oss returned empty responses under Ollama's
   structured-output mode (reasoning models emit to a separate field); the
   client scored this as "deliberately prescribed nothing" 150 times — a clean
   0.000 indistinguishable from a clinical decision.

---

## 7. Final results

All figures are held-out test performance. Model rows are mean ± s.d. over
3 seeds on the patient-level split unless stated. Sources:
`results/rx_generation/RESULTS.md` and the JSON files beside it.

### 7.1 Headline — the ladder from prior to full model

| System | Params | Micro-F1 | Category F1 | Jaccard | Exact | Empty-Rx F1 |
|---|---|---|---|---|---|---|
| always-empty (floor) | — | 0.0000 | 0.0000 | **0.2104** | **0.2104** | — |
| global drug prior @5 | — | 0.0688 | 0.2047 | 0.0355 | 0.0000 | — |
| tf-idf kNN on symptom text | — | 0.1376 | 0.2259 | 0.2259 | 0.1723 | — |
| prescriber prior @3 *(confounder)* | — | 0.1461 | 0.2647 | 0.0818 | 0.0014 | — |
| **prior_only** (demo + geo) | 4.23M | 0.0585 ± 0.0034 | 0.1260 | 0.1915 | 0.1729 | 0.4175 |
| **tabular_only** (+ vitals, history) | 4.43M | 0.1050 ± 0.0058 | 0.1927 | 0.2057 | 0.1682 | 0.4534 |
| **full** (+ symptom text) | 5.09M | 0.1915 ± 0.0082 | 0.3328 | 0.2527 | 0.1740 | 0.5941 |
| **no_fusion** *(best)* | 3.50M | **0.2076 ± 0.0041** | **0.3477** | 0.2643 | 0.1787 | **0.6139** |

Adding physiology then free text lifts micro-F1 0.059 → 0.105 → 0.192, i.e.
**3.3× the conditional-prior floor** and clear of the prescriber-identity
confounder (0.146). Note that Jaccard and exact-match must be read against the
always-empty floor of 0.210 — **no configuration beats it on exact match.**

### 7.2 Ablations — what each component is worth

| Variant | Params | Micro-F1 | Δ vs full | Category F1 |
|---|---|---|---|---|
| no_fusion | 3.50M | 0.2076 ± 0.0041 | **+0.0161** | 0.3477 |
| no_text_rnn | 4.69M | 0.1948 ± 0.0006 | +0.0033 | 0.3310 |
| gru_decoder | 3.90M | 0.1930 ± 0.0061 | +0.0015 | 0.3398 |
| **full** | 5.09M | 0.1915 ± 0.0082 | — | 0.3328 |
| no_history | 4.89M | 0.1913 ± 0.0038 | −0.0002 | 0.3313 |
| text_only | 4.89M | 0.1882 ± 0.0024 | −0.0033 | 0.3160 |
| no_vitals | 5.08M | 0.1853 ± 0.0010 | −0.0062 | 0.3141 |
| no_char_cnn | 4.94M | 0.1848 ± 0.0022 | −0.0067 | 0.3260 |
| tabular_only | 4.43M | 0.1050 ± 0.0058 | −0.0865 | 0.1927 |
| prior_only | 4.23M | 0.0585 ± 0.0034 | −0.1330 | 0.1260 |

Two components fail to justify themselves: **transformer fusion costs 0.016
micro-F1** (~4 s.d.), and the **history GRU is worth 0.0002** — indistinguishable
from noise. The char-CNN and vitals each contribute ~0.006, small but consistent
in sign across seeds.

### 7.3 Does the model use its inputs? (permutation on the trained model)

Intact micro-F1 **0.1905**.

| Shuffled across patients | Micro-F1 | Δ | Uses it? |
|---|---|---|---|
| symptom text | 0.0650 | **−0.1255** | yes, dominant |
| vitals | 0.1652 | −0.0253 | yes |
| demographics | 0.1715 | −0.0190 | yes |
| prior visits | 0.1861 | −0.0044 | **no** |
| all inputs | 0.0322 | −0.1583 | — |

Ablation and permutation agree on history (unused) and disagree productively on
vitals: `text_only` loses only 0.003, yet shuffling vitals costs 0.025. Vitals
carry real signal that is **largely redundant with the text**.

### 7.4 Temporal shift — the deployment-relevant number

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
> than the split: the `all` arm falls further (−37.6% era-clean) and closer to
> the prior, so feature sets must be selected on a **temporal** validation split,
> not a random one. (2) **Set-level agreement still collapses** for both arms —
> Jaccard and exact-match fall *below* the always-empty floor and tail-macro-F1
> drops −83.7%. Class-level prescribing transfers forward in time **as a ranked
> suggestion list, not as a whole-prescription draft.**

Train ≤2015 / test ≥2017:

| Level | Patient split | Temporal | Change |
|---|---|---|---|
| Brand (719) | 0.1915 | **0.0657 ± 0.0016** | **−66%** |
| Category (89) | 0.3328 | **0.1749** | **−47%** |

Brand-level falls to the global-frequency-prior baseline (0.069): **no useful
transfer.** Mechanism: only **2 of the top-10** brands are shared between eras,
drug-vocabulary Jaccard **0.462**, **12.2%** of later orders are for brands never
seen in training, **74%** of later encounters were written by a prescriber absent
from training and **58%** at a new site. The task itself did not drift
(empty-Rx rate 22.1% → 23.4%). Non-stationarity is in **formulary and staffing**,
not clinical presentation.

### 7.5 Where it fails

| Drug band (train freq) | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| head (≥100) | 0.2255 | 0.2274 | 0.2264 | 3,334 |
| mid (10–99) | 0.2245 | 0.0956 | 0.1341 | 2,208 |
| tail (<10) | 0.1818 | **0.0057** | 0.0110 | 353 |
| unseen | 0.0000 | 0.0000 | 0.0000 | 40 |

Macro-F1 ≈ 0.053. The tail is not learned. Output diversity: 972 distinct
prescriptions generated against 1,780 in the gold data; the modal generated
prescription (empty) covers 29.0% of encounters against a true 21.0%.

**Structured attributes** (on correctly predicted drugs): type 0.934,
instruction 0.659, duration 0.524, dose 0.484.
**Ranking:** P@1 0.282, P@5 0.166, **R@10 0.462**.
**Auxiliary heads:** advice micro-F1 0.144 (P 0.601 / R 0.082, ECE 0.004);
test micro-F1 0.051 (P 0.889 / R 0.026, ECE 0.001) — precise but insensitive at
a 0.5 threshold.

### 7.6 Local-LLM baselines (n=150 identical test encounters)

| System | Params | s/case | Mean drugs | Empty-F1 | Micro-F1 | Category F1 |
|---|---|---|---|---|---|---|
| medgemma, 0-shot, restraint prompt | 4B | 2.5 | 0.81 | 0.455 | 0.0342 | 0.165 |
| medgemma, 12-shot, restraint prompt | 4B | 2.4 | 0.06 | 0.417 | 0.0068 | 0.018 |
| medgemma, 12-shot, neutral prompt | 4B | 3.0 | 4.17 | 0.000 | 0.0504 | 0.147 |
| **gpt-oss, 12-shot, neutral prompt** | 20B | 28.5 | 1.61 | 0.500 | **0.1118** | **0.2517** |
| **PHC-RxGen** | **5M** | **~0.002** | 1.45 | **0.667** | **0.1865** | **0.3085** |
| *(gold)* | — | — | *1.91* | — | — | — |

gpt-oss figures exclude 3 of 150 unparseable calls (including them moves
micro-F1 by 0.001). **0% off-formulary names in every run** — no result is a
name-resolution artefact.

PHC-RxGen leads, but by **1.7× on micro-F1 and 1.23× at class level** — not a
rout. The warrant rests on **efficiency**: ~4,000× fewer parameters and
~14,000× faster inference, on hardware a rural programme can own.

### 7.7 Clinical NLP layer

**ICD-10 reconstruction** — 12,129 notes: 5.1% no-complaint (no code), **31.5%
confident (coded)**, 63.4% withheld. Final: **3,766 encounters coded (31.0%)**.
Retrieval-vs-LLM agreement 54.1% exact, 86.5% same-chapter; 60 of 3,817 LLM
codes (1.6%) were out-of-list and discarded. Corrected top categories: E11
diabetes (971), I10 hypertension (496), R51 headache (379), R10 abdominal pain
(227).

**NCD burden** (WHO thresholds, Asian BMI cut-offs) — hypertension 34.4%,
diabetes-range glucose 13.4%, obesity 18.9% (Asian ≥27.5) vs 9.5% (WHO ≥30),
anaemia 61.5% (n=3,651), multimorbidity 11.8%. Adults 30–79: hypertension
40.3%, diabetes-range 16.3% — against WHO GHO Bangladesh 28.8% and 9.8%.
Higher, as expected for care-seekers at screening camps; **orientation, not
validation**.

---

## 8. What the evidence supports — and what it does not

### Supported

- Encounter context predicts prescribing **above every null model tested**,
  including prescriber identity (0.192 vs 0.147). RQ1 answered affirmatively.
- The signal is **predominantly the symptom free-text**; vitals add
  information that is largely redundant with it; visit history contributes
  nothing on this corpus. RQ2 answered.
- **Forward-in-time performance collapses to the frequency-prior floor**, with
  a measured mechanism (formulary and staffing turnover, not clinical drift).
  RQ3 answered — and this is the deployment-relevant number.
- **Pharmacological class is the better target** on both predictability and
  temporal stability. RQ4 answered.
- A purpose-trained 5M-parameter model **outperforms both prompted LLMs
  tested**, but the margin over a capable instruction-tuned model (gpt-oss 20B)
  is modest — 1.7× on micro-F1, 1.23× at class level. RQ5 answered: the
  warrant for a purpose-trained model here rests primarily on **efficiency**
  (~4,000× fewer parameters, ~14,000× faster inference, runs on hardware a
  rural programme can own) rather than on a large accuracy advantage.

### Not supported, and must not be claimed

- **No clinical validity.** The model imitates historical prescribing. The
  corpus contains no outcome variable. Agreement with what a clinician did is
  not evidence that it helped.
- **No safety layer.** No drug–drug interaction checking, no contraindication
  model, no weight-based or renal dosing. Attribute heads predict the *modal
  recorded* dose string; that is not a dosing recommendation.
- **Not deployable as autonomous decision support.** Exact-prescription
  reproduction is below a trivial baseline. The defensible framing is
  retrieval — recall@10 = 0.462 — i.e. a suggestion list a clinician edits.
- **ICD codes are derived, not clinical.** Inferred from a symptom note with no
  examination or investigation behind them. Two-coder agreement measures
  *consistency*, not accuracy; no clinician-coded validation set exists.
- **Part of the measured signal is prescriber style**, not patient need.
- **Single-site, single-programme.** No external validation cohort.

---

## 9. Positioning as a thesis contribution

The honest framing is **not** "we built a prescription generator." The best
configuration reaches micro-F1 **0.208** on a patient-level split and **0.066**
forward in time, and no configuration beats an always-empty predictor on
exact-set match. That claim would not survive review.

The defensible contributions are:

1. **An empirical characterisation of the limits of prescription generation on
   LMIC community-telemedicine data** — a setting absent from a literature
   dominated by ICU/hospital corpora with rich diagnostic coding. The negative
   temporal result is the substantive finding: random-split evaluation
   overstates deployability by ~3× on this data.
2. **An evaluation protocol for prescribing models** whose instruments each
   caught a real error here — null-model floors, input permutation, temporal
   attribution, and LLM behaviour auditing. This transfers to any
   imitation-learning clinical model.
3. **A data-provenance methodology.** The finding that clinicians coded `R030`
   and `Z131` because I10 and E11 *were absent from the application's code
   table* reframes an apparent data-quality problem as a system-design
   artefact — the kind of finding that changes how a programme is evaluated.
4. **A reproducible open pipeline** from live database to trained model with
   every defect documented.

The strongest single claim available:

> On low-resource community telemedicine records without diagnostic coding,
> prescribing is predictable well above prior baselines **at the pharmacological
> class level** (micro-F1 0.4934 vs a 0.2345 frequency prior). **Brand-level
> prescribing does not transfer forward in time** (0.192 → 0.066, the prior
> floor), whereas class-level transfer degrades but survives (0.4615 → 0.3547,
> −23.1%, still +0.1245 above its floor): roughly four fifths of that difference
> comes from the target, one fifth from the estimator. A random split therefore
> overstates forward micro-F1 by **~1.3× at class level and ~3× at brand level**,
> and **set-level agreement collapses below the always-empty floor either way** —
> so the transferable product is a ranked suggestion list, not a
> whole-prescription draft. Two further results are operative in a low-resource
> setting: feature sets must be selected on a **temporal** validation split (the
> confounded `all` arm falls further, −37.6% era-clean), and a 5M-parameter
> purpose-trained model matches a prompted 20B LLM at ~14,000× lower inference
> cost.

### What would strengthen it materially

- A clinician-coded validation sample (≈300 encounters) — converts every
  agreement figure into an accuracy figure, and is the single highest-value
  addition.
- Appropriateness review of generated prescriptions by a physician, scoring
  clinical acceptability rather than string match against history.
- External validation on a second PHC site or programme.
- Class-level modelling as the primary target, with brand selection treated as
  a separate formulary-mapping step.

---

## 10. Practical recommendation from these numbers

If this system were taken forward, the evidence points to a configuration
different from the one the architecture section describes:

1. **Drop the autoregressive decoder, not the fusion encoder.** Reframing as
   multi-label classification is worth **+0.109 micro-F1** (`neural_mlc.md`) —
   the largest single effect measured in this project. Fusion costs 0.016 only
   *with* the decoder; under a multi-label head it helps (+0.015 to +0.026).
2. **Drop the history GRU.** Worth +0.0002 in ablation, −0.004 under
   permutation. It is unjustified until the corpus grows longitudinally —
   8,567 of 10,938 patients have a single visit.
3. **Model at pharmacological class, not brand.** Higher accuracy (0.348 vs
   0.208) and materially better temporal stability (−47% vs −66%). Treat brand
   selection as a separate formulary-mapping step that can be updated without
   retraining.
4. **Deploy as a ranked suggestion list, not a generated prescription.**
   R@10 = 0.462 supports "here are ten candidates"; exact-match below the
   trivial baseline does not support autonomous generation.
5. **Plan for retraining on a fixed cadence.** With 74% prescriber turnover and
   a 0.462 drug-vocabulary Jaccard across eras, any fixed model decays quickly.
   Monitor formulary drift directly rather than assuming stationarity.
6. **Do not deploy in any form without the clinician validation** in
   `docs/annotation_protocol.md`. Nothing measured here speaks to clinical
   appropriateness or safety.
