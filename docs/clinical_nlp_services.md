# Clinical NLP services — Hugging Face, Ollama, ICD-10, WHO NCD

Layer added on top of PHC-RxGen to fill the corpus's structural gaps: no usable
diagnosis codes, no pharmacological classification for 111 brands, and no
external benchmark for the NCD burden it measures.

**Everything runs locally.** The corpus contains real patient records (names,
national IDs, mobile numbers, addresses), so no clinical text is sent to a
hosted inference API. Hugging Face is used to *download model weights and a
reference vocabulary*; inference happens on the local GPU. Ollama serves models
on the same machine. The only network call carrying anything is the WHO GHO
request, which transmits no patient data — it downloads population statistics.

## Service inventory

| Service | Used for | Where it runs |
|---|---|---|
| **Hugging Face** — `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | biomedical entity linking: symptom text → ICD, brand → class | local GPU |
| **Hugging Face** — `TachyHealth/…icd10cm_order_April_2024` | complete ICD-10-CM reference (97,296 codes) | downloaded once |
| **Ollama** — `medgemma:latest` (4.3B) | ICD adjudication, prescription baseline | local (Windows host) |
| **Ollama** — `qwen3.6:35b`, `gpt-oss:20b` | optional stronger adjudicator | local |
| **WHO GHO** OData API | NCD prevalence benchmark for Bangladesh | public, no key, no PHI |
| **WHO ICD API** (`id.who.int`) | *not used* — returns 401, needs registration | — |

### Transport note

Ollama binds `127.0.0.1` on the Windows host, so WSL2 cannot reach it over TCP
(the same constraint as Postgres). Rather than ask for the service to be
rebound, [`ollama_client.py`](../src/phcrx/nlp/ollama_client.py) uses WSL
interop: it shells out to Windows `curl.exe`, which resolves `127.0.0.1` in the
Windows network namespace. The payload goes in on stdin — a temp file will not
work because `curl.exe` cannot open a Linux path, and inlining JSON breaks on
clinical text containing quotes and slashes.

---

## 1. The corpus has its own dialect, and general medical LLMs get it wrong

Asked to expand `"D/M, HTN for several years"`, **medgemma answered
"Dementia / Mild Cognitive Impairment"** and coded it F02 (dementia).

In this corpus D/M is **Diabetes Mellitus**. That is not a judgement call — it
is measurable, and [`glossary.py`](../src/phcrx/nlp/glossary.py) checks it
against independent physiology rather than asserting it:

| expansion | evidence | with mention | without | Δ |
|---|---|---|---|---|
| `d/m` → Diabetes Mellitus | blood glucose | 259.2 mg/dL | 131.0 | **+128.2** |
| `dm` → Diabetes Mellitus | blood glucose | 236.7 mg/dL | 120.0 | **+116.7** |
| `htn` → Hypertension | systolic BP | 149.4 mmHg | 122.8 | **+26.6** |

Encounters mentioning D/M sit squarely in the diabetic range. The glossary
(46 entries) is injected into every downstream prompt, and expansion is applied
before embedding — SapBERT has never seen `D/M` but matches
`Diabetes Mellitus` well.

**Implication for the thesis:** a site-specific glossary is not a nicety. An
off-the-shelf medical LLM applied to this corpus without one produces
confidently wrong codes.

---

## 2. ICD-10 coding — filling the corpus's biggest gap

Only 195 of 14,074 encounters carry an ICD code, and those are mostly not
diagnoses (the single most frequent is Y440, *"Iron preparations…"*, 128 of 216
rows). Meanwhile 86% of encounters have a symptom note.

### Why the existing codes are bad — a data-provenance finding

The application's own `tb_data_icd` table holds 5,127 codes but covers **only
chapters A, P, Q, R, S, T and V–Z. Chapters B through O are entirely absent.**
Every code this cohort needs is missing: E11 (diabetes), I10 (hypertension),
K21 (reflux), M54 (back pain), N39 (UTI), J06 (URTI).

So the clinicians were not coding badly — **they could not code correctly.**
`R030` "elevated blood-pressure reading" and `Z131` "screening for diabetes"
were the closest available options because I10 and E11 were not in the picker.
The reference is therefore taken from the CMS ICD-10-CM order file on
Hugging Face, cut to 3-character categories.

### Retrieval must be hybrid

Pure SapBERT retrieval failed on exact clinical terms:

| note | semantic-only | hybrid (α=0.5) |
|---|---|---|
| "Angina Pectoris" | R074 Chest pain, unspecified | **I20 Angina pectoris** (1.000) |
| "excessive menstrual bleeding" | R632 Polyphagia | **N92 Excessive menstruation** (0.671) |
| "Known case of D/M, HTN" | R030 Elevated BP reading | **E11 Type 2 diabetes** + I1A hypertension |
| "itching" | R21 Rash | **L29 Pruritus** |
| "Incresed level of b. glucose" | R730 Abnormal GTT | **R73 Elevated blood glucose** |

Blending SapBERT cosine with a character n-gram TF-IDF fixes exact-term recall
while keeping paraphrase matching. Note the last row: the hybrid survives the
typo *"Incresed"*, which is the same robustness argument that motivated the
char-CNN in PHC-RxGen.

### Retrieval will code anything you hand it — including "no complaint"

The first full run produced confident nonsense on notes that record *no*
complaint. Nearest-neighbour retrieval always returns a nearest neighbour:

| assigned | n | mean score | the actual note |
|---|---|---|---|
| I96 Gangrene | 192 | 0.327 | `"NO"` |
| T81 Complications of procedures | 243 | 0.291 | `"No complaint"` |
| R64 Cachexia | 152 | 0.374 | `"weakness"` |
| R61 Generalized hyperhidrosis | 321 | 0.403 | `"generalized weakness"` |

Shipping that would have asserted gangrene in 192 patients. Assignment is
therefore tiered rather than forced:

1. **no_complaint** — note matches a no-complaint pattern (`"NO"`,
   `"no complaints"`, `"general check up"`) → **no code**. These are screening
   visits, not diagnoses.
2. **confident** — hybrid score ≥ 0.50 → coded.
3. **low_confidence** — everything else → **withheld**, not guessed.

The gate is deliberately conservative: coverage is sacrificed so that what is
emitted can be relied on. Counts per tier are printed by the module and stored
in `icd_autocoded.parquet`, so a user can widen the gate knowingly.

Applied to the 12,129 notes:

| Tier | n | share | outcome |
|---|---|---|---|
| no complaint recorded | 621 | 5.1% | no code |
| confident (score ≥ 0.50) | 3,817 | **31.5%** | coded |
| low confidence | 7,691 | **63.4%** | withheld |

**Only about a third of notes get a code.** That is the honest coverage figure
and it should be quoted as such: the gate converts a silent-error problem into
a visible-coverage problem, which is the right trade for clinical data, but it
does not manufacture diagnoses for the remaining two thirds. Most withheld
notes are multi-complaint free text ("1. pain in both heels and back pain
2. generalized weakness") where no single 3-character category is defensible.

### Lexical matching has a systematic bias toward short "Other" descriptions

Gating fixed the no-complaint cases but exposed a second, subtler defect. The
character n-gram component rewards short descriptions that literally contain
the query token, which produced two systematic errors:

| retrieval said | n | should be | why it happened |
|---|---|---|---|
| **I1A** "Other hypertension" | 1,450 | **I10** "Essential (primary) hypertension" | I1A's shorter description overlaps "hypertension" more strongly |
| **A78** "Q fever" | 155 | **R50** "Fever of other and unknown origin" | any note containing "fever" matches the literal token |

The hypertension case matters most: it was the **largest single category in the
corpus**. And I1A is an **ICD-10-CM-only code added in 2022 — it does not exist
in WHO ICD-10**, so it should never have been a candidate for a WHO-coded
Bangladeshi cohort. Known CM-only additions (`I1A`, `E08`, `E09`, `D3A`, `M1A`,
`C7A`, `C7B`, `Z3A`) are now excluded from the reference.

**Excluding I1A did not fix the bias — it relocated it.** On the next run
`I15 "Secondary hypertension"` took the same slot (517 encounters), and is
equally wrong: secondary hypertension is hypertension caused by another
identified condition, not the default. Any short description containing the
query token wins. Curating the code list is therefore not a fix; the bias is
structural in lexical matching, which is what forced the design below.

medgemma got both of these right unprompted — it chose I10 over I1A with the
reasoning *"'Hypertension for 5 years' … consistent with essential
hypertension"*. That is the evidence behind the design decision below.

### The LLM, not retrieval, is the authoritative coder

Retrieval generates candidates; the LLM chooses among them. On the
disagreements the LLM was consistently the better coder:

| note | retrieval | medgemma |
|---|---|---|
| `"h/o TB 1 yr back"` | R61 hyperhidrosis | **B90** sequelae of TB |
| `"known case of DM"` | E13 other diabetes | **E11** type 2 diabetes |
| `"1.HTN for 5 years."` | I1A other hypertension | **I10** essential hypertension |
| `"yellow colorization of micturition"` | A01 typhoid | **R30** painful micturition |

Running `--sample -1` adjudicates the entire confident tier and writes
`icd_final` as the LLM's choice. Retrieval top-1 is retained in the output for
comparison, never as the shipped label.

### Result of full-tier adjudication (3,817 encounters)

The lexical artefacts are gone. Compare the top categories before and after:

| retrieval top-1 | n | → | LLM authoritative | n |
|---|---|---|---|---|
| I15 Secondary hypertension | 517 | → | **I10 Essential (primary) hypertension** | 496 |
| A78 Q fever | 40 | → | **R50 Fever of other/unknown origin** | 45 |
| E11 Type 2 diabetes | 628 | → | **E11 Type 2 diabetes** | 971 |

Final assignment: **3,766 encounters coded (31.0% of notes)**. Agreement with
retrieval was 54.1% exact and 86.5% same-chapter, with 1.3% of cases where the
LLM declined to code. Sixty out-of-list codes were produced across 3,817 calls
(1.6%) and discarded by the validity filter — the constraint is enforced in
code, never trusted to the prompt.

The corrected distribution matches the cohort: diabetes and hypertension
dominate, followed by headache, abdominal pain and chest pain. That is what a
rural NCD-screening programme should look like, and it is what the raw
retrieval output did not show.

### Validation design

There is no usable ground truth, so two independent coders are compared:
retrieval top-1 against medgemma constrained to the retrieved candidates. The
sample is drawn from the **confident tier only**, so the agreement figure
describes the assignments a user would actually receive rather than being
flattered by cases that never ship.

The LLM may only choose from the candidate list, and out-of-list codes are
counted as constraint violations and discarded — medgemma produced 12 in 150
cases, so the constraint is enforced in code rather than trusted to the prompt.

---

## 3. NCD stratification, benchmarked against WHO

[`ncd.py`](../src/phcrx/nlp/ncd.py) applies WHO diagnostic thresholds, plus
Asian-specific BMI cut-offs (WHO 2004 expert consultation), which matter
materially for a South Asian cohort:

| Condition | Measured | Prevalence |
|---|---|---|
| Hypertension (≥140/90) | 13,703 | **34.4%** |
| Hypertension stage 2 (≥160/100) | 13,703 | 13.1% |
| Diabetes-range glucose | 12,770 | **13.4%** |
| Pre-diabetes range | 12,770 | 13.8% |
| Overweight, Asian (BMI ≥23) | 12,609 | **53.6%** |
| Obese, Asian (BMI ≥27.5) | 12,609 | **18.9%** |
| Obese, WHO (BMI ≥30) | 12,609 | 9.5% |
| Anaemia (WHO) | 3,651 | 61.5% |
| NCD multimorbidity (≥2 of 3) | 14,074 | 11.8% |

The BMI rows carry the point: **the Asian threshold doubles measured obesity**
(18.9% vs 9.5%). Using WHO's ≥30 on this population would halve the apparent
burden.

### WHO GHO benchmark (Bangladesh)

| Indicator | WHO | Year | This cohort (30–79) |
|---|---|---|---|
| Hypertension, adults 30–79 | 28.8% | 2019 | **40.3%** |
| Raised BP (age-standardised) | 24.6% | 2019 | 40.3% |
| Raised fasting glucose | 9.8% | 2014 | **16.3%** |
| Obesity BMI ≥30 | 5.7% | 2024 | 9.5% |
| Overweight BMI ≥25 | 29.9% | 2024 | (see `ncd_summary.json`) |

**This is orientation, not validation.** The cohort is care-seekers at
screening camps, not a population sample; higher prevalence is expected and
does not indicate a WHO discrepancy. Anaemia in particular is measured on only
3,651 encounters (Hb coverage is 26%) and is almost certainly
ascertainment-biased toward symptomatic patients.

---

## 4. Local-LLM prescription baseline — and how easy it is to measure the prompt

The reviewer question this answers: does a purpose-trained 5M-parameter model
still earn its place when a 4B medical LLM can be prompted for free?

Getting an honest answer took three attempts, and the first two measured
**defects in my own prompt** rather than the model. Both are recorded here
because the failure modes are the transferable result.

**Attempt 1 — biased system prompt.** The system message ended with *"Many
screening visits need no medication at all — when the patient has no treatable
complaint, prescribe nothing."* medgemma then declined to prescribe on 96% of
encounters, including one with a recorded glucose of **410 mg/dL** whose own
stated reason was *"indicates diabetes mellitus"*. A prompt that instructs
restraint and then scores restraint measures nothing about the model.

**Attempt 2 — recency anchoring in the exemplars.** With few-shot examples
added, output collapsed to 0.07 drugs/encounter and the model echoed the
*wording* of the final exemplar (`"NO COMPLAINTS" → {"drugs": []}`) as its
stated reason on unrelated notes — including `"H/O HTN for 7 years. DM (under
medication)"`. Exemplars are now balanced to the corpus empty rate (21.5%) and
ordered to end on a prescribing case.

**Attempt 3 — neutral prompt.** Removing the restraint instruction swung the
model to the opposite extreme: 4.17 drugs per encounter against a gold mean of
1.91, and an empty-prescription F1 of exactly zero (it never withheld).

### Final comparison (n=150 identical test encounters)

| System | params | s/case | mean drugs | empty-F1 | micro-F1 | category F1 |
|---|---|---|---|---|---|---|
| medgemma, zero-shot, restraint prompt | 4B | 2.5 | 0.81 | 0.455 | 0.0342 | 0.165 |
| medgemma, 12-shot, restraint prompt | 4B | 2.4 | 0.06 | 0.417 | 0.0068 | 0.018 |
| medgemma, 12-shot, neutral prompt | 4B | 3.0 | 4.17 | 0.000 | 0.0504 | 0.147 |
| **gpt-oss, 12-shot, neutral prompt** | **20B** | **28.5** | **1.61** | **0.500** | **0.1118** | **0.2517** |
| **PHC-RxGen** | **5M** | **~0.002** | **1.45** | **0.667** | **0.1865** | **0.3085** |
| *(gold)* | — | — | *1.91* | — | — | — |

gpt-oss figures exclude 3 of 150 calls that returned nothing parseable
(including them changes micro-F1 by 0.001). Constraint compliance was perfect
throughout: **0% off-formulary names** in every run.

### A claim from the medgemma-only stage had to be withdrawn

On medgemma alone the natural reading was that *calibration of how much to
prescribe is learned from data and cannot be prompted*: volume swung from 0.06
to 4.17 drugs per encounter on prompt wording, never landing near the gold rate.

**gpt-oss refutes that.** At 20B with the same 12 exemplars it prescribes 1.61
drugs against a gold mean of 1.91 and reaches empty-Rx F1 0.500 — it calibrates
volume reasonably well from a handful of examples. The determinant was model
capability, not prompting as such. The earlier phrasing generalised from one
small model and is corrected here.

### What survives, and what it rests on

PHC-RxGen still leads on every accuracy measure, but the margin over a capable
instruction-tuned LLM is **modest**: 0.1865 vs 0.1118 micro-F1 (1.7×) and
0.3085 vs 0.2517 at category level (1.23×) — not the 3.7× that medgemma alone
suggested.

The stronger argument is therefore **efficiency, not accuracy**:

- 5M parameters against 20B — a factor of ~4,000
- ~2 ms per encounter on a 6 GB card, against 28.5 s CPU-offloaded — a factor
  of ~14,000
- runs on hardware a rural programme can actually own

For a deployment context defined by scarce compute and intermittent
connectivity, a 5M model that is 1.2× better at class level *and* runs in
milliseconds on commodity hardware is the defensible recommendation. Claiming
the LLM is simply incapable would not be.

**The methodological lesson for the thesis:** an LLM baseline measures the
prompt as much as the model. Reporting "the LLM scored 0.03" without auditing
refusal rate and exemplar ordering would have produced a confident wrong
conclusion — one that happened to flatter the proposed model. Any LLM
comparison should report *what the LLM did* (mean prediction size, refusal
rate, constraint violations), not only its score.

Constraint compliance was perfect throughout: **0% off-formulary drug names**
across all runs, so no result here is an artefact of failed name resolution.

## 5. Reproducing

```bash
python -m src.phcrx.nlp.glossary
```

```bash
python -m src.phcrx.nlp.icd_index --build
```

```bash
python -m src.phcrx.nlp.icd_code --sample 150
```

```bash
python -m src.phcrx.nlp.ncd
```

```bash
python -m src.phcrx.nlp.llm_rx --n 150 --model medgemma:latest --shots 12
```

Outputs land in `results/rx_generation/nlp/`.

## 6. Limitations

- **The auto-assigned ICD codes are derived, not clinical.** They are inferred
  from a symptom note by a retrieval model, with no examination, history or
  investigation behind them. They are suitable for cohort description and as
  model features; they are not a coded diagnosis and must not be presented as
  one.
- **No human-coded validation set exists.** Retrieval/LLM agreement measures
  consistency between two automated coders, not correctness. Establishing
  accuracy needs a clinician-coded sample, which this corpus does not have.
- **ICD-10-CM is the US clinical modification**, used here because it is
  complete and freely redistributable. At the 3-character category level it
  aligns with WHO ICD-10, but the two are not identical.
- **The glossary is site-specific** and was curated against this corpus. It
  should not be reused elsewhere without re-verification.
- **NCD flags are single-measurement.** Clinical diagnosis of hypertension and
  diabetes requires repeat readings on separate occasions; these flags mark
  *readings in the diagnostic range*, which overstates true prevalence.
