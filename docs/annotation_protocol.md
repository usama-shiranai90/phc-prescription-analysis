# Clinician annotation protocol

Converts the study's automated-agreement figures into **accuracy** figures, and
tests generated prescriptions for **clinical appropriateness** rather than
string match against history.

This is the single highest-value addition to the work. Without it:

- every ICD number is agreement between two automated coders, not accuracy;
- every prescription number is similarity to what one clinician happened to
  write, which the prescriber-prior baseline (micro-F1 0.147 from prescriber
  identity alone) shows is itself highly variable.

---

## 1. Design

### Two tasks

**Task A — ICD-10 coding (200 encounters).** The clinician codes the encounter
from the note and vitals. **No model output is shown**, so there is no
anchoring. This yields ground truth for the ICD pipeline.

**Task B — prescription appropriateness (150 encounters).** The model's
prescription and the historical one appear as *Option 1* and *Option 2* in
randomised order, with no indication of origin.

The pairwise-blinded design is the important choice. Rating the model against
history *as if history were correct* would bake clinician-to-clinician variation
into the result. A blinded A/B instead asks the question that matters for
deployment:

> Is the generated prescription **non-inferior to the attending clinician's**?

### Sampling

| Task A tier | n | why |
|---|---|---|
| confident (shipped) | 110 | measures accuracy of what the pipeline emits |
| low_confidence (withheld) | 70 | **over-sampled** — tests whether the gate is too tight |
| no_complaint (withheld) | 20 | verifies the no-complaint rule is correct |

The low-confidence tier is deliberately over-sampled relative to its share.
The open question is not only "is what we emit correct" but "is what we
suppress genuinely uncodable" — 63% of notes are currently withheld, and only a
clinician can say whether that is caution or waste.

| Task B stratum | n |
|---|---|
| gold prescription empty | 30 (20%, mirrors the corpus rate) |
| gold prescription non-empty | 120 |

**Double-annotation:** 25% of items (87) go to both annotators, giving
inter-rater reliability. This is not optional — clinician-vs-clinician
agreement is the **ceiling** on any accuracy an automated coder can
meaningfully claim. If two physicians agree on only 60% of primary codes, an
automated coder at 57% is at parity, not failing.

---

## 2. Running it

Build the sample and the blinded packets:

```bash
python -m src.phcrx.annotate.build_packets --n-icd 200 --n-rx 150
```

Generate one offline tool per annotator:

```bash
python -m src.phcrx.annotate.make_tool
```

This writes `results/rx_generation/annotation/annotate_<annotator>.html`
(~375 KB each). The annotator double-clicks the file — it runs offline in any
browser, needs no install and no network, autosaves to browser storage, and has
an **Export answers** button that downloads
`annotations_<annotator>.json`. Arrow keys navigate.

Score the returned files:

```bash
python -m src.phcrx.annotate.score_annotations
```

### What the annotator sees

Age, sex, the verbatim clinical note, and recorded vitals. For Task A: three
ICD fields with a searchable pick-list of all 1,547 three-character categories,
an explicit *"no code is appropriate"* option, a confidence rating, and a
**codability** judgement (clear single problem / multiple problems / too vague).
For Task B: the two prescriptions rendered as tables with drug, class, form,
dose, duration and instruction, then preference, an independent safety rating
for each option, and whether they would have prescribed differently.

### Confidentiality

The packets contain real patient notes and vitals. They carry an opaque
`ann_id`; the crosswalk to `prescription_id`, the model's own codes, and the
A/B key live in `KEY_do_not_share_with_annotators.json`.

**Withhold that file from annotators**, and keep the HTML tools local — do not
email or publish them.

---

## 3. What the analysis reports

**Task A**
- exact-code and same-chapter accuracy on the confident tier, with Wilson 95% CIs
- rate at which clinicians *could* code the withheld tiers — the gate-calibration
  test. A high rate means the 0.50 threshold is too tight and coverage is being
  wasted; a low rate means the withholding was correct
- distribution of codability judgements, which quantifies how much of the
  corpus is inherently un-codable at three-character granularity

**Task B**
- blinded head-to-head win rate with an exact CI, ties excluded
- **non-inferiority** at a 10-point margin (CI lower bound above 40%), not
  superiority — the model does not need to beat the physician, it needs to not
  be meaningfully worse
- unsafe-prescription rate for model *and* historical, each with a CI. This is
  the number that gates any deployment discussion, and having the historical
  rate alongside it is what makes it interpretable

**Reliability**
- Cohen's κ for primary ICD code, prescription preference and codability

---

## 4. Reading the results honestly

- **Interpret Task A accuracy against the κ ceiling, never in isolation.**
- **A model win rate near 50% is the good outcome.** It means generated
  prescriptions are indistinguishable in appropriateness from the attending
  clinician's. Anything much above 50% on n=150 should raise suspicion of a
  blinding leak — for instance, if model prescriptions are systematically
  shorter, an annotator may learn to spot them.
- **The unsafe rate is not a relative measure.** Even at parity with the
  historical rate, an absolute unsafe rate above a few percent is
  disqualifying for autonomous use and would confine the system to a
  suggestion-list role.
- **n=150 is powered for a coarse signal.** The head-to-head CI is roughly
  ±10 points. It can distinguish "clearly worse" from "roughly comparable"; it
  cannot resolve a 5-point difference. Do not over-read a small gap.
- **Two annotators is the minimum.** Three with adjudication of disagreements
  would materially strengthen every figure here.

---

## 5. Status

The instrument is built and the scoring path has been dry-run end-to-end with
synthetic responses (then deleted) to confirm packets → tool → scorer works.

**No clinician annotations have been collected.** Every accuracy, safety and
non-inferiority figure described above is currently unmeasured, and no claim
about clinical validity is supported until they are.
