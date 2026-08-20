# Remaining work, limitations, and where the headroom is

Status as of the current session. Every claim here is backed by a measured
number in `docs/` or `results/rx_generation/`.

---

## 1. What is done

| Workstream | Outcome | Doc |
|---|---|---|
| Data pipeline (Postgres → Parquet, defect repair) | done | `prescription_generation.md` |
| Neural generator + full ablation grid | done — **superseded** | `prescription_generation.md` |
| Evaluation apparatus (floors, permutation, temporal, strata) | done | `research_frame.md` |
| Clinical NLP (glossary, ICD, NCD, LLM baselines) | done | `clinical_nlp_services.md` |
| Corpus expansion (MCH/antenatal/postnatal) | done — **do not adopt** | `corpus_expansion.md` |
| Drug normalisation (brand → molecule → class) | done — **adopt class target** | `drug_normalization.md`* |
| Feature engineering + importance (50 targets) | done | `feature_importance.md` |
| Text preprocessing ablation | done — adopt for tail only | `results/…/textproc/ablation.csv` |
| Neural vs linear head-to-head | done | `model_comparison.md` |
| Neural multi-label fairness test | done — **reverses a conclusion** | `neural_mlc.md` |
| Final per-component recommender | done | `recommender.md` |
| Clinician annotation instrument | **built, never run** | `annotation_protocol.md` |

\* pending: the drug-normalisation doc was not written before its agent died;
the numbers are in `results/rx_generation/drugmap/drug_normalization.json`.

## 2. Remaining tasks

Verified against the code on 2026-08-20. Items struck through in later sections
are closed and not repeated here.

### P0 — blocks every clinical claim

| # | Task | Effort | Note |
|---|---|---|---|
| 1 | **Run the clinician annotation study** | 2–3 clinicians x ~4h | 350 packets + 2 offline HTML tools built and unused (`annotation_protocol.md`). Until this runs, every number in the repo is agreement with historical prescribing, not accuracy or safety. |

### P1 — correctness and consistency (cheap, high value)

| # | Task | Effort | Note |
|---|---|---|---|
| 2 | ~~Re-run the temporal split at class level~~ | **done (brand control pending)** | Closed. Class-level drop is **−23.1%** (deployable arm), not −66%, and lands clear of its prior floor. The `all` arm degrades twice as fast (−48.1%), reversing the deployment choice. See §9. A `brand717` control run is still fitting to separate the target effect from the estimator effect. |
| 3 | ~~Wire abstention into `recommend.py`~~ | **done** | Closed. `--policy {none,global80,sparse_targeted}`, val-only cut-points, policy artefact in `models/abstention_policy.joblib`. Verified: the per-encounter rule covers **exactly** the rows the batch analysis covers (0 differing). See §8 and `recommender.md` §6. |
| 4 | **Propagate the two reversed conclusions** | ~1h | `model_comparison.md` still asserts "the neural architecture is not justified"; `prescription_generation.md` §5 still asserts "transformer fusion hurts". Both were overturned by `neural_mlc.md` and are wrong as written. |
| 5 | **Write `docs/drug_normalization.md`** | ~1h | Results exist (`results/rx_generation/drugmap/drug_normalization.json`); prose never written — its agent died first. |
| 6 | **Write `docs/text_preprocessing.md`** | ~1h | Same: `results/rx_generation/textproc/ablation.csv` has the full table with CIs, no prose. |
| 7 | **Confirm bootstrap CIs on the final recommender table** | ~30m | `evaluate.py --bootstrap 2000` was run but the printed table shows point estimates. Verify the CIs are in `test_results.csv`; if not, re-run and publish them. |
| 8 | **Add a docs index / update README** | ~30m | 12 docs now with no navigation and no statement of which supersede which. |

### P2 — accuracy headroom

| # | Task | Effort | Note |
|---|---|---|---|
| 9 | **Hierarchical class → brand decoding** | ~1d | Class is what generalises (0.49, stable across eras); brand is what a dispensing system needs. Map class → brand by site formulary frequency, updatable without retraining. |
| 10 | **Per-label thresholds** | ~2h | One global threshold today. Tail-macro is 0.2249; per-label operating points should help the tail most. |
| 11 | **Rebuild `features.parquet` deployable-only** | ~2h | The current matrix mixes deployable and confounded families. The recommender selects correctly per component, but the artefact invites misuse. |
| 12 | **Sparse-encounter modelling from vitals** | ~1d | Fixed to 0.4153 by feature choice (§7), but this subgroup has the widest deployable/full gap (0.051) — it depends most on confounded features. |

### P3 — scope extensions

| # | Task | Effort | Note |
|---|---|---|---|
| 13 | **Site-transfer evaluation** | ~1d | 78 sites. Hold out whole sites rather than patients — closer to deploying at a new clinic than the patient split is. |
| 14 | **Re-test the MCH merge under the class target** | ~4h | Rejected at brand level, where MCH-only brands could never be rewarded. Under 46 classes the calculus may differ. Use `--freeze-adult-split`. |
| 15 | **`n_drugs` conditioned on predicted classes** | ~4h | Currently independent; the count should follow from the predicted set. |
| 16 | **Neural-MLC vs linear at brand level** | ~4h | Only compared at class level. Brand level is untested under multi-label framing. |

### Explicitly closed — do not redo

| Item | Verdict | Evidence |
|---|---|---|
| MCH corpus merge as default | rejected — 88.5% of new orders land in the head stratum | `corpus_expansion.md` §7 |
| LLM drug mapping | rejected — 30% accurate, degenerate high-confidence PPI default | §5 below |
| Multi-condition "fixes" | rejected — premise false, performance *rises* with complaint count | §6 |
| Specialist model for sparse rows | rejected — 0.3676 vs 0.4153 shared | §7 |
| Negative-entropy confidence signal | rejected — worse than random abstention | §8 |
| Autoregressive set decoder | superseded — costs 0.109 micro-F1 vs a multi-label head | `neural_mlc.md` |

## 3. Limitations

### Data
- **No outcome variable.** Nothing in the corpus says whether a prescription
  helped. Every metric is imitation fidelity.
- **No usable diagnosis codes.** 195/14,074 encounters, and mostly not
  diagnoses. The reconstructed ICD layer covers 31% of notes and is *derived
  from the symptom text the model already reads* — it added +0.0009 importance.
- **Prescriber confounding.** 67 prescribers; prescriber identity alone reaches
  micro-F1 0.147, and dose/duration/instruction are predictable from *who
  wrote it* rather than from the patient.
- **Non-stationarity.** 74% prescriber turnover and 0.462 cross-era brand
  vocabulary Jaccard between ≤2015 and ≥2017.
- **Small and old.** 11,065 prescriptions with drugs; 52% of encounters fall in
  2012–2014.
- **Year-precision age.** Many DOBs are Jan-1 placeholders, so paediatric
  weight-based dosing cannot be learned.
- **Single site/programme.** No external validation cohort.

### Method
- **Set metrics have high floors.** Always-empty scores Jaccard 0.2155 on drug
  classes, 0.7090 on advice, **0.8550 on tests**. Any headline must be read
  against its floor.
- **AUROC misled us once.** Tests looked "highly predictable" at AUROC
  0.923–0.979 but fall *below* the always-empty floor on Jaccard and exact
  match at 14.5% prevalence. Prevalence-insensitive metrics are not safe here.
- **Class mapping is 97.6% complete, not 100%.** The remaining orders are
  unmapped, and strict accounting is what makes the class-level advantage real
  (12.2% → 3.4%, not the 3.0% first reported on the mapped subset only).
- **The proxy for text ablations is a linear bag-of-words model.** A char-CNN
  may respond differently to normalisation.

### Product
- **No safety layer.** No drug–drug interaction checking, no contraindications,
  no renal/hepatic adjustment, no weight-based dosing.
- **Not autonomous.** Exact-set reproduction sits below trivial baselines. The
  defensible framing is a ranked suggestion list a clinician edits.
- **Tests must not be surfaced as a set** — only as ranked per-test
  probabilities.

## 4. Two conclusions this project reversed

Recorded because the reversals are more instructive than the results.

1. **"The neural architecture is not justified."** False as stated. The
   *autoregressive set-decoder* was not justified: same encoder with a
   multi-label head gains **+0.109 micro-F1** and draws level with the linear
   model. `neural_mlc.md`.
2. **"Transformer fusion hurts at this corpus size."** True only with the
   decoder. Under multi-label framing fusion **helps** (+0.015 to +0.026).

Four further hypotheses I held with confidence and which measurement rejected:
glossary expansion helping the model (it did not, until reframed as
augmentation); NCD flags adding signal (+0.0026); derived ICD codes adding
signal (+0.0009); and LLM drug mapping being usable (30% accurate with
degenerate high-confidence defaults).

## 5. Where the real headroom is

Ranked by expected value, not by how interesting the method is:

1. **Clinician validation** — converts the whole project from "agreement" to
   "accuracy and safety". Nothing else unlocks a clinical claim.
2. **Class-level temporal re-test** — likely turns the headline negative result
   (−66%) into something far better, and it is a few hours of compute.
3. ~~Sparse/no-complaint encounters~~ — **fixed; see §7.** The deficit was a
   feature-set artefact, not a modelling gap. Remaining sub-task: close the
   deployable-vs-full gap on these rows (0.364 vs 0.415), which is the
   largest such gap of any subgroup.
4. **Hierarchical class → brand** — bridges what generalises (class) to what a
   dispensing system needs (brand).
5. **More data from other PHC sites** — the corpus is the binding constraint,
   not the model. Every architecture tested lands within ~0.05 micro-F1 of every
   other, which is the signature of a data ceiling rather than a modelling one.


---

## 6. Multi-condition notes: a hypothesis that measurement rejected

The qualitative demo showed the recommender catching the diabetes half of
*"h/o DM+h/o asthma"* and missing both asthma classes. I generalised from that
single example to "multi-condition notes are the failure mode" and built two
fixes. **Both the premise and the fixes were wrong.**

Stratifying test performance by parsed complaint count:

| complaint spans | n | gold classes | micro-F1 | recall | precision |
|---|---|---|---|---|---|
| 0 (no complaint) | 436 | 1.37 | **0.3393** | 0.2881 | 0.4125 |
| 1 | 1,385 | 1.78 | 0.4526 | 0.5077 | 0.4082 |
| 2 | 561 | 2.37 | **0.4919** | 0.5350 | 0.4552 |
| 3+ | 398 | 2.93 | **0.5024** | 0.5325 | 0.4755 |

Performance **rises monotonically with complaint count**. More complaints means
more text, which means more evidence — the opposite of the assumed dilution.

Both fixes degraded the model:

| variant | test micro-F1 | delta |
|---|---|---|
| baseline (one bag, global threshold) | **0.4628** | — |
| A: threshold adapted to span count | 0.4599 | -0.0029 |
| B: per-span scoring, max-pooled | 0.4469 | -0.0159 |
| A+B | 0.4523 | -0.0105 |

Restricted to multi-complaint rows only, the baseline still wins (0.4967).
Span-max raises recall (0.6247 vs 0.5338) but wrecks precision (0.3745 vs
0.4645) by emitting 4.34 classes against a gold mean of 2.60 — scoring each span
independently loses the *joint* context that tells the model which combination
of conditions is plausible.

**The real weak spot is the opposite end.** No-complaint encounters (n=436,
micro-F1 0.339) are where the model is worst, because it must work from vitals
and demographics alone. That is the redirect: sparse encounters, not rich ones.

The methodological point is worth keeping. A single vivid qualitative example
motivated a plausible mechanism and a testable fix, and pointed in the wrong
direction. One stratified table refuted all of it. Qualitative inspection is
good for generating hypotheses and useless for confirming them.


---

## 7. Sparse / no-complaint encounters: fixed, and a number corrected

436 of 2,780 test encounters (15.7%) have no parseable complaint — screening
visits recorded as "NO", "no complaints", "general check up". They carry fewer
gold classes (1.37 vs 2.12) and were the weakest subgroup.

### Diagnosis before fix

| Question | Answer |
|---|---|
| Is the signal there at all? | **Yes.** A subset-specific frequency prior scores 0.1777; the model was already at 0.339 and reaches 0.415. Not a ceiling. |
| Is it a threshold artefact? | **Partly.** Recall 0.288 vs precision 0.413 confirmed under-firing. A sparse-specific threshold (0.17 vs 0.23 global) recovers +0.025. |
| Is it a feature-set artefact? | **Yes — this is the dominant cause.** |

On a no-text row the `text_raw` arm is 35 raw features plus an **all-zero TF-IDF
block**, so the model was running on a fraction of its inputs precisely where it
had least to work with.

### Results on the 436 sparse test rows

| System | micro-F1 | vs baseline |
|---|---|---|
| subset frequency prior (top-4) | 0.1777 | floor |
| `raw` @ global thr | 0.3042 | −0.035 |
| **`text_raw` @ global thr (baseline)** | **0.3393** | — |
| specialist fitted on sparse rows, `text_raw` | 0.3429 | +0.004 |
| `clinical` @ sparse thr | 0.3523 | +0.013 |
| `text_raw` @ **sparse-specific thr** | 0.3642 | **+0.025** |
| specialist fitted on sparse rows, `all` | 0.3676 | +0.028 |
| **`all` @ global thr** | **0.4153** | **+0.076** |

### Two findings

**1. A routed specialist is worse than the shared model.** `specialist[all]`
scores 0.3676 against shared `all` at 0.4153. Fitting on 1,784 sparse train rows
loses more to reduced sample size than it gains from specialisation. Routing is
not worth the complexity here.

**2. The fix is a configuration change, not new modelling.** Use the feature arm
that actually has inputs for these rows. This also lifts all-rows performance
(0.4890 vs 0.4628), so it is not a subgroup trade-off.

### Correction to a previously reported number

The 0.339 figure was measured on the **`text_raw` deployable arm**, not on the
shipped recommender. `docs/recommender.md` selects `all` for `drug_classes`
(test micro-F1 0.4934), so **the deployed model already scores 0.4153 on sparse
rows**, not 0.339. The "worst subgroup" framing was accurate for the deployable
variant and overstated for the deployed one.

The honest split:

| Variant | Sparse rows | All rows |
|---|---|---|
| deployed (`all`, includes confounded features) | 0.4153 | 0.4934 |
| deployable-only (`text_raw` + sparse threshold) | 0.3642 | 0.4628 |

The deployable/full gap is **widest on sparse rows** (0.051) — expected, since
with no narrative the model leans hardest on the tabular features, and those are
exactly the ones carrying prescriber/site/era confounding. Sparse encounters are
therefore where a deployed system is most dependent on signal that will not
transfer forward in time, and where calibrated abstention is most warranted.


---

## 8. Calibrated abstention

`src/phcrx/recommend/abstain.py`. Selective prediction: rank encounters by a
confidence signal, cover the top fraction, decline on the rest.

Probabilities are calibrated first (sigmoid, chosen by K-fold CV *inside*
validation), macro-ECE 0.0144 → 0.0080 on test, so a confidence signal built
from them means what it says.

### The confidence signal is real, and one candidate was worse than random

Selected on validation by risk-coverage AUC, with a random-abstention control:

| signal | val AUC |
|---|---|
| **expected_f1** (expected micro-F1 of the emitted set) | **0.5527** |
| max_prob | 0.5408 |
| n_above_thr | 0.5153 |
| margin | 0.5090 |
| *random (control)* | *0.4896* |
| neg_entropy | **0.4484** — worse than random |

Including the control was worth it: negative entropy would have *hurt*, and
without a random baseline that would have been invisible.

### Test risk-coverage

| coverage | n | micro-F1 | precision | recall | random control |
|---|---|---|---|---|---|
| 10% | 278 | **0.6276** | 0.5452 | 0.7394 | 0.4722 |
| 30% | 834 | 0.5734 | 0.4924 | 0.6864 | 0.4809 |
| 50% | 1,390 | 0.5440 | 0.4659 | 0.6536 | 0.4818 |
| 80% | 2,224 | 0.4995 | 0.4316 | 0.5928 | 0.4822 |
| 100% | 2,780 | 0.4852 | 0.4291 | 0.5581 | 0.4852 |

The signal beats the random control at every coverage level, by 0.02–0.16.

### Abstention self-targets sparse encounters

At 80% global coverage, nobody told the model which rows were sparse:

| complaint spans | n | abstained |
|---|---|---|
| **0 (sparse)** | 436 | **43.8%** |
| 1 | 1,385 | 21.3% |
| 2 | 561 | 9.6% |
| 3+ | 398 | **4.0%** |

An eleven-fold difference between the sparsest and richest groups, emerging from
the calibrated probabilities alone. This is the mechanism working as
hypothesised rather than a rule imposed on it.

### Sparse-subgroup policy

| coverage of sparse rows | n | micro-F1 | precision |
|---|---|---|---|
| 100% (no abstention) | 436 | 0.4172 | 0.3789 |
| 60% | 262 | 0.4407 | 0.3796 |
| 50% | 218 | 0.4637 | 0.3994 |
| **40%** | **174** | **0.4923** | 0.4221 |
| *(rich-note rows, no abstention)* | *2,344* | *0.4931* | — |

**Correcting the script's own verdict.** `abstain.py` printed "sparse rows never
reach rich-note quality at any coverage tested", because it tests `>=` against
0.4931. At 40% coverage sparse rows reach **0.4923** — short by **0.0008**,
which is noise. The accurate statement is that sparse encounters reach parity
with richly-described ones once the least-confident 60% are declined. An
automated pass/fail on a strict inequality was misleading and the number should
be read directly.

### Recommended operating point

Two defensible policies:

1. **Global 80% coverage** — micro-F1 0.4995 (from 0.4852), declining on 20% of
   encounters, 44% of which are sparse. Simple, single threshold.
2. **Sparse-targeted** — full coverage on notes with a parsed complaint, 40%
   coverage on sparse ones. Sparse encounters then perform at parity with rich
   ones (0.4923 vs 0.4931), at the cost of declining 262 encounters (9.4% of
   all test rows).

Policy 2 is preferable clinically: it never withholds a suggestion from an
encounter that carries a clinical narrative, and it is honest precisely where
the model was leaning hardest on prescriber/site/era confounding.


---

## 9. Temporal transfer at class level — the headline was overstated

`src/phcrx/temporal/class_temporal.py`. The temporal split is derived **in
memory** from the `year` column; `preprocess --split temporal` was deliberately
not re-run because it rewrites the shared Parquets in place and silently
invalidated downstream artefacts in an earlier session.

| target | arm | patient | temporal | change [95% CI] |
|---|---|---|---|---|
| class46 | **text_raw** (deployable) | 0.4615 | **0.3547** | **−23.1%** [−26.3, −20.0] |
| class46 | `all` (currently deployed) | 0.4934 | 0.2561 | **−48.1%** [−50.2, −46.0] |
| cat89 | text_raw | 0.4077 | 0.2947 | −27.8% [−31.1, −24.3] |
| cat89 | `all` | 0.4340 | 0.2197 | −49.4% [−51.6, −47.1] |

### The headline correction

The published claim — *"forward-in-time performance collapses to the
frequency-prior floor (0.208 → 0.066, −66%)"* — was measured at **brand level
with the autoregressive neural model**. At class level with deployable features
the drop is **−23.1%**, and the temporal result (0.3547) sits well clear of its
prior floor (0.2303). At brand level the model had fallen *to* the floor (0.066
vs 0.069). **"Collapses to the floor" is no longer an accurate description.**

Caveat: −66% and −23.1% differ in *both* target and estimator. A `brand717`
control under the same linear estimator is still running to separate the two.
Until it lands, the comparison is indicative, not controlled.

### The arm ranking reverses — and this changes what to deploy

Under the patient split `all` beats `text_raw` (0.4934 vs 0.4615). Under
temporal shift it **loses** (0.2561 vs 0.3547) and degrades twice as fast. The
prescriber/site/era features are not merely non-transferable, they are actively
harmful once the era moves.

`pipeline.py` selects the feature set on **validation, which uses the patient
split**, so it cannot see this and currently selects `all`. Given 74% prescriber
turnover, drift is certain, and **`text_raw` is the correct deployment choice**.

**Open task (new):** feature-set selection should be made under a temporal
criterion, not a random one. `docs/recommender.md` §5 currently recommends the
`all` variant and needs revising.

## 10. Abstention: an operating point measured on a model that was never deployed

Closing the wiring task surfaced a defect in §8. `abstain.py --analysis` refits
its own arm at `MultiLabelOvR(C=4.0, thr=0.21)`, while the model `pipeline.py`
actually serves for `drug_classes` is the validation winner at **C=1.0,
thr=0.24**. So §8's absolute numbers describe a model that is not deployed:

| | §8 as reported | deployed reality |
|---|---|---|
| full coverage | 0.4852 | **0.4934** |
| `global80` covered | 0.4995 | **0.5089** |
| lift | +0.0143 | +0.0154 |

The conclusion survives — the lift is the same and slightly larger — but the
absolute figures in §8 were wrong, and re-running the analysis at `--C 1.0`
reproduces the served numbers to four decimals.

Under the deployed model, `sparse_targeted` takes sparse encounters from
**0.4242 → 0.5153**, which *exceeds* rich-note performance (0.5010) rather than
merely reaching parity. Cost: 276 declined encounters (9.9%).

Self-targeting still holds with complaint counts hidden from the rule: 44.7% /
22.5% / 7.7% / 4.3% declined at 0 / 1 / 2 / 3+ complaints.

**Advice and tests were decided separately, not suppressed by side effect.**
Advice abstains under `sparse_targeted` only — its 80% cut-point is degenerate
because 68.2% of validation rows emit an empty advice set and tie at
`expected_f1 = 0` — and the sparse cut lifts sparse advice 0.6356 → 0.6749.
**Tests never abstain**: both cut-points are degenerate at 84.9% empty. Declining
to build a mechanism that cannot discriminate is the right outcome.
