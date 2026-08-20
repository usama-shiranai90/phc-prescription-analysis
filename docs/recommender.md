# PHC prescription-component recommender — final evaluation

A per-component recommender built from everything this project measured.
Held-out test: 2,780 encounters, patient-level split, no patient straddling.
Every vectoriser, scaler, threshold and calibrator fitted on train (thresholds
and calibrators on validation); test touched once.

```
python -m src.phcrx.recommend.pipeline     # fit + select feature set on val
python -m src.phcrx.recommend.evaluate     # held-out test, bootstrap CIs
python -m src.phcrx.recommend.recommend --demo --n 10
```

---

## 1. Results, against the floor that matters

Each component is scored against the *strongest trivial baseline for that
component*, not against zero. For sparse set targets the always-empty predictor
is a genuinely hard baseline and is reported alongside.

| Component | Metric | Trivial floor | **Recommender** | Verdict |
|---|---|---|---|---|
| **Any pharmacotherapy** | AUROC | 0.500 | **0.8743** | ship |
| | avg precision | 0.7896 | **0.9518** | |
| **Number of drugs** | MAE (train-median) | 1.3543 | **0.8802** | ship |
| **Drug classes** (46) | micro-F1 | 0.0000 | **0.4934** | ship |
| | macro-F1 | 0.0000 | **0.3379** | |
| | tail-macro-F1 | 0.0000 | **0.2249** | |
| | Jaccard | 0.2155 | **0.4138** | |
| **Advice** (32) | micro-F1 | 0.0000 | **0.5826** | ship |
| | Jaccard | 0.7090 | **0.7674** | |
| | exact-set | 0.7090 | 0.6583 | *below floor* |
| **Diagnostic tests** (19) | micro-F1 | 0.0000 | **0.3634** | **do not ship as a set** |
| | Jaccard | **0.8550** | 0.8423 | *below floor* |
| | exact-set | **0.8550** | 0.8219 | *below floor* |
| **Drug form** (4) | accuracy | 0.9235 | 0.9267 | marginal |
| | macro-F1 | 0.2401 | **0.3790** | |

### The drug-class result in context

micro-F1 **0.4934** on 46 pharmacological classes, against the project's history:

| Model | Label space | micro-F1 |
|---|---|---|
| Original neural, brand level | 719 brands | 0.208 |
| Original neural, category level | 89 categories | 0.348 |
| Neural autoregressive | 46 classes | 0.360 |
| Linear `lr_text_tab_all` | 46 classes | 0.487 |
| Neural multi-label `mlc_full_tab` | 46 classes | 0.486 |
| **This recommender** | **46 classes** | **0.4934** |

The tail matters more than the headline. The original model's tail recall was
**0.006**; tail-macro-F1 here is **0.2249**. Long-tail classes went from
effectively unpredicted to meaningfully predicted — the single biggest practical
change in the project.

## 2. A correction: diagnostic tests are NOT "highly predictable"

`docs/feature_importance.md` reported test-order AUROC of **0.923–0.979** and
concluded tests were among the most predictable components. **Under set
metrics that conclusion does not hold.**

85.5% of encounters have no test ordered. Always-empty therefore scores Jaccard
0.8550 and exact-set 0.8550, and the recommender comes in *below* both (0.8423 /
0.8219). AUROC is insensitive to prevalence; at 14.5% positive rate a strong
per-label ranker can still lose to "predict nothing" on set agreement.

Tests should be surfaced as a **ranked per-test probability** ("consider FBS,
p=0.64"), never as a predicted set. The same caution applies more weakly to
advice, which clears the floor on Jaccard (0.7674 vs 0.7090) but not on
exact-set match.

## 3. Components deliberately excluded

`dose`, `duration` and `instruction` are not modelled. Measured accuracy 0.506 /
0.503 / 0.725 against a strong majority class, falling to 0.474 / 0.438 / 0.636
once prescriber and site are removed. They are predictable **from the
prescriber, not the patient** — habit, not clinical decision. Surfacing them as
recommendations would misrepresent what the model knows.

## 4. Calibration

Probabilities are calibrated on validation (isotonic or sigmoid, chosen by
cross-validated ECE). Post-calibration test ECE:

| Component | Method | test ECE (raw → calibrated) |
|---|---|---|
| any_drug | isotonic | 0.0246 → 0.0223 |
| drug_classes | sigmoid | 0.0076 → **0.0028** |
| advice | sigmoid | 0.0086 → **0.0027** |
| tests | sigmoid | 0.0059 → **0.0027** |
| drug_form | isotonic | 0.0218 → **0.0051** |

Well-calibrated probabilities are what make a suggestion list usable: a clinician
can act on "p=0.92" differently from "p=0.25".

## 5. Deployable vs. full feature sets

Feature sets are split by whether they can be computed for a fresh encounter
without prescriber/site/temporal context:

| Component | Full | Deployable-only | Cost |
|---|---|---|---|
| drug_classes | 0.4934 | 0.4610 | −0.032 |
| advice | 0.5826 | 0.4535 | −0.129 |
| tests | 0.3634 | 0.2917 | −0.072 |
| any_drug (AUROC) | 0.8743 | 0.8522 | −0.022 |

The gap is real but the non-deployable features are the known confounders
(prescriber identity, site, era) that do not transfer forward in time. Note the
tail is *better* without them (drug-class tail-macro 0.2339 deployable vs 0.2249
full) — consistent with the earlier finding that confounders help the head and
hurt generalisation.

**Recommendation: deploy the deployable variant.** The full variant's advantage
is largely the prescriber-identity shortcut that collapses across eras.

## 6. Abstention: the recommender does not answer every encounter

`recommend.py` no longer recommends unconditionally. It loads
`results/rx_generation/recommend/abstention_policy.json` and, per component and
per encounter, either emits a suggestion set or prints `INSUFFICIENT
INFORMATION`. The policy is selected with `--policy`:

| `--policy` | behaviour |
|---|---|
| `none` | recommend on every encounter — the pre-abstention system |
| `global80` | cover the most confident 80% of encounters |
| **`sparse_targeted`** | cover every note with a parsed complaint; on a note with none, cover only the most confident 40% — **default** |

### The confidence signal, and why it is not interchangeable

Confidence is `expected_f1`: the expected micro-F1 of the set the model would
emit, computed from the calibrated probabilities alone
(`2·Σⱼpⱼ·[pⱼ≥thr] / (Σⱼ[pⱼ≥thr] + Σⱼpⱼ)`). It was chosen on validation by
risk-coverage AUC against a **random-abstention control**, and the control
earned its place: negative entropy scored 0.4484 against random's 0.4896, i.e.
*worse than declining at random*. Swapping the signal is not a free choice.

Calibration comes first (§4) — a confidence signal built from miscalibrated
probabilities would not mean what it says.

### Cut-points are absolute, and fixed on validation

The cut-point a row is compared against is a **number frozen on the validation
split**, not a quantile of the batch in hand. This is a deployment constraint,
not a stylistic one: a recommender scoring a single walk-in patient has no
batch to take a quantile of, and a quantile taken over the test split would be
a test statistic entering the decision rule. `abstain.py --policy` writes the
whole grid; these are the two the policies use:

| Component | mode | val AUC (`expected_f1` / random) | `global80` cut | `sparse40` cut |
|---|---|---|---|---|
| drug_classes | full | 0.5641 / 0.4984 | 0.2114 | 0.3497 |
| drug_classes | deployable | 0.5464 / 0.4799 | 0.1763 | 0.2125 |
| advice | full | 0.6154 / 0.5865 | *degenerate* | 0.4772 |
| advice | deployable | 0.5225 / 0.4854 | *degenerate* | 0.4033 |
| tests | full | 0.5015 / 0.4289 | *degenerate* | *degenerate* |
| tests | deployable | 0.4502 / 0.3448 | *degenerate* | *degenerate* |

The deployable models get their own cut-points because they are different
models with different thresholds; applying the full model's numbers to them
would put the rule on the wrong probability scale. If a component's recorded
threshold has drifted from the loaded model's, `recommend.py` warns and
disables abstention **for that component** — covering everything is the safe
direction on a stale artefact, since it degrades to the pre-abstention system
rather than withholding on a number that no longer refers to anything.

### Advice and tests are governed by their own decision, not by drug classes

An abstention on drug classes does **not** silence advice or tests. Each
component gets the same machinery run on its own validation confidences,
including its own random control, and the answer differs per component:

- **drug_classes** — abstains under both policies.
- **advice** — abstains under `sparse_targeted` only. Its 80%-coverage
  cut-point is *degenerate*: 68.2% of validation rows emit an empty advice set
  and are therefore tied at `expected_f1 = 0`, so no threshold can separate the
  bottom 20%. The 40%-coverage cut over sparse rows is not tied, and it works
  (below).
- **tests** — never abstains. Both cut-points are degenerate (84.9% of
  validation rows emit nothing), and §2 already rules tests unfit to surface as
  a set at all.

Marked *degenerate* above means exactly that tie at the floor, and it is
recorded in the artefact with its reason rather than silently passing through.

### Measured, through the inference path

`recommend.py --verify-policy` runs every policy over the whole test split
through `Recommender.recommend()` — the same single-encounter code path a
clinician's request would take — and writes `abstention_wired.json`:

| policy | component | coverage | micro-F1 on covered | sparse coverage | sparse micro-F1 | rich micro-F1 |
|---|---|---|---|---|---|---|
| `none` | drug_classes | 1.000 | 0.4934 | 1.000 | 0.4242 | 0.5010 |
| `global80` | drug_classes | 0.796 | **0.5089** | 0.553 | 0.4602 | 0.5138 |
| `sparse_targeted` | drug_classes | 0.901 | 0.5020 | 0.367 | **0.5153** | 0.5010 |
| `none` | advice | 1.000 | 0.5826 | 1.000 | 0.6356 | 0.5575 |
| `sparse_targeted` | advice | 0.903 | **0.5922** | 0.378 | **0.6749** | 0.5575 |
| any | tests | 1.000 | 0.3634 | 1.000 | 0.5012 | 0.3046 |

Two results worth stating plainly:

1. **`sparse_targeted` closes the sparse/rich gap and then some.** Sparse
   encounters go from micro-F1 0.4242 to **0.5153**, against 0.5010 for
   encounters that carry a clinical narrative. The subgroup that was the
   worst-served is, after declining its least confident 63%, no longer the
   worst-served. The cost is 276 declined encounters, 9.9% of the test split.
2. **Abstention self-targets sparsity without being told to.** Under
   `global80`, which knows nothing about complaint counts, the decline rate
   falls monotonically as the note gets richer — 44.7% (0 complaints, n=436),
   22.5% (1, n=1385), 7.7% (2, n=561), 4.3% (3+, n=398). A tenfold spread that
   nobody encoded; it falls out of the calibrated probabilities.

### Reproduction against the batch analysis

`docs/remaining_work.md` §8 reports this curve at 100% coverage 0.4852 and 80%
coverage 0.4995. The wired path lands at 0.4934 and 0.5089. **The gap is the
model, not the wiring.** `abstain.py --analysis` refits an arm of its own at
`MultiLabelOvR(C=4.0)`; the drug_classes model `pipeline.py` selected on
validation and `recommend.py` actually serves is `C=1.0`. Re-running the same
analysis at the served hyper-parameter (`--C 1.0 --out-tag _served`, written to
`abstention_served.json`) reproduces the inference path to the fourth decimal:

| quantity | batch (served C=1.0) | wired inference | Δ |
|---|---|---|---|
| drug_classes, full coverage | 0.4934 | 0.4934 | **0.0000** |
| sparse rows, no abstention | 0.4242 | 0.4242 | **0.0000** |
| rich rows, no abstention | 0.5010 | 0.5010 | **0.0000** |
| drug_classes, `global80` covered | 0.5081 | 0.5089 | +0.0007 |
| sparse rows, `sparse_targeted` | 0.5069 | 0.5153 | +0.0083 |

The two non-zero rows are the two places the deployed rule *cannot* match a
batch quantile by construction: the frozen cut-point lands at 79.6% rather than
80.0% coverage, and at 36.7% rather than 40.0% of sparse rows. So §8's absolute
numbers describe a model that was never deployed, while its *lift* (+0.0143 at
80% coverage) matches what deployment gets (+0.0154).

The stronger check is structural rather than numeric. `--verify-policy`
confirms that the rows the per-encounter rule covers are **exactly** the top-*k*
rows by confidence that the batch ranking would have taken — 0 rows differing,
for every component, over the population each policy actually ranks. The two
selection mechanisms are not merely close, they are identical; the only thing
separating deployment from the published curve is where on that curve a
validation-frozen cut-point lands.

### What the clinician sees

An abstained component prints the reason and its arithmetic, then the ranked
probabilities marked `~` as audit output — withholding a suggestion is not a
reason to hide what the model thought:

```
  DRUG CLASSES
    INSUFFICIENT INFORMATION -- no drug-class suggestion (confidence 0.28 < 0.35)
      policy 'sparse_targeted': expected_f1 0.2808 < 0.3497, the 40%-coverage cut-point fixed on sparse validation encounters
      ranked probabilities below are audit only -- nothing here was surfaced:
    ~ 0.27 ###.........  iron_supplement  <- in gold
```

That example is honest about the cost: `iron_supplement` was in gold and would
have been surfaced. Abstention trades recall on encounters the model cannot
read for precision on the ones it can, and 9.9% of encounters pay that price.

```
python -m src.phcrx.recommend.abstain              # analysis + policy artefact
python -m src.phcrx.recommend.abstain --policy     # artefact only (fast)
python -m src.phcrx.recommend.recommend --verify-policy
python -m src.phcrx.recommend.recommend --demo --policy none
```

## 7. Qualitative check

`recommend.py --demo` prints gold-vs-predicted with probability bars. Example
(`pid=11978`, 39y M, glucose 174.6 FBS, note *"h/o DM+h/o asthma"*):

- any pharmacotherapy p=0.72 → prescribe (gold: yes)
- top drug class `antidiabetic_biguanide` p=0.25
- advice: *"Avoid taking sweetened food"* 0.92, *"Follow the Diabetic
  food-chart"* 0.89, *"Walk or exercise regularly"* 0.70 — all three in gold
- test: FBS p=0.64

The diabetes signal is captured cleanly. The asthma half of the note is not —
`antihistamine` and `respiratory_other` were both in gold and both ranked below
threshold. Multi-condition notes remain the visible failure mode, consistent
with 39.5% of notes being multi-complaint.

## 8. What is fit to surface

| Output | Surface? | As what |
|---|---|---|
| Any pharmacotherapy | yes | a prescribe / withhold prompt |
| Number of drugs | yes | an expected-count hint |
| Drug classes | yes | ranked list with probabilities, top-6 |
| Advice | yes | ranked list |
| Diagnostic tests | **ranked probabilities only** | never a predicted set |
| Drug form | marginal | default-form hint only |
| Dose / duration / instruction | **no** | not modelled |

None of this is clinically validated. Every number is agreement with historical
prescribing; `docs/annotation_protocol.md` describes the clinician study that
would convert it into an accuracy and safety claim, and that study has not been
run.
