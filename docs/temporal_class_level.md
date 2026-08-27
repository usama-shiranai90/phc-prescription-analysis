# The temporal collapse, re-measured at class level with the deployed model


> **⚠ Figures on this page for the `all` arm are superseded.** They were
> computed from `features.parquet`, fitted on the patient-split train rows and
> therefore **not era-clean**. Refitting the engineered columns on ≤2015
> (`features_temporal.parquet`, `--features temporal`) gives:
>
> | arm | patient | temporal | drop |
> |---|---|---|---|
> | `text_raw` | 0.4615 | 0.3547 | **−23.1%** (unchanged — control) |
> | `all` — leaked | 0.4934 | 0.2561 | −48.1% |
> | **`all` — era-clean** | 0.4938 | **0.3081** | **−37.6%** [−40.1, −34.9] |
>
> The leak was **pessimistic**, not optimistic — the opposite of what was
> predicted. `text_raw` still wins under temporal shift, but by **0.047, not
> 0.099**, so the deployment argument rests on a much narrower margin than
> first reported. Note `all` has the better *tail*-macro under shift (0.0453 vs
> 0.0336).


**Verdict: the published −66% deployability headline is stale, and it is wrong
in both directions at once.** Measured on the target and the estimator this
project actually deploys, forward-in-time micro-F1 falls **0.4615 → 0.3547, a
−23.1% [−26.3, −20.0] drop** — not −66%, and nowhere near the frequency-prior
floor. But the *deployed* feature set does far worse than the deployable one:
`all` falls **0.4934 → 0.2561, −48.1% [−50.2, −46.0]**, ending up only +0.026
micro-F1 above a frequency prior. Under temporal shift the arm ordering
**reverses**: the feature set chosen on a random split is the worst to ship.

Roughly four fifths of the correction comes from the *target* (brand → class),
one fifth from the *estimator* (autoregressive decoder → linear multi-label);
a brand-target control run on the same code path separates the two (§6).

```
python -m src.phcrx.temporal.class_temporal
python -m src.phcrx.temporal.class_temporal --targets brand717 \
       --arms text_raw all --c-grid 1.0 2.0 --tag _brand
```

Raw: `results/rx_generation/temporal/` — `class_temporal.csv`,
`split_deltas.csv`, `class_temporal.json`, `run_main.log`, and the
`*_brand.csv` / `run_brand.log` counterparts.

---

## 1. Why this was re-run

`docs/research_frame.md` §7.4, `docs/prescription_generation.md` §5 and
`docs/model_comparison.md` all publish the same deployability claim:

> forward-in-time performance collapses to the frequency-prior floor — micro-F1
> 0.208 → 0.066, a −66% drop; a random-split evaluation overstates deployable
> performance by roughly 3×.

That number was measured on a **brand** target (719 labels) with the
**autoregressive** encoder–decoder. Both choices have since been superseded by
work in this same repository:

| Superseded choice | What replaced it | Evidence |
|---|---|---|
| 719 brands | 46 normalised pharmacological classes (97.6% of orders mapped) | `drug_normalization.parquet`; cross-era strict unseen-order rate **3.4%** for classes vs **12.2%** for brands |
| autoregressive set decoder | multi-label head / linear one-vs-rest | `docs/neural_mlc.md`: **+0.109 micro-F1** from dropping autoregression alone |

So the figure the project quotes as *the* deployment estimate is measured on a
target nobody proposes to predict any more, with an estimator nobody proposes
to ship. This document re-measures it on the target and the estimator that are
actually deployed (`src/phcrx/recommend/`, `drug_classes` component, test
micro-F1 0.4934 at class level).

## 2. Method

### 2.1 The split is derived in memory — and `preprocess` was not re-run

`python -m src.phcrx.preprocess --split temporal` **rewrites**
`data/processed/rxgen_*.parquet` in place and reshuffles the drug and word
vocabularies. Every downstream artefact is keyed to the current corpus —
`features.parquet`, the `drug_normalization.parquet` drug ids, the cached
neural predictions in `src/phcrx/bench/`, the fitted recommender in
`results/rx_generation/recommend/models/`. Regenerating the corpus silently
invalidates all of them, and a previous session did exactly that. **It was not
run here, in any form.**

It is also unnecessary. `rxgen_encounters.parquet` already carries a `year`
column, so the temporal split is one boolean over rows, and
`recommend.corpus.split_index` reads `df["split"]`. The entire change is one
overwritten column on a **copy** of the frame:

```python
out = df.copy()
out["split"] = np.where(year <= 2015, "train",
                        np.where(year <= 2016, "val", "test"))
```

Era boundaries are the ones the published brand-level result used
(`DataConfig.temporal_train_end = 2015`, `temporal_val_end = 2016`), so the
comparison holds the era definition fixed and varies only the target and the
estimator.

| Split | train | val | test | train years | test years |
|---|---|---|---|---|---|
| patient (existing `split` column) | 9,900 | 1,394 | 2,780 | 2012–2025 | 2012–2025 |
| temporal (derived here) | 9,961 | 1,233 | 2,880 | 2012–2015 | 2017–2025 |

The two are almost exactly the same size, which is what makes the comparison
readable: 2,780 vs 2,880 test encounters, of which **600 are shared**.

### 2.2 What is fitted, and where

One code path serves both splits. For every (target, feature set, split):

1. **fit on TRAIN** — the TF-IDF vocabulary and IDF weights, the typo
   corrector, the imputation medians, the standardisation constants and the
   one-hot levels are all learned inside `Pipeline.fit`, which only ever sees
   that split's train slice;
2. **select C on VAL** over {1, 2, 4, 8};
3. **calibrate on VAL** — method chosen by K-fold CV *inside* val;
4. **set the operating point on VAL** — one global threshold, tuned for
   micro-F1 on the calibrated val probabilities;
5. **read TEST once.**

The pipeline, the estimator and every metric are imported from
`src/phcrx/recommend/{blocks,corpus,metrics}.py`; nothing is reimplemented.

**The refit is evidenced, not asserted.** The fitted TF-IDF vocabulary is
**5,727 terms under the patient split and 5,914 under the temporal split** —
different fits on different rows, as required. `blocks._MEMO` keys on a SHA-1
digest of the exact training texts, so a patient-train fit cannot be served to
a temporal-train request. `_leak_report` additionally asserts, before anything
is fitted, that the three slices share no encounter, that temporal-train stops
at 2015, that temporal-val is exactly 2016 and that temporal-test starts at
2017.

### 2.3 One leak that could not be closed from here — and which way it biases

The `all` feature set reads 309 engineered columns from
`data/processed/features.parquet`. `features/build_features.py` built that
artefact with `is_train = (split == "train")` — the **patient** split. Its
TF-IDF/SVD text basis, ICD chapter levels, categorical levels and
leave-one-out site/prescriber target encodings therefore all saw later-era rows
during their offline fit. Rebuilding it is out of scope here (it is P2 item 11
in `docs/remaining_work.md`), so:

> **`all` under the temporal split is an optimistic upper bound.**

That bias runs in `all`'s favour, which matters for how §4 should be read: the
finding there is that `all` degrades *more* than the leak-free `text_raw`, and
it is established **despite** `all` being handed an advantage. Closing the leak
can only make `all` look worse.

The `all` set also contains `year`, `month`, `season`, `day_of_week` and
`days_since_cohort_start` as literal features. Under the temporal split every
test value of `year` and `days_since_cohort_start` lies strictly outside the
training range — pure extrapolation.

### 2.4 Uncertainty

Bootstrap, 2,000 resamples throughout.

- **Within a split** — one fixed set of resample indices is shared by every
  system, so recommender-vs-floor and arm-vs-arm differences are **paired**
  (`recommend.metrics.Resampler`; set metrics are recomputed exactly from
  per-row-per-label counts under each resample, not approximated).
- **Across splits** — the two splits do not share a test set, so the paired
  machinery does not apply. Each split gets its own **independent** resampler
  and the difference distribution is taken across independent draws. Where the
  test sets do overlap (600 rows) the two estimates are positively correlated,
  so treating them as independent **overstates** the width of the difference
  interval. The cross-split intervals below are therefore conservative.

---

## 3. Results — 46 pharmacological classes, both splits

All numbers are held-out test, 2,000-resample bootstrap, threshold and
calibration fitted on that split's validation rows only.

### Patient split (2,780 test encounters)

| System | micro-F1 [95% CI] | macro-F1 | tail-macro | Jaccard | exact | mean set |
|---|---|---|---|---|---|---|
| always-empty | 0.0000 | 0.0000 | 0.0000 | 0.2155 | 0.2155 | 0.00 |
| prior top-4 | 0.2345 [0.2249, 0.2434] | 0.0262 | 0.0000 | 0.1314 | 0.0004 | 4.00 |
| recommender `text_raw` *(deployable)* | 0.4615 [0.4501, 0.4730] | 0.3226 | 0.2064 | 0.3901 | 0.1831 | 2.16 |
| **recommender `all`** *(deployed)* | **0.4934** [0.4818, 0.5041] | **0.3379** | **0.2249** | **0.4138** | **0.1928** | 2.28 |

`all` at 0.4934 reproduces the deployed recommender exactly — same selected
C = 1.0, same sigmoid calibration, same 0.24 threshold, same micro-F1 0.4934 /
macro 0.3379 / tail-macro 0.2249 / Jaccard 0.4138 as `pipeline.json` and
`docs/recommender.md` §1. That is the check that this code path scores the
same thing the deployed pipeline scores.

### Temporal split (2,880 test encounters, all ≥2017)

| System | micro-F1 [95% CI] | macro-F1 | tail-macro | Jaccard | exact | mean set |
|---|---|---|---|---|---|---|
| always-empty | 0.0000 | 0.0000 | 0.0000 | **0.2375** | **0.2375** | 0.00 |
| prior top-4 | 0.2303 [0.2209, 0.2398] | 0.0258 | 0.0000 | 0.1281 | 0.0007 | 4.00 |
| **recommender `text_raw`** *(deployable)* | **0.3547** [0.3437, 0.3662] | **0.1601** | 0.0336 | 0.1897 | 0.0135 | 2.90 |
| recommender `all` *(deployed)* | 0.2561 [0.2475, 0.2649] | 0.1330 | **0.0801** | 0.1492 | 0.0017 | 5.70 |

### The drop, with intervals

Temporal minus patient. Unpaired bootstrap (§2.4), so these are conservative.

| System | metric | patient | temporal | delta [95% CI] | relative [95% CI] |
|---|---|---|---|---|---|
| `text_raw` | micro-F1 | 0.4615 | 0.3547 | −0.1068 [−0.1232, −0.0902] | **−23.1% [−26.3, −20.0]** |
| | macro-F1 | 0.3291 | 0.1639 | −0.1652 [−0.1901, −0.1405] | −50.1% [−55.1, −44.8] |
| | tail-macro | 0.2037 | 0.0331 | −0.1706 [−0.2020, −0.1420] | −83.7% [−89.4, −76.7] |
| | Jaccard | 0.3902 | 0.1897 | −0.2005 [−0.2165, −0.1847] | −51.4% [−54.1, −48.6] |
| | exact | 0.1833 | 0.0136 | −0.1697 [−0.1842, −0.1549] | −92.6% [−94.8, −90.1] |
| `all` | micro-F1 | 0.4934 | 0.2561 | −0.2374 [−0.2512, −0.2228] | **−48.1% [−50.2, −46.0]** |
| | macro-F1 | 0.3441 | 0.1360 | −0.2080 [−0.2349, −0.1831] | −60.4% [−64.5, −56.0] |
| | tail-macro | 0.2208 | 0.0791 | −0.1417 [−0.1783, −0.1062] | −64.0% [−71.8, −54.2] |
| | Jaccard | 0.4141 | 0.1492 | −0.2649 [−0.2803, −0.2501] | −64.0% [−65.9, −62.0] |
| | exact | 0.1932 | 0.0017 | −0.1915 [−0.2055, −0.1770] | −99.1% [−99.8, −98.2] |
| prior top-4 | micro-F1 | 0.2344 | 0.2301 | −0.0043 [−0.0171, +0.0088] | −1.8% [−7.1, +3.8] |

The prior moving by −1.8% with an interval straddling zero is the control: the
*task* is about as easy in the later era. What changes is what the model
learned, not what it is being asked to do.

### Against the floors, on the same rows

| Split | arm | delta vs prior top-k [95% CI] | p(better) |
|---|---|---|---|
| patient | `text_raw` | +0.2271 [+0.2147, +0.2399] | 1.000 |
| patient | `all` | +0.2590 [+0.2476, +0.2705] | 1.000 |
| temporal | `text_raw` | **+0.1245** [+0.1161, +0.1328] | 1.000 |
| temporal | `all` | **+0.0259** [+0.0197, +0.0323] | 1.000 |

**This is the single most important correction.** The published claim is that
forward-in-time performance *"collapses to the frequency-prior floor"*. At
class level with the deployable arm it does not: `text_raw` clears the prior by
**+0.1245 micro-F1** with an interval nowhere near zero, and clears the
always-empty floor by +0.3547. The deployable model still transfers forward in
time. It transfers worse, by a well-measured amount, but "no useful transfer"
is not what the data say.

### Where the collapse is real: set agreement and the tail

Two things genuinely do fall apart, and the corrected headline must not hide
them:

- **Jaccard and exact-set match fall *below* the always-empty floor.** Under
  the temporal split always-empty scores Jaccard 0.2375; `text_raw` scores
  0.1897 (delta −0.0480 [−0.0692, −0.0269]) and `all` scores 0.1492 (delta
  −0.0885 [−0.1081, −0.0683]). Micro-F1 survives because the model still gets
  individual classes right; whole-prescription agreement does not survive at
  all. **As a set generator, forward in time, the model is worse than proposing
  nothing.** It is usable as a ranked suggestion list, not as a draft
  prescription.
- **The tail is not learned forward in time.** tail-macro-F1 falls 0.2064 →
  0.0336 for `text_raw`, a −83.7% relative drop — far larger than the micro-F1
  headline. The rare-class gain that `docs/recommender.md` calls "the single
  biggest practical change in the project" is a same-era gain.

## 4. `all` vs `text_raw` — the deployed feature set is the wrong one to ship

**Yes: `all` degrades substantially more than `text_raw`, and the ordering
reverses.**

| Arm | patient | temporal | relative drop |
|---|---|---|---|
| `text_raw` (deployable, leak-free) | 0.4615 | **0.3547** | **−23.1%** [−26.3, −20.0] |
| `all` (deployed, carries temporal + site features) | **0.4934** | 0.2561 | **−48.1%** [−50.2, −46.0] |

On a random patient-level split `all` wins by +0.0319 micro-F1, which is why
`recommend/pipeline.py` selected it for the `drug_classes` component. Under
temporal shift `all` **loses to `text_raw` by 0.0986 micro-F1** — three times
the margin by which it won, in the opposite direction. The feature set chosen
on the random split is not merely suboptimal forward in time; it is actively
harmful.

Three mechanisms, all visible in the measured numbers:

1. **Extrapolation on literal time features.** `all` carries `year`, `month`,
   `season`, `day_of_week` and `days_since_cohort_start`. Under the temporal
   split every test value of `year` and `days_since_cohort_start` lies strictly
   outside the training range.
2. **Site and prescriber encodings evaporate.** `era_shift.json` measures 74%
   of later encounters written by a prescriber absent from training and 58% at
   a new site. The leave-one-out prescriber/site prescribing profiles — 33 of
   the 309 engineered columns — fall back to the train prior on three quarters
   of the test rows, so a feature the model leaned on becomes a constant.
3. **The operating point stops transferring.** `all` predicts a mean of
   **5.70 classes per encounter against a gold mean of 1.89**, at
   micro-precision 0.1705, and never predicts the empty set (empty-prediction
   rate 0.0000 against a 23.4% empty-prescription rate in the gold data). The
   threshold was tuned honestly on 2016 validation rows; the model's confidence
   on ≥2017 rows is simply inflated relative to that calibration. `text_raw`
   drifts the same way but far less (2.90 predicted, against 2.16 same-era).

And this is measured **with the leak of §2.3 still running in `all`'s favour**:
its engineered columns were fitted on the patient split and have already seen
later-era rows. A leak-free rebuild can only widen the gap.

**Practical consequence:** the feature set must be selected on a *temporal*
validation split, not a random one. The project's `[D]` deployable restriction
— introduced for an unrelated reason, that a walk-in encounter cannot produce
`features.parquet` — turns out to be the shift-robust choice as well.

## 5. Same estimator, two class-like targets

The legacy 89-category space is the closest available like-for-like to the
published *category* row (−47%), now measured with the linear multi-label
estimator instead of the decoder:

| Target | arm | patient | temporal | relative drop |
|---|---|---|---|---|
| class46 | `text_raw` | 0.4615 | 0.3547 | **−23.1%** [−26.3, −20.0] |
| class46 | `all` | 0.4934 | 0.2561 | −48.1% [−50.2, −46.0] |
| cat89 | `text_raw` | 0.4077 | 0.2947 | −27.8% [−31.1, −24.3] |
| cat89 | `all` | 0.4340 | 0.2197 | −49.4% [−51.6, −47.1] |

Holding the arm fixed, **class46 is both higher-scoring and more temporally
stable than cat89** at every point of comparison — the direction the
drug-normalisation workstream predicted, now measured with the deployed
estimator.

At cat89 under the temporal split, `all` is **statistically indistinguishable
from the frequency prior**: delta +0.0000 [−0.0059, +0.0059], p(better) =
0.514. For that one configuration the original "collapses to the
frequency-prior floor" language *is* accurate. It is not accurate for the
deployable arm at class level, which is the configuration that matters.

### 5.1 The label space itself, measured on these rows

`label_shift()` recomputes the cross-era stability figures on exactly the rows
being scored:

| Target | vocabulary Jaccard | unseen gold positives | top-10 shared |
|---|---|---|---|
| brand (717) | **0.462** | 12.1% | **2 / 10** |
| cat89 | 0.545 | 3.1% | 7 / 10 |
| class46 | **0.870** | **0.1%** | **8 / 10** |

Vocabulary Jaccard and top-10 overlap reproduce
`results/rx_generation/drugmap/drug_normalization.json` and
`results/rx_generation/era_shift.json` exactly (brand 0.4622 and 2-of-10;
class 0.8696 and 8-of-10) — the cross-check that the in-memory split is the
same split those artefacts were built from.

The unseen-rate column needs care, because three definitions are in
circulation and at class level they differ by a factor of thirty:

- **12.1% / 3.1% / 0.1% (this table)** — share of *gold multi-hot positives*
  in the test rows whose label has zero train positives. Encounter-level, so
  two orders of the same brand in one prescription count once.
- **12.2% / 0.12% (`unseen_order_rate`)** — the same quantity counted over
  orders rather than encounter-label pairs. The small gap is only that
  deduplication.
- **12.2% / 3.4% (`unseen_order_rate_strict`)** — additionally charges the
  2.4% of orders that normalisation could not map to a class as unseen.
  Unchanged at brand level (every brand is its own label), and it is the right
  figure to quote when arguing that class is the safer *target*, because an
  unmapped order is a real prescription the class model cannot express.

All three are correct. The label matrix scored in this document contains only
mapped orders, so 0.1% is the rate that applies to the numbers above and 3.4%
is the rate that applies to the target-selection argument.

## 6. Isolating the two changes: is it the target or the estimator?

The published −66% differs from the class-level number in **both** target and
estimator, so §3 on its own is indicative rather than controlled. To separate
them, the brand target was re-run through the *identical* code path — same
linear one-vs-rest estimator, same feature blocks, same rows, same val-fitted
calibration and threshold, same bootstrap. Only the label space changes.

Run: `python -m src.phcrx.temporal.class_temporal --targets brand717 --arms
text_raw all --c-grid 1.0 2.0 --tag _brand`. Raw:
`class_temporal_brand.csv`, `split_deltas_brand.csv`, `run_brand.log`.

### Estimator held fixed at linear + `text_raw`

| Target | labels | patient | temporal | relative drop [95% CI] |
|---|---|---|---|---|
| brand717 | 717 | 0.2317 | 0.0990 | **−57.2%** [−60.7, −53.7] |
| cat89 | 88 | 0.4077 | 0.2947 | −27.8% [−31.1, −24.3] |
| **class46** | 46 | 0.4615 | 0.3547 | **−23.1%** [−26.3, −20.0] |

### The decomposition

| Configuration | patient | temporal | relative drop |
|---|---|---|---|
| brand + autoregressive decoder *(published)* | 0.1915 | 0.0657 | **−66%** |
| brand + linear multi-label, `text_raw` | 0.2317 | 0.0990 | **−57.2%** |
| class + linear multi-label, `text_raw` | 0.4615 | 0.3547 | **−23.1%** |

Reading the two steps:

- **Changing the estimator, holding the target at brand: −66% → −57.2%.**
  Worth about 9 points of the correction. The linear model is also better at
  both ends (patient 0.192 → 0.232, temporal 0.066 → 0.099).
- **Changing the target, holding the estimator linear: −57.2% → −23.1%.**
  Worth about 34 points.

**So it is mostly the target, not the estimator.** Roughly four fifths of the
correction to the headline comes from predicting pharmacological classes
instead of brands; the remaining fifth comes from dropping autoregression.
That is exactly what the label-space statistics in §5.1 predict — brand
vocabulary Jaccard 0.462 with 12.2% of later orders for never-seen brands,
against 0.870 and 0.1% for classes.

### The "frequency-prior floor" claim, tested at brand level too

With the prior constructed identically for every system (order from that
split's train, k tuned on that split's val):

| Target | arm | temporal micro-F1 | prior | delta vs prior [95% CI] |
|---|---|---|---|---|
| brand717 | `text_raw` | 0.0990 | 0.0406 | +0.0585 [+0.0516, +0.0653] |
| brand717 | `all` | 0.0455 | 0.0406 | +0.0049 [+0.0013, +0.0085] |
| class46 | `text_raw` | 0.3547 | 0.2303 | +0.1245 [+0.1161, +0.1328] |
| class46 | `all` | 0.2561 | 0.2303 | +0.0259 [+0.0197, +0.0323] |

The published statement that brand-level performance falls *to* the prior floor
is accurate **for the decoder** (0.0657 against its own 0.069 prior). It is not
accurate for the linear estimator even at brand level, and it is a long way from
accurate for the deployable class-level configuration.

`all` at brand level under temporal shift is the most extreme instance of the
threshold-drift failure in §4: it predicts a mean of **16.07 brands per
encounter** against a gold mean of 2.00.

---

## 7. What the corrected deployability headline should be

Replacing the sentence currently in `docs/research_frame.md` §7.4,
`docs/prescription_generation.md` §5 and `docs/model_comparison.md`:

> **Forward in time (train ≤2015, test ≥2017) the deployed recommender's
> class-level micro-F1 falls from 0.4615 to 0.3547 — a −23.1% [−26.3, −20.0]
> drop, not the −66% previously published — and it stays clearly above the
> frequency-prior floor (+0.1245 [+0.1161, +0.1328]). A random patient-level
> split overstates forward-in-time performance by about **1.3×**, not 3×.
> Two qualifications are load-bearing. First, the *feature set* matters more
> than the split: the currently deployed `all` arm, which carries calendar,
> site and prescriber features, falls twice as far (0.4934 → 0.2561, −48.1%
> [−50.2, −46.0]) and lands almost on the prior — under temporal shift the
> leak-free deployable arm beats it by 0.0986 micro-F1, reversing the ordering
> the random split produced. Second, *set-level* agreement does collapse for
> both arms: Jaccard and exact-set match fall below the always-empty floor, and
> tail-macro-F1 falls −83.7%. The corrected claim is therefore that
> class-level prescribing **does** transfer forward in time as a ranked
> suggestion list, provided the model is restricted to deployable
> patient-level features — not that it transfers as a whole-prescription
> draft.**

Three sub-claims elsewhere in the docs also need correcting:

| Currently published | Should read |
|---|---|
| "random-split evaluation overstates deployability ~3×" | ~1.3× for the deployable class-level arm; ~1.9× for the currently deployed `all` arm |
| "falls to the frequency-prior floor — no useful transfer" | true for the brand-level decoder; false at class level, where the deployable arm clears the prior by +0.1245 |
| "Category-level scores degrade less (−47% vs −66%)" | direction confirmed and strengthened: with the deployed estimator, class46 −23.1% vs brand717 −57.2% on one code path |

`docs/remaining_work.md` P1 item 2 ("Re-run the temporal split at class level")
is closed by this document. Its own prediction — "this number is probably
materially wrong now" — is confirmed.

## 8. Limitations

1. **`all` under the temporal split is an optimistic upper bound.**
   `features.parquet` was fitted on the patient split (§2.3). The §4 conclusion
   survives this because the bias runs the wrong way for it, but the `all`
   temporal *point estimates* (0.2561 at class level, 0.0455 at brand level)
   should be read as ceilings, not measurements. Closing this is
   `remaining_work.md` P2 item 11.
2. **One threshold, tuned on 2016.** Every arm uses a single global operating
   point. Much of the `all` collapse is threshold drift rather than lost
   discrimination: micro-recall is essentially unchanged under temporal shift
   (`all`, class level: 0.5283 → 0.5147) while micro-precision collapses
   (0.4629 → 0.1705). A shift-aware operating point, or per-label thresholds
   (`remaining_work.md` P2 item 10), would recover an unknown part of the drop.
   This document measures the deployed configuration, not the best achievable
   one.
3. **The eras are not balanced.** 9,961 training encounters come from
   2012–2015 but 61% of them from 2012–2013, and the ≥2017 test era spans nine
   years with only 2,880 encounters, 40% of them from 2017 alone. The
   "forward-in-time" gap is not a fixed horizon.
4. **The cross-split comparison is not paired** (§2.4). Intervals on the
   difference are conservative; the point estimates are unaffected.
5. **`text_raw`, not `text_raw_sem`.** The best *deployable* arm on validation
   was `text_raw_sem` (val 0.4766 vs 0.4744). `text_raw` was used here because
   it is leak-free with no dependency on the SapBERT cache, and the difference
   on validation is smaller than the bootstrap width of anything in this
   document. The semantic arm was not tested under shift.
6. **Still agreement, not accuracy.** Every number here is agreement with what
   a clinician historically prescribed. Nothing in this document speaks to
   whether either era's prescribing was correct.
