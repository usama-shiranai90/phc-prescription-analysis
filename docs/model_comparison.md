# Neural vs. linear: a matched-protocol head-to-head

> **⚠ SUPERSEDED IN PART — read this first.**
> The verdict below is correct about the model **as it was built** (an
> autoregressive set-decoder) but **wrong as a statement about neural methods
> on this task**. `docs/neural_mlc.md` later removed the confound: the same
> encoder with a multi-label classification head instead of a decoder gains
> **+0.109 micro-F1** (0.3563 → 0.4650 on restricted47) and draws **level with
> the linear model** (norm46: neural 0.4860 vs linear 0.4865, CIs overlapping).
>
> The accurate claim is: **the autoregressive set-decoder is not justified.**
> The choice of neural vs linear estimator is not material to accuracy; the
> linear model is preferred here on *engineering* grounds (≈1,000× cheaper to
> fit, interpretable, trivially retrainable against formulary drift), not on
> accuracy.

**Verdict: the neural architecture is not justified. A TF-IDF + logistic
regression model beats PHC-RxGen by ~0.09 micro-F1 (≈28% relative) on identical
rows, labels and metric, under both evaluation protocols, with bootstrap CIs
that exclude zero.**

Run: `python -m src.phcrx.bench.head_to_head` (5,000 bootstrap resamples).
Raw table: `src/phcrx/bench/head_to_head.csv`.

---

## Why this was run

Two numbers did not agree. The neural model scored category-level micro-F1
**0.348**; a throwaway TF-IDF proxy built for the text-preprocessing ablation
scored **0.4204**. They were not comparable — different label sets, different
row filters, set-generation vs. independent thresholding — so the comparison was
redone under one fixed protocol.

Both protocols are reported because a ranking that flips between them would
itself be the finding:

- **full89** — all 2,780 test encounters, 89 classes, including encounters with
  no symptom text and empty gold prescriptions.
- **restricted47** — 1,923 text-bearing test encounters, 47 classes with ≥10
  training orders, no empty-gold rows.

Every vectoriser and scaler is fitted on train only. Linear thresholds are tuned
on validation. Because the neural decoder emits a *set* and the linear models
threshold independent probabilities, each linear system is also reported
**@sizematch** — threshold re-tuned so mean predicted set size matches the
neural model's — which removes set-size as an explanation.

## Results

### full89 — all 2,780 test encounters, 89 classes

| System | micro-F1 [95% CI] | macro-F1 | Jaccard | exact | mean set | Δ vs neural [CI] |
|---|---|---|---|---|---|---|
| always-empty | 0.0000 | 0.0000 | 0.2403 | 0.2403 | 0.00 | −0.331 |
| prior top-6 | 0.2193 | 0.0215 | 0.1230 | 0.0000 | 6.00 | −0.112 |
| **neural PHC-RxGen** | **0.3315** [0.318, 0.344] | 0.1464 | 0.3561 | 0.2335 | 1.33 | — |
| lr_tab_physio | 0.2885 | 0.0848 | 0.1846 | 0.0363 | 3.96 | −0.043 [−0.057, −0.028] |
| lr_tab_clinical | 0.3689 | 0.1630 | 0.3009 | 0.1234 | 3.24 | **+0.037** [+0.024, +0.050] |
| lr_text | 0.3714 | 0.1545 | 0.3166 | 0.1478 | 2.68 | **+0.040** [+0.026, +0.053] |
| lr_tab_all | 0.4104 | 0.2020 | 0.3809 | 0.2209 | 1.78 | **+0.079** [+0.065, +0.093] |
| **lr_text_tab_all** | **0.4240** [0.412, 0.436] | **0.2103** | 0.3860 | 0.2101 | 2.02 | **+0.092** [+0.079, +0.106] |
| lr_text_tab_all @sizematch | 0.3871 | 0.1901 | 0.3833 | **0.2468** | 1.31 | **+0.056** [+0.041, +0.070] |

### restricted47 — 1,923 text-bearing encounters, 47 classes

| System | micro-F1 [95% CI] | macro-F1 | Δ vs neural [CI] |
|---|---|---|---|
| **neural PHC-RxGen** | **0.3563** [0.342, 0.370] | 0.2256 | — |
| lr_tab_clinical | 0.4053 | 0.2596 | **+0.049** [+0.035, +0.063] |
| lr_text | 0.4204 | 0.2440 | **+0.064** [+0.050, +0.079] |
| lr_tab_all | 0.4406 | 0.2975 | **+0.084** [+0.070, +0.099] |
| **lr_text_tab_all** | **0.4503** [0.438, 0.462] | **0.3074** | **+0.094** [+0.080, +0.108] |

## What this establishes

1. **The ranking does not flip.** The linear model wins under both protocols,
   `p_better = 1.0000` in 5,000 resamples. This is not a protocol artefact.
2. **It is not a set-size artefact.** Size-matched, the linear model still wins
   by +0.056 [+0.041, +0.070] on full89, and additionally beats the neural model
   on exact-set match (0.2468 vs 0.2335) — the one metric where the neural
   decoder had looked better.
3. **It is not only micro-F1.** Macro-F1 is 0.2103 vs 0.1464 (full89) and
   0.3074 vs 0.2256 (restricted47). The linear model is *also* better on rare
   classes, where the neural model was weakest.
4. **Tabular features alone beat the neural model** (0.3689 vs 0.3315), as does
   text alone (0.3714). The neural model does not even match its own inputs
   handled linearly.

## Why the neural model loses

This is consistent with every earlier ablation rather than a surprise:

- removing the transformer fusion encoder **improved** micro-F1 by +0.016 (~4 s.d.)
- the history GRU contributed +0.0002 — indistinguishable from noise
- 9,900 training encounters is small for a 3.5–5.1M-parameter encoder–decoder
- autoregressive set generation with a canonical order imposes a sequence model
  on a target that is a *set*, and the decoder must additionally learn when to
  stop — the linear model gets set size for free from a tuned threshold

## Recommendation

**Replace the neural model with a regularised linear (or gradient-boosted)
multi-label classifier over TF-IDF text + clinical tabular features.** It is:

- **better** — +0.09 micro-F1, +0.06 macro-F1, and better exact match when
  size-matched
- **~1,000× cheaper** — seconds to fit on CPU vs. ~6 minutes on GPU per seed
- **interpretable** — per-class coefficients, which matters for a clinical
  audience and for the feature-importance analysis already done
- **trivially retrainable** — which is the operative property given 74%
  prescriber turnover and a 0.462 cross-era drug-vocabulary Jaccard

The **autoregressive encoder–decoder** should be reported as a measured
negative result: on a corpus of this size, forcing sequence generation onto a
set-valued target is outperformed by a linear bag-of-words baseline. This is
*not* a result about deep learning on this task — a neural multi-label
classifier matches the linear model (`docs/neural_mlc.md`). The useful
contribution is the **task formulation**, not the estimator family.

## Limitations

- Class-level targets only. Brand-level (719 labels) was not benchmarked here;
  the drug-normalisation workstream recommends class as the target anyway.
- The neural model was not re-tuned for this protocol — it was trained to
  generate sets autoregressively and is evaluated as such. A neural model built
  as a multi-label classifier (no decoder) was not tested and might close the
  gap; that is the fair next comparison, not a defence of the current design.
- Single checkpoint (`rxgen_full_patient_seed0.pt`, seed 0). Seed variance on
  the neural side is ±0.008 micro-F1, far smaller than the 0.09 gap.
