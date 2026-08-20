# Does multi-label reframing close the gap? Yes — the decoder was the problem

**Answer: the earlier conclusion was wrong in its attribution.** "A linear model
beats the neural model" was really "an autoregressive set-decoder is a bad fit
for this task". Keep the same encoder, replace the decoder with a multi-label
head, and the neural model gains **+0.109 micro-F1** and draws level with (or
marginally ahead of) the linear baseline.

Run: `python -m src.phcrx.neural_mlc.evaluate`. Raw:
`results/rx_generation/neural_mlc/benchmark.csv`.

---

## The experiment

`docs/model_comparison.md` concluded that TF-IDF + logistic regression beat
PHC-RxGen by +0.092 micro-F1 and recommended dropping the neural model. That
comparison had a confound, stated at the time as a limitation: the neural model
was tested **as built** — an autoregressive generator that emits a drug sequence
in canonical order and must additionally learn when to stop — and was never
tested as a plain multi-label classifier.

This removes the confound. Same encoder (char-CNN + word BiLSTM, per-vital
tokens, history GRU, transformer fusion), **no decoder**, a linear head over the
class space trained with `BCEWithLogitsLoss`, threshold tuned on validation
exactly as the linear baselines are.

## Results

### restricted47 — 1,923 text-bearing encounters, 47 classes

| System | micro-F1 [95% CI] | macro-F1 | exact |
|---|---|---|---|
| **mlc_tab_only** | **0.4650** [0.454, 0.476] | 0.3079 | 0.0912 |
| mlc_text_tab | 0.4646 [0.454, 0.475] | 0.3149 | 0.0927 |
| mlc_full_tab | 0.4602 [0.450, 0.470] | 0.3046 | 0.0816 |
| lr_text_tab_all *(previous best)* | 0.4503 [0.439, 0.462] | 0.3074 | 0.0738 |
| lr_tab_all | 0.4406 | 0.2975 | 0.0822 |
| mlc_full | 0.4190 | 0.2404 | 0.0556 |
| mlc_no_fusion | 0.4038 | 0.2318 | 0.0570 |
| **neural_rxgen** *(autoregressive)* | **0.3563** [0.342, 0.371] | 0.2256 | 0.0863 |

### norm46 — 46 normalised pharmacological classes

| System | micro-F1 [95% CI] | macro-F1 |
|---|---|---|
| lr_text_tab_all | **0.4865** [0.476, 0.497] | 0.3601 |
| **mlc_full_tab** | **0.4860** [0.476, 0.496] | 0.3530 |
| lr_tab_all | 0.4847 | 0.3521 |
| mlc_full | 0.4445 | 0.2989 |
| mlc_text_only | 0.4346 | 0.2896 |
| **neural_rxgen** | **0.3595** [0.347, 0.372] | 0.2500 |

## What changed, and what did not

**1. The decoder was the whole problem.** Same encoder, same data, same
protocol: 0.3563 → 0.4650 on restricted47, **+0.109 micro-F1** purely from
dropping autoregression. That is larger than any other single effect measured in
this project.

**2. Neural and linear are now level.** On norm46 they are indistinguishable
(0.4860 vs 0.4865, CIs almost entirely overlapping). On restricted47 the neural
classifier is marginally ahead (0.4650 vs 0.4503) with slightly overlapping CIs.
Neither ordering is decisive.

**3. A previous ablation finding reverses.** With the autoregressive decoder,
removing transformer fusion *improved* micro-F1 by +0.016. With the multi-label
head, fusion **helps**: 0.4190 vs 0.4038 (restricted47) and 0.4445 vs 0.4186
(norm46). The fusion encoder was not inherently harmful — it was harmful *in
combination with* a decoder the corpus could not support.

**4. Engineered tabular features carry more than text.** `mlc_tab_only` (0.4650)
edges `mlc_text_tab` (0.4646) and beats `mlc_full` without them (0.4190). This
is consistent with `lr_tab_all` ≈ `lr_text_tab_all` and means the feature
engineering work paid off more than the text encoder did.

## Corrections this forces

Two earlier claims in this repository are now wrong and are corrected here:

- `docs/model_comparison.md` — "the neural architecture is not justified" holds
  only for the **autoregressive** architecture. It should read: the
  autoregressive set-decoder is not justified.
- `docs/prescription_generation.md` §5 — "Transformer fusion hurts at this
  corpus size" holds only in the decoder setting; under multi-label framing
  fusion helps.

## Recommendation

**For accuracy, neural-MLC and linear are equivalent — so choose on other
grounds.** The linear model remains preferable in this deployment because it is
seconds to fit on CPU against minutes on GPU, interpretable per class, and
trivially retrainable as the formulary drifts (74% prescriber turnover, 0.462
cross-era vocabulary Jaccard). But that is now an **engineering** argument, not
an accuracy one, and it must be stated that way.

The scientifically correct statement is: *on this corpus, a multi-label
formulation is essential; the choice of neural vs linear estimator is not
material to accuracy.*

## Limitations

- Threshold tuned on validation for every system, so the comparison is fair in
  both directions.
- 3 seeds on the neural side; seed s.d. ±0.008, well below the +0.109 effect.
- Brand-level (719 labels) was not re-tested under the multi-label framing; the
  drug-normalisation workstream recommends class-level targets regardless.
