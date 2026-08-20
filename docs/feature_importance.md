# Feature importance for prescription-component recommendation

309 features, 10 families, 50 prescription-component targets. Gradient-boosted
trees fitted on the train split, permutation importance measured on the held-out
test split (2,780 encounters, patient-level split, no patient straddling).
All transforms fitted on train only.

Run: `python -m src.phcrx.features.build_features` then
`python -m src.phcrx.features.importance`.

---

## 1. The headline: not all prescription components are equally learnable

| Component | Best test performance | Verdict |
|---|---|---|
| **Advice set** (10 targets) | AUROC **0.928 – 0.985** | highly predictable |
| **Diagnostic tests** (8 targets) | AUROC **0.923 – 0.979** | highly predictable |
| **Drug type/form** | accuracy **0.933** | highly predictable |
| Condition-specific drug classes | AUROC 0.84 – 0.94 | predictable |
| Chronic-therapy flag | AUROC 0.880 | predictable |
| Any drug at all | AUROC 0.707 → **0.863** * | predictable once confounders dropped |
| Discretionary drug classes | AUROC 0.64 – 0.75 | weak |
| Number of drugs | MAE 1.18 → **0.98** *, R² **0.42** * | moderate |
| **Dose / duration** | accuracy **0.506 / 0.503** | near-unlearnable |

\* with non-clinical features removed — see §3.

**This is the actionable result.** The project has been evaluated end-to-end on
the hardest component (exact drug set) while the components that are actually
predictable — advice, diagnostic tests, drug form — were treated as auxiliary
heads and never reported as products in their own right.

## 2. Drug class is predictable exactly when the diagnostic is measured

| Drug class | AUROC | Why |
|---|---|---|
| Oral hypoglycaemics | **0.943** | blood glucose is measured on 91% of encounters |
| Antispasmodics | 0.932 | specific symptom vocabulary |
| Aceclofenac | 0.925 | pain complaints are explicit |
| Albendazole | 0.920 | distinctive presentation |
| Calcium-channel blockers | 0.913 | BP is measured on 97% |
| Domperidone | 0.900 | GI symptoms are explicit |
| ARB/ACEI | 0.837 | BP measured |
| — | — | — |
| Paracetamol | 0.751 | given for almost anything |
| Bromazepam | 0.744 | prescriber-preference drug |
| Beta blockers | 0.703 | overlaps other antihypertensives |
| H2 blockers | 0.697 | substitutes with PPI arbitrarily |
| PPI | 0.675 | near-universal co-prescription |
| Vitamins / minerals | **0.635 – 0.647** | discretionary supplementation |

The split is clean: **classes with a measured diagnostic are learnable; classes
prescribed by habit are not.** No amount of feature engineering fixes the second
group, because the information that determines them is not in the record.

## 3. The confounder result — and why "importance" ≠ "usefulness"

Mean permutation importance across all targets ranks the families:

| Family | Mean importance |
|---|---|
| temporal (year, month, season, gap) | **+0.1148** |
| prescriber | **+0.0959** |
| text | +0.0537 |
| site | +0.0242 |
| vitals | +0.0148 |
| derived clinical | +0.0079 |
| NCD flags | +0.0026 |
| demographics | +0.0022 |
| derived ICD | +0.0009 |
| history | +0.0006 |

The two largest families are **not clinical**. The fitted models lean hardest on
*when* the encounter happened and *who wrote it*.

But permutation importance measures what a fitted model **relies on**, not what
**helps**. Refitting with the non-clinical families removed:

| Variant | Mean AUROC (43 binary targets) |
|---|---|
| clinical_only (no temporal/prescriber/site) | **0.8885** |
| full (all 309 features) | 0.8802 |
| no_prescriber only | 0.8469 |

Dropping them **improves** mean AUROC, and the per-target effect is large and
systematic:

| Target | full → clinical_only |
|---|---|
| Vitamin+mineral | 0.647 → **0.809** (+0.162) |
| **Any drug prescribed** | 0.707 → **0.863** (+0.155) |
| Vitamin | 0.635 → **0.783** (+0.147) |
| Beta blocker | 0.703 → **0.846** (+0.143) |
| PPI | 0.675 → **0.802** (+0.127) |
| Paracetamol | 0.751 → **0.863** (+0.112) |
| Number of drugs (MAE) | 1.185 → **0.977** (R² 0.150 → **0.417**) |

High-cardinality prescriber and year features let the trees memorise
train-specific rules that misfire on held-out patients. Removing them forces
reliance on clinical signal, which generalises.

### But not for every component

| Target | full → clinical_only |
|---|---|
| Clonazepam | 0.881 → 0.800 (−0.082) |
| Aceclofenac | 0.925 → 0.865 (−0.060) |
| TCA/SSRI | 0.850 → 0.793 (−0.057) |
| Number of advice items (MAE) | 0.567 → 0.763 (worse) |
| Dose mode (accuracy) | 0.506 → 0.474 (worse) |
| Duration mode (accuracy) | 0.503 → 0.438 (worse) |
| Instruction mode (accuracy) | 0.725 → 0.636 (worse) |

The pattern is interpretable: **dose, duration, instruction and advice volume
are prescriber habits, not clinical decisions.** They are predictable *from
prescriber identity* and poorly predictable from the patient. That is a finding
about the data-generating process, not a modelling failure — and it means those
components should never be presented as clinical recommendations.

## 4. Feature families that did not earn their place

Two families I built specifically for this and expected to matter contribute
almost nothing:

- **NCD flags** (+0.0026) — 13 WHO/Asian-threshold flags. They are deterministic
  functions of the vitals already in the model, so they add no information; the
  trees can already split on `bp_sys >= 140`.
- **Derived ICD codes** (+0.0009) — 3,766 codes from the NLP layer. They are
  themselves derived from the symptom text, which the model already has, and
  only 31% of encounters carry one.

Both are honest negatives. Engineered features that are deterministic functions
of existing inputs do not add signal to a model that can learn the same split.

## 5. Recommendations

1. **Report advice and diagnostic-test recommendation as first-class outputs.**
   AUROC 0.92–0.99 versus 0.64–0.94 for drugs. This is the most defensible
   product in the dataset and it has been buried as an auxiliary loss term.
2. **Drop temporal, prescriber and site features for drug-presence and
   drug-count targets** — +0.155 AUROC and R² 0.15→0.42 respectively.
3. **Keep them only for dose/duration/instruction/advice-volume**, and label
   those outputs explicitly as "typical practice at this site" rather than
   clinical recommendations.
4. **Stop trying to predict dose and duration from clinical features.**
   Accuracy 0.506/0.503 against a strong majority class. They are habit.
5. **Do not invest further in NCD or ICD derived features.** Measured
   contribution is ~0.
6. **Segment the drug target by predictability**: model condition-specific
   classes (antidiabetic, antihypertensive, antispasmodic) as a recommendation
   product, and treat discretionary classes (vitamins, PPI) as a separate,
   lower-confidence suggestion.
