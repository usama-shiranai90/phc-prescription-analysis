# PHC Prescription Analysis

Analysis, modeling, and (upcoming) deep-learning work on the Portable Health Clinic (PHC)
dataset — prescription generation, prediction, and recommendation.

## Core dataset

The core dataset consists of the PHC database (which is excluded from this repository for privacy and security reasons).

- **Data Dictionary**: [docs/data_dictionary.md](docs/data_dictionary.md) — entity relationships, join keys, and
  clinical variable definitions/units for the analysis-relevant subgraph
  (patients → checkups → prescriptions → drugs/ICD-10/complaints).

*(Note: Raw database files, SQL dumps, patient data, and specific schema details are not tracked in this repository.)*

## Prescription generation (PHC-RxGen)

Deep multi-modal prescription generation — char-CNN + BiLSTM over symptom text,
per-vital tokens, a history GRU, Transformer fusion, and an autoregressive
Transformer decoder emitting drug orders with structured attributes.

- **Method, data findings, and limitations**: [docs/prescription_generation.md](docs/prescription_generation.md)
- **Code**: [src/phcrx/](src/phcrx/) · **SQL extract**: [src/sql/](src/sql/)
- **Results**: `results/rx_generation/` (`RESULTS.md`, figures, metrics JSON)

## Clinical NLP services (Hugging Face · Ollama · ICD-10 · WHO NCD)

Local-only NLP layer that fills the corpus's structural gaps: ICD-10 auto-coding
of symptom text (SapBERT + hybrid retrieval + medgemma adjudication), a
verified site abbreviation glossary, WHO NCD risk stratification benchmarked
against the WHO GHO API, and a local-LLM prescription baseline.

- **Method and limitations**: [docs/clinical_nlp_services.md](docs/clinical_nlp_services.md)
- **Code**: [src/phcrx/nlp/](src/phcrx/nlp/) · **Results**: `results/rx_generation/nlp/`

> No clinical text leaves the machine: Hugging Face supplies model weights and
> the ICD reference, inference is local GPU, and Ollama serves models on the
> host machine. The only outbound call is to the WHO GHO API, which downloads
> population statistics and transmits no patient data.

The data extraction step connects to the database locally and generates a frozen snapshot in
`data/interim/`, which is then processed into `data/processed/` for model training.

## Layout

```
data/          (Excluded) Raw databases, CSV extracts, and cleaned analysis tables
models/        (Excluded) Trained model checkpoints
results/       (Excluded) Analysis outputs (csv/png), grouped by study
reports/       (Excluded) Generated analysis reports (md + html)
docs/          Data dictionary, methodology, and open questions
notebooks/     Exploratory notebooks
src/
  sql/         Database profiling and extract SQL
  phcrx/       PHC-RxGen package: preprocess, data, model, train, baselines,
               metrics, report, predict, diagnose, era_shift
    nlp/       Clinical NLP services: glossary, icd_index, icd_code, ncd,
               llm_rx, ollama_client
```

## Environment

Modelling runs in a conda environment
(Python 3.13, PyTorch 2.10, CUDA). `environment.yml` captures an equivalent
spec for rebuilding elsewhere:

```bash
conda env create -f environment.yml
```
