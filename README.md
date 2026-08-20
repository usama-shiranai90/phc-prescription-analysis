# PHC Prescription Analysis

Analysis, modeling, and (upcoming) deep-learning work on the Portable Health Clinic (PHC)
`gramweb_ghealth` dataset — prescription generation, prediction, and recommendation.

## Core dataset

- **[data/raw/schema-only-gphc-postgreSQL.sql](data/raw/schema-only-gphc-postgreSQL.sql)** — canonical
  schema (PostgreSQL DDL, `gramweb_ghealth`, 135 tables) for the PHC database. Treat this as the
  reference schema for any new pipeline, migration, or feature-engineering work.
- **[data/raw/phc.db](data/raw/phc.db)** — SQLite load of the same database (from a MySQL/phpMyAdmin
  dump, 135 tables, 709,860 rows; see [load_log.txt](data/raw/load_log.txt) and
  [db_manifest.csv](data/raw/db_manifest.csv) for load details and per-table row counts).
- **[docs/data_dictionary.md](docs/data_dictionary.md)** — entity relationships, join keys, and
  clinical variable definitions/units for the analysis-relevant subgraph
  (patients → checkups → prescriptions → drugs/ICD-10/complaints).

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
> Windows host. The only outbound call is to the WHO GHO API, which downloads
> population statistics and transmits no patient data.

The live Postgres server (`gphc-fix`, schema `gramweb_ghealth`) runs on the
**Windows host** and its `pg_hba.conf` permits loopback only, so the extract
step runs from Windows and the WSL training env consumes a frozen snapshot in
`data/interim/` → `data/processed/`.

## Layout

```
data/
  raw/         Schema + original/loaded database, load metadata (immutable inputs)
  interim/     Frozen CSV extract from Postgres (src/sql/extract.sql output)
  processed/   Cleaned analysis tables + rxgen_* model inputs, vocab, norm stats (parquet/json)
  reference/   Lookup/reference tables (drug_reference.csv)
models/        Trained model checkpoints (hier_transformer.pt, ...)
results/       Analysis outputs (csv/png), grouped by study:
  cohort/              Cohort profile, missingness, data quality, vitals
  prescribing_patterns/ Rx anatomy, therapeutic classes, co-prescription, association rules
  epidemiology/        Diagnosis, geographic & temporal prevalence, phenotypes
  who_indicators/      WHO/INRUD prescribing indicators, AWaRe, audit & medication-safety scores
  clustering/          Patient clustering outputs
  modeling/            ML/DL metrics, ensembles, model comparison, SHAP
reports/       Generated analysis reports (md + html)
paper/         Manuscript draft (phc_paper.tex/.pdf, refs.bib)
docs/          Data dictionary, open questions
notebooks/     Exploratory notebooks
src/
  sql/         Postgres profiling + frozen-extract SQL
  phcrx/       PHC-RxGen package: preprocess, data, model, train, baselines,
               metrics, report, predict, diagnose, era_shift
    nlp/       Clinical NLP services: glossary, icd_index, icd_code, ncd,
               llm_rx, ollama_client
archive/       Superseded/duplicate exports (backup zips, prior session & plan JSON exports)
```

## Environment

Modelling runs in the WSL Ubuntu conda env
`/home/syedu/anaconda3/envs/collective-research` (Python 3.13, PyTorch 2.10
+cu128, CUDA on an RTX A2000 6GB). `environment.yml` captures an equivalent
spec for rebuilding elsewhere:

```bash
conda env create -f environment.yml
```
