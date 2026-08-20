# PHC Prescription Analysis

Analysis, modeling, and deep-learning pipeline for the Portable Health Clinic (PHC) dataset — focusing on prescription generation, clinical prediction, and recommendation systems.

> **Privacy Notice**: This is a public repository. All raw database files, private database snapshots/dumps, connection credentials, internal schema names, hostnames, and sensitive patient records are strictly excluded from version control for privacy and security reasons.

---

## Overview

This repository contains the machine learning and clinical NLP codebase for analyzing prescription patterns, generating multi-modal drug recommendations, and standardizing clinical text:

- **PHC-RxGen**: Multi-modal prescription generation model using char-CNN + BiLSTM for symptom text, tokenized vitals, patient history GRU, Transformer fusion, and autoregressive Transformer decoders for structured drug order emission.
- **Clinical NLP Services**: Local-only NLP pipeline including ICD-10 automated coding (SapBERT + hybrid retrieval + MedGemma adjudication), site glossary verification, WHO NCD risk stratification against WHO GHO benchmarks, and local LLM baselines.

---

## Repository Layout

```
data/          (Excluded from git) Raw data extracts, CSVs, and processed Parquet files
models/        (Excluded from git) Trained model checkpoints and serialized weights
results/       Analysis outputs, metrics JSON, and evaluation figures
reports/       Generated analysis reports (Markdown & HTML)
docs/          Data dictionary, methodology specifications, and research documentation
notebooks/     Exploratory research and analysis notebooks
src/
  sql/         Sanitized SQL extraction and profiling queries (generic interfaces)
  phcrx/       PHC-RxGen core package (preprocessing, modeling, training, evaluation)
    nlp/       Clinical NLP sub-package (ICD-10 coding, WHO NCD, Ollama LLM integration)
```

---

## Core Dataset & Privacy Controls

- **Data Dictionary**: [`docs/data_dictionary.md`](docs/data_dictionary.md) — High-level conceptual entity relationships and clinical variable definitions (Patients → Checkups → Prescriptions → Drugs / ICD-10 / Complaints).
- **Data Pipeline**: Raw database extraction is performed locally against an isolated database instance to output anonymized interim CSVs (`data/interim/`), which are subsequently cleaned into Parquet features (`data/processed/`) for model training.
- **Strict Anonymization**: No private database schema names, raw SQL dumps, database credentials, host IP addresses, or patient identifiers are stored or committed in this repository.

---

## Key Modules & Documentation

| Component | Description | Reference & Code |
| :--- | :--- | :--- |
| **Prescription Generation** | Deep multi-modal Rx generation architecture & evaluation results | [`docs/prescription_generation.md`](docs/prescription_generation.md) • [`src/phcrx/`](src/phcrx/) |
| **Clinical NLP Services** | Local GPU ICD-10 coding, WHO NCD risk profiling & local LLM baselines | [`docs/clinical_nlp_services.md`](docs/clinical_nlp_services.md) • [`src/phcrx/nlp/`](src/phcrx/nlp/) |
| **SQL Extract Interface** | Data extraction queries and profiling logic | [`src/sql/`](src/sql/) |
| **Results & Metrics** | Comprehensive evaluation metrics, tables, and figures | `results/rx_generation/` |

---

## Data Privacy & Local Execution Security

> **Zero External Transmission of Clinical Data**
> - All model weights (Hugging Face / SapBERT) and LLM inference (Ollama) run 100% locally on host GPU/CPU hardware.
> - The only external HTTP request is to the public WHO GHO API to fetch static population reference stats; no user or clinical data is ever transmitted externally.

---

## Environment Setup

Modelling runs in a Conda environment (Python 3.13, PyTorch, CUDA). Recreate the environment using `environment.yml`:

```bash
conda env create -f environment.yml
conda activate phc-rxgen
```

