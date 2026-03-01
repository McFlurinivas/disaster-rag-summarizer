# disaster-rag-summarizer

A **Retrieval-Augmented Generation (RAG)** system for querying and summarising historical disaster events. Given a natural-language query (e.g., *"recent volcano in Indonesia"*), the system retrieves the most relevant disaster records from a 17 000-event knowledge base and produces a concise, factually grounded summary.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Architecture](#architecture)
4. [Dataset](#dataset)
5. [Models](#models)
6. [Evaluation Results](#evaluation-results)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Configuration](#configuration)
10. [Notebooks](#notebooks)

---

## Overview

This project was developed as part of a research thesis at **Vellore Institute of Technology** on the topic *"RAG-Enhanced Disaster Summarization: A Large Language Model Approach"*.

It implements two complete pipelines:

| | Pipeline 1 | Pipeline 2 |
|---|---|---|
| **Retrieval** | Bi-encoder cosine similarity | Hybrid (semantic + TF-IDF) + cross-encoder reranking |
| **Generation** | Ollama LLaMA 3.1 8B | BART-large-CNN |
| **Best for** | Conversational answers | High-precision, evaluable summaries |

---

## Project Structure

```
disaster-rag-summarizer/
│
├── dataset_download.ipynb          # Download CrisisFACTS social-media messages
├── dataset_generation.ipynb        # Build & embed the disaster knowledge base
├── chatbot_training.ipynb          # RAG pipelines + evaluation
│
├── disaster_information.xlsx       # Raw EM-DAT disaster database (source)
├── extracted_disaster_info.csv     # Cleaned & structured disaster records
├── disaster_info_with_embeddings.csv  # Records + pre-computed 768-dim embeddings
├── disaster_messages.csv           # CrisisFACTS tweets (Pipeline social context)
│
├── disaster_classifier_model/      # Fine-tuned disaster classifier weights
├── fine_tuned_disaster_summarizer/ # Fine-tuned summarizer weights
├── results/                        # Evaluation outputs
│
├── ds_config.json                  # DeepSpeed / training configuration
├── requirements.txt                # All Python dependencies (pinned)
└── Pipeline Flow.drawio            # System architecture diagram
```

---

## Architecture

### Data Preparation (`dataset_generation.ipynb`)

```
disaster_information.xlsx
        │
        ▼
  Extract columns
  (Disaster Group, Type, Country, Dates, Damage, River Basin, ...)
        │
        ▼
  build_combined()  ←── clean NaN values, compose natural-language descriptions
        │
        ▼
  Embed with multi-qa-mpnet-base-dot-v1  (768-dim, batch_size=64)
        │
        ▼
  disaster_info_with_embeddings.csv  (17 325 rows)
```

### Pipeline 1 — Simple Retrieval + LLaMA

```
User Query
    │
    ▼
Bi-encoder embedding  (multi-qa-mpnet-base-dot-v1)
    │
    ▼
Cosine similarity over all 17 325 records
    │
    ▼
Top-1 event
    │
    ▼
LLaMA 3.1 8B (via Ollama)  →  Conversational summary
```

### Pipeline 2 — Hybrid Retrieval + BART (recommended)

```
User Query
    │
    ├─► Keyword expansion (DISASTER_SYNONYMS)
    │
    ├─► Stage 1a: Semantic similarity  (weight 0.65)
    │                  multi-qa-mpnet-base-dot-v1
    │
    ├─► Stage 1b: TF-IDF lexical similarity  (weight 0.35)
    │
    ▼
Hybrid score → Top-20 candidates
    │
    ▼
Stage 2: Cross-encoder reranking
    │   cross-encoder/ms-marco-MiniLM-L-6-v2
    │
    ▼
Top-5 results  (year filter applied if query contains a year)
    │
    ▼
Primary event (rank-1)  →  BART-large-CNN  →  Bullet-point summary
    │
    ▼
Evaluation: ROUGE · BLEU · BERTScore · Factual Consistency
```

---

## Dataset

### EM-DAT Disaster Database
- **Source:** `disaster_information.xlsx` — the International Disaster Database (EM-DAT)
- **Rows:** 17 325 disaster events spanning 1900–2024
- **Key fields:** Disaster Group/Subgroup/Type, Event Name, Country, Subregion, Location, Start/End Date, Total Damage, Insured Damage, AID Contribution, Reconstruction Costs, River Basin, Associated Types

### CrisisFACTS Social Media Dataset
- Downloaded via `dataset_download.ipynb` using the `ir_datasets` library
- Covers 11 real disaster events (wildfires, hurricanes, floods, explosions) from 2017–2020
- Saved as `disaster_messages.csv` for social media context augmentation

---

## Models

| Role | Model | Dimensions | Notes |
|---|---|---|---|
| Bi-encoder embedding | `sentence-transformers/multi-qa-mpnet-base-dot-v1` | 768 | Fine-tuned for QA retrieval |
| Cross-encoder reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | — | Re-scores (query, passage) pairs |
| Summarization | `facebook/bart-large-cnn` | — | Abstractive summarization |
| LLM (Pipeline 1) | `llama3.1:8b` via Ollama | — | Conversational generation |
| NER | `en_core_web_sm` (spaCy) | — | Entity extraction fallback |

---

## Evaluation Results

Evaluated on the **Lewotolo Volcano, Indonesia (2020)** event with query *"recent volcano in Indonesia"*:

| Metric | Score |
|---|---|
| ROUGE-1 F1 | **0.8846** |
| ROUGE-2 F1 | **0.8235** |
| ROUGE-L F1 | **0.8846** |
| BLEU (smoothed) | **0.6827** |
| BERTScore F1 | **0.9739** |
| Coverage | 70.59% |
| Hallucination Rate | **0.00%** |

> **Note:** Reference = the cleaned primary event text (same document BART was asked to summarise). ROUGE/BLEU measure faithfulness; BERTScore measures semantic similarity.

---

## Installation

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/) installed and running (for Pipeline 1 only)
- CUDA-capable GPU recommended (CPU works but embedding generation will be slow)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/disaster-rag-summarizer.git
cd disaster-rag-summarizer

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the spaCy language model
python -m spacy download en_core_web_sm

# 5. (Pipeline 1 only) Pull the LLaMA model via Ollama
ollama pull llama3.1:8b
```

### GPU / CPU PyTorch

The `requirements.txt` defaults to CUDA 12.1. Change the torch lines if needed:

```bash
# CPU-only
pip install torch==2.3.0 torchaudio==2.3.0 torchvision==0.18.0

# CUDA 11.8
pip install torch==2.3.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

---

## Usage

### Step 1 — Build the knowledge base

Open and run all cells in **`dataset_generation.ipynb`**:

```
Cell 1  →  Extract columns from disaster_information.xlsx
Cell 2  →  Build clean natural-language combined text (NaN-safe)
Cell 3  →  Generate 768-dim embeddings → disaster_info_with_embeddings.csv
           (~68 minutes on CPU for 17 325 rows)
```

> If `disaster_info_with_embeddings.csv` is already present and up to date, you can skip this step.

### Step 2 — Run the RAG pipelines

Open **`chatbot_training.ipynb`** and run the cells sequentially.

**Pipeline 1** (Cells 1–5):
```python
query = "Tell me about the disaster in Japan."
# Returns a conversational LLaMA summary
```

**Pipeline 2** (Cells 7–16):
```python
query = "recent volcano in Indonesia"
# Returns top-5 retrieved events + BART summary + evaluation metrics
```

#### Changing the query
Simply update the `query` variable in Cell 4 (Pipeline 1) or Cell 14 (Pipeline 2) and re-run from that cell.

---

## Configuration

All Pipeline 2 parameters are centralised in **Cell 7** of `chatbot_training.ipynb`. No other cell needs to be edited to change behaviour.

```python
# Models
SUMMARIZER_MODEL    = "facebook/bart-large-cnn"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval weights (must sum to 1)
HYBRID_WEIGHT_SEMANTIC = 0.65
HYBRID_WEIGHT_TFIDF    = 0.35

# How many events to retrieve / rerank
TOP_K       = 5
CANDIDATE_K = 20

# BART output length
SUMMARY_MAX_TOKENS = 150
SUMMARY_MIN_RATIO  = 0.67

# Keyword expansion — add synonyms here to improve recall for specific disaster types
DISASTER_SYNONYMS = {
    "earthquake": ["seismic activity", "quake", "tremor", "seismic"],
    "flood":      ["inundation", "deluge", "high water", "flooding", "flash flood"],
    # ... (15 disaster types covered)
}

# Stopwords excluded from factual-consistency scoring
CONSISTENCY_STOPWORDS = { "the", "a", "an", ... }
```

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `dataset_generation.ipynb` | Cleans EM-DAT data, builds combined text, generates embeddings |
| `chatbot_training.ipynb` | Two RAG pipelines: retrieval, summarization, and evaluation |
