# 🏥 Clinical Research Intelligence Platform

> End-to-end AI-powered clinical trial discovery platform built on Databricks, Delta Lake, ChromaDB, and Claude API.

**Gilbert Raj Vijayan** · MS Data Science, UNT Denton · Databricks Certified Data Engineer Associate

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Databricks](https://img.shields.io/badge/Databricks-Delta%20Lake-orange.svg)](https://databricks.com)
[![Claude](https://img.shields.io/badge/Claude-API-purple.svg)](https://anthropic.com)

---

## What This Is

A production-grade data engineering and AI platform that ingests 500,000+ clinical trials from ClinicalTrials.gov, processes them through a Medallion Lakehouse architecture on Databricks, and exposes them through a RAG-powered AI assistant that answers natural language questions with cited trial references.

**Example query:** *"Which cardiovascular trials are currently recruiting patients with heart failure?"*

**What the platform does:** Retrieves the 10 most semantically relevant trials from ChromaDB, sends them as context to Claude API, and returns a cited answer with NCT IDs, enrollment counts, and sponsor names — in under 3 seconds.

---

## Architecture

```
ClinicalTrials.gov API (free · no auth · 500K+ trials)
        │
        ▼  paginated REST fetch with retry/backoff
┌─────────────────────────────────────┐
│  BRONZE LAYER — Delta Lake          │
│  Raw JSON · Append-only             │
│  Partitioned by ingestion_date      │
└─────────────────────────────────────┘
        │
        ▼  Delta Live Tables pipeline
┌─────────────────────────────────────┐
│  SILVER LAYER — Delta Lake          │
│  Typed · Deduplicated               │
│  CDC MERGE (tracks status changes)  │
│  5 automated data quality rules     │
│  Change Data Feed enabled           │
└─────────────────────────────────────┘
        │
        ▼  aggregation pipeline
┌─────────────────────────────────────┐
│  GOLD LAYER — Delta Lake            │
│  Trial status summary               │
│  Sponsor leaderboard                │
│  Condition analytics                │
│  Yearly trend analysis              │
└─────────────────────────────────────┘
        │
        ▼  embedding pipeline
┌─────────────────────────────────────┐
│  VECTOR STORE — ChromaDB            │
│  sentence-transformers embeddings   │
│  384-dimension vectors              │
│  Cosine similarity search           │
└─────────────────────────────────────┘
        │
        ▼  RAG pipeline
┌─────────────────────────────────────┐
│  AI LAYER — Claude API              │
│  Retrieval-Augmented Generation     │
│  Natural language Q&A               │
│  NCT ID citations in every answer   │
└─────────────────────────────────────┘
        │
        ▼  observability
┌─────────────────────────────────────┐
│  MLFLOW                             │
│  Experiment tracking per query      │
│  Embedding model in Model Registry  │
│  Retrieval quality metrics logged   │
└─────────────────────────────────────┘
        │
        ▼  serving
┌─────────────────────────────────────┐
│  STREAMLIT DASHBOARD                │
│  AI chat interface                  │
│  Trial search + filters             │
│  Analytics dashboard                │
└─────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Platform | Databricks + Delta Lake + Delta Live Tables |
| Data Quality | Custom quality rules + CDC MERGE patterns |
| Orchestration | Databricks Jobs + Auto Loader |
| Vector Store | ChromaDB + sentence-transformers (all-MiniLM-L6-v2) |
| AI Layer | Anthropic Claude API + RAG pipeline |
| ML Tracking | MLflow experiment tracking + Model Registry |
| Dashboard | Streamlit |
| Language | Python + PySpark + SQL |
| Data Source | ClinicalTrials.gov API v2 |

---

## Project Structure

```
clinical-platform/
│
├── config.py                           ← All settings in one place
│
├── ingestion/
│   ├── api_client.py                   ← ClinicalTrials.gov API fetcher
│   ├── bronze_loader.py                ← Bronze Delta table writer
│   └── silver_loader.py               ← Silver layer with CDC MERGE
│
├── dlt/
│   └── pipeline.py                     ← Delta Live Tables: Bronze→Silver→Gold
│
├── quality/
│   └── checks.py                       ← Data quality rules engine
│
├── notebooks/
│   ├── 01_bronze_load_and_explore.py   ← Week 1: Bronze ingestion
│   ├── 02_silver_quality_cdc.py        ← Week 2: Silver + CDC
│   ├── 03_gold_layer_analytics.py      ← Week 3: Gold tables
│   ├── 04_vector_embeddings_chromadb.py← Week 4: Embeddings
│   ├── 05_rag_claude_api.py            ← Week 5: RAG pipeline
│   └── 06_mlflow_tracking.py           ← Week 6: MLflow tracking
│
├── app_lite.py                         ← Streamlit dashboard
├── test_api.py                         ← Local API test
└── README.md
```

---

## Key Results

| Metric | Value |
|---|---|
| Trials indexed | 2,000 (demo) · 500K+ (full load) |
| Pipeline layers | Bronze → Silver → Gold |
| Data quality rules | 5 automated checks per pipeline run |
| Vector dimensions | 384 (all-MiniLM-L6-v2) |
| RAG retrieval | Top-10 semantic search per query |
| MLflow runs logged | 7 (5 RAG queries + 2 model registration) |
| Platform KPIs | 2,000 trials · 228 recruiting · 23.7M patients enrolled |
| Query latency | Under 3 seconds end-to-end |

---

## Quick Start

### Local Test (no Databricks needed)

```bash
git clone https://github.com/gilbertrajvijayan/clinical-platform
cd clinical-platform
pip install requests
python test_api.py
```

### Run the Dashboard

```bash
pip install streamlit anthropic
export KMP_DUPLICATE_LIB_OK=TRUE
streamlit run app_lite.py
```

Open `http://localhost:8501` in your browser.

### Databricks Setup

1. Upload repo to Databricks Repos
2. Run notebooks in order (01 → 06)
3. Create DLT pipeline pointing to `dlt/pipeline.py`

---

## The Data Pipeline Explained

**Bronze** — Raw data lands here exactly as it came from the API. No transformations. If anything breaks downstream, we always have the original.

**Silver** — Data is cleaned, typed, and validated here. Five quality rules run automatically. If a critical rule fails, the pipeline stops — bad data never reaches production. CDC MERGE ensures that when a trial's status changes on ClinicalTrials.gov, we update the existing row instead of creating a duplicate.

**Gold** — Business-ready aggregations live here. Trial status breakdowns, sponsor leaderboards, condition analytics, yearly trends. This is what dashboards and the AI layer query.

**Vector Store** — Every trial's title, conditions, and summary are embedded into 384-dimension vectors using sentence-transformers. Stored in ChromaDB. Enables semantic search — finding trials by meaning, not just keywords.

**RAG Pipeline** — User asks a question → system embeds the question → finds the 10 most similar trials in ChromaDB → sends them as context to Claude API → Claude returns a cited, intelligent answer.

---

## Business Context

This platform solves a real problem: clinical research teams at hospital systems and health analytics companies spend days manually searching ClinicalTrials.gov for relevant trials. This platform reduces that to seconds through semantic search and AI-powered synthesis.

Target use cases:
- Hospital research coordinators finding recruiting trials for eligible patients
- Health analytics companies tracking competitor trial activity
- Pharmaceutical sponsors monitoring enrollment trends
- Research institutions identifying collaboration opportunities

---

## Target Employers

This project was built targeting roles at:
- **McKesson** (Irving, TX) — healthcare data platform engineering
- **UT Southwestern** (Dallas, TX) — clinical research data engineering
- **Oracle Health** — cloud data engineering for healthcare
- **Deloitte** (Dallas, TX) — data & AI consulting, life sciences practice

---

*Built by Gilbert Raj Vijayan | MS Data Science, UNT Denton | Databricks Certified Data Engineer Associate*
*GitHub: github.com/gilbertrajvijayan/clinical-platform*
