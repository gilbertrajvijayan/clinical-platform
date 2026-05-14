# Clinical Research Intelligence Platform
**Gilbert Raj Vijayan · MS Graduate, UNT · Databricks Certified Data Engineer Associate**

---

## Project Structure

```
clinical_platform/
│
├── config.py                          ← All settings in one place
│
├── ingestion/
│   ├── __init__.py
│   ├── api_client.py                  ← ClinicalTrials.gov API fetcher
│   └── bronze_loader.py               ← Writes to Bronze Delta table
│
├── dlt/
│   └── pipeline.py                    ← Delta Live Tables: Bronze→Silver→Gold
│
├── notebooks/
│   └── 01_bronze_load_and_explore.py  ← Run this on Databricks
│
├── tests/
│   └── (Week 2 — coming soon)
│
├── test_api.py                        ← Run this LOCALLY first
└── README.md
```

---

## Quick Start

### Step 1 — Local test (no Databricks needed)

```bash
# Create project folder
mkdir clinical_platform
cd clinical_platform

# Install dependency (only one needed locally)
pip install requests

# Copy all project files into this folder
# Then run the test:
python test_api.py
```

**Expected output:**
```
✅ SUCCESS — Fetched 10 records

[1] NCT04xxxxxx
    Title  : A Study of [Drug] in Patients With Heart Failure
    Status : RECRUITING
    Phase  : PHASE3
    ...
```

---

### Step 2 — Upload to Databricks

1. Open Databricks → **Workspace → Repos → Add Repo**
2. Connect your GitHub repo (push your local folder to GitHub first)
3. Or manually upload files to `/Workspace/Repos/<username>/clinical_platform/`

---

### Step 3 — Run the Bronze notebook

1. Open `notebooks/01_bronze_load_and_explore.py` in Databricks
2. Attach to a cluster (Standard, DBR 13.0+)
3. Update line: `sys.path.insert(0, "/Workspace/Repos/<your-username>/clinical_platform")`
4. Run all cells top to bottom

---

### Step 4 — Set up DLT Pipeline (optional for Week 1)

1. Databricks → **Workflows → Delta Live Tables → Create Pipeline**
2. Pipeline name: `clinical_research_dlt`
3. Source code: `/Workspace/Repos/<username>/clinical_platform/dlt/pipeline.py`
4. Target schema: `clinical_platform`
5. Mode: **Triggered**
6. Click **Start**

---

## Week-by-Week Plan

| Week | What gets built | Key concepts |
|------|----------------|--------------|
| 1 | API ingestion → Bronze Delta table | Delta Lake, partitioning, schema |
| 2 | Silver layer + data quality | CDC, Great Expectations, MERGE |
| 3 | Gold layer + dbt models | Star schema, aggregations, dbt |
| 4 | ChromaDB vector embeddings | Sentence transformers, vector DB |
| 5 | RAG pipeline + Claude API | LangChain, RAG, prompt engineering |
| 6 | MLflow experiment tracking | Model registry, eval metrics |
| 7 | FastAPI + Streamlit UI | REST API, demo dashboard |
| 8 | GitHub polish + portfolio | README, demo video, LinkedIn |

---

## Data Source

**ClinicalTrials.gov API v2** — free, no authentication required

```
https://clinicaltrials.gov/api/v2/studies?query.term=cardiovascular&pageSize=1000
```

- 500,000+ clinical trials
- Updates daily
- JSON format
- Pagination via `nextPageToken`

---

*Built by Gilbert Raj Vijayan | MS Data Science, UNT Denton | Databricks Certified Data Engineer Associate*
