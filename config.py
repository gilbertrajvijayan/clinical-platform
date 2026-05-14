# config.py — Central configuration for Clinical Research Intelligence Platform

# ── Databricks Catalog / Schema ───────────────────────────────────────────────
CATALOG       = "clinical_platform"
SCHEMA_BRONZE = "bronze"
SCHEMA_SILVER = "silver"
SCHEMA_GOLD   = "gold"

BRONZE_TABLE  = f"{CATALOG}.{SCHEMA_BRONZE}.clinical_trials"
SILVER_TABLE  = f"{CATALOG}.{SCHEMA_SILVER}.clinical_trials"
GOLD_TABLE    = f"{CATALOG}.{SCHEMA_GOLD}.trial_summary"

# ── Storage Paths (DBFS) ──────────────────────────────────────────────────────
BASE_PATH              = "/dbfs/clinical_platform"
LANDING_ZONE_PATH      = f"{BASE_PATH}/landing"
BRONZE_CHECKPOINT_PATH = f"{BASE_PATH}/checkpoints/bronze"
SILVER_CHECKPOINT_PATH = f"{BASE_PATH}/checkpoints/silver"
VECTOR_STORE_PATH      = f"{BASE_PATH}/chromadb"

# ── ClinicalTrials.gov API ────────────────────────────────────────────────────
CT_API_BASE_URL  = "https://clinicaltrials.gov/api/v2/studies"
CT_API_PAGE_SIZE = 1000
CT_API_DELAY     = 0.5   # seconds between pages (be polite to free API)

# ── Search Terms (cardiovascular focus matches your resume) ───────────────────
QUERY_TERMS = [
    "cardiovascular",
    "heart failure",
    "coronary artery disease",
    "atrial fibrillation",
    "hypertension",
]

# ── AI / Vector Layer (Week 4-5) ──────────────────────────────────────────────
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION = "clinical_trials_v1"
CLAUDE_MODEL      = "claude-sonnet-4-20250514"
CLAUDE_MAX_TOKENS = 1024
RAG_TOP_K         = 10
