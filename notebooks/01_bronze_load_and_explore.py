# Databricks notebook source
# Title: 01_bronze_load_and_explore
# Run this notebook on Databricks after uploading the project files.

# COMMAND ----------
# MAGIC %md
# MAGIC # Week 1 — Bronze Layer: Load & Explore
# MAGIC
# MAGIC This notebook does 4 things:
# MAGIC 1. Loads data from ClinicalTrials.gov API into the Bronze Delta table
# MAGIC 2. Validates the data landed correctly
# MAGIC 3. Profiles key columns
# MAGIC 4. Confirms you're ready for Week 2 (Silver layer)

# COMMAND ----------
# MAGIC %md ## Step 1 — Setup: add project to Python path

# COMMAND ----------

import sys
sys.path.insert(0, "/Workspace/Repos/<your-username>/clinical_platform")
# ↑ Replace <your-username> with your Databricks username

# COMMAND ----------
# MAGIC %md ## Step 2 — Create the catalog and schemas

# COMMAND ----------

# Run once to set up the Unity Catalog structure
spark.sql("CREATE CATALOG IF NOT EXISTS clinical_platform")
spark.sql("CREATE SCHEMA IF NOT EXISTS clinical_platform.bronze")
spark.sql("CREATE SCHEMA IF NOT EXISTS clinical_platform.silver")
spark.sql("CREATE SCHEMA IF NOT EXISTS clinical_platform.gold")

print("✅ Catalog and schemas ready")

# COMMAND ----------
# MAGIC %md ## Step 3 — Load Bronze table (start small: 2 pages = ~2,000 trials)

# COMMAND ----------

from ingestion.bronze_loader import run_bronze_load

run_bronze_load(
    spark       = spark,
    query_term  = "cardiovascular",
    max_pages   = 2,          # ~2,000 records — good for testing
    write_mode  = "overwrite" # overwrite on first run
)

# ✅ Once this works, change max_pages=None to load all 500K+ trials
# ⚠️  Full load takes ~30-45 minutes — run overnight

# COMMAND ----------
# MAGIC %md ## Step 4 — Validate the Bronze table

# COMMAND ----------

from config import BRONZE_TABLE
from pyspark.sql import functions as F

bronze = spark.table(BRONZE_TABLE)

print(f"Total rows     : {bronze.count():,}")
print(f"Columns        : {len(bronze.columns)}")
print(f"Null NCT IDs   : {bronze.filter(F.col('nct_id').isNull()).count()}")
print(f"Bad NCT format : {bronze.filter(~F.col('nct_id').startswith('NCT')).count()}")
# Expected: 0 nulls, 0 bad formats

# COMMAND ----------
# MAGIC %md ## Step 5 — Profile key columns

# COMMAND ----------

print("=== Status Distribution ===")
(
    bronze
    .groupBy("overall_status")
    .count()
    .orderBy(F.desc("count"))
    .show(10, truncate=False)
)

# COMMAND ----------

print("=== Phase Distribution ===")
(
    bronze
    .groupBy("phase")
    .count()
    .orderBy(F.desc("count"))
    .show(10, truncate=False)
)

# COMMAND ----------

print("=== Top 10 Sponsors ===")
(
    bronze
    .where(F.col("lead_sponsor").isNotNull())
    .groupBy("lead_sponsor")
    .count()
    .orderBy(F.desc("count"))
    .limit(10)
    .show(truncate=False)
)

# COMMAND ----------
# MAGIC %md ## Step 6 — Read 3 sample records

# COMMAND ----------

samples = (
    bronze
    .select("nct_id", "brief_title", "overall_status",
            "phase", "enrollment_count", "lead_sponsor")
    .limit(3)
    .collect()
)

for row in samples:
    print(f"\nNCT ID  : {row['nct_id']}")
    print(f"Title   : {row['brief_title']}")
    print(f"Status  : {row['overall_status']}")
    print(f"Phase   : {row['phase']}")
    print(f"Enroll  : {row['enrollment_count']}")
    print(f"Sponsor : {row['lead_sponsor']}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## ✅ Week 1 Complete
# MAGIC
# MAGIC - [x] ClinicalTrials.gov API connected
# MAGIC - [x] Bronze Delta table created and loaded
# MAGIC - [x] Data validated — no nulls, correct NCT format
# MAGIC - [x] DLT pipeline defined (Bronze → Silver → Gold)
# MAGIC
# MAGIC **Next: Week 2 — Silver layer with data quality + CDC**
