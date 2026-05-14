"""
bronze_loader.py
----------------
Writes raw ClinicalTrials data into the Bronze Delta table.
Run this on Databricks — not locally (needs PySpark + Delta).

Two modes:
  batch  → one-time full load (first run, or full refresh)
  append → add new records on top (daily incremental)

Usage inside a Databricks notebook:
    from ingestion.bronze_loader import run_bronze_load
    run_bronze_load(spark, query_term="cardiovascular", max_pages=5)
"""

import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, LongType
)
from ingestion.api_client import fetch_studies
from config import BRONZE_TABLE

log = logging.getLogger(__name__)


# ── Bronze Schema ─────────────────────────────────────────────────────────────
# Explicit schema = no surprises, no schema inference issues
BRONZE_SCHEMA = StructType([
    StructField("nct_id",            StringType(), False),
    StructField("brief_title",       StringType(), True),
    StructField("overall_status",    StringType(), True),
    StructField("phase",             StringType(), True),
    StructField("start_date",        StringType(), True),
    StructField("completion_date",   StringType(), True),
    StructField("last_update_date",  StringType(), True),
    StructField("study_type",        StringType(), True),
    StructField("enrollment_count",  LongType(),   True),
    StructField("conditions",        StringType(), True),
    StructField("brief_summary",     StringType(), True),
    StructField("lead_sponsor",      StringType(), True),
    StructField("raw_json",          StringType(), True),
    StructField("ingestion_ts",      StringType(), True),
    StructField("query_term",        StringType(), True),
    StructField("source",            StringType(), True),
])


def run_bronze_load(
    spark: SparkSession,
    query_term: str = "cardiovascular",
    max_pages: int = None,
    write_mode: str = "append",
):
    """
    Fetch from API → write to Bronze Delta table.

    Args:
        spark:       SparkSession (already exists on Databricks)
        query_term:  Search keyword
        max_pages:   Cap pages for testing. None = fetch all (~500 pages)
        write_mode:  "overwrite" for first run, "append" after that
    """
    log.info(f"Starting bronze load | query='{query_term}' | max_pages={max_pages}")

    # 1. Pull all records from the API into a Python list
    records = list(fetch_studies(
        query_term=query_term,
        max_pages=max_pages,
    ))

    if not records:
        log.warning("No records returned from API. Nothing written.")
        return

    log.info(f"Fetched {len(records):,} records. Creating Spark DataFrame...")

    # 2. Convert to Spark DataFrame with enforced schema
    df = spark.createDataFrame(records, schema=BRONZE_SCHEMA)

    # 3. Add date partition column (for efficient querying later)
    df = df.withColumn(
        "ingestion_date",
        F.to_date(F.col("ingestion_ts"))
    )

    # 4. Write to Delta table, partitioned by date
    (
        df.write
        .format("delta")
        .mode(write_mode)
        .partitionBy("ingestion_date")
        .option("mergeSchema", "true")
        .saveAsTable(BRONZE_TABLE)
    )

    log.info(f"Done. Wrote {len(records):,} rows to {BRONZE_TABLE}")

    # 5. Quick summary
    _print_summary(spark)


def _print_summary(spark: SparkSession):
    """Print a quick count by status so you can see what landed."""
    df = spark.table(BRONZE_TABLE)
    print(f"\n── Bronze Table Summary ──────────────────")
    print(f"   Total rows : {df.count():,}")
    print(f"\n   Top statuses:")
    (
        df.groupBy("overall_status")
        .count()
        .orderBy(F.desc("count"))
        .limit(6)
        .show(truncate=False)
    )
