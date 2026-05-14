"""
dlt/pipeline.py
---------------
Delta Live Tables pipeline — Bronze → Silver → Gold.
Deploy this file as the source in a Databricks DLT pipeline.

Databricks DLT setup:
  1. Go to Workflows → Delta Live Tables → Create Pipeline
  2. Source code: point to this file
  3. Target schema: clinical_platform
  4. Mode: Triggered (run daily)
"""

import dlt
from pyspark.sql import functions as F
from pyspark.sql.types import LongType


# ── BRONZE ────────────────────────────────────────────────────────────────────
# Reads raw JSON files dropped into the landing zone by the API client.
# Append-only. No transformations. Keep everything raw.

@dlt.table(
    name="bronze_clinical_trials",
    comment="Raw data from ClinicalTrials.gov — append only, no changes",
    table_properties={
        "quality": "bronze",
        "pipelines.autoOptimize.managed": "true",
    },
)
def bronze_clinical_trials():
    return (
        spark.readStream
        .format("cloudFiles")                               # Auto Loader
        .option("cloudFiles.format", "json")
        .option("cloudFiles.schemaLocation",
                "/dbfs/clinical_platform/schema/bronze")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
        .load("/dbfs/clinical_platform/landing/")
        .withColumn("_ingestion_ts",  F.current_timestamp())
        .withColumn("_source_file",   F.input_file_name())
    )


# ── SILVER ────────────────────────────────────────────────────────────────────
# Clean, typed, validated data.
# Bad records are dropped and logged — not silently lost.

# Data quality rules
RULES = {
    "valid_nct_id":        "nct_id IS NOT NULL AND nct_id LIKE 'NCT%'",
    "has_title":           "brief_title IS NOT NULL AND LENGTH(brief_title) > 5",
    "positive_enrollment": "enrollment_count IS NULL OR enrollment_count >= 0",
}


@dlt.table(
    name="silver_clinical_trials",
    comment="Cleaned, typed, validated clinical trials — ready for analytics",
    table_properties={
        "quality": "silver",
        "delta.enableChangeDataFeed": "true",   # needed for CDC in Week 2
        "pipelines.autoOptimize.managed": "true",
    },
)
@dlt.expect_or_drop("valid_nct_id",        RULES["valid_nct_id"])
@dlt.expect_or_drop("has_title",           RULES["has_title"])
@dlt.expect("positive_enrollment",         RULES["positive_enrollment"])  # warn, don't drop
def silver_clinical_trials():
    bronze = dlt.read_stream("bronze_clinical_trials")
    return (
        bronze
        # ── Cast types ───────────────────────────────────────────────────
        .withColumn("enrollment_count",  F.col("enrollment_count").cast(LongType()))
        .withColumn("start_date",        F.to_date("start_date"))
        .withColumn("completion_date",   F.to_date("completion_date"))
        .withColumn("last_update_date",  F.to_date("last_update_date"))
        .withColumn("ingestion_ts",      F.to_timestamp("ingestion_ts"))

        # ── Normalize strings ────────────────────────────────────────────
        .withColumn("overall_status",    F.upper(F.trim("overall_status")))
        .withColumn("study_type",        F.upper(F.trim("study_type")))
        .withColumn("brief_title",       F.trim("brief_title"))
        .withColumn("lead_sponsor",      F.trim("lead_sponsor"))

        # ── Derived fields ───────────────────────────────────────────────
        .withColumn("is_recruiting",     F.col("overall_status") == "RECRUITING")
        .withColumn("is_completed",      F.col("overall_status") == "COMPLETED")
        .withColumn("is_interventional", F.col("study_type") == "INTERVENTIONAL")
        .withColumn("duration_days",     F.datediff("completion_date", "start_date"))

        # ── Record hash for CDC change detection ─────────────────────────
        .withColumn("_record_hash",
            F.md5(F.concat_ws("|",
                "nct_id", "overall_status",
                "enrollment_count", "completion_date"
            ))
        )
        .withColumn("_silver_ts", F.current_timestamp())

        # ── Final column selection ────────────────────────────────────────
        .select(
            "nct_id", "brief_title", "overall_status", "phase",
            "study_type", "enrollment_count", "conditions",
            "start_date", "completion_date", "last_update_date",
            "brief_summary", "lead_sponsor", "query_term", "source",
            "ingestion_ts", "is_recruiting", "is_completed",
            "is_interventional", "duration_days",
            "_record_hash", "_silver_ts",
        )
    )


# ── GOLD ──────────────────────────────────────────────────────────────────────
# Business-ready aggregations. Feeds dashboards and the AI layer.

@dlt.table(
    name="gold_trial_status_summary",
    comment="Trial counts by status and phase — for dashboards",
    table_properties={"quality": "gold"},
)
def gold_trial_status_summary():
    silver = dlt.read("silver_clinical_trials")
    return (
        silver
        .groupBy("overall_status", "phase", "study_type")
        .agg(
            F.count("*").alias("trial_count"),
            F.avg("enrollment_count").alias("avg_enrollment"),
            F.sum("enrollment_count").alias("total_enrollment"),
            F.countDistinct("lead_sponsor").alias("unique_sponsors"),
        )
        .withColumn("_gold_ts", F.current_timestamp())
    )


@dlt.table(
    name="gold_sponsor_leaderboard",
    comment="Top sponsors by number of trials and total enrollment",
    table_properties={"quality": "gold"},
)
def gold_sponsor_leaderboard():
    silver = dlt.read("silver_clinical_trials")
    return (
        silver
        .where(F.col("lead_sponsor").isNotNull())
        .groupBy("lead_sponsor")
        .agg(
            F.count("*").alias("total_trials"),
            F.sum(F.when(F.col("is_recruiting"), 1).otherwise(0)).alias("active_trials"),
            F.sum(F.when(F.col("is_completed"), 1).otherwise(0)).alias("completed_trials"),
            F.sum("enrollment_count").alias("total_enrollment"),
        )
        .orderBy(F.desc("total_trials"))
        .withColumn("_gold_ts", F.current_timestamp())
    )
