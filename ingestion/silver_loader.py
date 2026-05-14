"""
ingestion/silver_loader.py
--------------------------
Builds the Silver layer from Bronze data.

Two key things happen here that didn't happen in Bronze:

1. DATA QUALITY — every record is validated before it lands in Silver.
   Bad records are counted and logged, not silently dropped.

2. CDC (Change Data Capture) — we use Delta MERGE to track changes.
   If a trial's status changes from RECRUITING to COMPLETED,
   we UPDATE the existing row rather than creating a duplicate.
   We also log what changed and when.

This is the pattern that makes interviewers nod — it shows you
understand that data changes over time and you handle it properly.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, LongType, BooleanType, DateType, TimestampType, IntegerType
)
from delta.tables import DeltaTable
from quality.checks import run_quality_checks, check_duplicates
from config import BRONZE_TABLE, SILVER_TABLE
import logging

log = logging.getLogger(__name__)


# ── Silver Schema ─────────────────────────────────────────────────────────────
SILVER_SCHEMA = StructType([
    StructField("nct_id",            StringType(),   False),
    StructField("brief_title",       StringType(),   True),
    StructField("overall_status",    StringType(),   True),
    StructField("phase",             StringType(),   True),
    StructField("study_type",        StringType(),   True),
    StructField("enrollment_count",  LongType(),     True),
    StructField("conditions",        StringType(),   True),
    StructField("start_date",        DateType(),     True),
    StructField("completion_date",   DateType(),     True),
    StructField("last_update_date",  DateType(),     True),
    StructField("brief_summary",     StringType(),   True),
    StructField("lead_sponsor",      StringType(),   True),
    StructField("query_term",        StringType(),   True),
    StructField("is_recruiting",     BooleanType(),  True),
    StructField("is_completed",      BooleanType(),  True),
    StructField("is_interventional", BooleanType(),  True),
    StructField("duration_days",     IntegerType(),  True),
    StructField("_record_hash",      StringType(),   True),
    StructField("_silver_created_at",TimestampType(),True),
    StructField("_silver_updated_at",TimestampType(),True),
    StructField("_update_count",     IntegerType(),  True),
])


def run_silver_load(spark: SparkSession):
    """
    Read Bronze → clean → quality check → MERGE into Silver.

    The MERGE (CDC) logic:
    - If nct_id does NOT exist in Silver → INSERT new row
    - If nct_id EXISTS but record hash changed → UPDATE the row
    - If nct_id EXISTS and nothing changed → skip (no write needed)

    This means Silver always has exactly one row per trial,
    always showing the most current state.
    """
    log.info("Starting Silver load...")

    # 1. Read from Bronze
    bronze_df = spark.table(BRONZE_TABLE)
    log.info(f"Bronze rows: {bronze_df.count():,}")

    # 2. Transform to Silver
    silver_new = _transform_to_silver(bronze_df)
    log.info(f"Transformed rows: {silver_new.count():,}")

    # 3. Run quality checks — stop if critical rules fail
    passed, results = run_quality_checks(silver_new)
    if not passed:
        raise ValueError(
            "Critical data quality checks failed. Silver table not updated. "
            "Fix the Bronze data and re-run."
        )

    # 4. Check duplicates
    dupes = check_duplicates(silver_new)
    if dupes > 0:
        # Deduplicate — keep latest ingestion per nct_id
        silver_new = _deduplicate(silver_new)
        log.info(f"After dedup: {silver_new.count():,} rows")

    # 5. MERGE into Silver (CDC)
    _merge_into_silver(spark, silver_new)

    log.info("Silver load complete.")
    _print_silver_summary(spark)


def _transform_to_silver(df):
    """Apply all type casts, normalizations, and derived columns."""
    return (
        df
        # ── Type casts ────────────────────────────────────────────────
        .withColumn("enrollment_count",  F.col("enrollment_count").cast(LongType()))
        .withColumn("start_date",        F.to_date("start_date"))
        .withColumn("completion_date",   F.to_date("completion_date"))
        .withColumn("last_update_date",  F.to_date("last_update_date"))

        # ── Normalize strings ─────────────────────────────────────────
        .withColumn("overall_status",    F.upper(F.trim("overall_status")))
        .withColumn("study_type",        F.upper(F.trim("study_type")))
        .withColumn("brief_title",       F.trim("brief_title"))
        .withColumn("lead_sponsor",      F.trim("lead_sponsor"))

        # ── Derived columns ───────────────────────────────────────────
        .withColumn("is_recruiting",     F.col("overall_status") == "RECRUITING")
        .withColumn("is_completed",      F.col("overall_status") == "COMPLETED")
        .withColumn("is_interventional", F.col("study_type") == "INTERVENTIONAL")
        .withColumn("duration_days",
            F.datediff("completion_date", "start_date").cast(IntegerType())
        )

        # ── Record hash (used by CDC to detect changes) ───────────────
        # If any of these fields change, the hash changes → we update
        .withColumn("_record_hash",
            F.md5(F.concat_ws("|",
                F.col("nct_id"),
                F.col("overall_status"),
                F.col("enrollment_count").cast(StringType()),
                F.col("completion_date").cast(StringType()),
                F.col("lead_sponsor"),
            ))
        )

        # ── Timestamps ────────────────────────────────────────────────
        .withColumn("_silver_created_at", F.current_timestamp())
        .withColumn("_silver_updated_at", F.current_timestamp())
        .withColumn("_update_count",      F.lit(0).cast(IntegerType()))

        # ── Select final columns ──────────────────────────────────────
        .select(
            "nct_id", "brief_title", "overall_status", "phase",
            "study_type", "enrollment_count", "conditions",
            "start_date", "completion_date", "last_update_date",
            "brief_summary", "lead_sponsor", "query_term",
            "is_recruiting", "is_completed", "is_interventional",
            "duration_days", "_record_hash",
            "_silver_created_at", "_silver_updated_at", "_update_count",
        )
    )


def _deduplicate(df):
    """Keep only the latest row per nct_id."""
    from pyspark.sql.window import Window
    window = Window.partitionBy("nct_id").orderBy(F.desc("_silver_updated_at"))
    return (
        df
        .withColumn("_rank", F.row_number().over(window))
        .filter(F.col("_rank") == 1)
        .drop("_rank")
    )


def _merge_into_silver(spark: SparkSession, new_df):
    """
    CDC MERGE — the core of the Silver layer.

    WHEN MATCHED AND hash changed   → UPDATE all fields + increment update_count
    WHEN MATCHED AND hash unchanged → do nothing (no write = saves cost)
    WHEN NOT MATCHED               → INSERT new row
    """
    silver_table_path = SILVER_TABLE.replace(".", "/")

    # Check if Silver table exists
    table_exists = spark._jvm.scala.util.Try(
        lambda: spark.table(SILVER_TABLE)
    ) if False else _table_exists(spark, SILVER_TABLE)

    if not table_exists:
        # First run — just write directly
        log.info("Silver table does not exist. Creating with initial load...")
        (
            new_df.write
            .format("delta")
            .mode("overwrite")
            .option("mergeSchema", "true")
            .saveAsTable(SILVER_TABLE)
        )
        log.info(f"Silver table created: {SILVER_TABLE}")
        return

    # Subsequent runs — MERGE (CDC)
    log.info("Silver table exists. Running MERGE (CDC)...")

    silver_delta = DeltaTable.forName(spark, SILVER_TABLE)

    (
        silver_delta.alias("existing")
        .merge(
            new_df.alias("incoming"),
            "existing.nct_id = incoming.nct_id"
        )
        # If the record hash changed → update all fields
        .whenMatchedUpdate(
            condition="existing._record_hash != incoming._record_hash",
            set={
                "brief_title":        "incoming.brief_title",
                "overall_status":     "incoming.overall_status",
                "phase":              "incoming.phase",
                "enrollment_count":   "incoming.enrollment_count",
                "completion_date":    "incoming.completion_date",
                "last_update_date":   "incoming.last_update_date",
                "is_recruiting":      "incoming.is_recruiting",
                "is_completed":       "incoming.is_completed",
                "duration_days":      "incoming.duration_days",
                "_record_hash":       "incoming._record_hash",
                "_silver_updated_at": "incoming._silver_updated_at",
                "_update_count":      "existing._update_count + 1",
            }
        )
        # New trial not seen before → insert
        .whenNotMatchedInsertAll()
        .execute()
    )

    log.info("MERGE complete.")


def _table_exists(spark: SparkSession, table_name: str) -> bool:
    """Check if a Delta table exists."""
    try:
        spark.table(table_name)
        return True
    except Exception:
        return False


def _print_silver_summary(spark: SparkSession):
    """Print a summary of the Silver table."""
    df = spark.table(SILVER_TABLE)
    print(f"\n── Silver Table Summary ──────────────────────────")
    print(f"   Total trials    : {df.count():,}")
    print(f"   Recruiting now  : {df.filter('is_recruiting').count():,}")
    print(f"   Completed       : {df.filter('is_completed').count():,}")
    print(f"   Updated records : {df.filter('_update_count > 0').count():,}")
    print(f"\n   Status breakdown:")
    (
        df.groupBy("overall_status")
        .count()
        .orderBy(F.desc("count"))
        .show(8, truncate=False)
    )
