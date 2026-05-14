# Databricks notebook source
# Title: 03_gold_layer_analytics
# Week 3 — Gold Layer: Business-Ready Analytics Tables
#
# What this builds:
# 1. Gold aggregation tables from Silver data
# 2. Business metrics — recruitment rates, completion rates
# 3. Top sponsors leaderboard
# 4. Databricks SQL queries you can turn into dashboards

# COMMAND ----------
# MAGIC %md
# MAGIC # Week 3 — Gold Layer: Business Analytics
# MAGIC
# MAGIC Silver = clean, one row per trial
# MAGIC Gold = answers to business questions
# MAGIC
# MAGIC Every Gold table here answers a specific question:
# MAGIC - How many trials are recruiting right now by phase?
# MAGIC - Which sponsors run the most trials?
# MAGIC - What is the average enrollment by condition?
# MAGIC - How has trial completion trended over time?

# COMMAND ----------
# MAGIC %md ## Step 1 — Setup

# COMMAND ----------

import sys
sys.path.insert(0, "/Workspace/Users/gilbertrajvijayan@gmail.com/clinical-platform")

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from config import SILVER_TABLE, CATALOG

print("Setup complete")

# COMMAND ----------
# MAGIC %md ## Step 2 — Read Silver table

# COMMAND ----------

silver = spark.table(SILVER_TABLE)
print(f"Silver rows: {silver.count():,}")
silver.printSchema()

# COMMAND ----------
# MAGIC %md ## Step 3 — Gold Table 1: Trial Status Summary
# MAGIC
# MAGIC **Business question:** How many trials are in each status and phase?
# MAGIC Used by: Product team, dashboard homepage

# COMMAND ----------

gold_status = (
    silver
    .groupBy("overall_status", "phase", "study_type")
    .agg(
        F.count("*").alias("trial_count"),
        F.avg("enrollment_count").alias("avg_enrollment"),
        F.sum("enrollment_count").alias("total_enrollment"),
        F.countDistinct("lead_sponsor").alias("unique_sponsors"),
        F.sum(F.when(F.col("duration_days").isNotNull(), 1).otherwise(0))
         .alias("trials_with_duration"),
        F.avg("duration_days").alias("avg_duration_days"),
    )
    .withColumn("avg_enrollment",    F.round("avg_enrollment", 0))
    .withColumn("avg_duration_days", F.round("avg_duration_days", 0))
    .withColumn("_gold_updated_at",  F.current_timestamp())
    .orderBy(F.desc("trial_count"))
)

gold_status.show(10, truncate=False)

# Write to Gold table
(
    gold_status.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(f"{CATALOG}.gold.trial_status_summary")
)
print(f"Written: {CATALOG}.gold.trial_status_summary")

# COMMAND ----------
# MAGIC %md ## Step 4 — Gold Table 2: Sponsor Leaderboard
# MAGIC
# MAGIC **Business question:** Which sponsors run the most trials?
# MAGIC Used by: Sales team to identify top hospital accounts

# COMMAND ----------

gold_sponsors = (
    silver
    .where(F.col("lead_sponsor").isNotNull())
    .groupBy("lead_sponsor")
    .agg(
        F.count("*").alias("total_trials"),
        F.sum(F.when(F.col("is_recruiting"),     1).otherwise(0)).alias("recruiting_trials"),
        F.sum(F.when(F.col("is_completed"),      1).otherwise(0)).alias("completed_trials"),
        F.sum(F.when(F.col("is_interventional"), 1).otherwise(0)).alias("interventional_trials"),
        F.sum("enrollment_count").alias("total_enrollment"),
        F.avg("enrollment_count").alias("avg_enrollment_per_trial"),
    )
    .withColumn("completion_rate",
        F.round(F.col("completed_trials") / F.col("total_trials") * 100, 1)
    )
    .withColumn("avg_enrollment_per_trial", F.round("avg_enrollment_per_trial", 0))
    .withColumn("_gold_updated_at", F.current_timestamp())
    .orderBy(F.desc("total_trials"))
)

print("Top 15 sponsors by trial count:")
gold_sponsors.limit(15).show(truncate=False)

# Write to Gold table
(
    gold_sponsors.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(f"{CATALOG}.gold.sponsor_leaderboard")
)
print(f"Written: {CATALOG}.gold.sponsor_leaderboard")

# COMMAND ----------
# MAGIC %md ## Step 5 — Gold Table 3: Condition Analytics
# MAGIC
# MAGIC **Business question:** Which conditions have the most trial activity?
# MAGIC Used by: Product team for content prioritization

# COMMAND ----------

# Explode conditions (some trials have multiple conditions)
from pyspark.sql.functions import explode, split

gold_conditions = (
    silver
    .where(F.col("conditions").isNotNull())
    # Split comma-separated conditions into individual rows
    .withColumn("condition", explode(split(F.col("conditions"), ", ")))
    .withColumn("condition", F.trim(F.col("condition")))
    .where(F.length("condition") > 2)
    .groupBy("condition")
    .agg(
        F.count("*").alias("trial_count"),
        F.sum(F.when(F.col("is_recruiting"), 1).otherwise(0)).alias("recruiting_now"),
        F.sum("enrollment_count").alias("total_patients_enrolled"),
        F.avg("enrollment_count").alias("avg_enrollment"),
    )
    .withColumn("avg_enrollment", F.round("avg_enrollment", 0))
    .withColumn("_gold_updated_at", F.current_timestamp())
    .orderBy(F.desc("trial_count"))
)

print("Top 15 conditions by trial activity:")
gold_conditions.limit(15).show(truncate=False)

# Write to Gold table
(
    gold_conditions.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(f"{CATALOG}.gold.condition_analytics")
)
print(f"Written: {CATALOG}.gold.condition_analytics")

# COMMAND ----------
# MAGIC %md ## Step 6 — Gold Table 4: Yearly Trend
# MAGIC
# MAGIC **Business question:** How has trial activity changed over time?
# MAGIC Used by: Executive dashboard, investor reporting

# COMMAND ----------

gold_trend = (
    silver
    .where(F.col("start_date").isNotNull())
    .withColumn("start_year", F.year("start_date"))
    .where(F.col("start_year") >= 2000)
    .where(F.col("start_year") <= 2025)
    .groupBy("start_year")
    .agg(
        F.count("*").alias("trials_started"),
        F.sum(F.when(F.col("is_interventional"), 1).otherwise(0))
         .alias("interventional_trials"),
        F.sum("enrollment_count").alias("total_enrollment"),
        F.avg("enrollment_count").alias("avg_enrollment"),
    )
    .withColumn("avg_enrollment", F.round("avg_enrollment", 0))
    .withColumn("_gold_updated_at", F.current_timestamp())
    .orderBy("start_year")
)

print("Trial activity by year:")
gold_trend.show(30, truncate=False)

# Write to Gold table
(
    gold_trend.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(f"{CATALOG}.gold.yearly_trend")
)
print(f"Written: {CATALOG}.gold.yearly_trend")

# COMMAND ----------
# MAGIC %md ## Step 7 — Key Business Metrics (KPIs)
# MAGIC
# MAGIC These are the numbers that go on the homepage dashboard

# COMMAND ----------

total_trials      = silver.count()
recruiting_now    = silver.filter("is_recruiting").count()
completed_trials  = silver.filter("is_completed").count()
total_enrollment  = silver.agg(F.sum("enrollment_count")).collect()[0][0] or 0
unique_sponsors   = silver.select("lead_sponsor").distinct().count()
avg_duration      = silver.filter(F.col("duration_days").isNotNull()) \
                          .agg(F.avg("duration_days")).collect()[0][0] or 0

recruitment_rate  = round(recruiting_now / total_trials * 100, 1)
completion_rate   = round(completed_trials / total_trials * 100, 1)

print("=" * 50)
print("  PLATFORM KPIs — Clinical Research Intelligence")
print("=" * 50)
print(f"  Total trials indexed    : {total_trials:,}")
print(f"  Currently recruiting    : {recruiting_now:,} ({recruitment_rate}%)")
print(f"  Completed trials        : {completed_trials:,} ({completion_rate}%)")
print(f"  Total patients enrolled : {int(total_enrollment):,}")
print(f"  Unique sponsors         : {unique_sponsors:,}")
print(f"  Avg trial duration      : {int(avg_duration):,} days")
print("=" * 50)

# COMMAND ----------
# MAGIC %md ## Step 8 — Validate all Gold tables exist

# COMMAND ----------

gold_tables = [
    f"{CATALOG}.gold.trial_status_summary",
    f"{CATALOG}.gold.sponsor_leaderboard",
    f"{CATALOG}.gold.condition_analytics",
    f"{CATALOG}.gold.yearly_trend",
]

for table in gold_tables:
    count = spark.table(table).count()
    print(f"  {table}: {count:,} rows")

# COMMAND ----------
# MAGIC %md ## Step 9 — SQL queries for Databricks dashboard
# MAGIC
# MAGIC Copy these into Databricks SQL Editor to create your dashboard charts

# COMMAND ----------

print("""
-- Chart 1: Trial Status Breakdown (Pie chart)
SELECT overall_status, SUM(trial_count) as total
FROM clinical_platform.gold.trial_status_summary
GROUP BY overall_status
ORDER BY total DESC;

-- Chart 2: Top 10 Sponsors (Bar chart)
SELECT lead_sponsor, total_trials, recruiting_trials, completion_rate
FROM clinical_platform.gold.sponsor_leaderboard
LIMIT 10;

-- Chart 3: Trial Activity by Year (Line chart)
SELECT start_year, trials_started, total_enrollment
FROM clinical_platform.gold.yearly_trend
ORDER BY start_year;

-- Chart 4: Top Conditions (Horizontal bar)
SELECT condition, trial_count, recruiting_now
FROM clinical_platform.gold.condition_analytics
LIMIT 15;
""")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Week 3 Complete
# MAGIC
# MAGIC - [x] Gold Table 1: Trial status summary by phase and study type
# MAGIC - [x] Gold Table 2: Sponsor leaderboard with completion rates
# MAGIC - [x] Gold Table 3: Condition analytics with patient enrollment
# MAGIC - [x] Gold Table 4: Yearly trend of trial activity
# MAGIC - [x] Platform KPIs calculated and displayed
# MAGIC - [x] SQL queries ready for Databricks dashboard
# MAGIC
# MAGIC **Next: Week 4 — ChromaDB vector embeddings for semantic search**
