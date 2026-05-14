"""
quality/checks.py
-----------------
Data quality checks for the Silver layer.
Runs on every pipeline execution and logs results.

A 4-year engineer writes quality checks because:
- Bad data causes bad business decisions
- Silent failures are worse than loud failures
- Every rule here has a real business reason behind it
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from datetime import datetime
from typing import List, Tuple


# ── Rules: (name, sql_expression, is_critical, business_reason) ──────────────
# is_critical=True → pipeline stops if this fails

SILVER_RULES = [
    (
        "nct_id_not_null",
        "nct_id IS NOT NULL",
        True,
        "Every trial must have an NCT ID — it is the unique identifier"
    ),
    (
        "nct_id_format_valid",
        "nct_id LIKE 'NCT%'",
        True,
        "NCT IDs always start with NCT — anything else is corrupt data"
    ),
    (
        "brief_title_not_null",
        "brief_title IS NOT NULL AND LENGTH(brief_title) > 5",
        True,
        "A trial without a title cannot be displayed or searched"
    ),
    (
        "status_is_known",
        """overall_status IN (
            'RECRUITING', 'COMPLETED', 'TERMINATED', 'SUSPENDED',
            'WITHDRAWN', 'NOT_YET_RECRUITING', 'ACTIVE_NOT_RECRUITING',
            'ENROLLING_BY_INVITATION', 'UNKNOWN'
        )""",
        False,
        "Unknown statuses warn us the API added a new value we have not handled"
    ),
    (
        "enrollment_non_negative",
        "enrollment_count IS NULL OR enrollment_count >= 0",
        False,
        "Negative enrollment is impossible — indicates a data type issue"
    ),
    (
        "completion_after_start",
        """start_date IS NULL OR completion_date IS NULL
           OR completion_date >= start_date""",
        False,
        "A trial cannot complete before it starts — catches date parsing bugs"
    ),
]


def run_quality_checks(
    df: DataFrame,
    table_name: str = "silver_clinical_trials",
) -> Tuple[bool, List[dict]]:
    """
    Run all quality rules against a DataFrame.
    Returns (all_critical_passed, list_of_results).

    Usage:
        passed, results = run_quality_checks(silver_df)
        if not passed:
            raise ValueError("Critical quality checks failed")
    """
    results = []
    total_rows = df.count()
    all_critical_passed = True

    print(f"\n{'='*55}")
    print(f"  Data Quality Report: {table_name}")
    print(f"  Rows checked : {total_rows:,}")
    print(f"  Run at       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}\n")

    for rule_name, expression, is_critical, reason in SILVER_RULES:
        passing = df.filter(expression).count()
        failing = total_rows - passing
        rate    = (passing / total_rows * 100) if total_rows > 0 else 0
        passed  = failing == 0

        result = {
            "rule":         rule_name,
            "passed":       passed,
            "is_critical":  is_critical,
            "passing_rows": passing,
            "failing_rows": failing,
            "pass_rate":    round(rate, 2),
            "reason":       reason,
        }
        results.append(result)

        icon = "OK " if passed else ("CRITICAL FAIL" if is_critical else "WARNING")
        print(f"[{icon}] {rule_name}")
        print(f"         {passing:,}/{total_rows:,} rows passed ({rate:.1f}%)")
        print(f"         {reason}\n")

        if is_critical and not passed:
            all_critical_passed = False

    passed_count = sum(1 for r in results if r["passed"])
    print(f"{'='*55}")
    print(f"  Result : {passed_count}/{len(results)} rules passed")
    print(f"  Status : {'PASSED' if all_critical_passed else 'FAILED - pipeline stopped'}")
    print(f"{'='*55}\n")

    return all_critical_passed, results


def check_duplicates(df: DataFrame) -> int:
    """Return count of duplicate NCT IDs. Should always be 0."""
    total    = df.count()
    distinct = df.select("nct_id").distinct().count()
    dupes    = total - distinct
    if dupes > 0:
        print(f"WARNING: {dupes} duplicate NCT IDs found")
    else:
        print(f"OK: No duplicate NCT IDs")
    return dupes
