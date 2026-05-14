"""
test_api.py
-----------
Run this LOCALLY first to confirm the API is working.
No Databricks needed. Just Python + requests.

Run:
    python test_api.py
"""

import sys
import os

# Make sure Python can find the ingestion package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion.api_client import fetch_studies

print("=" * 55)
print("  Clinical Research Intelligence Platform")
print("  Week 1 — API Connection Test")
print("=" * 55)
print()
print("Connecting to ClinicalTrials.gov API...")
print("Fetching 10 cardiovascular trials...\n")

# Fetch just 10 records for a quick test
records = list(fetch_studies(
    query_term="cardiovascular",
    max_pages=1,
    page_size=10,
))

print()
print(f"✅ SUCCESS — Fetched {len(records)} records")
print()
print("─" * 55)

for i, r in enumerate(records, 1):
    print(f"\n[{i}] {r['nct_id']}")
    print(f"    Title  : {r['brief_title']}")
    print(f"    Status : {r['overall_status']}")
    print(f"    Phase  : {r['phase']}")
    print(f"    Enroll : {r['enrollment_count']}")
    print(f"    Sponsor: {r['lead_sponsor']}")

print()
print("─" * 55)
print("✅ API client is working correctly.")
print("✅ Next step: run the Databricks notebook to load Bronze table.")
print()
