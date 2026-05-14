"""
api_client.py
-------------
Fetches clinical trial data from ClinicalTrials.gov API v2.
- Handles pagination automatically (nextPageToken)
- Retries on failure with exponential backoff
- Returns clean Python dicts ready for Spark or local use
- No API key needed — completely free and public

Usage (local test):
    from ingestion.api_client import fetch_studies
    records = list(fetch_studies("cardiovascular", max_pages=1, page_size=10))
"""

import requests
import time
import json
import logging
from typing import Generator, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


def fetch_studies(
    query_term: str = "cardiovascular",
    page_size: int = 1000,
    max_pages: Optional[int] = None,
    delay_seconds: float = 0.5,
) -> Generator[dict, None, None]:
    """
    Generator — yields one study dict per call.
    Automatically handles all pagination.

    Args:
        query_term:     Search keyword for ClinicalTrials.gov
        page_size:      Records per page, max 1000
        max_pages:      Stop after N pages (None = fetch everything)
        delay_seconds:  Pause between pages (respect the free API)

    Yields:
        dict with flattened study fields + ingestion metadata
    """
    from datetime import datetime, timezone

    params = {
        "query.term": query_term,
        "pageSize":   page_size,
        "format":     "json",
    }

    page_num      = 0
    total_fetched = 0
    next_token    = None

    while True:

        # Stop if we've hit max_pages
        if max_pages and page_num >= max_pages:
            log.info(f"Reached max_pages={max_pages}. Stopping.")
            break

        # Add pagination token after page 1
        if next_token:
            params["pageToken"] = next_token
        elif "pageToken" in params:
            del params["pageToken"]

        # Fetch page
        response = _get_with_retry(BASE_URL, params)
        if response is None:
            log.error("Failed after all retries. Stopping.")
            break

        data    = response.json()
        studies = data.get("studies", [])

        if not studies:
            log.info("No more studies returned. Done.")
            break

        ingestion_ts = datetime.now(timezone.utc).isoformat()

        for study in studies:
            proto  = study.get("protocolSection", {})
            id_mod = proto.get("identificationModule", {})
            st_mod = proto.get("statusModule", {})
            de_mod = proto.get("designModule", {})
            ds_mod = proto.get("descriptionModule", {})
            sp_mod = proto.get("sponsorCollaboratorsModule", {})
            co_mod = proto.get("conditionsModule", {})

            yield {
                # Core identifiers
                "nct_id":       id_mod.get("nctId"),
                "brief_title":  id_mod.get("briefTitle"),

                # Status & phase
                "overall_status": st_mod.get("overallStatus"),
                "phase":          _list_to_str(de_mod.get("phases")),

                # Dates
                "start_date":           _safe_date(st_mod, "startDateStruct"),
                "completion_date":      _safe_date(st_mod, "completionDateStruct"),
                "last_update_date":     _safe_date(st_mod, "lastUpdatePostDateStruct"),

                # Design
                "study_type":       de_mod.get("studyType"),
                "enrollment_count": de_mod.get("enrollmentInfo", {}).get("count"),

                # Content
                "conditions":    _list_to_str(co_mod.get("conditions")),
                "brief_summary": ds_mod.get("briefSummary"),
                "lead_sponsor":  sp_mod.get("leadSponsor", {}).get("name"),

                # Raw JSON preserved for bronze completeness
                "raw_json": json.dumps(study),

                # Ingestion metadata
                "ingestion_ts": ingestion_ts,
                "query_term":   query_term,
                "source":       "clinicaltrials_gov_v2",
            }
            total_fetched += 1

        page_num += 1
        log.info(f"Page {page_num} done | +{len(studies)} studies | total: {total_fetched}")

        next_token = data.get("nextPageToken")
        if not next_token:
            log.info(f"All pages fetched. Grand total: {total_fetched}")
            break

        time.sleep(delay_seconds)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_with_retry(
    url: str,
    params: dict,
    max_retries: int = 5
) -> Optional[requests.Response]:
    """GET with exponential backoff. Returns None after all retries fail."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)

            if resp.status_code == 200:
                return resp

            elif resp.status_code == 429:
                wait = 2 ** attempt * 5
                log.warning(f"Rate limited (429). Waiting {wait}s... (attempt {attempt+1})")
                time.sleep(wait)

            elif resp.status_code >= 500:
                wait = 2 ** attempt
                log.warning(f"Server error {resp.status_code}. Waiting {wait}s... (attempt {attempt+1})")
                time.sleep(wait)

            else:
                log.error(f"Unexpected status {resp.status_code}: {resp.text[:300]}")
                return None

        except requests.RequestException as e:
            wait = 2 ** attempt
            log.warning(f"Request failed: {e}. Waiting {wait}s... (attempt {attempt+1})")
            time.sleep(wait)

    return None


def _list_to_str(val) -> Optional[str]:
    """Convert list → comma-separated string. Handles None gracefully."""
    if val is None:
        return None
    if isinstance(val, list):
        return ", ".join(str(v) for v in val)
    return str(val)


def _safe_date(module: dict, key: str) -> Optional[str]:
    """Safely extract date string from a date struct."""
    return module.get(key, {}).get("date")
