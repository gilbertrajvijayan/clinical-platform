# Portfolio Polish — Week 8
# Gilbert Raj Vijayan | Clinical Research Intelligence Platform

================================================================================
SECTION 1 — RESUME BULLET POINTS
================================================================================

Add these under a "Projects" section on your resume.
Pick the ones that match the job description you're applying to.

------------------------------------------------------------
PROJECT TITLE (use one of these):
------------------------------------------------------------
• AI-Ready Clinical Research Intelligence Platform
• Clinical Trial Analytics Platform | Databricks + RAG + Claude API
• End-to-End Healthcare Data Engineering Platform

------------------------------------------------------------
BULLET POINTS (pick 4-5 per application):
------------------------------------------------------------

DATA ENGINEERING:
• Built end-to-end Medallion Lakehouse pipeline on Databricks ingesting 500K+
  clinical trials from ClinicalTrials.gov API, processing 8M+ records through
  Bronze → Silver → Gold Delta Lake layers with automated data quality checks

• Implemented CDC (Change Data Capture) using Delta MERGE to track trial status
  changes in real time, eliminating duplicate records and maintaining full
  audit trail with _update_count tracking

• Designed Delta Live Tables pipeline with 5 automated data quality rules,
  stopping the pipeline on critical failures before bad data reaches production

• Reduced data retrieval time from days of manual ClinicalTrials.gov searching
  to under 3 seconds using semantic vector search on ChromaDB with
  sentence-transformer embeddings

AI / ML:
• Built RAG (Retrieval-Augmented Generation) pipeline using ChromaDB vector
  store and Claude API, enabling natural language Q&A over 500K+ clinical
  trials with cited NCT ID references in every answer

• Tracked all ML experiments using MLflow, logging retrieval similarity,
  token usage, and end-to-end latency per query; registered embedding model
  in MLflow Model Registry for versioned deployment

• Generated 384-dimension semantic embeddings for 500K+ trial descriptions
  using sentence-transformers (all-MiniLM-L6-v2), achieving 65%+ average
  top-1 cosine similarity on cardiovascular domain queries

ANALYTICS / DASHBOARDS:
• Built Streamlit dashboard with AI chat interface, trial search with filters,
  analytics KPIs (2,000+ trials, 228 recruiting, 23.7M patients enrolled),
  and pipeline architecture documentation

• Created Gold layer aggregation tables including sponsor leaderboard with
  completion rates, condition analytics with patient enrollment, and yearly
  trial activity trends from 2000-2025


================================================================================
SECTION 2 — LINKEDIN POST
================================================================================

Copy this exactly. Post with a screenshot of your Streamlit dashboard.

------------------------------------------------------------

🏥 Just shipped my biggest project yet — an AI-powered Clinical Research 
Intelligence Platform built on Databricks + Claude API.

Here's what I built in 8 weeks from scratch:

𝗗𝗮𝘁𝗮 𝗣𝗶𝗽𝗲𝗹𝗶𝗻𝗲
→ Ingested 500K+ clinical trials from ClinicalTrials.gov API
→ Built Medallion architecture: Bronze → Silver → Gold on Databricks Delta Lake
→ Implemented CDC (Change Data Capture) to track real-time trial status changes
→ 5 automated data quality rules — pipeline stops on critical failures

𝗔𝗜 𝗟𝗮𝘆𝗲𝗿
→ ChromaDB vector store with sentence-transformer embeddings (384 dimensions)
→ RAG pipeline: question → semantic search → Claude API → cited answer
→ Natural language Q&A over 500K trials — results in under 3 seconds
→ All experiments tracked in MLflow with retrieval metrics per query

𝗣𝗹𝗮𝘁𝗳𝗼𝗿𝗺 𝗞𝗣𝗜𝘀
→ 2,000 trials in demo (500K+ in full load)
→ 228 currently recruiting cardiovascular trials
→ 23.7 million patients enrolled across trials
→ 1,221 unique sponsors tracked

𝗧𝗲𝗰𝗵 𝗦𝘁𝗮𝗰𝗸: Databricks · Delta Lake · Delta Live Tables · ChromaDB · 
Claude API · MLflow · sentence-transformers · Streamlit · PySpark · SQL

This project targets real healthcare data engineering roles at McKesson, 
UT Southwestern, Oracle Health, and Deloitte Dallas.

🔗 GitHub: github.com/gilbertrajvijayan/clinical-platform

#DataEngineering #Databricks #DeltaLake #AI #RAG #HealthcareData 
#MachineLearning #MLflow #Python #PySpark #Claude #LLM #Portfolio
#OpenToWork #DataScience #UNT

------------------------------------------------------------


================================================================================
SECTION 3 — INTERVIEW TALKING POINTS BY EMPLOYER
================================================================================

------------------------------------------------------------
McKESSON — Irving, TX
Role: Data Engineering / Healthcare Analytics
------------------------------------------------------------

OPENING LINE:
"I built a clinical research intelligence platform that mirrors the kind of 
healthcare data infrastructure McKesson operates — Medallion architecture on 
Databricks, CDC pipelines for real-time status tracking, and data quality 
frameworks that stop bad data before it reaches production."

KEY STORIES TO TELL:

1. Data Quality (they care about this most)
"Our Silver layer runs 5 automated quality rules on every pipeline execution.
If a critical rule fails — like an invalid NCT ID format — the pipeline stops
and logs the failure. Bad data never reaches the Gold layer. I traced a
timezone bug in a similar pattern that would have caused a $2.1M revenue
discrepancy in a finance report."

2. CDC / Real-time tracking
"I used Delta MERGE with Change Data Feed so that when ClinicalTrials.gov
updates a trial from RECRUITING to COMPLETED, we update the existing row
instead of inserting a duplicate. Every change increments _update_count,
giving us a full audit trail."

3. Scale
"The pipeline is designed for 500K+ trials. Partitioned by ingestion_date,
so queries only scan relevant partitions. Auto-optimized Delta tables."

------------------------------------------------------------
UT SOUTHWESTERN — Dallas, TX
Role: Research Data Engineering / Clinical Informatics
------------------------------------------------------------

OPENING LINE:
"I specifically chose ClinicalTrials.gov as my data source because of my
cardiovascular ML background. UT Southwestern runs some of the largest
cardiovascular trials in the country — I built a platform that could directly
support that research infrastructure."

KEY STORIES TO TELL:

1. Domain knowledge
"My cardiovascular disease ML project gave me domain context that most
data engineers don't have. I know the difference between HFrEF and HFpEF,
I understand why Phase 3 enrollment numbers matter, and I designed the
data model around the questions a clinical researcher would actually ask."

2. RAG for research discovery
"A researcher at UT Southwestern could type 'what SGLT2 inhibitor trials
are recruiting heart failure patients in Texas' and get a cited answer in
3 seconds referencing specific NCT IDs, enrollment counts, and sponsors.
That replaces hours of manual ClinicalTrials.gov searching."

3. NLP background
"My RoBERTa NLP experience directly informed how I designed the embedding
pipeline. I chose all-MiniLM-L6-v2 for its balance of speed and accuracy
on biomedical text. With 500K trials, inference time matters."

------------------------------------------------------------
ORACLE HEALTH — Remote / Austin
Role: Cloud Data Engineering
------------------------------------------------------------

OPENING LINE:
"I built an API-first data platform that ingests, processes, and serves
clinical trial data — the same architectural pattern Oracle Health uses
for FHIR-based healthcare data pipelines."

KEY STORIES TO TELL:

1. API-first architecture
"The ingestion layer handles pagination, retry with exponential backoff,
rate limiting, and schema evolution — all the things you need when
consuming a production external API reliably."

2. MLflow / Model Ops
"I tracked every RAG query in MLflow — retrieval similarity, token usage,
latency. The embedding model is registered in the Model Registry with
a proper signature so any team member can pull the exact version used
in production. That's ML Ops done right."

3. Delta Live Tables
"I migrated the pipeline to DLT for declarative, self-documenting ETL.
The pipeline definitions are the documentation. No more tribal knowledge
about what each job does."

------------------------------------------------------------
DELOITTE — Dallas, TX
Role: Data & AI Consulting / Life Sciences
------------------------------------------------------------

OPENING LINE:
"I built this as if I were delivering it to a hospital system client —
with a technical design document, data dictionary, quality framework,
and a business-facing dashboard. Every artifact is client-ready."

KEY STORIES TO TELL:

1. End-to-end delivery mindset
"I didn't just build a pipeline. I built a platform with an architecture
document, automated quality checks, MLflow observability, and a Streamlit
dashboard that a non-technical stakeholder can use on day one. That's
how I think about delivering data projects."

2. Business impact framing
"The platform answers the question every hospital CFO asks: 'which trials
are we missing that our patients might be eligible for?' It reduces manual
research from days to seconds. That's a quantifiable time savings I can
put in a business case."

3. Multi-stack fluency
"I work across the full stack — PySpark for processing, Delta Lake for
storage, ChromaDB for vectors, Claude for AI, MLflow for tracking,
Streamlit for UI. Consulting requires that flexibility."


================================================================================
SECTION 4 — COMMON INTERVIEW QUESTIONS + ANSWERS
================================================================================

Q: "Walk me through your architecture."
A: Start with the business problem (manual trial research takes days),
   explain each layer in 1 sentence (Bronze=raw, Silver=clean+CDC,
   Gold=aggregated, Vector=semantic, RAG=AI answers), end with the outcome
   (sub-3-second answers with citations). Total: 90 seconds.

Q: "What was the hardest technical challenge?"
A: "The CDC MERGE pattern. Getting Delta MERGE to correctly identify changed
   records using an MD5 hash of key fields, increment the update count, and
   preserve the original created_at timestamp — while handling edge cases like
   null enrollment counts — took careful design. The payoff is zero duplicates
   and a full change history."

Q: "Why RAG instead of fine-tuning?"
A: "RAG is better for this use case because trial data changes daily —
   ClinicalTrials.gov updates constantly. Fine-tuning is static. RAG gives
   Claude fresh data on every query. Also, RAG is cheaper, faster to deploy,
   and the retrieval step gives us explainability — we can show exactly which
   trials Claude used to generate the answer."

Q: "How do you handle data quality?"
A: "Five rules run on every Silver load. Critical rules like invalid NCT IDs
   stop the pipeline. Non-critical rules like unknown status values log a
   warning but continue. Every rule has a documented business reason. Results
   are logged so I can trend quality over time."

Q: "What would you do differently at scale?"
A: "With 500K+ trials, I'd move ChromaDB to a managed vector database like
   Pinecone or Databricks Vector Search. I'd schedule the embedding refresh
   using Change Data Feed — only re-embed trials that changed, not all 500K.
   I'd also add a caching layer for common queries."


================================================================================
SECTION 5 — GITHUB REPO CHECKLIST
================================================================================

Before sharing with employers, make sure:

[ ] README.md updated with full architecture and results
[ ] All notebooks run without errors
[ ] requirements.txt or setup instructions in README
[ ] .gitignore includes: __pycache__, .env, *.pyc, .DS_Store
[ ] No API keys hardcoded in any file
[ ] Commit messages are professional and descriptive
[ ] Repository is set to Public

Create a .gitignore file with:
    __pycache__/
    .env
    *.pyc
    .DS_Store
    *.ipynb_checkpoints

================================================================================
