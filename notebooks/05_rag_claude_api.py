# Databricks notebook source
# Title: 05_rag_claude_api
# Week 5 — RAG Pipeline + Claude API
#
# What this builds:
# 1. Retrieval-Augmented Generation (RAG) pipeline
# 2. User asks a question in plain English
# 3. System retrieves top 10 most relevant trials from ChromaDB
# 4. Claude API synthesizes a smart, cited answer
# 5. Full conversational clinical research assistant

# COMMAND ----------
# MAGIC %md
# MAGIC # Week 5 — RAG Pipeline + Claude API
# MAGIC
# MAGIC **What RAG means:**
# MAGIC Retrieval-Augmented Generation — instead of asking Claude to answer
# MAGIC from its training data alone, we first RETRIEVE relevant trials
# MAGIC from our database, then give them to Claude as context.
# MAGIC
# MAGIC **Why this matters:**
# MAGIC - Claude trained on data up to a cutoff date — trials change daily
# MAGIC - Our ChromaDB has current, real trial data
# MAGIC - RAG = Claude's intelligence + our fresh data = better answers
# MAGIC
# MAGIC **The flow:**
# MAGIC User question → embed → ChromaDB search → top 10 trials → Claude → cited answer

# COMMAND ----------
# MAGIC %md ## Step 1 — Install dependencies

# COMMAND ----------

%pip install sentence-transformers chromadb anthropic --quiet
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md ## Step 2 — Setup after restart

# COMMAND ----------

import sys
sys.path.insert(0, "/Workspace/Users/gilbertrajvijayan@gmail.com/clinical-platform")

import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
from config import CHROMA_COLLECTION, EMBEDDING_MODEL, CLAUDE_MODEL, RAG_TOP_K
import os

print(f"Claude model    : {CLAUDE_MODEL}")
print(f"Embedding model : {EMBEDDING_MODEL}")
print(f"Top K results   : {RAG_TOP_K}")

# COMMAND ----------
# MAGIC %md ## Step 3 — Set your Claude API key
# MAGIC
# MAGIC Get your free API key at: https://console.anthropic.com

# COMMAND ----------

# Set your Claude API key here
# Get it free at: https://console.anthropic.com → API Keys → Create Key
ANTHROPIC_API_KEY = "YOUR_API_KEY_HERE"   # ← replace this

os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

# Test the key
try:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    print("Claude API client created successfully")
except Exception as e:
    print(f"Error: {e}")
    print("Make sure you replaced YOUR_API_KEY_HERE with your actual key")

# COMMAND ----------
# MAGIC %md ## Step 4 — Reload ChromaDB and embedding model

# COMMAND ----------

# Reload the embedding model
print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)
print(f"Model loaded — dimension: {model.get_sentence_embedding_dimension()}")

# Reconnect to ChromaDB
chroma_path = "/tmp/clinical_chromadb"
chroma_client = chromadb.PersistentClient(path=chroma_path)

try:
    collection = chroma_client.get_collection(CHROMA_COLLECTION)
    print(f"ChromaDB collection loaded: {collection.count():,} trials")
except Exception as e:
    print(f"ChromaDB error: {e}")
    print("Make sure you ran Week 4 notebook first to populate ChromaDB")

# COMMAND ----------
# MAGIC %md ## Step 5 — Build the RAG retrieval function

# COMMAND ----------

def retrieve_trials(query: str, n_results: int = RAG_TOP_K) -> list:
    """
    Step 1 of RAG: RETRIEVE
    Embed the query, search ChromaDB, return top N trials.
    """
    query_embedding = model.encode([query], convert_to_numpy=True)

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results,
        include=["metadatas", "distances", "documents"],
    )

    trials = []
    for meta, dist, doc in zip(
        results["metadatas"][0],
        results["distances"][0],
        results["documents"][0],
    ):
        trials.append({
            "nct_id":           meta.get("nct_id", ""),
            "brief_title":      meta.get("brief_title", ""),
            "overall_status":   meta.get("overall_status", ""),
            "phase":            meta.get("phase", ""),
            "conditions":       meta.get("conditions", ""),
            "lead_sponsor":     meta.get("lead_sponsor", ""),
            "enrollment_count": meta.get("enrollment_count", 0),
            "is_recruiting":    meta.get("is_recruiting", False),
            "similarity":       round((1 - dist) * 100, 1),
            "document":         doc[:500],  # first 500 chars of text
        })

    return trials


def format_trials_for_claude(trials: list) -> str:
    """
    Format retrieved trials as structured context for Claude.
    Clear formatting helps Claude give better, more cited answers.
    """
    context_parts = []
    for i, trial in enumerate(trials, 1):
        recruiting = "YES - Currently Recruiting" if trial["is_recruiting"] else "No"
        context_parts.append(f"""
TRIAL {i} (Relevance: {trial['similarity']}%)
NCT ID     : {trial['nct_id']}
Title      : {trial['brief_title']}
Status     : {trial['overall_status']}
Recruiting : {recruiting}
Phase      : {trial['phase']}
Conditions : {trial['conditions']}
Sponsor    : {trial['lead_sponsor']}
Enrollment : {trial['enrollment_count']:,} patients
Summary    : {trial['document'][:400]}
""")
    return "\n---\n".join(context_parts)

# COMMAND ----------
# MAGIC %md ## Step 6 — Build the Claude RAG function

# COMMAND ----------

SYSTEM_PROMPT = """You are a clinical research intelligence assistant for a healthcare analytics platform.

You have access to a curated database of cardiovascular clinical trials from ClinicalTrials.gov.

Your job:
1. Answer the researcher's question accurately using the trial data provided
2. Always cite specific NCT IDs when referencing trials (e.g., NCT04334447)
3. Highlight trials that are currently RECRUITING when relevant
4. Be concise but thorough — researchers are busy
5. If the data doesn't contain enough information to answer fully, say so honestly
6. Never make up trial details — only use what is in the provided context

Format your response clearly with:
- A direct answer to the question
- Specific trial citations with NCT IDs
- Key insights from the data
- Recruiting opportunities if relevant"""


def ask_clinical_rag(question: str, verbose: bool = True) -> str:
    """
    Full RAG pipeline:
    1. Retrieve relevant trials from ChromaDB
    2. Format them as context
    3. Send to Claude API with system prompt
    4. Return Claude's cited answer

    Args:
        question: Natural language question about clinical trials
        verbose:  If True, show retrieved trials before Claude's answer

    Returns:
        Claude's answer as a string
    """
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")

    # Step 1: RETRIEVE
    print("Retrieving relevant trials from ChromaDB...")
    trials = retrieve_trials(question)
    print(f"Found {len(trials)} relevant trials\n")

    if verbose:
        print("Top 3 retrieved trials:")
        for t in trials[:3]:
            print(f"  [{t['similarity']}%] {t['nct_id']}: {t['brief_title'][:60]}")
        print()

    # Step 2: AUGMENT — format trials as context
    context = format_trials_for_claude(trials)

    # Step 3: GENERATE — call Claude API
    print("Sending to Claude API...")

    user_message = f"""Please answer this clinical research question using the trial data below.

QUESTION: {question}

RELEVANT TRIALS FROM OUR DATABASE:
{context}

Please provide a clear, cited answer."""

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_message}
        ],
    )

    answer = response.content[0].text

    print("\nCLAUDE'S ANSWER:")
    print("-" * 60)
    print(answer)
    print("-" * 60)
    print(f"\nTokens used: {response.usage.input_tokens} in / {response.usage.output_tokens} out")

    return answer

# COMMAND ----------
# MAGIC %md ## Step 7 — Run your first RAG query

# COMMAND ----------

# Query 1 — What a cardiologist would actually ask
answer1 = ask_clinical_rag(
    "What cardiovascular trials are currently recruiting patients "
    "and what treatments are they testing?"
)

# COMMAND ----------
# MAGIC %md ## Step 8 — Query about specific condition

# COMMAND ----------

answer2 = ask_clinical_rag(
    "What does the clinical evidence show about heart failure "
    "treatment in Phase 3 trials? Which sponsors are leading this research?"
)

# COMMAND ----------
# MAGIC %md ## Step 9 — Query a business stakeholder would ask

# COMMAND ----------

answer3 = ask_clinical_rag(
    "Which research institutions are running the most cardiovascular trials "
    "and what is their focus area?"
)

# COMMAND ----------
# MAGIC %md ## Step 10 — Multi-turn conversation
# MAGIC
# MAGIC This shows the system can handle follow-up questions

# COMMAND ----------

def rag_conversation(questions: list) -> None:
    """
    Run multiple questions and show how the system
    handles different types of clinical queries.
    """
    print("CLINICAL RESEARCH ASSISTANT — CONVERSATION DEMO")
    print("=" * 60)

    for i, question in enumerate(questions, 1):
        print(f"\n[Question {i}/{len(questions)}]")
        ask_clinical_rag(question, verbose=False)
        print()


conversation_questions = [
    "Are there any Phase 4 heart failure trials with large enrollment?",
    "Which trials focus on elderly patients with cardiovascular disease?",
    "What is the average enrollment size in cardiovascular Phase 3 trials?",
]

rag_conversation(conversation_questions)

# COMMAND ----------
# MAGIC %md ## Step 11 — Evaluate RAG quality

# COMMAND ----------

# Basic RAG evaluation metrics
# In Week 6 we track these in MLflow properly

print("RAG Pipeline Quality Metrics")
print("=" * 40)

# Test 5 queries and measure retrieval quality
test_queries = [
    "heart failure medication",
    "atrial fibrillation treatment",
    "hypertension clinical trial",
    "coronary artery disease intervention",
    "cardiovascular risk reduction elderly",
]

avg_top1_similarity  = 0
avg_recruiting_found = 0

for query in test_queries:
    trials = retrieve_trials(query, n_results=5)
    top1_sim   = trials[0]["similarity"] if trials else 0
    recruiting = sum(1 for t in trials if t["is_recruiting"])

    avg_top1_similarity  += top1_sim
    avg_recruiting_found += recruiting

n = len(test_queries)
print(f"Queries tested          : {n}")
print(f"Avg top-1 similarity    : {avg_top1_similarity/n:.1f}%")
print(f"Avg recruiting trials   : {avg_recruiting_found/n:.1f} per query")
print(f"ChromaDB collection size: {collection.count():,} trials")
print()
print("These metrics get tracked in MLflow in Week 6")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Week 5 Complete
# MAGIC
# MAGIC - [x] Claude API connected and tested
# MAGIC - [x] RAG retrieval function built (ChromaDB → top 10 trials)
# MAGIC - [x] Context formatting for Claude (structured, cited)
# MAGIC - [x] Full RAG pipeline: question → retrieve → augment → generate
# MAGIC - [x] Multi-turn conversation demonstrated
# MAGIC - [x] Basic RAG quality metrics calculated
# MAGIC
# MAGIC **Next: Week 6 — MLflow experiment tracking**
# MAGIC Track every RAG query, log quality metrics, register the embedding model
