# Databricks notebook source
# Title: 04_vector_embeddings_chromadb
# Week 4 — Vector Embeddings + ChromaDB
#
# What this builds:
# 1. Install sentence-transformers and chromadb
# 2. Convert trial descriptions into vector embeddings
# 3. Store embeddings in ChromaDB
# 4. Test semantic search — find similar trials by meaning not keywords

# COMMAND ----------
# MAGIC %md
# MAGIC # Week 4 — Vector Embeddings + ChromaDB
# MAGIC
# MAGIC **The problem this solves:**
# MAGIC A user types: "heart failure drugs with reduced side effects"
# MAGIC The old way: keyword search — misses trials that say "cardiac dysfunction" or "HFrEF"
# MAGIC The new way: vector search — understands MEANING, finds semantically similar trials
# MAGIC
# MAGIC **How it works:**
# MAGIC 1. Take each trial's title + summary + conditions
# MAGIC 2. Convert text into a 384-number vector (embedding) using a pre-trained model
# MAGIC 3. Store all vectors in ChromaDB
# MAGIC 4. At query time: embed the question, find the closest trial vectors
# MAGIC 5. Return the most semantically similar trials

# COMMAND ----------
# MAGIC %md ## Step 1 — Install dependencies

# COMMAND ----------

# Install required packages
# This takes 2-3 minutes on first run
%pip install sentence-transformers chromadb --quiet
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md ## Step 2 — Setup after restart

# COMMAND ----------

import sys
sys.path.insert(0, "/Workspace/Users/gilbertrajvijayan@gmail.com/clinical-platform")

from config import SILVER_TABLE, VECTOR_STORE_PATH, CHROMA_COLLECTION, EMBEDDING_MODEL
from pyspark.sql import functions as F
import pandas as pd

print(f"Embedding model : {EMBEDDING_MODEL}")
print(f"Collection name : {CHROMA_COLLECTION}")
print(f"Vector store    : {VECTOR_STORE_PATH}")

# COMMAND ----------
# MAGIC %md ## Step 3 — Load Silver data for embedding

# COMMAND ----------

silver = spark.table(SILVER_TABLE)

# Select only the fields we embed
# We combine title + conditions + summary into one text block per trial
embedding_df = (
    silver
    .where(F.col("brief_summary").isNotNull())
    .where(F.col("nct_id").isNotNull())
    .select(
        "nct_id",
        "brief_title",
        "overall_status",
        "phase",
        "conditions",
        "lead_sponsor",
        "enrollment_count",
        "brief_summary",
        "is_recruiting",
    )
    .withColumn(
        "text_to_embed",
        F.concat_ws(" | ",
            F.coalesce("brief_title",    F.lit("")),
            F.coalesce("conditions",     F.lit("")),
            F.coalesce("phase",          F.lit("")),
            F.coalesce("overall_status", F.lit("")),
            # Use first 500 chars of summary to keep it focused
            F.substring(F.coalesce("brief_summary", F.lit("")), 1, 500),
        )
    )
)

total = embedding_df.count()
print(f"Trials to embed: {total:,}")
embedding_df.select("nct_id", "text_to_embed").show(3, truncate=80)

# COMMAND ----------
# MAGIC %md ## Step 4 — Convert to Pandas for embedding
# MAGIC
# MAGIC sentence-transformers works on CPU with Pandas — no GPU needed

# COMMAND ----------

# Convert to Pandas — we have 2,000 trials, easily fits in memory
pdf = embedding_df.toPandas()
print(f"Loaded {len(pdf):,} trials into Pandas")
print(f"Sample text:\n{pdf['text_to_embed'].iloc[0][:300]}")

# COMMAND ----------
# MAGIC %md ## Step 5 — Load the embedding model

# COMMAND ----------

from sentence_transformers import SentenceTransformer
import time

print(f"Loading model: {EMBEDDING_MODEL}")
print("This takes 30-60 seconds on first run (downloading model weights)...")

start = time.time()
model = SentenceTransformer(EMBEDDING_MODEL)
elapsed = round(time.time() - start, 1)

print(f"Model loaded in {elapsed}s")
print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
# Expected: 384 dimensions for all-MiniLM-L6-v2

# COMMAND ----------
# MAGIC %md ## Step 6 — Generate embeddings
# MAGIC
# MAGIC This is the core step — converts text into 384-number vectors

# COMMAND ----------

print(f"Generating embeddings for {len(pdf):,} trials...")
print("Estimated time: 1-3 minutes for 2,000 trials...")

start = time.time()

embeddings = model.encode(
    pdf["text_to_embed"].tolist(),
    batch_size=64,          # process 64 at a time
    show_progress_bar=True,
    convert_to_numpy=True,
)

elapsed = round(time.time() - start, 1)
print(f"\nDone in {elapsed}s")
print(f"Embeddings shape: {embeddings.shape}")
# Expected: (2000, 384) — 2000 trials, 384 dimensions each

# COMMAND ----------
# MAGIC %md ## Step 7 — Store in ChromaDB

# COMMAND ----------

import chromadb
import os

# Use persistent storage so embeddings survive notebook restarts
chroma_path = "/tmp/clinical_chromadb"
os.makedirs(chroma_path, exist_ok=True)

# Create ChromaDB client with persistent storage
client = chromadb.PersistentClient(path=chroma_path)

# Delete collection if it exists (fresh start)
try:
    client.delete_collection(CHROMA_COLLECTION)
    print(f"Deleted existing collection: {CHROMA_COLLECTION}")
except:
    pass

# Create new collection
collection = client.create_collection(
    name=CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"}  # cosine similarity for text
)

print(f"Created collection: {CHROMA_COLLECTION}")

# COMMAND ----------
# MAGIC %md ## Step 8 — Add embeddings to ChromaDB in batches

# COMMAND ----------

BATCH_SIZE = 500
total_added = 0

for i in range(0, len(pdf), BATCH_SIZE):
    batch_pdf        = pdf.iloc[i:i+BATCH_SIZE]
    batch_embeddings = embeddings[i:i+BATCH_SIZE]

    collection.add(
        ids        = batch_pdf["nct_id"].tolist(),
        embeddings = batch_embeddings.tolist(),
        metadatas  = [
            {
                "nct_id":           row["nct_id"] or "",
                "brief_title":      (row["brief_title"] or "")[:200],
                "overall_status":   row["overall_status"] or "",
                "phase":            row["phase"] or "",
                "conditions":       (row["conditions"] or "")[:200],
                "lead_sponsor":     (row["lead_sponsor"] or "")[:100],
                "enrollment_count": int(row["enrollment_count"])
                                    if pd.notna(row["enrollment_count"]) else 0,
                "is_recruiting":    bool(row["is_recruiting"]),
            }
            for _, row in batch_pdf.iterrows()
        ],
        documents = batch_pdf["text_to_embed"].tolist(),
    )

    total_added += len(batch_pdf)
    print(f"Added batch {i//BATCH_SIZE + 1}: {total_added:,}/{len(pdf):,} trials")

print(f"\nAll {total_added:,} trials stored in ChromaDB")
print(f"Collection count: {collection.count()}")

# COMMAND ----------
# MAGIC %md ## Step 9 — Test semantic search
# MAGIC
# MAGIC This is the exciting part — ask questions in plain English

# COMMAND ----------

def search_trials(query: str, n_results: int = 5) -> None:
    """
    Search clinical trials by semantic meaning.
    Returns the most relevant trials even if they
    don't contain the exact words in your query.
    """
    print(f"\nQuery: '{query}'")
    print("=" * 60)

    # Embed the query using the same model
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Search ChromaDB for similar vectors
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results,
        include=["metadatas", "distances", "documents"],
    )

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (meta, dist) in enumerate(zip(metadatas, distances), 1):
        similarity = round((1 - dist) * 100, 1)
        print(f"\n[{rank}] {meta['nct_id']} — {similarity}% match")
        print(f"     Title   : {meta['brief_title'][:70]}")
        print(f"     Status  : {meta['overall_status']}")
        print(f"     Phase   : {meta['phase']}")
        print(f"     Enroll  : {meta['enrollment_count']:,}")
        print(f"     Sponsor : {meta['lead_sponsor'][:50]}")

# COMMAND ----------

# Test 1 — Natural language clinical question
search_trials("heart failure treatment with reduced hospitalization")

# COMMAND ----------

# Test 2 — Recruiting trials for a specific condition
search_trials("currently recruiting atrial fibrillation patients")

# COMMAND ----------

# Test 3 — Specific drug class
search_trials("SGLT2 inhibitor cardiovascular outcomes")

# COMMAND ----------

# Test 4 — Patient safety focus
search_trials("elderly patients cardiovascular risk reduction")

# COMMAND ----------
# MAGIC %md ## Step 10 — Compare keyword vs semantic search

# COMMAND ----------

# This demonstrates WHY vector search is better than keyword search

query = "cardiac dysfunction in diabetic patients"

print("KEYWORD SEARCH (old way):")
print("Would only find trials containing exact words: 'cardiac', 'dysfunction', 'diabetic'")
print()

keyword_results = (
    silver
    .filter(
        F.lower("brief_summary").contains("cardiac") |
        F.lower("brief_summary").contains("diabetes")
    )
    .select("nct_id", "brief_title", "overall_status")
    .limit(3)
    .collect()
)
for r in keyword_results:
    print(f"  {r['nct_id']}: {r['brief_title'][:60]}")

print()
print("SEMANTIC SEARCH (our way):")
print("Finds trials about the same CONCEPT even with different words")
search_trials(query, n_results=3)

# COMMAND ----------
# MAGIC %md ## Step 11 — Save ChromaDB path for Week 5

# COMMAND ----------

print("ChromaDB is ready for Week 5 (RAG + Claude API)")
print(f"Path      : {chroma_path}")
print(f"Collection: {CHROMA_COLLECTION}")
print(f"Vectors   : {collection.count():,}")
print()
print("In Week 5 we connect this to Claude API:")
print("  User question → embed → ChromaDB search → top 10 trials → Claude synthesizes answer")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Week 4 Complete
# MAGIC
# MAGIC - [x] sentence-transformers model loaded (all-MiniLM-L6-v2)
# MAGIC - [x] 2,000 trial descriptions converted to 384-dimension vectors
# MAGIC - [x] All vectors stored in ChromaDB with metadata
# MAGIC - [x] Semantic search working — finds trials by meaning not keywords
# MAGIC - [x] Demonstrated semantic vs keyword search advantage
# MAGIC
# MAGIC **Next: Week 5 — RAG pipeline + Claude API**
# MAGIC The AI layer that turns this into a conversational research assistant
