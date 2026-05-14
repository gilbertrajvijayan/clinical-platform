# Databricks notebook source
# Title: 06_mlflow_tracking
# Week 6 — MLflow Experiment Tracking
#
# What this builds:
# 1. MLflow experiment to track every RAG query
# 2. Log retrieval quality metrics per run
# 3. Register the embedding model in MLflow Model Registry
# 4. Compare runs to find best configuration
# 5. Production-ready ML observability

# COMMAND ----------
# MAGIC %md
# MAGIC # Week 6 — MLflow Experiment Tracking
# MAGIC
# MAGIC **Why MLflow matters:**
# MAGIC Without tracking, you have no idea:
# MAGIC - Which embedding model gives better retrieval?
# MAGIC - Does TOP_K=5 or TOP_K=10 give better answers?
# MAGIC - Is retrieval quality improving or degrading over time?
# MAGIC - When did a model change break something?
# MAGIC
# MAGIC MLflow answers all of these. Every run is logged,
# MAGIC compared, and reproducible. This is what separates
# MAGIC a data scientist from a machine learning engineer.

# COMMAND ----------
# MAGIC %md ## Step 1 — Install dependencies

# COMMAND ----------

%pip install sentence-transformers chromadb anthropic mlflow --quiet
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md ## Step 2 — Setup

# COMMAND ----------

import sys
sys.path.insert(0, "/Workspace/Users/gilbertrajvijayan@gmail.com/clinical-platform")

import mlflow
import mlflow.pyfunc
import anthropic
import chromadb
import os
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pyspark.sql import functions as F
from config import (
    SILVER_TABLE, CHROMA_COLLECTION, EMBEDDING_MODEL,
    CLAUDE_MODEL, RAG_TOP_K, MLFLOW_EXPERIMENT_NAME
)

# Set your API key
ANTHROPIC_API_KEY = "YOUR_API_KEY_HERE"  # ← paste your key
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

print("Setup complete")

# COMMAND ----------
# MAGIC %md ## Step 3 — Set up MLflow experiment

# COMMAND ----------

# Create or get the experiment
experiment_name = "/Users/gilbertrajvijayan@gmail.com/clinical_platform_rag"

mlflow.set_experiment(experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)

print(f"Experiment name : {experiment_name}")
print(f"Experiment ID   : {experiment.experiment_id}")
print(f"Artifact store  : {experiment.artifact_location}")

# COMMAND ----------
# MAGIC %md ## Step 4 — Rebuild ChromaDB and load model

# COMMAND ----------

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)
print(f"Model ready — dim: {model.get_sentence_embedding_dimension()}")

# Rebuild ChromaDB from Silver
print("Rebuilding ChromaDB...")
chroma_path = "/tmp/clinical_chromadb"
os.makedirs(chroma_path, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=chroma_path)

try:
    chroma_client.delete_collection(CHROMA_COLLECTION)
except:
    pass

collection = chroma_client.create_collection(
    name=CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

silver = spark.table(SILVER_TABLE)
pdf = (
    silver
    .where(F.col("brief_summary").isNotNull())
    .select(
        "nct_id", "brief_title", "overall_status", "phase",
        "conditions", "lead_sponsor", "enrollment_count",
        "brief_summary", "is_recruiting",
    )
    .withColumn("text_to_embed",
        F.concat_ws(" | ",
            F.coalesce("brief_title",    F.lit("")),
            F.coalesce("conditions",     F.lit("")),
            F.coalesce("phase",          F.lit("")),
            F.coalesce("overall_status", F.lit("")),
            F.substring(F.coalesce("brief_summary", F.lit("")), 1, 500),
        )
    )
    .toPandas()
)

embeddings = model.encode(
    pdf["text_to_embed"].tolist(),
    batch_size=64, show_progress_bar=True, convert_to_numpy=True
)

BATCH_SIZE = 500
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
        documents=batch_pdf["text_to_embed"].tolist(),
    )

print(f"ChromaDB ready: {collection.count():,} trials")

# COMMAND ----------
# MAGIC %md ## Step 5 — Define tracked RAG function
# MAGIC
# MAGIC Every RAG query now logs metrics to MLflow automatically

# COMMAND ----------

SYSTEM_PROMPT = """You are a clinical research intelligence assistant.
Answer questions using the trial data provided.
Always cite NCT IDs. Be concise and accurate.
If data is insufficient, say so honestly."""


def tracked_rag_query(
    question: str,
    top_k: int = RAG_TOP_K,
    run_name: str = None,
) -> dict:
    """
    RAG pipeline with full MLflow tracking.

    Logs:
    - Parameters: model names, top_k, question length
    - Metrics: retrieval similarity, latency, token usage
    - Artifacts: retrieved trials, Claude's answer
    - Tags: query category, recruiting trials found

    Returns:
        dict with answer and all metrics
    """
    run_name = run_name or f"rag_{int(time.time())}"

    with mlflow.start_run(run_name=run_name):

        # ── Log parameters ────────────────────────────────────────────
        mlflow.log_params({
            "embedding_model": EMBEDDING_MODEL,
            "claude_model":    CLAUDE_MODEL,
            "top_k":           top_k,
            "question_length": len(question),
            "collection_size": collection.count(),
        })

        # ── Step 1: Retrieve ──────────────────────────────────────────
        retrieve_start = time.time()
        query_embedding = model.encode([question], convert_to_numpy=True)
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=["metadatas", "distances", "documents"],
        )
        retrieve_time = time.time() - retrieve_start

        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        similarities = [(1 - d) * 100 for d in distances]

        # ── Log retrieval metrics ─────────────────────────────────────
        recruiting_count = sum(
            1 for m in metadatas if m.get("is_recruiting", False)
        )
        mlflow.log_metrics({
            "retrieval_time_seconds":  round(retrieve_time, 3),
            "top1_similarity":         round(similarities[0], 2),
            "avg_similarity":          round(np.mean(similarities), 2),
            "min_similarity":          round(min(similarities), 2),
            "recruiting_trials_found": recruiting_count,
            "trials_retrieved":        len(metadatas),
        })

        # ── Step 2: Format context ────────────────────────────────────
        context_parts = []
        for i, (meta, sim) in enumerate(zip(metadatas, similarities), 1):
            context_parts.append(
                f"TRIAL {i} ({sim:.1f}% match)\n"
                f"NCT ID: {meta['nct_id']}\n"
                f"Title: {meta['brief_title']}\n"
                f"Status: {meta['overall_status']}\n"
                f"Phase: {meta['phase']}\n"
                f"Enrollment: {meta['enrollment_count']:,}\n"
                f"Sponsor: {meta['lead_sponsor']}\n"
            )
        context = "\n---\n".join(context_parts)

        # ── Step 3: Generate with Claude ──────────────────────────────
        generate_start = time.time()
        response = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Question: {question}\n\nTrials:\n{context}"
            }],
        )
        generate_time = time.time() - generate_start
        answer = response.content[0].text

        # ── Log generation metrics ────────────────────────────────────
        mlflow.log_metrics({
            "generation_time_seconds": round(generate_time, 3),
            "total_time_seconds":      round(retrieve_time + generate_time, 3),
            "input_tokens":            response.usage.input_tokens,
            "output_tokens":           response.usage.output_tokens,
            "answer_length":           len(answer),
        })

        # ── Log tags ──────────────────────────────────────────────────
        mlflow.set_tags({
            "query":             question[:100],
            "recruiting_found":  str(recruiting_count > 0),
            "high_confidence":   str(similarities[0] > 60),
        })

        # ── Log artifacts ─────────────────────────────────────────────
        # Save the full answer as a text artifact
        answer_path = f"/tmp/answer_{run_name}.txt"
        with open(answer_path, "w") as f:
            f.write(f"QUESTION:\n{question}\n\n")
            f.write(f"RETRIEVED TRIALS:\n{context}\n\n")
            f.write(f"CLAUDE'S ANSWER:\n{answer}")
        mlflow.log_artifact(answer_path, "rag_outputs")

        # Print results
        print(f"\nQuery: {question[:70]}...")
        print(f"Top-1 similarity : {similarities[0]:.1f}%")
        print(f"Retrieve time    : {retrieve_time:.2f}s")
        print(f"Generate time    : {generate_time:.2f}s")
        print(f"Tokens used      : {response.usage.input_tokens} in / {response.usage.output_tokens} out")
        print(f"\nAnswer preview   : {answer[:200]}...")
        print(f"MLflow run       : {run_name}")

        return {
            "question":    question,
            "answer":      answer,
            "top1_sim":    similarities[0],
            "retrieve_ms": round(retrieve_time * 1000),
            "generate_ms": round(generate_time * 1000),
            "tokens_in":   response.usage.input_tokens,
            "tokens_out":  response.usage.output_tokens,
        }

# COMMAND ----------
# MAGIC %md ## Step 6 — Run tracked queries
# MAGIC
# MAGIC Each query becomes a logged MLflow run you can compare

# COMMAND ----------

# Run 5 different queries — each creates a separate MLflow run
test_queries = [
    ("heart failure treatment recruiting trials",         "query_heart_failure"),
    ("atrial fibrillation Phase 3 completed studies",    "query_afib"),
    ("elderly cardiovascular risk reduction programs",    "query_elderly"),
    ("SGLT2 inhibitor diabetes cardiovascular outcomes", "query_sglt2"),
    ("coronary artery disease intervention surgery",     "query_cad"),
]

results = []
for question, run_name in test_queries:
    result = tracked_rag_query(question, run_name=run_name)
    results.append(result)
    print()

print(f"\nAll {len(results)} queries logged to MLflow")

# COMMAND ----------
# MAGIC %md ## Step 7 — Compare runs as a summary table

# COMMAND ----------

print("\nRun Comparison Summary")
print("=" * 75)
print(f"{'Query':<45} {'Top-1%':>6} {'Ret ms':>7} {'Gen ms':>7} {'Tok-in':>7}")
print("-" * 75)

for r in results:
    q_short = r["question"][:44]
    print(f"{q_short:<45} {r['top1_sim']:>6.1f} {r['retrieve_ms']:>7} "
          f"{r['generate_ms']:>7} {r['tokens_in']:>7}")

print("=" * 75)

avg_top1    = np.mean([r["top1_sim"]    for r in results])
avg_ret     = np.mean([r["retrieve_ms"] for r in results])
avg_gen     = np.mean([r["generate_ms"] for r in results])
avg_tok_in  = np.mean([r["tokens_in"]   for r in results])

print(f"\nAverages:")
print(f"  Top-1 similarity : {avg_top1:.1f}%")
print(f"  Retrieval time   : {avg_ret:.0f}ms")
print(f"  Generation time  : {avg_gen:.0f}ms")
print(f"  Input tokens     : {avg_tok_in:.0f}")

# COMMAND ----------
# MAGIC %md ## Step 8 — Register the embedding model in MLflow Model Registry

# COMMAND ----------

# Log the embedding model as a registered MLflow model
# This is what "production-grade ML" looks like

class ClinicalEmbeddingModel(mlflow.pyfunc.PythonModel):
    """
    Wrapper to serve the embedding model via MLflow.
    Input: list of text strings
    Output: numpy array of embeddings
    """
    def load_context(self, context):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def predict(self, context, model_input):
        texts = model_input["text"].tolist()
        return self.model.encode(texts, convert_to_numpy=True)


# Register the model
with mlflow.start_run(run_name="embedding_model_registration"):
    mlflow.log_params({
        "model_name":    EMBEDDING_MODEL,
        "dimensions":    model.get_sentence_embedding_dimension(),
        "corpus_size":   collection.count(),
    })
    mlflow.log_metrics({
        "avg_top1_similarity": round(avg_top1, 2),
        "avg_retrieval_ms":    round(avg_ret, 2),
    })

    # Log the model
    mlflow.pyfunc.log_model(
        artifact_path="clinical_embedding_model",
        python_model=ClinicalEmbeddingModel(),
        registered_model_name="clinical_trial_embedder",
    )
    print("Embedding model registered as: clinical_trial_embedder")

# COMMAND ----------
# MAGIC %md ## Step 9 — View experiments in MLflow UI

# COMMAND ----------

print("To view your MLflow experiments:")
print()
print("1. In Databricks left sidebar → click 'Experiments'")
print("2. Find 'clinical_platform_rag'")
print("3. Click it to see all your runs")
print("4. Compare runs side by side")
print("5. Click any run to see parameters, metrics, and artifacts")
print()
print("What you will see:")
print("  - 5 RAG query runs with retrieval + generation metrics")
print("  - 1 model registration run")
print("  - Comparison charts for top-1 similarity across queries")
print("  - Answer artifacts saved per run")
print()
print(f"Experiment: /Users/gilbertrajvijayan@gmail.com/clinical_platform_rag")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Week 6 Complete
# MAGIC
# MAGIC - [x] MLflow experiment created for RAG tracking
# MAGIC - [x] Every query logs: retrieval time, similarity, token usage, latency
# MAGIC - [x] 5 test queries run and compared in MLflow
# MAGIC - [x] Embedding model registered in MLflow Model Registry
# MAGIC - [x] Full ML observability pipeline in place
# MAGIC
# MAGIC **Next: Week 7 — Streamlit dashboard + FastAPI**
# MAGIC The demo UI you show in interviews
