"""
app.py — Streamlit Dashboard
-----------------------------
Clinical Research Intelligence Platform
Interactive demo UI for interviews and portfolio

Run locally:
    pip install streamlit anthropic chromadb sentence-transformers
    streamlit run app.py

This is the demo you show during interviews.
It runs entirely on your laptop — no Databricks needed.
"""

import streamlit as st
import anthropic
import chromadb
import pandas as pd
import numpy as np
import os
import time
import requests
import json
from sentence_transformers import SentenceTransformer

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical Research Intelligence Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .trial-card {
        background: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .recruiting-badge {
        background: #28a745;
        color: white;
        padding: 2px 8px;
        border-radius: 99px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .completed-badge {
        background: #6c757d;
        color: white;
        padding: 2px 8px;
        border-radius: 99px;
        font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital.png", width=60)
    st.markdown("### Clinical Research Intelligence")
    st.markdown("*Powered by Databricks + Claude AI*")
    st.divider()

    # API Key input
    st.markdown("**Configuration**")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get your free key at console.anthropic.com"
    )

    st.divider()
    st.markdown("**Built by**")
    st.markdown("Gilbert Raj Vijayan")
    st.markdown("MS Graduate · UNT Denton")
    st.markdown("Databricks Certified DE Associate")
    st.divider()
    st.markdown("**Stack**")
    st.markdown("🔶 Databricks + Delta Lake")
    st.markdown("🧠 ChromaDB + sentence-transformers")
    st.markdown("🤖 Claude API (RAG)")
    st.markdown("📊 MLflow tracking")


# ── Initialize session state ──────────────────────────────────────────────────
if "collection" not in st.session_state:
    st.session_state.collection   = None
if "model" not in st.session_state:
    st.session_state.model        = None
if "trials_df" not in st.session_state:
    st.session_state.trials_df    = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ── Data Loading Functions ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_data(show_spinner="Fetching trials from ClinicalTrials.gov...")
def fetch_trials(query_term: str = "cardiovascular", max_pages: int = 1) -> pd.DataFrame:
    """Fetch trials directly from ClinicalTrials.gov API."""
    records = []
    params = {
        "query.term": query_term,
        "pageSize":   100,
        "format":     "json",
    }
    next_token = None
    pages = 0

    while pages < max_pages:
        if next_token:
            params["pageToken"] = next_token
        resp = requests.get(
            "https://clinicaltrials.gov/api/v2/studies",
            params=params, timeout=30
        )
        if resp.status_code != 200:
            break

        data    = resp.json()
        studies = data.get("studies", [])
        if not studies:
            break

        for study in studies:
            proto  = study.get("protocolSection", {})
            id_mod = proto.get("identificationModule", {})
            st_mod = proto.get("statusModule", {})
            de_mod = proto.get("designModule", {})
            ds_mod = proto.get("descriptionModule", {})
            sp_mod = proto.get("sponsorCollaboratorsModule", {})
            co_mod = proto.get("conditionsModule", {})

            records.append({
                "nct_id":           id_mod.get("nctId", ""),
                "brief_title":      id_mod.get("briefTitle", ""),
                "overall_status":   st_mod.get("overallStatus", ""),
                "phase":            ", ".join(de_mod.get("phases", [])),
                "enrollment_count": de_mod.get("enrollmentInfo", {}).get("count", 0),
                "conditions":       ", ".join(co_mod.get("conditions", [])),
                "brief_summary":    ds_mod.get("briefSummary", ""),
                "lead_sponsor":     sp_mod.get("leadSponsor", {}).get("name", ""),
                "start_date":       st_mod.get("startDateStruct", {}).get("date", ""),
                "completion_date":  st_mod.get("completionDateStruct", {}).get("date", ""),
            })

        next_token = data.get("nextPageToken")
        pages += 1
        if not next_token:
            break

    return pd.DataFrame(records)


def build_chromadb(df: pd.DataFrame):
    """Build ChromaDB collection from DataFrame."""
    model      = load_embedding_model()
    client     = chromadb.Client()

    try:
        client.delete_collection("clinical_trials_demo")
    except:
        pass

    collection = client.create_collection(
        "clinical_trials_demo",
        metadata={"hnsw:space": "cosine"}
    )

    df_valid = df[df["brief_summary"].str.len() > 10].copy()
    df_valid["text_to_embed"] = (
        df_valid["brief_title"].fillna("") + " | " +
        df_valid["conditions"].fillna("") + " | " +
        df_valid["brief_summary"].fillna("").str[:400]
    )

    embeddings = model.encode(
        df_valid["text_to_embed"].tolist(),
        batch_size=32, show_progress_bar=False, convert_to_numpy=True
    )

    BATCH = 200
    for i in range(0, len(df_valid), BATCH):
        batch = df_valid.iloc[i:i+BATCH]
        batch_emb = embeddings[i:i+BATCH]
        collection.add(
            ids        = batch["nct_id"].tolist(),
            embeddings = batch_emb.tolist(),
            metadatas  = [
                {
                    "nct_id":           r["nct_id"],
                    "brief_title":      r["brief_title"][:200],
                    "overall_status":   r["overall_status"],
                    "phase":            r["phase"],
                    "conditions":       r["conditions"][:200],
                    "lead_sponsor":     r["lead_sponsor"][:100],
                    "enrollment_count": int(r["enrollment_count"] or 0),
                    "is_recruiting":    r["overall_status"] == "RECRUITING",
                }
                for _, r in batch.iterrows()
            ],
            documents = batch["text_to_embed"].tolist(),
        )

    return collection, model


def semantic_search(query: str, collection, model, n: int = 8) -> list:
    """Search ChromaDB for semantically similar trials."""
    q_emb    = model.encode([query], convert_to_numpy=True)
    results  = collection.query(
        query_embeddings=q_emb.tolist(),
        n_results=n,
        include=["metadatas", "distances", "documents"],
    )
    trials = []
    for meta, dist, doc in zip(
        results["metadatas"][0],
        results["distances"][0],
        results["documents"][0],
    ):
        trials.append({**meta, "similarity": round((1-dist)*100, 1), "document": doc})
    return trials


def ask_claude_rag(question: str, trials: list, api_key: str) -> str:
    """Send retrieved trials + question to Claude API."""
    client = anthropic.Anthropic(api_key=api_key)

    context = "\n---\n".join([
        f"NCT ID: {t['nct_id']}\n"
        f"Title: {t['brief_title']}\n"
        f"Status: {t['overall_status']} | Phase: {t['phase']}\n"
        f"Enrollment: {t['enrollment_count']:,} | Sponsor: {t['lead_sponsor']}\n"
        f"Match: {t['similarity']}%"
        for t in trials
    ])

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=800,
        system="""You are a clinical research intelligence assistant.
Answer questions using the trial data provided.
Always cite NCT IDs. Be concise, accurate, and helpful.
Highlight recruiting trials when relevant.""",
        messages=[{
            "role": "user",
            "content": f"Question: {question}\n\nRelevant trials:\n{context}"
        }],
    )
    return response.content[0].text


# ── Main App ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🏥 Clinical Research Intelligence Platform</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered clinical trial discovery · '
            'Built on Databricks + Delta Lake + Claude API</div>',
            unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🤖 AI Research Assistant",
    "🔍 Trial Search",
    "📊 Analytics Dashboard",
    "⚙️ Data Pipeline",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — AI Research Assistant (RAG)
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Ask anything about cardiovascular clinical trials")
    st.markdown("*Powered by ChromaDB semantic search + Claude AI*")

    # Load data button
    col1, col2 = st.columns([3, 1])
    with col1:
        query_term = st.text_input(
            "Data focus area",
            value="cardiovascular",
            label_visibility="collapsed"
        )
    with col2:
        load_btn = st.button("Load Data", type="primary", use_container_width=True)

    if load_btn:
        with st.spinner("Fetching trials and building vector index..."):
            df = fetch_trials(query_term, max_pages=2)
            collection, emb_model = build_chromadb(df)
            st.session_state.collection = collection
            st.session_state.model      = emb_model
            st.session_state.trials_df  = df
            st.success(f"✅ Loaded {len(df):,} trials · "
                       f"{collection.count():,} indexed in ChromaDB")

    st.divider()

    # Chat interface
    if st.session_state.collection is None:
        st.info("👆 Click 'Load Data' first to initialize the AI assistant")
    else:
        # Show chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("Ask about cardiovascular clinical trials..."):
            if not api_key:
                st.error("Please enter your Anthropic API key in the sidebar")
            else:
                # Add user message
                st.session_state.chat_history.append({
                    "role": "user", "content": prompt
                })
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Searching trials and generating answer..."):
                        start = time.time()
                        trials  = semantic_search(
                            prompt,
                            st.session_state.collection,
                            st.session_state.model
                        )
                        answer  = ask_claude_rag(prompt, trials, api_key)
                        elapsed = round(time.time() - start, 1)

                    st.markdown(answer)
                    st.caption(f"Retrieved {len(trials)} trials · "
                               f"Top match: {trials[0]['similarity']}% · "
                               f"{elapsed}s total")

                    # Show retrieved trials in expander
                    with st.expander("View retrieved trials"):
                        for t in trials[:5]:
                            status_color = (
                                "🟢" if t["overall_status"] == "RECRUITING"
                                else "⚫"
                            )
                            st.markdown(
                                f"{status_color} **{t['nct_id']}** — "
                                f"{t['brief_title'][:80]} "
                                f"*({t['similarity']}% match)*"
                            )

                st.session_state.chat_history.append({
                    "role": "assistant", "content": answer
                })

        # Quick question buttons
        st.markdown("**Quick questions:**")
        qcols = st.columns(3)
        quick_qs = [
            "Which trials are currently recruiting?",
            "What Phase 3 trials have the most patients?",
            "Which sponsors run the most trials?",
        ]
        for i, (col, q) in enumerate(zip(qcols, quick_qs)):
            with col:
                if st.button(q, key=f"quick_{i}", use_container_width=True):
                    st.session_state.chat_history.append(
                        {"role": "user", "content": q}
                    )
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Trial Search
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Search & Filter Trials")

    if st.session_state.trials_df is None:
        st.info("Load data in the AI Assistant tab first")
    else:
        df = st.session_state.trials_df.copy()

        # Filters
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            status_filter = st.multiselect(
                "Status",
                options=sorted(df["overall_status"].dropna().unique()),
                default=["RECRUITING"],
            )
        with fcol2:
            phase_filter = st.multiselect(
                "Phase",
                options=sorted(df["phase"].dropna().unique()),
            )
        with fcol3:
            search_text = st.text_input("Search title/conditions", "")

        # Apply filters
        filtered = df.copy()
        if status_filter:
            filtered = filtered[filtered["overall_status"].isin(status_filter)]
        if phase_filter:
            filtered = filtered[filtered["phase"].isin(phase_filter)]
        if search_text:
            mask = (
                filtered["brief_title"].str.contains(search_text, case=False, na=False) |
                filtered["conditions"].str.contains(search_text, case=False, na=False)
            )
            filtered = filtered[mask]

        st.markdown(f"**{len(filtered):,} trials match your filters**")

        # Display trials
        for _, row in filtered.head(20).iterrows():
            with st.expander(
                f"{'🟢' if row['overall_status'] == 'RECRUITING' else '⚫'} "
                f"{row['nct_id']} — {row['brief_title'][:80]}"
            ):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Status:** {row['overall_status']}")
                    st.markdown(f"**Phase:** {row['phase'] or 'N/A'}")
                    st.markdown(f"**Enrollment:** {int(row['enrollment_count'] or 0):,}")
                with c2:
                    st.markdown(f"**Sponsor:** {row['lead_sponsor']}")
                    st.markdown(f"**Start:** {row['start_date'] or 'N/A'}")
                    st.markdown(f"**Conditions:** {row['conditions'][:100]}")
                if row["brief_summary"]:
                    st.markdown(f"**Summary:** {row['brief_summary'][:300]}...")
                st.markdown(
                    f"[View on ClinicalTrials.gov]"
                    f"(https://clinicaltrials.gov/study/{row['nct_id']})"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Analytics Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Analytics Dashboard")

    if st.session_state.trials_df is None:
        st.info("Load data in the AI Assistant tab first")
    else:
        df = st.session_state.trials_df.copy()

        # KPI row
        total    = len(df)
        recruit  = (df["overall_status"] == "RECRUITING").sum()
        complete = (df["overall_status"] == "COMPLETED").sum()
        sponsors = df["lead_sponsor"].nunique()
        enroll   = df["enrollment_count"].sum()

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Trials",       f"{total:,}")
        k2.metric("Recruiting",         f"{recruit:,}",
                  f"{recruit/total*100:.1f}%")
        k3.metric("Completed",          f"{complete:,}",
                  f"{complete/total*100:.1f}%")
        k4.metric("Unique Sponsors",    f"{sponsors:,}")
        k5.metric("Patients Enrolled",  f"{int(enroll):,}")

        st.divider()

        # Charts
        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown("**Trial Status Distribution**")
            status_counts = (
                df["overall_status"].value_counts()
                  .reset_index()
                  .rename(columns={"overall_status": "Status", "count": "Count"})
            )
            st.bar_chart(status_counts.set_index("Status"))

        with ch2:
            st.markdown("**Top 10 Sponsors by Trial Count**")
            sponsor_counts = (
                df["lead_sponsor"]
                .value_counts()
                .head(10)
                .reset_index()
                .rename(columns={"lead_sponsor": "Sponsor", "count": "Trials"})
            )
            st.bar_chart(sponsor_counts.set_index("Sponsor"))

        st.divider()

        # Phase breakdown
        st.markdown("**Phase Distribution**")
        phase_counts = (
            df["phase"].fillna("Unknown")
            .value_counts()
            .reset_index()
            .rename(columns={"phase": "Phase", "count": "Count"})
        )
        st.bar_chart(phase_counts.set_index("Phase"))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Data Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Data Pipeline Architecture")

    st.markdown("""
    This platform is built on a production-grade Databricks Medallion architecture:
    """)

    pipeline_cols = st.columns(5)
    stages = [
        ("🌐", "Source", "ClinicalTrials.gov API v2", "#e3f2fd"),
        ("🥉", "Bronze", "Raw Delta table\nAppend-only\nFull JSON preserved", "#fff3e0"),
        ("🥈", "Silver", "Cleaned + typed\nCDC MERGE\nData quality checks", "#e8f5e9"),
        ("🥇", "Gold", "Aggregations\nBusiness metrics\nSQL dashboards", "#fce4ec"),
        ("🧠", "AI Layer", "ChromaDB vectors\nRAG + Claude API\nMLflow tracking", "#f3e5f5"),
    ]

    for col, (icon, name, desc, color) in zip(pipeline_cols, stages):
        with col:
            st.markdown(
                f'<div style="background:{color};padding:12px;'
                f'border-radius:8px;text-align:center;">'
                f'<div style="font-size:2rem">{icon}</div>'
                f'<div style="font-weight:700;margin:4px 0">{name}</div>'
                f'<div style="font-size:12px;color:#555">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Tech Stack**")
        stack = {
            "Data Platform":    "Databricks + Delta Lake + Delta Live Tables",
            "Data Quality":     "Custom checks + CDC MERGE patterns",
            "Vector Store":     "ChromaDB + sentence-transformers",
            "AI Layer":         "Claude API (claude-opus-4-5) + RAG",
            "ML Tracking":      "MLflow experiment tracking + Model Registry",
            "Data Source":      "ClinicalTrials.gov API v2 (free, no auth)",
            "Language":         "Python + PySpark + SQL",
        }
        for k, v in stack.items():
            st.markdown(f"- **{k}:** {v}")

    with col2:
        st.markdown("**Key Numbers**")
        numbers = {
            "Trials available":      "500,000+ (2,000 in demo)",
            "Vector dimensions":     "384 (all-MiniLM-L6-v2)",
            "Pipeline layers":       "Bronze → Silver → Gold",
            "Data quality rules":    "5 automated checks per run",
            "MLflow runs logged":    "7 (5 RAG + 2 model registration)",
            "CDC tracked fields":    "status, enrollment, completion date",
            "RAG retrieval":         "Top-10 semantic search per query",
        }
        for k, v in numbers.items():
            st.markdown(f"- **{k}:** {v}")

    st.divider()
    st.markdown("**GitHub Repository**")
    st.code("https://github.com/gilbertrajvijayan/clinical-platform", language=None)
