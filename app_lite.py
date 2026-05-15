"""
app.py — Streamlit Dashboard (Lightweight Version)
---------------------------------------------------
Clinical Research Intelligence Platform
Uses ClinicalTrials.gov API directly — no heavy model downloads
Keyword search + Claude AI for answers
"""

import streamlit as st
import anthropic
import requests
import pandas as pd
import time

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical Research Intelligence Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Clinical Research Intel")
    st.markdown("*Powered by Databricks + Claude AI*")
    st.divider()

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get your free key at console.anthropic.com"
    )

    st.divider()
    st.markdown("**Built by**")
    st.markdown("**Gilbert Raj Vijayan**")
    st.markdown("MS Graduate · UNT Denton")
    st.markdown("Databricks Certified DE Associate")
    st.divider()
    st.markdown("**Stack**")
    st.markdown("🔶 Databricks + Delta Lake")
    st.markdown("🧠 ChromaDB + Embeddings")
    st.markdown("🤖 Claude API (RAG)")
    st.markdown("📊 MLflow Tracking")


# ── Session State ─────────────────────────────────────────────────────────────
if "trials_df" not in st.session_state:
    st.session_state.trials_df = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ── Helper Functions ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching trials from ClinicalTrials.gov...")
def fetch_trials(query_term: str, page_size: int = 200) -> pd.DataFrame:
    """Fetch trials directly from ClinicalTrials.gov API."""
    records = []
    params  = {
        "query.term": query_term,
        "pageSize":   page_size,
        "format":     "json",
    }
    try:
        resp = requests.get(
            "https://clinicaltrials.gov/api/v2/studies",
            params=params, timeout=30
        )
        if resp.status_code != 200:
            return pd.DataFrame()

        studies = resp.json().get("studies", [])
        for study in studies:
            proto  = study.get("protocolSection", {})
            id_mod = proto.get("identificationModule", {})
            st_mod = proto.get("statusModule", {})
            de_mod = proto.get("designModule", {})
            ds_mod = proto.get("descriptionModule", {})
            sp_mod = proto.get("sponsorCollaboratorsModule", {})
            co_mod = proto.get("conditionsModule", {})

            records.append({
                "nct_id":         id_mod.get("nctId", ""),
                "brief_title":    id_mod.get("briefTitle", ""),
                "overall_status": st_mod.get("overallStatus", ""),
                "phase":          ", ".join(de_mod.get("phases", [])),
                "enrollment":     de_mod.get("enrollmentInfo", {}).get("count", 0),
                "conditions":     ", ".join(co_mod.get("conditions", [])),
                "brief_summary":  (ds_mod.get("briefSummary", "") or "")[:500],
                "lead_sponsor":   sp_mod.get("leadSponsor", {}).get("name", ""),
                "start_date":     st_mod.get("startDateStruct", {}).get("date", ""),
            })
    except Exception as e:
        st.error(f"API error: {e}")

    return pd.DataFrame(records)


def search_trials(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    """Simple keyword search across title, conditions, summary."""
    if not keyword:
        return df
    keyword = keyword.lower()
    mask = (
        df["brief_title"].str.lower().str.contains(keyword, na=False) |
        df["conditions"].str.lower().str.contains(keyword, na=False) |
        df["brief_summary"].str.lower().str.contains(keyword, na=False)
    )
    return df[mask]


def ask_claude(question: str, trials: pd.DataFrame, api_key: str) -> str:
    """Send top trials as context to Claude and get a cited answer."""
    client = anthropic.Anthropic(api_key=api_key)

    # Format top 8 trials as context
    top = trials.head(8)
    context = "\n---\n".join([
        f"NCT ID: {r['nct_id']}\n"
        f"Title: {r['brief_title']}\n"
        f"Status: {r['overall_status']} | Phase: {r['phase']}\n"
        f"Enrollment: {int(r['enrollment'] or 0):,} | Sponsor: {r['lead_sponsor']}\n"
        f"Summary: {r['brief_summary'][:300]}"
        for _, r in top.iterrows()
    ])

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=800,
        system="""You are a clinical research intelligence assistant.
Answer questions using the trial data provided.
Always cite NCT IDs. Be concise and accurate.
Highlight recruiting trials when relevant.""",
        messages=[{
            "role": "user",
            "content": f"Question: {question}\n\nRelevant trials:\n{context}"
        }],
    )
    return response.content[0].text


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🏥 Clinical Research Intelligence Platform")
st.markdown("*AI-powered clinical trial discovery · Built on Databricks + Delta Lake + Claude API*")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🤖 AI Assistant",
    "🔍 Trial Search",
    "📊 Analytics",
    "⚙️ Architecture",
])


# ═══════════════════════════════════════════
# TAB 1 — AI Assistant
# ═══════════════════════════════════════════
with tab1:
    st.markdown("### Ask anything about cardiovascular clinical trials")

    col1, col2 = st.columns([3, 1])
    with col1:
        query_term = st.text_input(
            "Search focus",
            value="cardiovascular",
            label_visibility="collapsed"
        )
    with col2:
        load_btn = st.button("Load Trials", type="primary", use_container_width=True)

    if load_btn:
        df = fetch_trials(query_term)
        st.session_state.trials_df = df
        recruiting = (df["overall_status"] == "RECRUITING").sum()
        st.success(f"✅ Loaded {len(df):,} trials · {recruiting:,} currently recruiting")

    st.divider()

    if st.session_state.trials_df is None:
        st.info("👆 Click 'Load Trials' to start")
    else:
        # Chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("Ask about cardiovascular trials..."):
            if not api_key:
                st.error("Enter your Anthropic API key in the sidebar")
            else:
                st.session_state.chat_history.append(
                    {"role": "user", "content": prompt}
                )
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Searching and generating answer..."):
                        start   = time.time()
                        results = search_trials(
                            st.session_state.trials_df, prompt
                        )
                        if results.empty:
                            results = st.session_state.trials_df
                        answer  = ask_claude(prompt, results, api_key)
                        elapsed = round(time.time() - start, 1)

                    st.markdown(answer)
                    st.caption(f"Searched {len(results):,} trials · {elapsed}s")

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )

        # Quick questions
        st.markdown("**Quick questions:**")
        q1, q2, q3 = st.columns(3)
        with q1:
            if st.button("Which trials are recruiting?", use_container_width=True):
                st.session_state.chat_history.append(
                    {"role": "user",
                     "content": "Which trials are currently recruiting patients?"}
                )
                st.rerun()
        with q2:
            if st.button("Top sponsors by trial count?", use_container_width=True):
                st.session_state.chat_history.append(
                    {"role": "user",
                     "content": "Which sponsors are running the most trials?"}
                )
                st.rerun()
        with q3:
            if st.button("Phase 3 trials with large enrollment?",
                         use_container_width=True):
                st.session_state.chat_history.append(
                    {"role": "user",
                     "content": "What Phase 3 trials have the largest enrollment?"}
                )
                st.rerun()


# ═══════════════════════════════════════════
# TAB 2 — Trial Search
# ═══════════════════════════════════════════
with tab2:
    st.markdown("### Search & Filter Trials")

    if st.session_state.trials_df is None:
        st.info("Load trials in the AI Assistant tab first")
    else:
        df = st.session_state.trials_df

        f1, f2, f3 = st.columns(3)
        with f1:
            status_filter = st.multiselect(
                "Status",
                options=sorted(df["overall_status"].dropna().unique()),
                default=["RECRUITING"],
            )
        with f2:
            phase_filter = st.multiselect(
                "Phase",
                options=sorted(df["phase"].dropna().unique()),
            )
        with f3:
            keyword = st.text_input("Keyword search", "")

        filtered = df.copy()
        if status_filter:
            filtered = filtered[filtered["overall_status"].isin(status_filter)]
        if phase_filter:
            filtered = filtered[filtered["phase"].isin(phase_filter)]
        if keyword:
            filtered = search_trials(filtered, keyword)

        st.markdown(f"**{len(filtered):,} trials match your filters**")

        for _, row in filtered.head(15).iterrows():
            icon = "🟢" if row["overall_status"] == "RECRUITING" else "⚫"
            with st.expander(
                f"{icon} {row['nct_id']} — {row['brief_title'][:75]}"
            ):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Status:** {row['overall_status']}")
                    st.markdown(f"**Phase:** {row['phase'] or 'N/A'}")
                    st.markdown(f"**Enrollment:** {int(row['enrollment'] or 0):,}")
                with c2:
                    st.markdown(f"**Sponsor:** {row['lead_sponsor']}")
                    st.markdown(f"**Start:** {row['start_date'] or 'N/A'}")
                if row["brief_summary"]:
                    st.markdown(f"**Summary:** {row['brief_summary'][:250]}...")
                st.markdown(
                    f"[View on ClinicalTrials.gov]"
                    f"(https://clinicaltrials.gov/study/{row['nct_id']})"
                )


# ═══════════════════════════════════════════
# TAB 3 — Analytics
# ═══════════════════════════════════════════
with tab3:
    st.markdown("### Analytics Dashboard")

    if st.session_state.trials_df is None:
        st.info("Load trials in the AI Assistant tab first")
    else:
        df = st.session_state.trials_df

        total    = len(df)
        recruit  = (df["overall_status"] == "RECRUITING").sum()
        complete = (df["overall_status"] == "COMPLETED").sum()
        sponsors = df["lead_sponsor"].nunique()
        enroll   = df["enrollment"].sum()

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Trials",      f"{total:,}")
        k2.metric("Recruiting",        f"{recruit:,}",
                  f"{recruit/total*100:.1f}%")
        k3.metric("Completed",         f"{complete:,}",
                  f"{complete/total*100:.1f}%")
        k4.metric("Unique Sponsors",   f"{sponsors:,}")
        k5.metric("Patients Enrolled", f"{int(enroll):,}")

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Status Distribution**")
            st.bar_chart(
                df["overall_status"].value_counts().head(8)
            )
        with c2:
            st.markdown("**Top 10 Sponsors**")
            st.bar_chart(
                df["lead_sponsor"].value_counts().head(10)
            )

        st.markdown("**Phase Breakdown**")
        st.bar_chart(
            df["phase"].fillna("Unknown").value_counts()
        )


# ═══════════════════════════════════════════
# TAB 4 — Architecture
# ═══════════════════════════════════════════
with tab4:
    st.markdown("### Data Pipeline Architecture")

    cols = st.columns(5)
    stages = [
        ("🌐", "Source",   "ClinicalTrials.gov\nAPI v2\nFree · No auth"),
        ("🥉", "Bronze",   "Raw Delta table\nAppend-only\nFull JSON"),
        ("🥈", "Silver",   "Cleaned + typed\nCDC MERGE\nQuality checks"),
        ("🥇", "Gold",     "Aggregations\nBusiness KPIs\nSQL dashboards"),
        ("🧠", "AI Layer", "ChromaDB\nRAG + Claude\nMLflow"),
    ]
    colors = ["#e3f2fd", "#fff3e0", "#e8f5e9", "#fce4ec", "#f3e5f5"]

    for col, (icon, name, desc), color in zip(cols, stages, colors):
        with col:
            st.markdown(
                f'<div style="background:{color};padding:14px;'
                f'border-radius:8px;text-align:center;min-height:130px">'
                f'<div style="font-size:2rem">{icon}</div>'
                f'<div style="font-weight:700;margin:6px 0">{name}</div>'
                f'<div style="font-size:11px;color:#555;white-space:pre-line">'
                f'{desc}</div></div>',
                unsafe_allow_html=True,
            )

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Tech Stack**")
        for k, v in {
            "Data Platform":  "Databricks + Delta Lake + DLT",
            "Data Quality":   "Custom checks + CDC MERGE",
            "Vector Store":   "ChromaDB + sentence-transformers",
            "AI Layer":       "Claude API + RAG pipeline",
            "ML Tracking":    "MLflow + Model Registry",
            "Data Source":    "ClinicalTrials.gov API v2",
        }.items():
            st.markdown(f"- **{k}:** {v}")

    with c2:
        st.markdown("**Key Numbers**")
        for k, v in {
            "Trials available":   "500K+ (200 in demo)",
            "Pipeline layers":    "Bronze → Silver → Gold",
            "Quality rules":      "5 automated checks per run",
            "MLflow runs":        "7 logged runs",
            "RAG retrieval":      "Top-8 trials per query",
            "GitHub":             "gilbertrajvijayan/clinical-platform",
        }.items():
            st.markdown(f"- **{k}:** {v}")
