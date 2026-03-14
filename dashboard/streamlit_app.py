import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd

from core.data_loader import DataLoader
from core.pipeline_intelligence import *
from core.root_cause_analysis import generate_root_cause
from tools.visualization import *
from agents.supervisor_agent import SupervisorAgent
from memory.episodic_memory import EpisodicMemory
from memory.semantic_memory import SemanticMemory
from core.insight_engine import generate_pipeline_insights


# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(
    page_title="RosterIQ",
    layout="wide"
)

st.title("RosterIQ: Provider Roster Intelligence Dashboard")


# ===============================
# LOAD DATA
# ===============================

@st.cache_data
def load_data():
    loader = DataLoader(
        "data/roster_processing_details.csv",
        "data/aggregated_operational_metrics.csv"
    )

    roster_df, market_df = loader.load_data()

    return roster_df, market_df, loader


roster_df, market_df, loader = load_data()


# ===============================
# LOAD AGENTS
# ===============================

@st.cache_resource
def load_agent(roster_df):
    return SupervisorAgent(roster_df)

agent = load_agent(roster_df)


# ===============================
# LOAD MEMORY SYSTEMS
# ===============================

@st.cache_resource
def load_episodic_memory():
    return EpisodicMemory()

episodic_memory = load_episodic_memory()


@st.cache_resource
def load_semantic_memory():
    return SemanticMemory()

semantic_memory = load_semantic_memory()


# ===============================
# SYSTEM OVERVIEW
# ===============================

st.header("System Overview")

summary = loader.basic_summary()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Roster Files", summary["total_roster_files"])
col2.metric("Organizations", summary["organizations"])
col3.metric("Failed Files", summary["failed_files"])
col4.metric("Stuck Files", summary["stuck_files"])


# ===============================
# AUTONOMOUS INSIGHTS
# ===============================

st.divider()
st.header("🧠 Autonomous Pipeline Insights")

insights = generate_pipeline_insights(roster_df)

for insight in insights:
    st.info(insight)


# ===============================
# STUCK OPERATIONS
# ===============================

st.divider()
st.header("🚨 Stuck Roster Operations")

stuck_df = detect_stuck_operations(roster_df)

if stuck_df.empty:
    st.success("No stuck operations detected")
else:
    st.dataframe(stuck_df, width="stretch")


# ===============================
# PIPELINE BOTTLENECKS
# ===============================

st.divider()
st.header("⚠️ Pipeline Bottlenecks")

bottlenecks = detect_stage_bottlenecks(roster_df)

if bottlenecks.empty:
    st.success("No bottlenecks detected")
else:
    st.dataframe(bottlenecks.head(20), width="stretch")


# ===============================
# FAILURE ANALYSIS
# ===============================

st.divider()
st.header("📉 Top Organizations Causing Failures")

failure_fig = failure_distribution(roster_df)

st.plotly_chart(failure_fig, width="stretch")


# ===============================
# DURATION ANOMALY
# ===============================

st.divider()
st.header("⏱ Stage Duration Anomalies")

fig = duration_anomaly_chart(roster_df)

st.plotly_chart(fig, width="stretch")


# ===============================
# ROOT CAUSE PANEL
# ===============================

st.divider()
st.header("🧠 Root Cause Insights")

root = generate_root_cause(roster_df)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Top Failure Organizations")
    st.dataframe(root["top_failure_orgs"])

with col2:
    st.subheader("Failure Stages")
    st.dataframe(root["failure_stages"])

with col3:
    st.subheader("Source System Failures")
    st.dataframe(root["source_systems"])


# ===============================
# MARKET SUCCESS TRENDS
# ===============================

st.divider()
st.header("📊 Market Success Trends")

import plotly.express as px

market_fig = px.line(
    market_df,
    x="MONTH",
    y="SCS_PERCENT",
    color="MARKET",
    title="Market Success Rate Over Time"
)

st.plotly_chart(market_fig, width="stretch")


# ===============================
# AI QUERY PANEL
# ===============================

st.divider()
st.header("🤖 RosterIQ AI Assistant")

st.markdown("""
Example questions you can ask:

• Which organizations cause most failures?  
• Show stuck roster operations  
• What source systems fail most?  
• Why are pipelines failing?  
• Show pipeline bottlenecks  
""")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


query = st.text_input("Ask a question about the roster pipeline")


# ===============================
# QUERY EXECUTION
# ===============================

if st.button("Run Query"):

    if query:

        with st.spinner("AI agent analyzing pipeline data..."):

            try:

                # Retrieve episodic memory
                past_context = episodic_memory.fetch_recent()

                # Retrieve semantic knowledge
                semantic_context = semantic_memory.retrieve(query)

                if not semantic_context:
                    semantic_context = "No additional domain knowledge found."


                # Build contextual prompt (for agent reasoning)
                full_query = f"""
You are an AI system analyzing provider roster pipelines.

Relevant domain knowledge:
{semantic_context}

Previous interactions:
{past_context}

User question:
{query}

Provide a clear analytical answer using the available tools if necessary.
"""


                # ===================================
                # SUPERVISOR DECISION
                # ===================================

                procedure = agent.choose_procedure(query)

                print("Selected procedure:", procedure)


                # ===================================
                # VISUALIZATION ROUTE
                # ===================================

                if procedure == "visualization_analysis":

                    st.info("Visualization generated based on the query.")

                    fig = generate_visualization(roster_df, query)

                    if fig is not None:
                        st.plotly_chart(
                            fig,
                            width="stretch",
                            key=f"viz_{len(st.session_state.chat_history)}"
                        )

                    answer = "Visualization generated based on your request."
                    agent_used = "Visualization Engine"


                # ===================================
                # NORMAL AGENT ROUTE
                # ===================================

                else:

                    response = agent.route_query(query)

                    procedure = response["procedure"]
                    agent_used = response["agent"]
                    answer = response["output"]

                    # Clean reasoning traces
                    if "Final Answer:" in answer:
                        answer = answer.split("Final Answer:")[-1].strip()

                    elif "Thought:" in answer:
                        answer = answer.split("Thought:")[0].strip()

                    answer = answer.replace("Action:", "")
                    answer = answer.replace("Action Input:", "")
                    answer = answer.strip()

                    if not answer:
                        answer = "The agent completed the analysis but no formatted answer was produced."


                # ===================================
                # STORE EPISODIC MEMORY
                # ===================================

                episodic_memory.store_interaction(query, answer)


                # ===================================
                # UPDATE CHAT HISTORY
                # ===================================

                st.session_state.chat_history.append((query, answer))


                # ===================================
                # SHOW ROUTING INFO
                # ===================================

                st.success(f"Supervisor selected procedure: {procedure}")
                st.success(f"System component used: {agent_used}")

                st.divider()


                # ===================================
                # DISPLAY HISTORY
                # ===================================

                for q, a in st.session_state.chat_history[::-1]:
                    st.markdown(f"**You:** {q}")
                    st.markdown(f"**Agent:** {a}")


            except Exception as e:
                st.error(f"Agent error: {str(e)}")


    else:
        st.warning("Please enter a query.")