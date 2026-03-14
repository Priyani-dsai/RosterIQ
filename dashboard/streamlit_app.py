import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# print("PROJECT ROOT:", PROJECT_ROOT)
# print("SYS PATH:", sys.path)

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
from tools.visualization import *



# Page config

st.set_page_config(
page_title="RosterIQ",
layout="wide"
)
st.title("RosterIQ: Provider Roster Intelligence Dashboard")


# Load data

@st.cache_data
def load_data():
    loader = DataLoader(
    "data/roster_processing_details.csv",
    "data/aggregated_operational_metrics.csv"
)

    roster_df, market_df = loader.load_data()

    return roster_df, market_df, loader


roster_df, market_df, loader = load_data()


# Load agent

@st.cache_resource
def load_agent(roster_df):
    return SupervisorAgent(roster_df)

agent = load_agent(roster_df)

# initialize episodic memory once per session
@st.cache_resource
def load_episodic_memory():
    return EpisodicMemory()

episodic_memory = load_episodic_memory()

# initialize semantic memory once per session
@st.cache_resource
def load_semantic_memory():
    return SemanticMemory()

semantic_memory = load_semantic_memory()


# Data Summary

st.header("System Overview")

summary = loader.basic_summary()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Roster Files", summary["total_roster_files"])
col2.metric("Organizations", summary["organizations"])
col3.metric("Failed Files", summary["failed_files"])
col4.metric("Stuck Files", summary["stuck_files"])

# Autonomous Insights

st.divider()
st.header("🧠 Autonomous Pipeline Insights")

insights = generate_pipeline_insights(roster_df)

for insight in insights:
    st.info(insight)

# Stuck Operations

st.divider()
st.header("🚨 Stuck Roster Operations")

stuck_df = detect_stuck_operations(roster_df)

if stuck_df.empty:
    st.success("No stuck operations detected")
else:
    st.dataframe(stuck_df, width="stretch")



# Pipeline Bottlenecks

st.divider()
st.header("⚠️ Pipeline Bottlenecks")

bottlenecks = detect_stage_bottlenecks(roster_df)

if bottlenecks.empty:
    st.success("No bottlenecks detected")
else:
    st.dataframe(bottlenecks.head(20), width="stretch")


# Failure Analysis

st.divider()
st.header("📉 Top Organizations Causing Failures")

failure_fig = failure_distribution(roster_df)

st.plotly_chart(failure_fig, width="stretch")


# Duration Anomaly Visualization

st.divider()
st.header("⏱ Stage Duration Anomalies")

fig = duration_anomaly_chart(roster_df)

st.plotly_chart(fig, width="stretch")


# Root Cause Panel

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


# Market Success Trends

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



# AI Query Panel

st.divider()
st.header("🤖 RosterIQ AI Assistant")

st.markdown("""
Example questions you can ask:

* Which organizations cause most failures?
* Show stuck roster operations
* What source systems fail most?
* Why are pipelines failing?
* Show pipeline bottlenecks
  """)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask a question about the roster pipeline")

if st.button("Run Query"):

    if query:

        with st.spinner("AI agent analyzing pipeline data..."):

            try:

                # Retrieve episodic memory
                past_context = episodic_memory.fetch_recent()

                semantic_context = semantic_memory.retrieve(query)
                if not semantic_context:
                    semantic_context = "No additional domain knowledge found."

                # Build context-aware query
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

                # Invoke agent
                response = agent.route_query(full_query)

                procedure = response["procedure"]
                agent_used = response["agent"]
                answer = response["output"]

                # Only generate visualization if the supervisor selected visualization procedure
                if procedure == "visualization_analysis":

                    st.info("Visualization generated based on the query.")

                    if "failure" in query.lower():
                        fig = failure_distribution(roster_df)
                        st.plotly_chart(fig, width="stretch", key="viz_failure")

                    elif "duration" in query.lower() or "anomaly" in query.lower():
                        fig = duration_anomaly_chart(roster_df)
                        st.plotly_chart(fig, width="stretch", key="viz_duration")

                    elif "health" in query.lower():
                        fig = pipeline_stage_health_heatmap(roster_df)
                        st.plotly_chart(fig, width="stretch", key="viz_health")

                # Clean LangChain reasoning traces

                # Prefer Final Answer if available
                if "Final Answer:" in answer:
                    answer = answer.split("Final Answer:")[-1].strip()

                # Remove reasoning blocks
                elif "Thought:" in answer:
                    answer = answer.split("Thought:")[0].strip()

                # Remove tool traces
                answer = answer.replace("Action:", "")
                answer = answer.replace("Action Input:", "")
                answer = answer.strip()

                if not answer:
                    answer = "The agent completed the analysis but no formatted answer was produced."

    

                # Store interaction in episodic memory
                episodic_memory.store_interaction(query, answer)


                # Store for UI display
                st.session_state.chat_history.append((query, answer))

                # Show routing information for the latest response
                st.success(f"Supervisor selected procedure: {procedure}")
                st.success(f"Specialist agent used: {agent_used}")

                st.divider()

                # Display conversation history
                for q, a in st.session_state.chat_history[::-1]:
                    st.markdown(f"**You:** {q}")
                    st.markdown(f"**Agent:** {a}")
                    

            except Exception as e:
                st.error(f"Agent error: {str(e)}")

    else:
        st.warning("Please enter a query.")