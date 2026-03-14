import os
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq

from core.pipeline_intelligence import (
    organization_failure_analysis,
    source_system_failure_analysis
)

from core.root_cause_analysis import generate_root_cause
from tools.web_search import web_search_tool


def create_quality_agent(roster_df):

    # ---------------------------------------------------
    # TOOL 1 — ORGANIZATION FAILURE ANALYSIS
    # ---------------------------------------------------

    def org_failure_tool(query=""):
        df = organization_failure_analysis(roster_df).head(10)

        results = []
        for org, count in df.items():
            results.append(f"{org} — {count} failures")

        return "Top organizations causing failures:\n" + "\n".join(results)


    # ---------------------------------------------------
    # TOOL 2 — SOURCE SYSTEM FAILURE ANALYSIS
    # ---------------------------------------------------

    def source_failure_tool(query=""):
        df = source_system_failure_analysis(roster_df)

        results = []
        for system, ratio in df.items():
            results.append(f"{system} — {round(ratio*100,2)}% failure rate")

        return "Source systems contributing most to pipeline failures:\n" + "\n".join(results[:10])


    # ---------------------------------------------------
    # TOOL 3 — ROOT CAUSE ANALYSIS
    # ---------------------------------------------------

    def root_cause_tool(query=""):

        root = generate_root_cause(roster_df)

        orgs = root["top_failure_orgs"]
        stages = root["failure_stages"]
        systems = root["source_systems"]

        org_text = []
        for org, count in orgs.head(5).items():
            org_text.append(f"{org} ({count} failures)")

        stage_text = []
        for stage, count in stages.items():
            stage_text.append(f"{stage} ({count})")

        system_text = []
        for sys, val in systems.head(5).items():
            system_text.append(f"{sys} ({round(val*100,2)}%)")

        return f"""
Root cause analysis of pipeline failures:

Top failing organizations:
{', '.join(org_text)}

Failure stages:
{', '.join(stage_text)}

Source systems contributing to failures:
{', '.join(system_text)}
"""


    # ---------------------------------------------------
    # TOOL DEFINITIONS
    # ---------------------------------------------------

    tools = [

        Tool(
            name="get_failure_organizations",
            func=org_failure_tool,
            description="""
Use this tool when the user asks about which organizations are causing pipeline failures.

Example queries:
- Which organizations cause most failures?
- Top failing organizations
- Organizations with highest failure counts
"""
        ),

        Tool(
            name="get_source_system_failures",
            func=source_failure_tool,
            description="""
Use this tool when the user asks about source systems responsible for failures.

Example queries:
- Which source systems fail most?
- Source system failure rates
- Systems causing ingestion failures
"""
        ),

        Tool(
            name="get_root_cause",
            func=root_cause_tool,
            description="""
Use this tool when the user asks about WHY pipelines are failing.

Example queries:
- Why are pipelines failing?
- Root cause of failures
- Causes of ingestion failures
"""
        ),

        Tool(
            name="web_search",
            func=web_search_tool,
            description="""
Use this tool ONLY when the user explicitly asks to search the web
or asks about general ETL/data engineering concepts not present in the dataset.

Example queries:
- Search causes of ETL pipeline bottlenecks
- Search debugging methods for data pipelines
- Look up ETL pipeline best practices
"""
        )

    ]


    # ---------------------------------------------------
    # LLM INITIALIZATION
    # ---------------------------------------------------

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )


    # ---------------------------------------------------
    # AGENT PROMPT
    # ---------------------------------------------------

    agent_kwargs = {
        "prefix": """
You are an AI assistant that analyzes provider roster pipeline data.

You have access to specialized analytical tools.

Rules:
- Always prefer dataset analysis tools before web search.
- Use web_search ONLY when the user explicitly asks to search the internet.
- Never invent tools.
- Use ONLY the provided tools.

When using a tool follow this format:

Thought: reasoning
Action: tool_name
Action Input: input

After the observation you may continue reasoning.

If you have enough information, output:

Final Answer: <your answer>

Do NOT output Action and Final Answer together.
"""
    }


    # ---------------------------------------------------
    # CREATE AGENT
    # ---------------------------------------------------

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs=agent_kwargs,
        verbose=False,
        max_iterations=4,
        early_stopping_method="generate",
        handle_parsing_errors=True
    )

    return agent