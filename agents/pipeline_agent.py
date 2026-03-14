import os
from dotenv import load_dotenv

load_dotenv()

from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq

from core.pipeline_intelligence import detect_stuck_operations, detect_stage_bottlenecks


def create_pipeline_agent(roster_df):

    # ------------------------------------------------
    # TOOL 1 — STUCK PIPELINE OPERATIONS
    # ------------------------------------------------
    def stuck_tool(query=""):

        df = detect_stuck_operations(roster_df).head(10)

        if df.empty:
            return "No stuck pipeline operations detected."

        results = []

        for _, row in df.iterrows():
            org = row.get("ORG_NM", "Unknown organization")
            stage = row.get("LATEST_STAGE_NM", "Unknown stage")

            results.append(f"{org} pipeline stuck at stage {stage}")

        return "Detected stuck pipeline operations:\n" + "\n".join(results)


    # ------------------------------------------------
    # TOOL 2 — PIPELINE BOTTLENECK DETECTION
    # ------------------------------------------------
    def bottleneck_tool(query=""):

        df = detect_stage_bottlenecks(roster_df).head(10)

        if df.empty:
            return "No pipeline bottlenecks detected."

        results = []

        for _, row in df.iterrows():

            org = row.get("ORG_NM", "Unknown organization")
            state = row.get("CNT_STATE", "Unknown state")

            duration = row.get("DART_GEN_DURATION", "unknown")
            avg = row.get("AVG_DART_GENERATION_DURATION", "unknown")
            ratio = row.get("DART_GEN_RATIO", "unknown")

            results.append(
                f"{org} ({state}) has slow DART generation "
                f"(duration={duration}, avg={avg}, ratio={ratio})"
            )

        return "Pipeline bottlenecks detected:\n" + "\n".join(results)


    # ------------------------------------------------
    # TOOL DEFINITIONS
    # ------------------------------------------------
    tools = [

        Tool(
            name="get_stuck_operations",
            func=stuck_tool,
            description="""
Use this tool when the user asks about pipelines that are stuck or halted.

Examples:
- Show stuck roster operations
- Which pipelines are stuck
- Jobs not progressing
- Pipelines halted at a stage

This tool returns organizations and the pipeline stage where execution stopped.
"""
        ),

        Tool(
            name="get_pipeline_bottlenecks",
            func=bottleneck_tool,
            description="""
Use this tool when the user asks about slow pipeline processing or performance bottlenecks.

Examples:
- Show pipeline bottlenecks
- Which pipeline stages are slow
- Pipeline performance issues
- Slow pipeline processing
- Long stage execution times

This tool returns organizations where pipeline stages are unusually slow.
"""
        )

    ]


    # ------------------------------------------------
    # LLM INITIALIZATION
    # ------------------------------------------------
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )


    # ------------------------------------------------
    # AGENT PROMPT
    # ------------------------------------------------
    agent_kwargs = {
        "prefix": """
You are an AI assistant specialized in diagnosing provider roster pipeline operations.

You have access to analytical tools that inspect pipeline execution.

Rules:
- Use ONLY the provided tools.
- Never invent tools.
- Choose the most relevant tool depending on the user question.

Guidelines:
- If the pipeline is halted → use get_stuck_operations
- If the pipeline is slow → use get_pipeline_bottlenecks

When using tools follow this format:

Thought: reasoning
Action: tool_name
Action Input: input

After observing the tool output, continue reasoning.

If you have enough information, respond with:

Final Answer: <your answer>

Never output both an Action and Final Answer together.
"""
    }


    # ------------------------------------------------
    # CREATE AGENT
    # ------------------------------------------------
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