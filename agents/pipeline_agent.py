from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq

from core.pipeline_intelligence import detect_stuck_operations, detect_stage_bottlenecks


def create_pipeline_agent(roster_df):

    # -------------------------
    # Tool 1: Stuck Operations
    # -------------------------
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


    # -------------------------
    # Tool 2: Pipeline Bottlenecks
    # -------------------------
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


    # -------------------------
    # Tools registry
    # -------------------------
    tools = [

    Tool(
        name="get_stuck_operations",
        func=stuck_tool,
        description="""
Use this tool ONLY to detect roster pipeline jobs that are stuck or not progressing.

A pipeline is considered stuck when it is halted at a stage and not moving forward.

Use this tool when the user asks:
- Show stuck roster operations
- Which pipelines are stuck
- Jobs not progressing
- Pipelines halted at a stage
- Pipeline jobs not completing

DO NOT use this tool for slow pipelines or performance issues.

The tool returns a list of organizations and the pipeline stage where processing is stuck.
After calling this tool, clearly explain which operations are stuck.
"""
    ),

    Tool(
        name="get_pipeline_bottlenecks",
        func=bottleneck_tool,
        description="""
Use this tool ONLY to detect pipeline performance bottlenecks or slow processing stages.

A bottleneck means the pipeline is working but taking unusually long to process.

Use this tool when the user asks:
- Show pipeline bottlenecks
- Which pipeline stages are slow
- Pipeline performance issues
- Stage delays
- Slow pipeline processing
- Long processing times

DO NOT use this tool for pipelines that are stuck or halted.

The tool returns organizations and stage duration metrics indicating abnormal processing delays.
After calling this tool, summarize which pipelines or stages are slow.
"""
    )

]


    # -------------------------
    # LLM
    # -------------------------
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )


    # -------------------------
    # Agent
    # -------------------------
    agent_kwargs = {
    "prefix": """
You are an AI assistant that analyzes provider roster pipeline data.

You must use ONLY the tools provided below.

Rules:
- Never invent new tools.
- Only use tools that are explicitly listed.
- If a tool is required, you MUST follow this format:

Thought: what you should do
Action: tool_name
Action Input: input to the tool

After the tool returns an Observation, you must decide the next step.

If you have enough information, respond ONLY with:

Final Answer: <your answer>

Never output both an Action and Final Answer in the same step.
"""
}
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