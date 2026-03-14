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

    def org_failure_tool(query=""):
        df = organization_failure_analysis(roster_df).head(10)

        results = []
        for org, count in df.items():
           results.append(f"{org} — {count} failures")

        return "Top organizations causing failures:\n" + "\n".join(results)

    def source_failure_tool(query=""):
        df = source_system_failure_analysis(roster_df)

        results = []
        for system, ratio in df.items():
            results.append(f"{system} — {round(ratio*100,2)}% failure rate")

        return "Source systems contributing most to pipeline failures:\n" + "\n".join(results[:10])

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

    tools = [

        Tool(
            name="get_failure_organizations",
            func=org_failure_tool,
            description="""
Returns the organizations responsible for the highest number of provider roster pipeline failures.

Use this tool when the user asks questions like:
- Which organizations cause most failures?
- Top failing organizations
- Organizations with highest pipeline failure counts

The tool returns a ranked list of organizations and their failure counts.
After calling this tool, summarize the results clearly for the user.
"""
        ),

        Tool(
            name="get_source_system_failures",
            func=source_failure_tool,
            description="""
Identifies which source systems are responsible for pipeline failures.

Use when the user asks questions like:
- Which source systems fail most?
- Source systems causing pipeline failures
- Failure distribution by source system

The tool returns source systems and their failure ratios.
After using the tool, explain which systems contribute most to failures.
"""
        ),

        Tool(
            name="get_root_cause",
            func=root_cause_tool,
            description="""
Performs root cause analysis for provider roster pipeline failures.

Use when the user asks:
- Why are pipelines failing?
- Root cause of failures
- Failure stage analysis

The tool returns insights including failure organizations, failure stages, and source system issues.
After calling the tool, summarize the root causes clearly.
"""
        ),

        Tool(
    name="web_search",
    func=web_search_tool,
    description="""
Search the internet for external technical information.

Use this tool when the user asks about:
- ETL pipelines
- data engineering concepts
- debugging data pipelines
- causes of pipeline bottlenecks
- search something
- look up external information
- research causes or best practices
- find explanations outside the dataset

The tool returns web article titles and links.
After using the tool, summarize the findings for the user.
"""
)

]


    llm = ChatGroq(
       model="llama-3.1-8b-instant",
       temperature=0,
       groq_api_key=os.getenv("GROQ_API_KEY")
     )
    
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