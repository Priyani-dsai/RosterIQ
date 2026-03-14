import os
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

from core.pipeline_intelligence import *
from core.root_cause_analysis import generate_root_cause


def create_agent(roster_df, market_df):

    def stuck_tool(_input: str):
        return detect_stuck_operations(roster_df).head(20).to_string()

    def bottleneck_tool(_input: str):
        return detect_stage_bottlenecks(roster_df).head(20).to_string()

    def org_failure_tool(query: str = ""):
        df = organization_failure_analysis(roster_df).head(10)
        result = "\n".join(
        [f"{org} — {count} failures" for org, count in df.items()])
        
        return f"Top organizations causing failures:\n{result}"
    
    def source_failure_tool(_input: str):
        return source_system_failure_analysis(roster_df).to_string()

    def root_cause_tool(_input: str):
        return str(generate_root_cause(roster_df))


    tools = [

        Tool(
            name="get_stuck_operations",
            func=stuck_tool,
            description="Find stuck roster operations in the pipeline"
        ),

        Tool(
            name="get_pipeline_bottlenecks",
            func=bottleneck_tool,
            description="Find abnormal pipeline stage durations"
        ),

        Tool(
            name="get_failure_organizations",
            func=org_failure_tool,
            description="Find organizations causing pipeline failures"
        ),

        Tool(
            name="get_source_system_failures",
            func=source_failure_tool,
            description="Find source systems causing failures"
        ),

        Tool(
            name="get_root_cause",
            func=root_cause_tool,
            description="Analyze root causes of pipeline failures"
        )
    ]


    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )


    agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    max_iterations=3,
    early_stopping_method="generate",
    handle_parsing_errors=True
)

    return agent