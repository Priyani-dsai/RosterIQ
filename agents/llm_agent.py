from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain_groq import ChatGroq

from core.pipeline_intelligence import *
from core.root_cause_analysis import generate_root_cause


def create_agent(roster_df, market_df):

    def stuck_tool(_):
        return detect_stuck_operations(roster_df).head(20).to_string()

    def bottleneck_tool(_):
        return detect_stage_bottlenecks(roster_df).head(20).to_string()

    def org_failure_tool(_):
        return organization_failure_analysis(roster_df).head(10).to_string()

    def source_failure_tool(_):
        return source_system_failure_analysis(roster_df).to_string()

    def root_cause_tool(_):
        return str(generate_root_cause(roster_df))

    tools = [

        Tool(
            name="get_stuck_operations",
            func=stuck_tool,
            description="Find stuck roster operations"
        ),

        Tool(
            name="get_pipeline_bottlenecks",
            func=bottleneck_tool,
            description="Find abnormal pipeline stage durations"
        ),

        Tool(
            name="get_failure_organizations",
            func=org_failure_tool,
            description="Find organizations with highest failures"
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
        model="llama3-70b-8192",
        temperature=0
    )

    agent = initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description",
        verbose=True
    )

    return agent