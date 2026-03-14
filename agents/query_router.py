import pandas as pd

from core.pipeline_intelligence import *
from core.root_cause_analysis import generate_root_cause


def handle_query(query, roster_df, market_df):

    q = query.lower()

    # --------------------------------
    # stuck operations
    # --------------------------------
    if "stuck" in q:

        result = detect_stuck_operations(roster_df)

        explanation = f"Found {len(result)} stuck roster operations in the pipeline."

        return explanation, result


    # --------------------------------
    # pipeline bottlenecks
    # --------------------------------
    if "bottleneck" in q or "slow" in q:

        result = detect_stage_bottlenecks(roster_df)

        explanation = "These roster operations show abnormal stage durations."

        return explanation, result


    # --------------------------------
    # organization failures
    # --------------------------------
    if "organization" in q or "org" in q:

        result = organization_failure_analysis(roster_df)

        explanation = "These organizations have the highest pipeline failure counts."

        return explanation, result


    # --------------------------------
    # source system reliability
    # --------------------------------
    if "source system" in q or "source" in q:

        result = source_system_failure_analysis(roster_df)

        explanation = "Failure rates by source system."

        return explanation, result


    # --------------------------------
    # root cause
    # --------------------------------
    if "why" in q or "root cause" in q:

        result = generate_root_cause(roster_df)

        explanation = "Root cause analysis of pipeline failures."

        return explanation, result


    return "I couldn't understand the query. Try asking about failures, bottlenecks, or stuck pipelines.", None