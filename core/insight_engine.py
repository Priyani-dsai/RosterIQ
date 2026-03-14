import pandas as pd


def generate_pipeline_insights(roster_df):

    insights = []

    # 1. Stage failure analysis
    stage_failures = roster_df["LATEST_STAGE_NM"].value_counts()

    if not stage_failures.empty:
        top_stage = stage_failures.idxmax()
        count = stage_failures.max()

        insights.append(
            f"Pipeline Insight: The stage '{top_stage}' contributes the highest number of failures ({count})."
        )

    # 2. Organization failures
    org_failures = roster_df["ORG_NM"].value_counts()

    if not org_failures.empty:
        top_org = org_failures.idxmax()
        count = org_failures.max()

        insights.append(
            f"Operational Risk: Organization '{top_org}' is responsible for the highest failure count ({count})."
        )

    # 3. Source system failures
    src_failures = roster_df["SRC_SYS"].value_counts()

    if not src_failures.empty:
        top_src = src_failures.idxmax()
        count = src_failures.max()

        insights.append(
            f"System Warning: Source system '{top_src}' contributes the highest number of pipeline failures ({count})."
        )

    return insights