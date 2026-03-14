def detect_stuck_operations(roster_df):

    stuck = roster_df[roster_df["IS_STUCK"] == 1]

    return stuck[
        [
            "RO_ID",
            "ORG_NM",
            "CNT_STATE",
            "SRC_SYS",
            "LATEST_STAGE_NM",
            "IS_FAILED"
        ]
    ]
# This finds jobs currently stuck in pipeline stages.

def detect_stage_bottlenecks(roster_df):

    bottlenecks = roster_df[
        roster_df["DART_GEN_RATIO"] > 2
    ]

    return bottlenecks[
        [
            "RO_ID",
            "ORG_NM",
            "CNT_STATE",
            "DART_GEN_DURATION",
            "AVG_DART_GENERATION_DURATION",
            "DART_GEN_RATIO"
        ]
    ]
# This finds stages running 2× slower than expected.

def organization_failure_analysis(roster_df):

    failures = roster_df.groupby("ORG_NM")["IS_FAILED"].sum()

    failures = failures.sort_values(ascending=False)

    return failures.head(20)

# This shows organizations causing most failures.

def source_system_failure_analysis(roster_df):

    stats = roster_df.groupby("SRC_SYS")["IS_FAILED"].mean()

    stats = stats.sort_values(ascending=False)

    return stats
# This tells us which ingestion systems are unreliable.

def pipeline_health_summary(roster_df):

    health_cols = [
        "PRE_PROCESSING_HEALTH",
        "MAPPING_APROVAL_HEALTH",
        "ISF_GEN_HEALTH",
        "DART_GEN_HEALTH",
        "DART_REVIEW_HEALTH",
        "DART_UI_VALIDATION_HEALTH",
        "SPS_LOAD_HEALTH"
    ]

    summary = {}

    for col in health_cols:
        summary[col] = roster_df[col].value_counts()

    return summary

# This shows the health of each pipeline stage

