def generate_root_cause(roster_df):

    failures = roster_df[roster_df["IS_FAILED"] == 1]

    org_failures = (
        failures.groupby("ORG_NM")
        .size()
        .sort_values(ascending=False)
        .head(5)
    )

    stage_failures = (
        failures.groupby("LATEST_STAGE_NM")
        .size()
        .sort_values(ascending=False)
    )

    source_failures = (
        failures.groupby("SRC_SYS")
        .size()
        .sort_values(ascending=False)
    )

    return {
        "top_failure_orgs": org_failures,
        "failure_stages": stage_failures,
        "source_systems": source_failures
    }