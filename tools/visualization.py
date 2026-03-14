import plotly.express as px


# ---------------------------------------------------
# PIPELINE HEALTH HEATMAP
# ---------------------------------------------------
def pipeline_stage_health_heatmap(df):

    health_cols = [
        "PRE_PROCESSING_HEALTH",
        "MAPPING_APROVAL_HEALTH",
        "ISF_GEN_HEALTH",
        "DART_GEN_HEALTH",
        "DART_REVIEW_HEALTH",
        "DART_UI_VALIDATION_HEALTH",
        "SPS_LOAD_HEALTH"
    ]

    heatmap_data = df[health_cols].apply(lambda col: col.value_counts())

    fig = px.imshow(
        heatmap_data.fillna(0),
        title="Pipeline Stage Health Distribution",
        aspect="auto"
    )

    return fig


# ---------------------------------------------------
# STAGE DURATION ANOMALY CHART
# ---------------------------------------------------
def duration_anomaly_chart(df):

    fig = px.scatter(
        df,
        x="AVG_DART_GENERATION_DURATION",
        y="DART_GEN_DURATION",
        color="CNT_STATE",
        title="DART Generation Duration vs Average",
        labels={
            "AVG_DART_GENERATION_DURATION": "Average Duration",
            "DART_GEN_DURATION": "Actual Duration"
        }
    )

    return fig


# ---------------------------------------------------
# FAILURE DISTRIBUTION
# ---------------------------------------------------
def failure_distribution(df):

    failure_counts = (
        df.groupby("ORG_NM")["IS_FAILED"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
    )

    fig = px.bar(
        x=failure_counts.values,
        y=failure_counts.index,
        orientation="h",
        title="Top Organizations by Pipeline Failures",
        labels={"x": "Failure Count", "y": "Organization"}
    )

    fig.update_layout(yaxis={"categoryorder": "total ascending"})

    return fig


# ---------------------------------------------------
# VISUALIZATION ROUTER
# ---------------------------------------------------
def generate_visualization(df, query=""):

    query = query.lower()

    # Failure charts
    if any(word in query for word in [
        "failure",
        "failures",
        "failure distribution",
        "failing organizations"
    ]):
        return failure_distribution(df)

    # Bottleneck / duration charts
    if any(word in query for word in [
        "duration",
        "bottleneck",
        "slow",
        "pipeline delay"
    ]):
        return duration_anomaly_chart(df)

    # Health charts
    if any(word in query for word in [
        "health",
        "stage health",
        "pipeline health"
    ]):
        return pipeline_stage_health_heatmap(df)

    return None