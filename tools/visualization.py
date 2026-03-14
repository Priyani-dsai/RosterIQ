import plotly.express as px

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
        title="Pipeline Stage Health Distribution"
    )

    return fig


def duration_anomaly_chart(df):

    fig = px.scatter(
        df,
        x="AVG_DART_GENERATION_DURATION",
        y="DART_GEN_DURATION",
        color="CNT_STATE",
        title="DART Generation Duration vs Average"
    )

    return fig

def failure_distribution(df):

    failure_counts = df.groupby("ORG_NM")["IS_FAILED"].sum().sort_values(ascending=False).head(20)

    fig = px.bar(
        failure_counts,
        title="Top Organizations by Pipeline Failures"
    )

    return fig

def generate_visualization(df, query=""):

    query = query.lower()

    if "failure" in query:
        return failure_distribution(df)

    if "duration" in query or "bottleneck" in query:
        return duration_anomaly_chart(df)

    if "health" in query or "stage health" in query:
        return pipeline_stage_health_heatmap(df)

    return None