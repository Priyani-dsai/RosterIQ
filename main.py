from core.data_loader import DataLoader

loader = DataLoader(
    "data/roster_processing_details.csv",
    "data/aggregated_operational_metrics.csv"
)


roster_df, market_df = loader.load_data()

print("\n=== DATA SUMMARY ===")
print(loader.basic_summary())

from core.pipeline_intelligence import *

print("\nSTUCK OPERATIONS")
print(detect_stuck_operations(roster_df))

print("\nPIPELINE BOTTLENECKS")
print(detect_stage_bottlenecks(roster_df).head())

print("\nTOP FAILURE ORGANIZATIONS")
print(organization_failure_analysis(roster_df).head())

print("\nSOURCE SYSTEM FAILURE RATE")
print(source_system_failure_analysis(roster_df))

from core.root_cause_analysis import generate_root_cause

print("\nROOT CAUSE ANALYSIS")
print(generate_root_cause(roster_df))