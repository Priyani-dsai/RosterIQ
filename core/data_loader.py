import pandas as pd


class DataLoader:

    def __init__(self, roster_path, market_path):
        self.roster_path = roster_path
        self.market_path = market_path

        self.roster_df = None
        self.market_df = None

    def load_data(self):

        # Load CSVs
        self.roster_df = pd.read_csv(self.roster_path)
        self.market_df = pd.read_csv(self.market_path)

        # Normalize column names
        self.roster_df.columns = (
            self.roster_df.columns
            .str.strip()
            .str.upper()
            .str.replace(" ", "_")
        )

        self.market_df.columns = (
            self.market_df.columns
            .str.strip()
            .str.upper()
            .str.replace(" ", "_")
        )



        # Parse date columns
        self._parse_dates()

        # Create derived analytics columns
        self._create_derived_features()

        print("Roster columns after normalization:")
        print(self.roster_df.columns.tolist())

        return self.roster_df, self.market_df

    def _parse_dates(self):

        roster_date_cols = [
            "FILE_RECEIVED_DT",
            "PRE_PROCESSING_START_DT",
            "PRE_PROCESSING_END_DT",
            "ISF_GEN_START_DT",
            "ISF_GEN_END_DT",
            "DART_GEN_START_DT",
            "DART_GEN_END_DT",
            "SPS_LOAD_START_DT",
            "SPS_LOAD_END_DT"
        ]

        for col in roster_date_cols:
            if col in self.roster_df.columns:
                self.roster_df[col] = pd.to_datetime(
                    self.roster_df[col],
                    errors="coerce"
                )

        if "MONTH" in self.market_df.columns:
            self.market_df["MONTH"] = pd.to_datetime(self.market_df["MONTH"],format="%b-%y",errors="coerce")

    def _create_derived_features(self):

        df = self.roster_df

        # Total pipeline duration
        duration_cols = [
            "PRE_PROCESSING_DURATION",
            "MAPPING_APROVAL_DURATION",
            "ISF_GEN_DURATION",
            "DART_GEN_DURATION",
            "DART_REVIEW_DURATION",
            "DART_UI_VALIDATION_DURATION",
            "SPS_LOAD_DURATION"
        ]

        existing_cols = [c for c in duration_cols if c in df.columns]

        if existing_cols:
            df["TOTAL_PIPELINE_DURATION"] = df[existing_cols].sum(axis=1)

        # Duration anomaly ratios
        if "DART_GEN_DURATION" in df.columns and "AVG_DART_GENERATION_DURATION" in df.columns:
            df["DART_GEN_RATIO"] = (
                df["DART_GEN_DURATION"] /
                df["AVG_DART_GENERATION_DURATION"]
            )

        if "SPS_LOAD_DURATION" in df.columns and "AVG_SPS_LOAD_DURATION" in df.columns:
            df["SPS_LOAD_RATIO"] = (
                df["SPS_LOAD_DURATION"] /
                df["AVG_SPS_LOAD_DURATION"]
            )

        # Flag slow pipeline jobs
        if "TOTAL_PIPELINE_DURATION" in df.columns:
            threshold = df["TOTAL_PIPELINE_DURATION"].median() * 2
            df["IS_PIPELINE_SLOW"] = df["TOTAL_PIPELINE_DURATION"] > threshold

        self.roster_df = df

    def basic_summary(self):

        if self.roster_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        cols = self.roster_df.columns
        summary = {
        "total_roster_files": len(self.roster_df),
        "states": self.roster_df["CNT_STATE"].nunique() if "CNT_STATE" in cols else "missing",
        "organizations": self.roster_df["ORG_NM"].nunique() if "ORG_NM" in cols else "missing",
        "source_systems": self.roster_df["SRC_SYS"].nunique() if "SRC_SYS" in cols else "missing",
        "failed_files": int(self.roster_df["IS_FAILED"].sum()) if "IS_FAILED" in cols else "missing",
        "stuck_files": int(self.roster_df["IS_STUCK"].sum()) if "IS_STUCK" in cols else "missing"
    }

        return summary