"""Data validation — consistency checks for the unified dataset."""

import pandas as pd


def validate_campaigns(df: pd.DataFrame) -> dict:
    """
    Run consistency checks on the unified DataFrame
    and return a structured validation report.
    """
    report = {}

    # Check 1: date range coverage per channel
    date_range = df.groupby("channel")["date"].agg(["min", "max"])
    report["date_range"] = date_range.to_dict("index")

    # Check 2: campaigns with zero revenue but non-zero spend (budget leak)
    budget_leak = df[(df["spend"] > 0) & (df["revenue"] == 0)]
    report["zero_revenue_campaigns"] = (
        budget_leak["campaign_name"].unique().tolist()
    )

    # Check 3: duplicate rows (same campaign + date)
    dupes = df.duplicated(subset=["channel", "campaign_id", "date"], keep=False)
    report["duplicate_rows"] = int(dupes.sum())

    # Check 4: negative spend or revenue
    report["negative_spend_rows"] = int((df["spend"] < 0).sum())
    report["negative_revenue_rows"] = int((df["revenue"] < 0).sum())

    # Check 5: total record counts per channel
    report["record_counts"] = df["channel"].value_counts().to_dict()

    # Check 6: missing values summary
    missing = df.isnull().sum()
    report["missing_values"] = {
        col: int(cnt) for col, cnt in missing.items() if cnt > 0
    }

    return report
