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
    # Original:
    # report["zero_revenue_campaigns"] = (
    #     budget_leak["campaign_name"].unique().tolist()
    # )
    report["zero_revenue_campaigns"] = (
        budget_leak["campaign_name"].unique().tolist()
    )
    report["zero_revenue_campaign_count"] = int(budget_leak["campaign_name"].nunique())
    report["zero_revenue_rows"] = int(len(budget_leak))
    report["zero_revenue_top_spend"] = (
        budget_leak.groupby(["channel", "campaign_name"], dropna=False)["spend"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .round(2)
        .reset_index()
        .to_dict("records")
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

    # Check 7: missing observed dates per channel
    missing_dates_by_channel = {}
    for channel, ch_df in df.groupby("channel"):
        expected_dates = pd.date_range(ch_df["date"].min(), ch_df["date"].max(), freq="D")
        observed_dates = pd.DatetimeIndex(ch_df["date"].dropna().unique()).normalize()
        missing_dates_by_channel[channel] = int(len(expected_dates.difference(observed_dates)))
    report["missing_dates_by_channel"] = missing_dates_by_channel

    # Check 8: channel date coverage mismatch
    global_min = df["date"].min()
    global_max = df["date"].max()
    coverage_mismatch = {}
    for channel, values in date_range.to_dict("index").items():
        coverage_mismatch[channel] = {
            "starts_after_global_min_days": int((values["min"] - global_min).days),
            "ends_before_global_max_days": int((global_max - values["max"]).days),
        }
    report["channel_coverage_mismatch"] = coverage_mismatch

    # Check 9: estimated revenue visibility
    if "revenue_is_estimated" in df.columns:
        estimated = df[df["revenue_is_estimated"].fillna(False)]
        report["estimated_revenue_rows"] = int(len(estimated))
        report["estimated_revenue_rows_by_channel"] = (
            estimated["channel"].value_counts().to_dict()
        )
        if "revenue_source" in df.columns:
            report["revenue_sources_by_channel"] = (
                df.groupby("channel")["revenue_source"]
                .agg(lambda values: sorted(map(str, values.dropna().unique())))
                .to_dict()
            )
    else:
        report["estimated_revenue_rows"] = 0
        report["estimated_revenue_rows_by_channel"] = {}
        report["revenue_sources_by_channel"] = {}

    # Check 10: zero spend rows
    zero_spend = df[df["spend"] == 0]
    report["zero_spend_rows"] = int(len(zero_spend))
    report["zero_spend_rows_by_channel"] = zero_spend["channel"].value_counts().to_dict()

    # Check 11: extreme ROAS outliers
    positive_spend = df[df["spend"] > 0].copy()
    positive_spend["row_roas"] = positive_spend["revenue"] / positive_spend["spend"]
    extreme_roas = positive_spend[positive_spend["row_roas"] > 50]
    report["extreme_roas_rows"] = int(len(extreme_roas))
    report["extreme_roas_top_rows"] = (
        extreme_roas.sort_values("row_roas", ascending=False)
        [["channel", "campaign_name", "date", "spend", "revenue", "row_roas"]]
        .head(10)
        .assign(date=lambda rows: rows["date"].dt.strftime("%Y-%m-%d"))
        .round({"spend": 2, "revenue": 2, "row_roas": 2})
        .to_dict("records")
    )

    return report
