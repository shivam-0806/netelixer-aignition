"""Channel-level metric computations for dashboards and LLM context."""

import pandas as pd


def compute_channel_metrics(df: pd.DataFrame, lookback_days: int = 90) -> pd.DataFrame:
    """
    Compute per-channel summary metrics over the last ``lookback_days`` days.
    Returns a DataFrame with one row per channel.
    """
    cutoff = df["date"].max() - pd.Timedelta(days=lookback_days)
    recent = df[df["date"] >= cutoff]

    metrics = (
        recent.groupby("channel")
        .agg(
            total_spend=("spend", "sum"),
            total_revenue=("revenue", "sum"),
            total_clicks=("clicks", "sum"),
            total_impressions=("impressions", "sum"),
            total_conversions=("conversions", "sum"),
            avg_daily_spend=("spend", "mean"),
            avg_daily_revenue=("revenue", "mean"),
        )
        .reset_index()
    )

    metrics["roas"] = metrics["total_revenue"] / metrics["total_spend"].replace(0, pd.NA)
    metrics["cpc"] = metrics["total_spend"] / metrics["total_clicks"].replace(0, pd.NA)
    metrics["cvr"] = metrics["total_conversions"] / metrics["total_clicks"].replace(0, pd.NA)

    metrics[["roas", "cpc", "cvr"]] = metrics[["roas", "cpc", "cvr"]].fillna(0).round(2)

    return metrics
