"""Feature engineering — prepare forecast-ready aggregated data."""

import pandas as pd


def prepare_forecast_input(df: pd.DataFrame, grain: str = "channel") -> pd.DataFrame:
    """
    Aggregate the unified daily data to weekly granularity per grain.

    Parameters
    ----------
    df : pd.DataFrame
        Unified harmonized DataFrame with daily rows.
    grain : str
        One of "channel", "campaign_type", "campaign".

    Returns
    -------
    pd.DataFrame
        Weekly-aggregated data suitable for Prophet / XGBoost training.
    """
    group_keys = {
        "channel":       ["channel"],
        "campaign_type": ["channel", "campaign_type"],
        "campaign":      ["channel", "campaign_id", "campaign_name"],
    }[grain]

    df = df.copy()
    df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)

    result = (
        df.groupby(group_keys + ["week"])
        .agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
            conversions=("conversions", "sum"),
        )
        .reset_index()
    )

    # Original:
    # result["roas"] = result["revenue"] / result["spend"].replace(0, pd.NA)
    # result["roas"] = result["roas"].fillna(0)
    spend = pd.to_numeric(result["spend"], errors="coerce").replace(0, float("nan"))
    revenue = pd.to_numeric(result["revenue"], errors="coerce")
    result["roas"] = (revenue / spend).fillna(0.0).astype(float)

    return result
