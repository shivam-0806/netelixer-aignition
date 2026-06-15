"""Google Ads CSV loader — maps raw schema to unified format."""

import pandas as pd

# Canonical unified schema keys (used for column selection)
UNIFIED_COLS = [
    "date", "channel", "campaign_id", "campaign_name", "campaign_type",
    "spend", "revenue", "clicks", "impressions", "conversions", "daily_budget",
]


def load_google(filepath) -> pd.DataFrame:
    """
    Load a Google Ads campaign stats CSV and return a DataFrame
    conforming to the unified schema.

    Parameters
    ----------
    filepath : str or file-like
        Path to the CSV or an uploaded file object (e.g. from Streamlit).
    """
    df = pd.read_csv(filepath)

    df = df.rename(columns={
        "segments_date":                     "date",
        "metrics_clicks":                    "clicks",
        "metrics_conversions":               "conversions",
        "metrics_cost_micros":               "spend",
        "metrics_impressions":               "impressions",
        "metrics_conversions_value":         "revenue",
        "campaign_advertising_channel_type": "campaign_type",
        "campaign_budget_amount":            "daily_budget",
    })

    # Critical: convert micros to standard currency
    df["spend"] = df["spend"] / 1_000_000
    df["daily_budget"] = df["daily_budget"] / 1_000_000  # budget is also in micros

    df["channel"] = "Google"
    df["campaign_id"] = df["campaign_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"])

    return df[UNIFIED_COLS]
