"""Bing / Microsoft Ads CSV loader — maps raw schema to unified format."""

import pandas as pd

UNIFIED_COLS = [
    "date", "channel", "campaign_id", "campaign_name", "campaign_type",
    "spend", "revenue", "clicks", "impressions", "conversions", "daily_budget",
    "revenue_is_estimated", "revenue_source",
]


def load_bing(filepath) -> pd.DataFrame:
    """
    Load a Bing / MS Ads campaign stats CSV and return a DataFrame
    conforming to the unified schema.

    Parameters
    ----------
    filepath : str or file-like
        Path to the CSV or an uploaded file object.
    """
    df = pd.read_csv(filepath)

    df = df.rename(columns={
        "CampaignId":   "campaign_id",
        "TimePeriod":   "date",
        "Revenue":      "revenue",
        "Spend":        "spend",
        "Clicks":       "clicks",
        "Impressions":  "impressions",
        "Conversions":  "conversions",
        "CampaignType": "campaign_type",
        "DailyBudget":  "daily_budget",
        "CampaignName": "campaign_name",
    })

    df["channel"] = "Bing"
    df["campaign_id"] = df["campaign_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    df["revenue_is_estimated"] = False
    df["revenue_source"] = "Revenue"

    return df[UNIFIED_COLS]
