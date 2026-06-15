"""Meta (Facebook) Ads CSV loader — maps raw schema to unified format."""

import pandas as pd

UNIFIED_COLS = [
    "date", "channel", "campaign_id", "campaign_name", "campaign_type",
    "spend", "revenue", "clicks", "impressions", "conversions", "daily_budget",
]


def infer_meta_campaign_type(name: str) -> str:
    """Infer campaign type from Meta campaign name conventions."""
    if not isinstance(name, str):
        return "Other"
    name_lower = name.lower()
    if "prospecting" in name_lower:
        return "Prospecting"
    elif "retargeting" in name_lower or "remarketing" in name_lower:
        return "Retargeting"
    elif "dpa" in name_lower:
        return "DPA"
    elif "generic" in name_lower:
        return "Generic"
    else:
        return "Other"


def load_meta(filepath) -> pd.DataFrame:
    """
    Load a Meta Ads campaign stats CSV and return a DataFrame
    conforming to the unified schema.

    Notes
    -----
    - campaign_id is read as string to avoid scientific notation truncation.
    - If no ``revenue`` column exists, revenue is derived from conversions
      times ``conversion_value`` (or set to 0).
    - daily_budget may be NaN in Meta exports; filled with 0.

    Parameters
    ----------
    filepath : str or file-like
        Path to the CSV or an uploaded file object.
    """
    df = pd.read_csv(filepath, dtype={"campaign_id": str})

    df = df.rename(columns={
        "date_start":  "date",
        "conversion":  "conversions",
        # spend, clicks, impressions, daily_budget, campaign_name already match
    })

    # Revenue derivation — Meta doesn't always have a direct revenue field
    if "revenue" not in df.columns:
        if "conversion_value" in df.columns:
            df["revenue"] = df["conversions"] * df["conversion_value"]
        else:
            # Fallback: assume an industry ROAS of 2.0
            df["revenue"] = df["spend"] * 2.0

    # Derive campaign_type from campaign_name heuristics
    df["campaign_type"] = df["campaign_name"].apply(infer_meta_campaign_type)

    df["channel"] = "Meta"
    df["campaign_id"] = df["campaign_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"])

    # Fill missing daily_budget
    df["daily_budget"] = df["daily_budget"].fillna(0.0)

    return df[UNIFIED_COLS]
