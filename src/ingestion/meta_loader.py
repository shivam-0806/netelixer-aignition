"""Meta (Facebook) Ads CSV loader — maps raw schema to unified format."""

import pandas as pd

UNIFIED_COLS = [
    "date", "channel", "campaign_id", "campaign_name", "campaign_type",
    "spend", "revenue", "clicks", "impressions", "conversions", "daily_budget",
    "revenue_is_estimated", "revenue_source",
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


def load_meta(filepath, estimated_aov: float | None = None,
              fallback_roas: float = 2.0) -> pd.DataFrame:
    """
    Load a Meta Ads campaign stats CSV and return a DataFrame
    conforming to the unified schema.

    Notes
    -----
    - campaign_id is read as string to avoid scientific notation truncation.
    - If a direct ``revenue`` column exists, it is treated as observed revenue.
    - If ``conversion_value`` exists, revenue is derived from conversions times
      conversion_value.
    - If neither exists, revenue is estimated from ``estimated_aov`` when
      supplied, otherwise from ``spend * fallback_roas``. This fallback remains
      explicit via ``revenue_is_estimated`` and ``revenue_source``.
    - daily_budget may be NaN in Meta exports; retained as missing.

    Parameters
    ----------
    filepath : str or file-like
        Path to the CSV or an uploaded file object.
    estimated_aov : float, optional
        Average order value to use when Meta has conversions but no revenue.
    fallback_roas : float, default 2.0
        Last-resort ROAS assumption when no AOV or revenue field is available.
    """
    df = pd.read_csv(filepath, dtype={"campaign_id": str})

    df = df.rename(columns={
        "date_start":  "date",
        "conversion":  "conversions",
        # spend, clicks, impressions, daily_budget, campaign_name already match
    })

    # Revenue derivation: Meta exports often do not contain ecommerce revenue.
    if "revenue" in df.columns:
        df["revenue_is_estimated"] = False
        df["revenue_source"] = "revenue"
    elif "conversion_value" in df.columns:
        df["revenue"] = df["conversions"] * df["conversion_value"]
        df["revenue_is_estimated"] = False
        df["revenue_source"] = "conversion_value"
    elif estimated_aov is not None:
        df["revenue"] = df["conversions"] * estimated_aov
        df["revenue_is_estimated"] = True
        df["revenue_source"] = f"estimated_aov_{estimated_aov:g}"
    else:
        df["revenue"] = df["spend"] * fallback_roas
        df["revenue_is_estimated"] = True
        df["revenue_source"] = f"fallback_roas_{fallback_roas:g}"

    # Derive campaign_type from campaign_name heuristics
    df["campaign_type"] = df["campaign_name"].apply(infer_meta_campaign_type)

    df["channel"] = "Meta"
    df["campaign_id"] = df["campaign_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"])

    return df[UNIFIED_COLS]
