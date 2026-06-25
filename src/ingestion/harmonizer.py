"""Harmonizer — merges per-channel DataFrames into a single unified dataset."""

import pandas as pd


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return numeric ratio with zero/invalid denominators converted to 0.0."""
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, float("nan"))
    return (numerator / denominator).fillna(0.0).astype(float)


def harmonize(google_df: pd.DataFrame, bing_df: pd.DataFrame,
              meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate all channel DataFrames, derive ROAS / CPC / CVR,
    and return a single sorted DataFrame.
    """
    unified = pd.concat([google_df, bing_df, meta_df], ignore_index=True)

    # Derived metrics (used by LLM summary & UI)
    unified["roas"] = _safe_divide(unified["revenue"], unified["spend"])
    unified["cpc"] = _safe_divide(unified["spend"], unified["clicks"])
    unified["cvr"] = _safe_divide(unified["conversions"], unified["clicks"])

    return unified.sort_values("date").reset_index(drop=True)
