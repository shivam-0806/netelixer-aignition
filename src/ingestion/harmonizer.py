"""Harmonizer — merges per-channel DataFrames into a single unified dataset."""

import pandas as pd


def harmonize(google_df: pd.DataFrame, bing_df: pd.DataFrame,
              meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate all channel DataFrames, derive ROAS / CPC / CVR,
    and return a single sorted DataFrame.
    """
    unified = pd.concat([google_df, bing_df, meta_df], ignore_index=True)

    # Derived metrics (used by LLM summary & UI)
    unified["roas"] = unified["revenue"] / unified["spend"].replace(0, pd.NA)
    unified["cpc"]  = unified["spend"]   / unified["clicks"].replace(0, pd.NA)
    unified["cvr"]  = unified["conversions"] / unified["clicks"].replace(0, pd.NA)

    # Fill NaN derived metrics with 0
    unified[["roas", "cpc", "cvr"]] = unified[["roas", "cpc", "cvr"]].fillna(0)

    return unified.sort_values("date").reset_index(drop=True)
