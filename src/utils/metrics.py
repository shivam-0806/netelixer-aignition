"""Channel-level metric computations for dashboards and LLM context."""

import numpy as np
import pandas as pd


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return numeric ratio with zero/invalid denominators converted to 0.0."""
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return (numerator / denominator).fillna(0.0).astype(float)


def compute_channel_metrics(df: pd.DataFrame, lookback_days: int = 90) -> pd.DataFrame:
    """
    Compute per-channel summary metrics over the last ``lookback_days`` days.
    Returns a DataFrame with one row per channel.
    """
    cutoff = df["date"].max() - pd.Timedelta(days=lookback_days)
    recent = df[df["date"] >= cutoff]

    agg_spec = {
        "total_spend": ("spend", "sum"),
        "total_revenue": ("revenue", "sum"),
        "total_clicks": ("clicks", "sum"),
        "total_impressions": ("impressions", "sum"),
        "total_conversions": ("conversions", "sum"),
        "avg_daily_spend": ("spend", "mean"),
        "avg_daily_revenue": ("revenue", "mean"),
    }

    if "revenue_is_estimated" in recent.columns:
        agg_spec["estimated_revenue_rows"] = ("revenue_is_estimated", "sum")

    metrics = recent.groupby("channel").agg(**agg_spec).reset_index()

    # Original:
    # metrics["roas"] = metrics["total_revenue"] / metrics["total_spend"].replace(0, pd.NA)
    # metrics["cpc"] = metrics["total_spend"] / metrics["total_clicks"].replace(0, pd.NA)
    # metrics["cvr"] = metrics["total_conversions"] / metrics["total_clicks"].replace(0, pd.NA)
    # metrics[["roas", "cpc", "cvr"]] = metrics[["roas", "cpc", "cvr"]].fillna(0).round(2)
    metrics["roas"] = _safe_divide(metrics["total_revenue"], metrics["total_spend"])
    metrics["cpc"] = _safe_divide(metrics["total_spend"], metrics["total_clicks"])
    metrics["cvr"] = _safe_divide(metrics["total_conversions"], metrics["total_clicks"])

    if "estimated_revenue_rows" not in metrics.columns:
        metrics["estimated_revenue_rows"] = 0

    channel_rows = recent.groupby("channel").size().rename("rows_in_lookback")
    metrics = metrics.merge(channel_rows, on="channel", how="left")
    metrics["estimated_revenue_rows"] = metrics["estimated_revenue_rows"].astype(int)
    metrics["rows_in_lookback"] = metrics["rows_in_lookback"].astype(int)
    metrics["revenue_is_estimated"] = metrics["estimated_revenue_rows"] > 0
    metrics["estimated_revenue_share"] = _safe_divide(
        metrics["estimated_revenue_rows"], metrics["rows_in_lookback"]
    )

    if "revenue_source" in recent.columns:
        sources = (
            recent.groupby("channel")["revenue_source"]
            .agg(lambda values: ", ".join(sorted(map(str, values.dropna().unique()))))
            .rename("revenue_source")
        )
        metrics = metrics.merge(sources, on="channel", how="left")
    else:
        metrics["revenue_source"] = "unknown"

    metrics[["roas", "cpc", "cvr", "estimated_revenue_share"]] = (
        metrics[["roas", "cpc", "cvr", "estimated_revenue_share"]].round(2)
    )

    return metrics


def calculate_backtest_metrics(backtest_results: list[dict]) -> dict:
    """
    Evaluates forecast accuracy across multiple historical backtest windows.
    
    Expected format for backtest_results:
    [
        {"actual": 50000, "p10": 45000, "p50": 51000, "p90": 58000},
        ...
    ]
    """
    if not backtest_results:
        return {}

    errors = []
    coverage_count = 0
    
    for res in backtest_results:
        actual = res["actual"]
        median = res["p50"]
        low = res["p10"]
        high = res["p90"]
        
        # Calculate Absolute Percentage Error for this window
        if actual > 0:
            error = abs(actual - median) / actual
            errors.append(error)
            
        # Check if actual falls within the 80% confidence interval
        if low <= actual <= high:
            coverage_count += 1
            
    aape = float(np.mean(errors)) if errors else 0.0
    coverage_prob = coverage_count / len(backtest_results)
    
    return {
        "aape": float(round(aape, 4)),               # Aim for < 0.15 (15% error)
        "coverage_prob": float(round(coverage_prob, 4)), # Aim for ~ 0.80 (80% coverage)
        "windows_tested": len(backtest_results)
    }
