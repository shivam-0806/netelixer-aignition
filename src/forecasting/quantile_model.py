"""Quantile Regression via XGBoost — budget-sensitivity simulation."""

import xgboost as xgb
import numpy as np
import pandas as pd

QUANTILES = [0.10, 0.50, 0.90]

FEATURE_COLS = [
    "spend", "month", "quarter", "week_of_year",
    "spend_lag1", "spend_lag4", "spend_rolling4",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for the quantile model."""
    df = df.copy()
    df["month"]          = df["week"].dt.month
    df["quarter"]        = df["week"].dt.quarter
    df["week_of_year"]   = df["week"].dt.isocalendar().week.astype(int)
    df["spend_lag1"]     = df["spend"].shift(1).fillna(df["spend"].mean())
    df["spend_lag4"]     = df["spend"].shift(4).fillna(df["spend"].mean())
    df["spend_rolling4"] = df["spend"].rolling(4, min_periods=1).mean()
    return df


def train_quantile_models(weekly_df: pd.DataFrame, channel: str) -> dict:
    """
    Train XGBoost quantile regression models (P10, P50, P90) for a channel.

    Parameters
    ----------
    weekly_df : pd.DataFrame
        Output of ``prepare_forecast_input()``.
    channel : str
        Channel name.

    Returns
    -------
    dict
        Mapping {quantile_value: trained_xgb_model}.
    """
    channel_df = weekly_df[weekly_df["channel"] == channel].sort_values("week").copy()
    channel_df = build_features(channel_df)

    X = channel_df[FEATURE_COLS]
    y = channel_df["revenue"]

    models = {}
    for q in QUANTILES:
        model = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=q,
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            verbosity=0,
        )
        model.fit(X, y)
        models[q] = model

    return models


def simulate_budget(models: dict, planned_spend: float,
                    horizon_days: int, channel_df: pd.DataFrame) -> dict:
    """
    Simulate revenue outcomes for a given planned spend and horizon.

    Parameters
    ----------
    models : dict
        Output of ``train_quantile_models()``.
    planned_spend : float
        Total planned spend for the horizon.
    horizon_days : int
        30, 60, or 90.
    channel_df : pd.DataFrame
        Weekly channel data (with build_features already applied).

    Returns
    -------
    dict
        Keys: revenue_low, revenue_median, revenue_high
    """
    n_weeks = max(horizon_days // 7, 1)
    weekly_spend = planned_spend / n_weeks

    # Build feature row from recent historical averages
    channel_df = build_features(channel_df.sort_values("week"))
    last_row = channel_df.iloc[-1]

    next_week = last_row["week"] + pd.Timedelta(weeks=1)
    feature_row = {
        "spend":           weekly_spend,
        "month":           next_week.month,
        "quarter":         next_week.quarter,
        "week_of_year":    next_week.isocalendar()[1],
        "spend_lag1":      last_row["spend"],
        "spend_lag4":      channel_df["spend"].iloc[-4:].mean(),
        "spend_rolling4":  channel_df["spend"].iloc[-4:].mean(),
    }
    X_pred = pd.DataFrame([feature_row])

    weekly_predictions = {q: max(0, models[q].predict(X_pred)[0]) for q in QUANTILES}

    return {
        "revenue_low":    round(weekly_predictions[0.10] * n_weeks, 2),
        "revenue_median": round(weekly_predictions[0.50] * n_weeks, 2),
        "revenue_high":   round(weekly_predictions[0.90] * n_weeks, 2),
    }
