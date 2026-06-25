"""Quantile Regression via XGBoost — budget-sensitivity simulation."""

import math
import xgboost as xgb
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
            n_estimators=400,          # Increased to learn more complex interactions
            max_depth=5,               # Deeper trees to capture non-linear scale
            learning_rate=0.03,        # Lower rate for smoother convergence
            subsample=0.85,            # Let it see slightly more data per tree
            verbosity=0,
        )
        model.fit(X, y)
        models[q] = model

    return models


def _ordered_quantile_predictions(predictions: dict) -> dict:
    """Clamp predictions to non-negative values and enforce p10 <= p50 <= p90."""
    low = max(0.0, float(predictions[0.10]))
    median = max(0.0, float(predictions[0.50]))
    high = max(0.0, float(predictions[0.90]))
    low, median, high = sorted([low, median, high])
    return {0.10: low, 0.50: median, 0.90: high}


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
    # Original:
    # n_weeks = max(horizon_days // 7, 1)
    # weekly_spend = planned_spend / n_weeks
    #
    # Corrected: include partial weeks and convert exact-period spend into a
    # full-week spend rate. Forecast totals are scaled back to the exact window.
    n_weeks = max(math.ceil(horizon_days / 7), 1)
    equivalent_weeks = horizon_days / 7
    horizon_scale = equivalent_weeks / n_weeks
    weekly_spend = planned_spend / equivalent_weeks if equivalent_weeks else 0

    # Original:
    # Build one synthetic feature row from recent historical averages, predict
    # once, then multiply by n_weeks.
    #
    # Corrected: step through each future week so calendar features such as
    # month, quarter, and week_of_year can change across 60/90-day forecasts.
    channel_df = build_features(channel_df.sort_values("week"))
    last_row = channel_df.iloc[-1]
    spend_history = channel_df["spend"].tolist()
    totals = {q: 0.0 for q in QUANTILES}

    for step in range(1, n_weeks + 1):
        next_week = last_row["week"] + pd.Timedelta(weeks=step)
        recent_spend = spend_history[-4:] if spend_history else [weekly_spend]
        feature_row = {
            "spend": weekly_spend,
            "month": next_week.month,
            "quarter": next_week.quarter,
            "week_of_year": next_week.isocalendar()[1],
            "spend_lag1": spend_history[-1] if spend_history else weekly_spend,
            "spend_lag4": sum(recent_spend) / len(recent_spend),
            "spend_rolling4": sum(recent_spend) / len(recent_spend),
        }
        X_pred = pd.DataFrame([feature_row])
        predictions = {
            q: models[q].predict(X_pred)[0]
            for q in QUANTILES
        }
        ordered = _ordered_quantile_predictions(predictions)
        for q in QUANTILES:
            totals[q] += ordered[q]
        spend_history.append(weekly_spend)

    return {
        "revenue_low":    round(totals[0.10] * horizon_scale, 2),
        "revenue_median": round(totals[0.50] * horizon_scale, 2),
        "revenue_high":   round(totals[0.90] * horizon_scale, 2),
    }
